"""Multi-song DEAM dataset labeled with the official valence/arousal annotations.

Moods come from DEAM's human annotations (see annotations.py) when an
annotations directory is available. The old audio-feature heuristic
(tempo/RMS/centroid median splits) is kept only as a fallback for running
without the annotation CSVs.

Because the full dataset is ~1800 songs, this module also:
  * clips each song from 15 s onward, matching the region the dynamic
    annotations actually cover (the first 15 s of DEAM is unannotated)
  * cuts each song into `clips_per_song` consecutive short clips, each
    labeled with the valence/arousal of its own window, and shuffles the
    result so no mood is clustered anywhere in the tensor
  * extracts melody (top-k CQT) during loading so raw waveforms don't
    have to be kept in memory
  * caches mels + melodies + labels to disk, so re-runs skip the
    ~30 min mp3 decode + CQT pass
"""

import glob
import os

import numpy as np
import torch
import librosa
from tqdm import tqdm

from pipeline import bigvgan_mel_spectrogram, FixedMelNormalizer
from annotations import (load_annotation_windows, mood_from_va,
                         song_id_from_filename, MOOD_QUADRANTS)


def _heuristic_song_features(wav: np.ndarray, sr: int):
    tempo = float(np.atleast_1d(librosa.feature.tempo(y=wav, sr=sr))[0])
    rms = float(librosa.feature.rms(y=wav).mean())
    centroid = float(librosa.feature.spectral_centroid(y=wav, sr=sr).mean())
    return tempo, rms, centroid


def _heuristic_labels(features: list) -> list:
    """Fallback: mood per song from median splits of audio features."""
    tempos = np.array([f[0] for f in features])
    rmss = np.array([f[1] for f in features])
    cents = np.array([f[2] for f in features])

    def z(x):
        return (x - x.mean()) / (x.std() + 1e-8)

    arousal = z(tempos) + z(rmss)
    valence = z(cents)

    high_a = arousal > np.median(arousal)
    high_v = valence > np.median(valence)

    labels = []
    dark_cutoff = np.percentile(valence, 25)
    for a, v, val in zip(high_a, high_v, valence):
        if not a and val < dark_cutoff:
            labels.append("dark and mysterious")
        else:
            labels.append(MOOD_QUADRANTS[(bool(a), bool(v))])
    return labels


def _load_span(path: str, sr: int, start_seconds: float,
               span_seconds: float):
    """Load one song's [start, start+span) region as mono audio.

    Returns:
        wav: (span_samples,) zero-padded to the full span
        real_samples: how many samples are actual audio (rest is padding)
    """
    wav, _ = librosa.load(path, sr=sr, mono=True,
                          offset=start_seconds, duration=span_seconds)
    if len(wav) == 0:  # song shorter than the offset: take from the start
        wav, _ = librosa.load(path, sr=sr, mono=True, duration=span_seconds)
    max_samples = int(span_seconds * sr)
    if len(wav) > max_samples:
        wav = wav[:max_samples]
    real_samples = len(wav)
    if len(wav) < max_samples:
        wav = np.pad(wav, (0, max_samples - len(wav)))
    return wav, real_samples


def _cache_path(cache_dir: str, n_songs, clip_start: float, clip_len: float,
                clips_per_song: int, labeled: bool) -> str:
    tag = "all" if n_songs is None else str(n_songs)
    src = "annot" if labeled else "heur"
    return os.path.join(
        cache_dir,
        f"deam_{tag}songs_{clip_start:g}s+{clip_len:g}sx{clips_per_song}"
        f"_{src}.npz")


def build_dataset(audio_dir: str, n_songs, bigvgan_model,
                  clip_seconds: float = 5, annotations_dir: str = None,
                  clip_start_seconds: float = 15, melody_extractor=None,
                  cache_dir: str = None, clips_per_song: int = 1,
                  shuffle_seed: int = 0):
    """
    Load DEAM clips with annotation-derived mood labels.

    Each song contributes up to `clips_per_song` consecutive clips of
    `clip_seconds`, each labeled with the valence/arousal averaged over
    that clip's own window (dynamic annotations are per-second, so one
    song can yield e.g. both a "happy" and a "sad" clip). The finished
    dataset is shuffled with a fixed seed so clips from the same song —
    and runs of the same mood — never sit next to each other.

    Args:
        n_songs: number of songs to use, or None for the entire dataset.
                 Subsets are spread evenly across the collection.
        annotations_dir: root of the DEAM annotations download. If None or
                 the files are missing, falls back to heuristic labels.
        clip_start_seconds: where the first clip starts. Default 15 s,
                 the start of the annotated region.
        melody_extractor: optional MelodyExtractor; when given, melodies
                 are computed here and returned instead of raw waveforms.
        cache_dir: optional directory for caching the processed dataset.
        clips_per_song: consecutive clips cut from each song. Clips that
                 fall mostly past the end of a short song are dropped.

    Returns:
        mel_batch: (N, 1, n_mels, T) globally-normalized mel spectrograms (CPU)
        melodies: (N, top_k, T_cqt) long tensor, or None if no extractor
        mood_texts: list of N mood label strings
        names: list of N "filename#clip" strings
    """
    sr = bigvgan_model.h.sampling_rate
    windows = [(clip_start_seconds + k * clip_seconds,
                clip_start_seconds + (k + 1) * clip_seconds)
               for k in range(clips_per_song)]

    va = None
    if annotations_dir:
        try:
            va = load_annotation_windows(annotations_dir, windows)
        except FileNotFoundError as e:
            print(f"[DATA] WARNING: {e}\n"
                  f"[DATA] Falling back to heuristic mood labels — "
                  f"conditioning will be much weaker.")

    cache_file = None
    if cache_dir:
        cache_file = _cache_path(cache_dir, n_songs, clip_start_seconds,
                                 clip_seconds, clips_per_song,
                                 labeled=va is not None)
        if os.path.exists(cache_file):
            print(f"[DATA] Loading cached dataset: {cache_file}")
            data = np.load(cache_file, allow_pickle=False)
            mel_batch = torch.from_numpy(data["mels"].astype(np.float32))
            melodies = (torch.from_numpy(data["melodies"].astype(np.int64))
                        if "melodies" in data else None)
            mood_texts = [str(s) for s in data["moods"]]
            names = [str(s) for s in data["names"]]
            _print_distribution(mood_texts, len(names))
            return mel_batch, melodies, mood_texts, names

    files = sorted(glob.glob(os.path.join(audio_dir, "*.mp3")))
    if not files:
        raise FileNotFoundError(f"No MP3 files found in {audio_dir}")

    if va is not None:
        annotated = [f for f in files if song_id_from_filename(f) in va]
        skipped = len(files) - len(annotated)
        if skipped:
            print(f"[DATA] Skipping {skipped} songs without annotations")
        files = annotated

    if n_songs is not None and n_songs < len(files):
        idxs = np.unique(np.linspace(0, len(files) - 1, n_songs).astype(int))
        files = [files[i] for i in idxs]

    normalizer = FixedMelNormalizer()
    seg_samples = int(clip_seconds * sr)
    span_seconds = clips_per_song * clip_seconds
    mels, melodies, features, mood_texts, names = [], [], [], [], []
    for path in tqdm(files, desc="Loading songs"):
        span, real_samples = _load_span(path, sr, clip_start_seconds,
                                        span_seconds)
        for k in range(clips_per_song):
            # Keep a clip only if at least half of it is real audio
            # (always keep the first so short songs still contribute)
            if k > 0 and real_samples - k * seg_samples < seg_samples // 2:
                break
            wav = span[k * seg_samples:(k + 1) * seg_samples]

            wav_tensor = torch.FloatTensor(wav).unsqueeze(0)
            mel = bigvgan_mel_spectrogram(wav_tensor, bigvgan_model)  # (1, M, T)
            mels.append(normalizer.normalize(mel))

            if melody_extractor is not None:
                melodies.append(
                    torch.from_numpy(melody_extractor.extract(wav)))

            if va is not None:
                valence, arousal = va[song_id_from_filename(path)][k]
                mood_texts.append(mood_from_va(valence, arousal))
            else:
                features.append(_heuristic_song_features(wav, sr))
            names.append(f"{os.path.basename(path)}#{k}")

    if va is None:
        mood_texts = _heuristic_labels(features)

    # Shuffle so consecutive clips never share a song or a mood run;
    # fixed seed keeps the order (and the cache) reproducible.
    order = np.random.default_rng(shuffle_seed).permutation(len(names))
    mels = [mels[i] for i in order]
    if melodies:
        melodies = [melodies[i] for i in order]
    mood_texts = [mood_texts[i] for i in order]
    names = [names[i] for i in order]

    mel_batch = torch.stack(mels, dim=0)  # (N, 1, M, T)
    melody_batch = torch.stack(melodies, dim=0) if melodies else None

    if cache_file:
        os.makedirs(cache_dir, exist_ok=True)
        arrays = {
            "mels": mel_batch.numpy().astype(np.float16),  # [0,1], fp16 is fine
            "moods": np.array(mood_texts),
            "names": np.array(names),
        }
        if melody_batch is not None:
            arrays["melodies"] = melody_batch.numpy().astype(np.int16)
        np.savez(cache_file, **arrays)
        print(f"[DATA] Cached dataset to {cache_file}")

    _print_distribution(mood_texts, len(names))
    return mel_batch, melody_batch, mood_texts, names


def _print_distribution(mood_texts, n):
    from collections import Counter
    print(f"[DATA] Loaded {n} clips, mood distribution: "
          f"{dict(Counter(mood_texts))}")
