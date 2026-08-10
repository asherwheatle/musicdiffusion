"""Multi-song DEAM dataset labeled with the official valence/arousal annotations.

Moods come from DEAM's human annotations (see annotations.py) when an
annotations directory is available. The old audio-feature heuristic
(tempo/RMS/centroid median splits) is kept only as a fallback for running
without the annotation CSVs.

Because the full dataset is ~1800 songs, this module also:
  * clips each song from 15 s onward, matching the region the dynamic
    annotations actually cover (the first 15 s of DEAM is unannotated)
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
from annotations import (load_annotations, mood_from_va, song_id_from_filename,
                         MOOD_QUADRANTS)


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


def _load_clip(path: str, sr: int, clip_start_seconds: float,
               clip_seconds: float) -> np.ndarray:
    """Load one song as a fixed-length mono clip starting at clip_start."""
    wav, _ = librosa.load(path, sr=sr, mono=True,
                          offset=clip_start_seconds, duration=clip_seconds)
    if len(wav) == 0:  # song shorter than the offset: take from the start
        wav, _ = librosa.load(path, sr=sr, mono=True, duration=clip_seconds)
    max_samples = int(clip_seconds * sr)
    if len(wav) > max_samples:
        wav = wav[:max_samples]
    elif len(wav) < max_samples:
        wav = np.pad(wav, (0, max_samples - len(wav)))
    return wav


def _cache_path(cache_dir: str, n_songs, clip_start: float, clip_len: float,
                labeled: bool) -> str:
    tag = "all" if n_songs is None else str(n_songs)
    src = "annot" if labeled else "heur"
    return os.path.join(
        cache_dir, f"deam_{tag}songs_{clip_start:g}s+{clip_len:g}s_{src}.npz")


def build_dataset(audio_dir: str, n_songs, bigvgan_model,
                  clip_seconds: float = 15, annotations_dir: str = None,
                  clip_start_seconds: float = 15, melody_extractor=None,
                  cache_dir: str = None):
    """
    Load DEAM clips with annotation-derived mood labels.

    Args:
        n_songs: number of songs to use, or None for the entire dataset.
                 Subsets are spread evenly across the collection.
        annotations_dir: root of the DEAM annotations download. If None or
                 the files are missing, falls back to heuristic labels.
        clip_start_seconds: where each training clip starts. Default 15 s,
                 the start of the annotated region.
        melody_extractor: optional MelodyExtractor; when given, melodies
                 are computed here and returned instead of raw waveforms.
        cache_dir: optional directory for caching the processed dataset.

    Returns:
        mel_batch: (N, 1, n_mels, T) globally-normalized mel spectrograms (CPU)
        melodies: (N, top_k, T_cqt) long tensor, or None if no extractor
        mood_texts: list of N mood label strings
        names: list of N filenames
    """
    sr = bigvgan_model.h.sampling_rate

    va = None
    if annotations_dir:
        try:
            va = load_annotations(annotations_dir, clip_start_seconds,
                                  clip_seconds)
        except FileNotFoundError as e:
            print(f"[DATA] WARNING: {e}\n"
                  f"[DATA] Falling back to heuristic mood labels — "
                  f"conditioning will be much weaker.")

    cache_file = None
    if cache_dir:
        cache_file = _cache_path(cache_dir, n_songs, clip_start_seconds,
                                 clip_seconds, labeled=va is not None)
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
    mels, melodies, features, mood_texts, names = [], [], [], [], []
    for path in tqdm(files, desc="Loading songs"):
        wav = _load_clip(path, sr, clip_start_seconds, clip_seconds)

        wav_tensor = torch.FloatTensor(wav).unsqueeze(0)
        mel = bigvgan_mel_spectrogram(wav_tensor, bigvgan_model)  # (1, M, T)
        mels.append(normalizer.normalize(mel))

        if melody_extractor is not None:
            melodies.append(torch.from_numpy(melody_extractor.extract(wav)))

        if va is not None:
            valence, arousal = va[song_id_from_filename(path)]
            mood_texts.append(mood_from_va(valence, arousal))
        else:
            features.append(_heuristic_song_features(wav, sr))
        names.append(os.path.basename(path))

    if va is None:
        mood_texts = _heuristic_labels(features)

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
    print(f"[DATA] Loaded {n} songs, mood distribution: "
          f"{dict(Counter(mood_texts))}")
