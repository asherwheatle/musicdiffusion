"""Multi-song DEAM dataset with heuristic mood labels.

DEAM ships valence/arousal annotations, but they are not in this repo, so
moods are derived from audio features instead:
  arousal proxy = tempo + RMS energy (median split over the loaded set)
  valence proxy = spectral centroid / brightness (median split)
Quadrants map to the mood vocabulary used for text conditioning.
"""

import glob
import os

import numpy as np
import torch
import librosa
from tqdm import tqdm

from pipeline import bigvgan_mel_spectrogram, FixedMelNormalizer


MOOD_QUADRANTS = {
    (True, True): "happy and uplifting",       # high arousal, high valence
    (True, False): "energetic and powerful",   # high arousal, low valence
    (False, True): "calm and peaceful",        # low arousal, high valence
    (False, False): "sad and melancholic",     # low arousal, low valence
}


def _song_features(wav: np.ndarray, sr: int):
    tempo = float(np.atleast_1d(librosa.feature.tempo(y=wav, sr=sr))[0])
    rms = float(librosa.feature.rms(y=wav).mean())
    centroid = float(librosa.feature.spectral_centroid(y=wav, sr=sr).mean())
    return tempo, rms, centroid


def label_moods(features: list[tuple]) -> list[str]:
    """Assign a mood text per song from median splits of the feature set."""
    tempos = np.array([f[0] for f in features])
    rmss = np.array([f[1] for f in features])
    cents = np.array([f[2] for f in features])

    # z-score so tempo and energy contribute equally to arousal
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
            labels.append("dark and mysterious")  # darkest low-arousal songs
        else:
            labels.append(MOOD_QUADRANTS[(bool(a), bool(v))])
    return labels


def build_dataset(audio_dir: str, n_songs: int, bigvgan_model,
                  clip_seconds: int = 15):
    """
    Load n_songs DEAM clips evenly spread across the collection.

    Returns:
        mel_batch: (N, 1, n_mels, T) globally-normalized mel spectrograms
        wavs_np: list of N (T_samples,) float32 waveforms
        mood_texts: list of N mood label strings
        names: list of N filenames
    """
    sr = bigvgan_model.h.sampling_rate
    files = sorted(glob.glob(os.path.join(audio_dir, "*.mp3")))
    if not files:
        raise FileNotFoundError(f"No MP3 files found in {audio_dir}")

    # Evenly spaced indices for stylistic diversity
    idxs = np.unique(np.linspace(0, len(files) - 1, n_songs).astype(int))
    max_samples = clip_seconds * sr

    normalizer = FixedMelNormalizer()
    mels, wavs_np, features, names = [], [], [], []
    for i in tqdm(idxs, desc="Loading songs"):
        path = files[i]
        wav, _ = librosa.load(path, sr=sr, mono=True)
        if len(wav) > max_samples:
            wav = wav[:max_samples]
        elif len(wav) < max_samples:
            wav = np.pad(wav, (0, max_samples - len(wav)))

        wav_tensor = torch.FloatTensor(wav).unsqueeze(0)
        mel = bigvgan_mel_spectrogram(wav_tensor, bigvgan_model)  # (1, M, T)
        mels.append(normalizer.normalize(mel))
        wavs_np.append(wav)
        features.append(_song_features(wav, sr))
        names.append(os.path.basename(path))

    mood_texts = label_moods(features)

    from collections import Counter
    print(f"[DATA] Loaded {len(mels)} songs, mood distribution: "
          f"{dict(Counter(mood_texts))}")

    mel_batch = torch.stack(mels, dim=0)  # (N, 1, M, T)
    return mel_batch, wavs_np, mood_texts, names
