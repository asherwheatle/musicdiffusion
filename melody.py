"""Melody extraction and encoding (paper section III-B: Top-k CQT)."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
from scipy.signal import butter, sosfilt


class MelodyExtractor:
    """
    Extract top-k CQT melody features from mono audio.

    Pipeline:
      1. Biquadratic high-pass filter at Middle C (261.2 Hz)
      2. CQT: 128 bins, 12 bins/octave, hop=512, fmin=8.18 Hz (MIDI 0)
      3. Top-k bins per frame by magnitude
      4. Return pitch-bin indices (1-128) per frame
    """

    def __init__(self, sr=44100, n_bins=128, bins_per_octave=12,
                 hop_length=512, fmin=8.18, top_k=4, highpass_cutoff=261.2):
        self.sr = sr
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave
        self.hop_length = hop_length
        self.fmin = fmin
        self.top_k = top_k
        self.highpass_cutoff = highpass_cutoff

    def _highpass(self, wav: np.ndarray) -> np.ndarray:
        sos = butter(2, self.highpass_cutoff, btype="high",
                     fs=self.sr, output="sos")
        return sosfilt(sos, wav).astype(np.float32)

    def extract(self, waveform: np.ndarray) -> np.ndarray:
        """
        Args:
            waveform: (T_samples,) mono float32 in [-1, 1]
        Returns:
            melody_indices: (top_k, T_cqt) int array, values in [1, 128]
        """
        filtered = self._highpass(waveform)
        cqt = librosa.cqt(
            filtered, sr=self.sr, hop_length=self.hop_length,
            n_bins=self.n_bins, bins_per_octave=self.bins_per_octave,
            fmin=self.fmin,
        )
        mag = np.abs(cqt)  # (128, T_cqt)
        top_k_idx = np.argsort(mag, axis=0)[-self.top_k:][::-1]  # (top_k, T_cqt)
        return (top_k_idx + 1).astype(np.int64)  # 1-indexed pitch bins


class MelodyEncoder(nn.Module):
    """
    Embed melody pitch indices and downsample to match latent temporal resolution.

    Paper section III-B step 5:
      nn.Embedding(129, d_melody) for pitches (128 + 1 mask token at index 0)
      followed by 1D conv stack to downsample from CQT frame rate to latent rate.

    Output is interpolated to exactly match the latent's temporal width (W_lat).
    """

    def __init__(self, d_model=256, top_k=4):
        super().__init__()
        self.top_k = top_k
        d_melody = d_model // top_k

        self.pitch_embed = nn.Embedding(129, d_melody, padding_idx=0)

        self.downsample = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=4, stride=4, padding=0),
            nn.SiLU(),
            nn.Conv1d(d_model, d_model, kernel_size=4, stride=4, padding=0),
            nn.SiLU(),
        )

    def forward(self, melody_indices: torch.LongTensor,
                target_len: int) -> torch.Tensor:
        """
        Args:
            melody_indices: (B, top_k, T_cqt) long tensor, values 0-128
            target_len: int, latent temporal width W_lat to match
        Returns:
            (B, target_len, d_model) melody conditioning (per-timeframe)
        """
        B, K, T = melody_indices.shape
        emb = self.pitch_embed(melody_indices)       # (B, K, T, d_melody)
        emb = emb.permute(0, 2, 1, 3).reshape(B, T, -1)  # (B, T, K*d_melody)
        emb = emb.permute(0, 2, 1)                  # (B, d_model, T)
        down = self.downsample(emb)                  # (B, d_model, T_down)
        down = F.interpolate(down, size=target_len, mode="linear",
                             align_corners=False)
        return down.permute(0, 2, 1)                 # (B, target_len, d_model)
