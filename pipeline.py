"""
Audio Reconstruction Pipeline: DEAM -> Spectrogram -> VAE -> BigVGAN -> WAV
===========================================================================
Tests how well a song survives the full encode/decode pipeline.

Pipeline:
  1. Load a DEAM song (15-second clip)
  2. Convert to Mel-Spectrogram (using BigVGAN's mel config for consistency)
  3. Encode through VAE encoder -> latent space
  4. Decode through VAE decoder -> reconstructed spectrogram
  5. PatchGAN discriminator sharpens reconstruction during training
  6. BigVGAN vocoder (nvidia/bigvgan_v2_44khz_128band_512x) -> output WAV

Usage:
  python pipeline.py [--audio_dir PATH] [--epochs_vae N] [--epochs_gan N] [--song_index I]

Requirements:
  pip install -r requirements.txt
"""

import os
import sys
import glob
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import librosa
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm


# ============================================================================
# Configuration
# ============================================================================

class Config:
    # BigVGAN v2 44khz model expects 44100 Hz, 128 mel bands, hop=512
    sample_rate = 44100
    clip_seconds = 15
    # These will be read from BigVGAN's config (model.h) at runtime,
    # but we set defaults here for the VAE architecture
    n_mels = 128
    latent_dim = 128
    lr_vae = 1e-3
    lr_disc = 4e-4
    beta_kl = 0.0001          # KL weight (low to prioritize reconstruction)
    adv_weight = 0.01         # adversarial loss weight for generator
    epochs_vae = 300          # Phase 1: VAE-only warmup
    epochs_gan = 150          # Phase 2: VAE + GAN adversarial
    log_interval = 25
    device = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# BigVGAN Wrapper (HuggingFace local clone)
# ============================================================================

# Path to the cloned HuggingFace BigVGAN repo (contains bigvgan.py, meldataset.py, weights)
BIGVGAN_LOCAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "bigvgan_v2_44khz_128band_512x")

# Add the cloned repo to sys.path so `import bigvgan` and `from meldataset import ...` resolve
if BIGVGAN_LOCAL_DIR not in sys.path:
    sys.path.insert(0, BIGVGAN_LOCAL_DIR)


def load_bigvgan(device: str = "cuda"):
    """Load pretrained BigVGAN v2 44kHz vocoder from the local HuggingFace clone."""
    import bigvgan

    if not os.path.isdir(BIGVGAN_LOCAL_DIR):
        raise FileNotFoundError(
            f"BigVGAN repo not found at {BIGVGAN_LOCAL_DIR}.\n"
            f"Clone it first:\n"
            f"  git lfs install\n"
            f"  git clone https://huggingface.co/nvidia/bigvgan_v2_44khz_128band_512x"
        )

    print(f"[BIGVGAN] Loading from local clone: {BIGVGAN_LOCAL_DIR} ...")
    model = bigvgan.BigVGAN.from_pretrained(
        BIGVGAN_LOCAL_DIR,
        use_cuda_kernel=False,
    )
    model.remove_weight_norm()
    model = model.eval().to(device)
    print(f"[BIGVGAN] Loaded. SR={model.h.sampling_rate}, "
          f"n_mels={model.h.num_mels}, hop={model.h.hop_size}")
    return model


def bigvgan_mel_spectrogram(wav_tensor: torch.Tensor, bigvgan_model):
    """
    Compute mel spectrogram using BigVGAN's own mel function.
    wav_tensor: (1, T_time) float tensor in [-1, 1]
    Returns: (1, n_mels, T_frame) mel spectrogram
    """
    from meldataset import get_mel_spectrogram
    mel = get_mel_spectrogram(wav_tensor, bigvgan_model.h)
    return mel


# ============================================================================
# Data Loading
# ============================================================================

def load_deam_song(audio_dir: str, song_index: int = 0, sample_rate: int = 44100,
                   clip_seconds: int = 15):
    """Load a single DEAM song using librosa, crop to clip_seconds, mono."""
    pattern = os.path.join(audio_dir, "*.mp3")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No MP3 files found in {audio_dir}")

    song_index = song_index % len(files)
    path = files[song_index]
    print(f"[DATA] Loading: {os.path.basename(path)} (index {song_index})")

    # librosa loads as float32 in [-1, 1], mono by default
    wav, sr = librosa.load(path, sr=sample_rate, mono=True)

    # Crop to clip_seconds
    max_samples = clip_seconds * sr
    if len(wav) > max_samples:
        wav = wav[:max_samples]

    wav_tensor = torch.FloatTensor(wav).unsqueeze(0)  # (1, T_time)
    print(f"[DATA] Waveform shape: {wav_tensor.shape}, SR: {sr}, "
          f"Duration: {len(wav)/sr:.2f}s")
    return wav_tensor, sr


# ============================================================================
# Spectrogram Normalization (for VAE [0,1] range)
# ============================================================================

class MelNormalizer:
    """Normalize/denormalize mel spectrograms for VAE training."""

    def __init__(self):
        self.mel_min = None
        self.mel_max = None

    def normalize(self, mel: torch.Tensor) -> torch.Tensor:
        """Normalize mel spectrogram to [0, 1] range."""
        self.mel_min = mel.min()
        self.mel_max = mel.max()
        return (mel - self.mel_min) / (self.mel_max - self.mel_min + 1e-8)

    def denormalize(self, mel_norm: torch.Tensor) -> torch.Tensor:
        """Denormalize back to original mel scale."""
        return mel_norm * (self.mel_max - self.mel_min + 1e-8) + self.mel_min


# ============================================================================
# VAE Model
# ============================================================================

class Encoder(nn.Module):
    """Convolutional encoder: spectrogram -> (mu, logvar)."""

    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.LeakyReLU(0.2),

            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
        )
        self._flat_size = None
        self.fc_mu = None
        self.fc_logvar = None
        self.latent_dim = latent_dim

    def _init_fc(self, x: torch.Tensor):
        self._flat_size = x.shape[1] * x.shape[2] * x.shape[3]
        device = x.device
        self.fc_mu = nn.Linear(self._flat_size, self.latent_dim).to(device)
        self.fc_logvar = nn.Linear(self._flat_size, self.latent_dim).to(device)

    def forward(self, x: torch.Tensor):
        h = self.conv(x)
        if self.fc_mu is None:
            self._init_fc(h)
        h_flat = h.view(h.size(0), -1)
        return self.fc_mu(h_flat), self.fc_logvar(h_flat), h.shape


class Decoder(nn.Module):
    """Transposed convolutional decoder: z -> reconstructed spectrogram."""

    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        self._fc = None
        self._conv_shape = None

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),

            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),

            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),

            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def _init_fc(self, conv_shape, device):
        self._conv_shape = conv_shape
        flat_size = conv_shape[1] * conv_shape[2] * conv_shape[3]
        self._fc = nn.Linear(self.latent_dim, flat_size).to(device)

    def forward(self, z: torch.Tensor, conv_shape: tuple):
        if self._fc is None:
            self._init_fc(conv_shape, z.device)
        h = self._fc(z)
        h = h.view(-1, conv_shape[1], conv_shape[2], conv_shape[3])
        return self.deconv(h)


class SpectrogramVAE(nn.Module):
    """Variational Autoencoder for mel-spectrograms."""

    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar, conv_shape = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z, conv_shape)
        return x_recon, mu, logvar


# ============================================================================
# PatchGAN Discriminator
# ============================================================================

class PatchDiscriminator(nn.Module):
    """PatchGAN discriminator for spectrograms."""

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2),

            nn.Conv2d(128, 1, 4, stride=1, padding=1),
        )

    def forward(self, x):
        return self.model(x)


# ============================================================================
# Loss & Helpers
# ============================================================================

def vae_loss(x, x_recon, mu, logvar, beta=0.0001):
    """Combined reconstruction (MSE) + KL divergence loss."""
    recon = F.mse_loss(x_recon, x, reduction="sum") / x.size(0)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon + beta * kl, recon, kl


def pad_spectrogram(spec: torch.Tensor, divisor: int = 16):
    """Pad spectrogram so H and W are divisible by divisor."""
    _, _, h, w = spec.shape
    pad_h = (divisor - h % divisor) % divisor
    pad_w = (divisor - w % divisor) % divisor
    padded = F.pad(spec, (0, pad_w, 0, pad_h), mode="reflect")
    return padded, (h, w)


def unpad_spectrogram(spec: torch.Tensor, orig_hw: tuple):
    """Remove padding to restore original spatial dims."""
    h, w = orig_hw
    return spec[:, :, :h, :w]


# ============================================================================
# Training
# ============================================================================

def train_pipeline(spectrogram: torch.Tensor, cfg: Config):
    """
    Two-phase training on normalized mel spectrogram:
      Phase 1 - VAE warmup (reconstruction + KL only)
      Phase 2 - VAE-GAN (add adversarial discriminator)
    """
    device = cfg.device
    spec_padded, orig_hw = pad_spectrogram(spectrogram)
    spec_padded = spec_padded.to(device)

    print(f"[TRAIN] Spectrogram shape (padded): {spec_padded.shape}")
    print(f"[TRAIN] Device: {device}")

    vae = SpectrogramVAE(cfg.latent_dim).to(device)
    disc = PatchDiscriminator().to(device)
    opt_vae = optim.Adam(vae.parameters(), lr=cfg.lr_vae)
    opt_disc = optim.Adam(disc.parameters(), lr=cfg.lr_disc)
    bce_logits = nn.BCEWithLogitsLoss()

    # ---- Phase 1: VAE warmup ----
    print(f"\n{'='*60}")
    print(f" Phase 1: VAE Warmup ({cfg.epochs_vae} epochs)")
    print(f"{'='*60}")

    vae.train()
    for epoch in tqdm(range(1, cfg.epochs_vae + 1), desc="Phase 1 (VAE)"):
        opt_vae.zero_grad()
        x_recon, mu, logvar = vae(spec_padded)

        if x_recon.shape != spec_padded.shape:
            x_recon = F.interpolate(x_recon, size=spec_padded.shape[2:],
                                    mode="bilinear", align_corners=False)

        loss, recon_loss, kl_loss = vae_loss(
            spec_padded, x_recon, mu, logvar, cfg.beta_kl
        )
        loss.backward()
        opt_vae.step()

        if epoch % cfg.log_interval == 0 or epoch == 1:
            tqdm.write(f"  Epoch {epoch:4d} | Loss: {loss.item():.4f} | "
                       f"Recon: {recon_loss.item():.4f} | KL: {kl_loss.item():.4f}")

    # ---- Phase 2: VAE-GAN adversarial ----
    print(f"\n{'='*60}")
    print(f" Phase 2: VAE-GAN Adversarial ({cfg.epochs_gan} epochs)")
    print(f"{'='*60}")

    for epoch in tqdm(range(1, cfg.epochs_gan + 1), desc="Phase 2 (GAN)"):
        # --- Train Discriminator ---
        disc.train()
        vae.eval()
        opt_disc.zero_grad()

        with torch.no_grad():
            x_recon, mu, logvar = vae(spec_padded)
            if x_recon.shape != spec_padded.shape:
                x_recon = F.interpolate(x_recon, size=spec_padded.shape[2:],
                                        mode="bilinear", align_corners=False)

        pred_real = disc(spec_padded)
        pred_fake = disc(x_recon.detach())
        loss_d = 0.5 * (bce_logits(pred_real, torch.ones_like(pred_real)) +
                        bce_logits(pred_fake, torch.zeros_like(pred_fake)))
        loss_d.backward()
        opt_disc.step()

        # --- Train Generator (VAE decoder) ---
        vae.train()
        disc.eval()
        opt_vae.zero_grad()

        x_recon, mu, logvar = vae(spec_padded)
        if x_recon.shape != spec_padded.shape:
            x_recon = F.interpolate(x_recon, size=spec_padded.shape[2:],
                                    mode="bilinear", align_corners=False)

        loss_vae, recon_loss, kl_loss = vae_loss(
            spec_padded, x_recon, mu, logvar, cfg.beta_kl
        )
        pred_fake_for_g = disc(x_recon)
        loss_g_adv = bce_logits(pred_fake_for_g, torch.ones_like(pred_fake_for_g))

        total_loss = loss_vae + cfg.adv_weight * loss_g_adv
        total_loss.backward()
        opt_vae.step()

        if epoch % cfg.log_interval == 0 or epoch == 1:
            tqdm.write(f"  Epoch {epoch:4d} | VAE: {loss_vae.item():.4f} | "
                       f"D: {loss_d.item():.4f} | G_adv: {loss_g_adv.item():.4f}")

    # ---- Final reconstruction ----
    vae.eval()
    with torch.no_grad():
        x_recon, mu, logvar = vae(spec_padded)
        if x_recon.shape != spec_padded.shape:
            x_recon = F.interpolate(x_recon, size=spec_padded.shape[2:],
                                    mode="bilinear", align_corners=False)

    recon_padded = x_recon.cpu()
    recon_spec = unpad_spectrogram(recon_padded, orig_hw)
    return recon_spec, vae


# ============================================================================
# Evaluation Metrics
# ============================================================================

def compute_metrics(original_spec, recon_spec, original_wav, recon_wav):
    """Compute reconstruction quality metrics."""
    spec_mse = F.mse_loss(recon_spec, original_spec).item()

    diff_norm = torch.norm(original_spec - recon_spec, p="fro")
    orig_norm = torch.norm(original_spec, p="fro")
    spectral_conv = (diff_norm / (orig_norm + 1e-8)).item()

    min_len = min(original_wav.shape[-1], recon_wav.shape[-1])
    wav_mse = F.mse_loss(
        recon_wav[..., :min_len], original_wav[..., :min_len]
    ).item()

    return {
        "spectrogram_mse": spec_mse,
        "spectral_convergence": spectral_conv,
        "waveform_mse": wav_mse,
    }


# ============================================================================
# Visualization
# ============================================================================

def plot_comparison(original_spec, recon_spec, metrics, output_path):
    """Side-by-side spectrogram comparison with metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    im0 = axes[0].imshow(original_spec.squeeze().cpu().numpy(),
                         aspect="auto", origin="lower", cmap="magma")
    axes[0].set_title("Original Spectrogram", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Time Frames")
    axes[0].set_ylabel("Mel Bins")
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    im1 = axes[1].imshow(recon_spec.squeeze().cpu().numpy(),
                         aspect="auto", origin="lower", cmap="magma")
    axes[1].set_title("Reconstructed (VAE-GAN + BigVGAN)", fontsize=14,
                      fontweight="bold")
    axes[1].set_xlabel("Time Frames")
    axes[1].set_ylabel("Mel Bins")
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    metrics_text = (
        f"Spec MSE: {metrics['spectrogram_mse']:.6f}  |  "
        f"Spectral Conv: {metrics['spectral_convergence']:.4f}  |  "
        f"Wav MSE: {metrics['waveform_mse']:.6f}"
    )
    fig.suptitle(metrics_text, fontsize=11, y=0.02, color="gray")
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIS] Saved comparison plot: {output_path}")


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="DEAM Audio Reconstruction: Spectrogram -> VAE -> BigVGAN -> WAV"
    )
    parser.add_argument("--audio_dir", type=str,
                        default=os.path.join("data", "DEAM_audio", "MEMD_audio"),
                        help="Path to DEAM MP3 files")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Directory for output files")
    parser.add_argument("--song_index", type=int, default=0,
                        help="Index of the song to load from sorted file list")
    parser.add_argument("--epochs_vae", type=int, default=None,
                        help="VAE warmup epochs (default: 300)")
    parser.add_argument("--epochs_gan", type=int, default=None,
                        help="GAN adversarial epochs (default: 150)")
    parser.add_argument("--latent_dim", type=int, default=128,
                        help="VAE latent dimension")
    args = parser.parse_args()

    cfg = Config()
    cfg.latent_dim = args.latent_dim
    if args.epochs_vae is not None:
        cfg.epochs_vae = args.epochs_vae
    if args.epochs_gan is not None:
        cfg.epochs_gan = args.epochs_gan

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print(" DEAM Audio Reconstruction Pipeline")
    print(" Spectrogram -> VAE -> BigVGAN Vocoder -> WAV")
    print("=" * 60)
    print(f" Device       : {cfg.device}")
    print(f" Latent dim   : {cfg.latent_dim}")
    print(f" VAE epochs   : {cfg.epochs_vae}")
    print(f" GAN epochs   : {cfg.epochs_gan}")
    print(f" Clip length  : {cfg.clip_seconds}s")
    print(f" Sample rate  : {cfg.sample_rate} Hz")
    print("=" * 60)

    # --- Step 1: Load BigVGAN vocoder ---
    print("\n[STEP 1] Loading BigVGAN vocoder...")
    bigvgan_model = load_bigvgan(cfg.device)

    # --- Step 2: Load DEAM song ---
    print("\n[STEP 2] Loading DEAM song...")
    waveform, sr = load_deam_song(
        args.audio_dir, args.song_index,
        sample_rate=bigvgan_model.h.sampling_rate,
        clip_seconds=cfg.clip_seconds,
    )

    # Save original as WAV
    orig_wav_path = os.path.join(args.output_dir, "original.wav")
    sf.write(orig_wav_path, waveform.squeeze(0).numpy(), sr)
    print(f"[STEP 2] Saved original: {orig_wav_path}")

    # --- Step 3: Waveform -> Mel Spectrogram (BigVGAN's mel config) ---
    print("\n[STEP 3] Computing mel spectrogram (BigVGAN config)...")
    mel = bigvgan_mel_spectrogram(waveform, bigvgan_model)
    print(f"[STEP 3] Mel shape: {mel.shape} "
          f"(range: [{mel.min():.3f}, {mel.max():.3f}])")

    # --- Step 4: Normalize mel for VAE training ---
    normalizer = MelNormalizer()
    mel_norm = normalizer.normalize(mel)  # (1, 128, T) -> [0, 1]

    # Add channel dim for 2D conv: (1, 1, 128, T)
    mel_input = mel_norm.unsqueeze(0)

    # --- Step 5: Train VAE-GAN ---
    print("\n[STEP 5] Training VAE-GAN pipeline...")
    recon_norm, vae = train_pipeline(mel_input, cfg)
    recon_norm = recon_norm.squeeze(0)  # back to (1, 128, T)

    # --- Step 6: Denormalize and run BigVGAN vocoder ---
    print("\n[STEP 6] BigVGAN: reconstructed mel -> waveform...")
    recon_mel = normalizer.denormalize(recon_norm)  # back to BigVGAN mel scale

    with torch.inference_mode():
        recon_mel_device = recon_mel.to(cfg.device)
        wav_gen = bigvgan_model(recon_mel_device)  # (1, 1, T_time)

    wav_gen_float = wav_gen.squeeze(0).cpu()  # (1, T_time)

    recon_wav_path = os.path.join(args.output_dir, "reconstructed.wav")
    sf.write(recon_wav_path, wav_gen_float.squeeze(0).numpy(), sr)
    print(f"[STEP 6] Saved reconstructed: {recon_wav_path}")

    # --- Step 7: Also generate a "baseline" BigVGAN output (no VAE) ---
    print("\n[STEP 7] BigVGAN baseline: original mel -> waveform (no VAE)...")
    with torch.inference_mode():
        mel_device = mel.to(cfg.device)
        wav_baseline = bigvgan_model(mel_device)
    wav_baseline_float = wav_baseline.squeeze(0).cpu()

    baseline_wav_path = os.path.join(args.output_dir, "baseline_bigvgan.wav")
    sf.write(baseline_wav_path, wav_baseline_float.squeeze(0).numpy(), sr)
    print(f"[STEP 7] Saved baseline: {baseline_wav_path}")

    # --- Step 8: Evaluate ---
    print("\n[STEP 8] Computing metrics...")
    metrics = compute_metrics(mel_norm, recon_norm, waveform, wav_gen_float)

    print("\n" + "=" * 60)
    print(" RECONSTRUCTION METRICS (Original vs VAE-GAN + BigVGAN)")
    print("=" * 60)
    print(f"  Spectrogram MSE       : {metrics['spectrogram_mse']:.6f}")
    print(f"  Spectral Convergence  : {metrics['spectral_convergence']:.4f}")
    print(f"  Waveform MSE          : {metrics['waveform_mse']:.6f}")
    print("=" * 60)

    # --- Step 9: Visualize ---
    print("\n[STEP 9] Generating comparison plot...")
    plot_path = os.path.join(args.output_dir, "comparison.png")
    plot_comparison(mel, recon_mel, metrics, plot_path)

    print("\n" + "=" * 60)
    print(" PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Original WAV          : {orig_wav_path}")
    print(f"  BigVGAN baseline WAV  : {baseline_wav_path}")
    print(f"  Reconstructed WAV     : {recon_wav_path}")
    print(f"  Comparison plot       : {plot_path}")
    print()
    print("  Compare all 3 files:")
    print("    original.wav         = source audio")
    print("    baseline_bigvgan.wav = mel -> BigVGAN only (no VAE)")
    print("    reconstructed.wav    = mel -> VAE -> BigVGAN (full pipeline)")
    print()
    print("  This lets you isolate VAE degradation from vocoder degradation.")


if __name__ == "__main__":
    main()
