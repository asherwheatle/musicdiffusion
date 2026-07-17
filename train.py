"""Training loops for the latent autoencoder and diffusion model."""

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from config import DiffusionConfig
from autoencoder import LatentAutoencoder
from melody import MelodyExtractor, MelodyEncoder
from text_encoder import TextEncoder
from dit import MoodDiT
from diffusion import GaussianDiffusion
from pipeline import pad_spectrogram, unpad_spectrogram


def train_autoencoder(mel_batch: torch.Tensor, cfg: DiffusionConfig):
    """
    Train the latent autoencoder on a batch of normalized mel spectrograms.

    Args:
        mel_batch: (N, 1, n_mels, T) normalized mels, one per song

    Preserves spatial structure (no flatten bottleneck) so the latent
    is suitable for 2D diffusion.

    Returns:
        ae: trained LatentAutoencoder
        orig_hw: (H, W) before padding, needed for unpadding later
    """
    device = cfg.device
    mel_padded, orig_hw = pad_spectrogram(mel_batch)
    mel_padded = mel_padded.to(device)
    n_songs = mel_padded.shape[0]

    ae = LatentAutoencoder(cfg.ae_channels).to(device)
    optimizer = optim.Adam(ae.parameters(), lr=cfg.ae_lr)

    print(f"\n{'='*60}")
    print(f" Training Latent Autoencoder ({cfg.ae_epochs} epochs)")
    print(f" Input: {mel_padded.shape} ({n_songs} songs)  Device: {device}")
    print(f"{'='*60}")

    ae.train()
    for epoch in tqdm(range(1, cfg.ae_epochs + 1), desc="Autoencoder"):
        perm = torch.randperm(n_songs, device=device)
        epoch_loss, n_batches = 0.0, 0
        for i in range(0, n_songs, cfg.batch_size):
            batch = mel_padded[perm[i:i + cfg.batch_size]]
            optimizer.zero_grad()
            recon, z = ae(batch)
            if recon.shape != batch.shape:
                recon = F.interpolate(recon, size=batch.shape[2:],
                                      mode="bilinear", align_corners=False)
            loss = F.mse_loss(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        if epoch % cfg.log_interval == 0 or epoch == 1:
            tqdm.write(f"  Epoch {epoch:4d} | MSE: {epoch_loss / n_batches:.6f}")

    ae.eval()
    return ae, orig_hw


def train_diffusion(ae: LatentAutoencoder, mel_batch: torch.Tensor,
                    waveforms_np: list[np.ndarray], mood_texts: list[str],
                    cfg: DiffusionConfig):
    """
    Train the DiT + ControlNet diffusion model in the autoencoder's latent space.

    Args:
        mel_batch: (N, 1, n_mels, T) normalized mels, one per song
        waveforms_np: list of N waveforms (for melody extraction)
        mood_texts: list of N mood strings, one per song

    Each step:
      1. Sample a minibatch of songs' latents z0
      2. Sample random timesteps, add noise -> z_t
      3. Predict v conditioned on each song's text + melody
      4. MSE loss against v-target

    CFG dropout randomly drops text conditioning with probability cfg_dropout
    to enable classifier-free guidance at inference.

    Returns:
        dit, melody_enc, text_enc, diffusion, (latent_mean, latent_std)
    """
    device = cfg.device
    mel_padded, orig_hw = pad_spectrogram(mel_batch)
    mel_padded = mel_padded.to(device)
    n_songs = mel_padded.shape[0]

    ae.eval()
    with torch.no_grad():
        z0_all = ae.encoder(mel_padded)
    # Standardize latents so the diffusion's unit-variance noise assumption
    # holds (the encoder's GroupNorm+SiLU output is skewed, std != 1)
    latent_mean = z0_all.mean()
    latent_std = z0_all.std()
    z0_all = (z0_all - latent_mean) / latent_std
    print(f"[DIFF] Latents: {z0_all.shape} "
          f"(mean={latent_mean.item():.4f}, std={latent_std.item():.4f})")

    extractor = MelodyExtractor(
        sr=cfg.sample_rate, n_bins=cfg.cqt_bins,
        bins_per_octave=cfg.cqt_bins_per_octave,
        hop_length=cfg.cqt_hop, fmin=cfg.cqt_fmin,
        top_k=cfg.melody_top_k, highpass_cutoff=cfg.highpass_cutoff,
    )
    melody_all = torch.stack([
        torch.from_numpy(extractor.extract(w))
        for w in tqdm(waveforms_np, desc="Extracting melodies")
    ], dim=0).to(device)  # (N, top_k, T_cqt)
    print(f"[DIFF] Melody shape: {melody_all.shape}")

    _, C_lat, H_lat, W_lat = z0_all.shape
    dit = MoodDiT(
        latent_channels=C_lat, latent_h=H_lat,
        d_model=cfg.d_model, n_heads=cfg.n_heads,
        n_blocks=cfg.n_dit_blocks, n_control_blocks=cfg.n_controlnet_blocks,
        mlp_ratio=cfg.mlp_ratio, dropout=cfg.dropout,
    ).to(device)

    melody_enc = MelodyEncoder(d_model=cfg.d_model, top_k=cfg.melody_top_k).to(device)
    text_enc = TextEncoder(d_model=cfg.d_model, max_len=cfg.text_max_len).to(device)
    diffusion = GaussianDiffusion(cfg.num_train_timesteps, device)

    all_params = (list(dit.parameters()) +
                  list(melody_enc.parameters()) +
                  list(text_enc.parameters()))
    optimizer = optim.AdamW(all_params, lr=cfg.diff_lr)

    tokens_all = torch.stack([
        TextEncoder.tokenize(t, cfg.text_max_len) for t in mood_texts
    ], dim=0).to(device)  # (N, text_max_len)
    null_tokens = TextEncoder.tokenize("", cfg.text_max_len).to(device)

    from collections import Counter
    print(f"\n{'='*60}")
    print(f" Training Diffusion Model ({cfg.diff_epochs} epochs, "
          f"{n_songs} songs, batch {cfg.batch_size})")
    print(f" DiT blocks: {cfg.n_dit_blocks} ({cfg.n_controlnet_blocks} w/ ControlNet)")
    print(f" d_model: {cfg.d_model}  heads: {cfg.n_heads}")
    print(f" Mood distribution: {dict(Counter(mood_texts))}")
    print(f"{'='*60}")

    dit.train()
    melody_enc.train()
    text_enc.train()

    for epoch in tqdm(range(1, cfg.diff_epochs + 1), desc="Diffusion"):
        optimizer.zero_grad()

        idx = torch.randint(0, n_songs, (cfg.batch_size,), device=device)
        z0 = z0_all[idx]
        t = torch.randint(0, cfg.num_train_timesteps,
                          (cfg.batch_size,), device=device)
        noise = torch.randn_like(z0)
        z_t = diffusion.q_sample(z0, t, noise)

        mel_emb = melody_enc(melody_all[idx], W_lat)

        tokens = tokens_all[idx].clone()
        drop = torch.rand(cfg.batch_size, device=device) < cfg.cfg_dropout
        tokens[drop] = null_tokens
        text_emb = text_enc(tokens)

        v_pred = dit(z_t, t, text_emb, mel_emb)
        v_tgt = diffusion.v_target(z0, noise, t)

        loss = F.mse_loss(v_pred, v_tgt)
        loss.backward()
        optimizer.step()

        if epoch % cfg.log_interval == 0 or epoch == 1:
            tqdm.write(f"  Epoch {epoch:4d} | Loss: {loss.item():.6f}")

    dit.eval()
    melody_enc.eval()
    text_enc.eval()
    return dit, melody_enc, text_enc, diffusion, (latent_mean, latent_std)
