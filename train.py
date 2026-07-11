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


def train_autoencoder(mel_norm: torch.Tensor, cfg: DiffusionConfig):
    """
    Train the latent autoencoder on a normalized mel spectrogram.

    Preserves spatial structure (no flatten bottleneck) so the latent
    is suitable for 2D diffusion.

    Returns:
        ae: trained LatentAutoencoder
        orig_hw: (H, W) before padding, needed for unpadding later
    """
    device = cfg.device
    mel_padded, orig_hw = pad_spectrogram(mel_norm.unsqueeze(0))
    mel_padded = mel_padded.to(device)

    ae = LatentAutoencoder(cfg.ae_channels).to(device)
    optimizer = optim.Adam(ae.parameters(), lr=cfg.ae_lr)

    print(f"\n{'='*60}")
    print(f" Training Latent Autoencoder ({cfg.ae_epochs} epochs)")
    print(f" Input: {mel_padded.shape}  Device: {device}")
    print(f"{'='*60}")

    ae.train()
    for epoch in tqdm(range(1, cfg.ae_epochs + 1), desc="Autoencoder"):
        optimizer.zero_grad()
        recon, z = ae(mel_padded)
        if recon.shape != mel_padded.shape:
            recon = F.interpolate(recon, size=mel_padded.shape[2:],
                                  mode="bilinear", align_corners=False)
        loss = F.mse_loss(recon, mel_padded)
        loss.backward()
        optimizer.step()

        if epoch % cfg.log_interval == 0 or epoch == 1:
            tqdm.write(f"  Epoch {epoch:4d} | MSE: {loss.item():.6f} | "
                       f"Latent: {z.shape}")

    ae.eval()
    return ae, orig_hw


def train_diffusion(ae: LatentAutoencoder, mel_norm: torch.Tensor,
                    waveform_np: np.ndarray, mood_texts: list[str],
                    cfg: DiffusionConfig):
    """
    Train the DiT + ControlNet diffusion model in the autoencoder's latent space.

    Each epoch:
      1. Encode mel -> latent z0
      2. Sample random timestep, add noise -> z_t
      3. Predict v conditioned on text + melody
      4. MSE loss against v-target

    CFG dropout randomly drops text conditioning with probability cfg_dropout
    to enable classifier-free guidance at inference.

    Returns:
        dit, melody_enc, text_enc, diffusion
    """
    device = cfg.device
    mel_padded, orig_hw = pad_spectrogram(mel_norm.unsqueeze(0))
    mel_padded = mel_padded.to(device)

    ae.eval()
    with torch.no_grad():
        z0 = ae.encoder(mel_padded)
    print(f"[DIFF] Latent shape: {z0.shape}")

    extractor = MelodyExtractor(
        sr=cfg.sample_rate, n_bins=cfg.cqt_bins,
        bins_per_octave=cfg.cqt_bins_per_octave,
        hop_length=cfg.cqt_hop, fmin=cfg.cqt_fmin,
        top_k=cfg.melody_top_k, highpass_cutoff=cfg.highpass_cutoff,
    )
    melody_indices = extractor.extract(waveform_np)
    melody_tensor = torch.from_numpy(melody_indices).unsqueeze(0).to(device)
    print(f"[DIFF] Melody shape: {melody_tensor.shape}")

    _, C_lat, H_lat, W_lat = z0.shape
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

    text_tokens_list = [
        TextEncoder.tokenize(t, cfg.text_max_len) for t in mood_texts
    ]
    null_tokens = TextEncoder.tokenize("", cfg.text_max_len).to(device)

    print(f"\n{'='*60}")
    print(f" Training Diffusion Model ({cfg.diff_epochs} epochs)")
    print(f" DiT blocks: {cfg.n_dit_blocks} ({cfg.n_controlnet_blocks} w/ ControlNet)")
    print(f" d_model: {cfg.d_model}  heads: {cfg.n_heads}")
    print(f" Mood texts: {mood_texts}")
    print(f"{'='*60}")

    dit.train()
    melody_enc.train()
    text_enc.train()

    for epoch in tqdm(range(1, cfg.diff_epochs + 1), desc="Diffusion"):
        optimizer.zero_grad()

        t = torch.randint(0, cfg.num_train_timesteps, (1,), device=device)
        noise = torch.randn_like(z0)
        z_t = diffusion.q_sample(z0, t, noise)

        mel_emb = melody_enc(melody_tensor, W_lat)

        text_idx = epoch % len(text_tokens_list)
        tokens = text_tokens_list[text_idx].unsqueeze(0).to(device)
        if torch.rand(1).item() < cfg.cfg_dropout:
            tokens = null_tokens.unsqueeze(0)
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
    return dit, melody_enc, text_enc, diffusion
