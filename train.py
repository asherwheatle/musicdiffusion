"""Training loops for the latent autoencoder and diffusion model.

Data tensors (mels, latents, melodies) stay on CPU so the full DEAM set fits
in memory; only the active minibatch is moved to the GPU. Batches are served
through a pinned-memory DataLoader and copied with non_blocking=True so the
host->device transfer overlaps compute instead of stalling each step.
"""

import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from config import DiffusionConfig
from autoencoder import LatentAutoencoder
from text_encoder import ClapTextEncoder
from dit import MoodDiT
from melody import MelodyEncoder
from diffusion import GaussianDiffusion
from pipeline import pad_spectrogram


def _atomic_save(obj, path: str):
    """Write to a temp file then rename, so an eviction mid-write can never
    leave a truncated (unloadable) checkpoint at `path`."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def train_autoencoder(mel_batch: torch.Tensor, cfg: DiffusionConfig):
    """
    Train the latent autoencoder on a batch of normalized mel spectrograms.

    Args:
        mel_batch: (N, 1, n_mels, T) normalized mels, one per clip (CPU)

    Preserves spatial structure (no flatten bottleneck) so the latent
    is suitable for 2D diffusion.

    Returns:
        ae: trained LatentAutoencoder
        orig_hw: (H, W) before padding, needed for unpadding later
    """
    device = cfg.device
    mel_padded, orig_hw = pad_spectrogram(mel_batch)
    n_songs = mel_padded.shape[0]

    ae = LatentAutoencoder(cfg.ae_channels).to(device)
    optimizer = optim.Adam(ae.parameters(), lr=cfg.ae_lr)

    # Resume from an interrupted run if a checkpoint is present, so an
    # eviction doesn't cost every completed epoch.
    ckpt_path = os.path.join(cfg.output_dir, "autoencoder_ckpt.pt")
    start_epoch = 1
    if getattr(cfg, "resume", True) and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        ae.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        start_epoch = ck["epoch"] + 1
        orig_hw = ck.get("orig_hw", orig_hw)
        print(f"[RESUME] Autoencoder resumed from epoch {ck['epoch']} "
              f"-> continuing at {start_epoch} ({ckpt_path})")

    # DataLoader with pinned memory so H2D copies overlap compute and kill the
    # per-batch copy stall; workers prefetch/collate the next batch off the
    # training thread.
    dataset = TensorDataset(mel_padded)
    num_workers = getattr(cfg, "num_workers", 2)
    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=(num_workers > 0),
    )

    print(f"\n{'='*60}")
    print(f" Training Latent Autoencoder ({cfg.ae_epochs} epochs)")
    print(f" Input: {tuple(mel_padded.shape)} ({n_songs} songs)  Device: {device}")
    print(f" Batch size: {cfg.batch_size}")
    print(f"{'='*60}")

    ckpt_interval = getattr(cfg, "ae_ckpt_interval", cfg.ae_epochs)

    ae.train()
    for epoch in tqdm(range(start_epoch, cfg.ae_epochs + 1), desc="Autoencoder",
                      initial=start_epoch - 1, total=cfg.ae_epochs):
        epoch_loss, n_batches = 0.0, 0
        for (batch,) in loader:
            batch = batch.to(device, non_blocking=True)
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
            tqdm.write(f"  Epoch {epoch:4d} | MSE: {epoch_loss / max(n_batches, 1):.6f}")

        if epoch % ckpt_interval == 0 or epoch == cfg.ae_epochs:
            # Resumable checkpoint (model + optimizer + epoch) ...
            _atomic_save({"model": ae.state_dict(),
                          "optimizer": optimizer.state_dict(),
                          "epoch": epoch, "orig_hw": orig_hw}, ckpt_path)
            # ... plus the inference-ready copy the rest of the code loads,
            # so even an interrupted run leaves a usable (if under-trained) AE.
            _atomic_save(ae.state_dict(),
                         os.path.join(cfg.output_dir, "autoencoder.pt"))

    ae.eval()
    return ae, orig_hw


@torch.no_grad()
def _encode_latents(ae: LatentAutoencoder, mel_padded: torch.Tensor,
                    device: str, chunk: int = 32) -> torch.Tensor:
    """Encode all mels to latents in chunks; result stays on CPU."""
    zs = []
    for i in tqdm(range(0, mel_padded.shape[0], chunk), desc="Encoding latents"):
        zs.append(ae.encoder(mel_padded[i:i + chunk].to(device)).cpu())
    return torch.cat(zs, dim=0)


def train_diffusion(ae: LatentAutoencoder, mel_batch: torch.Tensor,
                    melody_all: torch.Tensor, mood_texts: list[str],
                    cfg: DiffusionConfig):
    """
    Train the DiT + ControlNet diffusion model in the autoencoder's latent space.

    Args:
        mel_batch: (N, 1, n_mels, T) normalized mels, one per clip (CPU)
        melody_all: (N, top_k, T_cqt) precomputed melody pitch indices (CPU)
        mood_texts: list of N mood strings, one per clip

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
    n_songs = mel_padded.shape[0]

    ae.eval()
    z0_all = _encode_latents(ae, mel_padded, device)
    del mel_padded   # free the padded mel copy; only latents are needed now
    # Standardize latents so the diffusion's unit-variance noise assumption
    # holds (the encoder's GroupNorm+SiLU output is skewed, std != 1).
    latent_mean = z0_all.mean()
    latent_std = z0_all.std()
    z0_all = (z0_all - latent_mean) / latent_std
    print(f"[DIFF] Latents: {tuple(z0_all.shape)} "
          f"(mean={latent_mean.item():.4f}, std={latent_std.item():.4f})")
    print(f"[DIFF] Melody shape: {tuple(melody_all.shape)}")

    _, C_lat, H_lat, W_lat = z0_all.shape
    dit = MoodDiT(
        latent_channels=C_lat, latent_h=H_lat,
        d_model=cfg.d_model, n_heads=cfg.n_heads,
        n_blocks=cfg.n_dit_blocks, n_control_blocks=cfg.n_controlnet_blocks,
        mlp_ratio=cfg.mlp_ratio, dropout=cfg.dropout,
    ).to(device)

    melody_enc = MelodyEncoder(d_model=cfg.d_model, top_k=cfg.melody_top_k).to(device)
    # Frozen CLAP text tower + trainable projection (Lever A). Only the
    # projection is optimized; CLAP is frozen and off the training hot path.
    text_enc = ClapTextEncoder(cfg.d_model, clap_ckpt=cfg.clap_ckpt,
                               n_tokens=cfg.text_n_tokens, device=device).to(device)
    diffusion = GaussianDiffusion(cfg.num_train_timesteps, device)

    # Precompute each song's frozen CLAP text embedding once (there are only a
    # handful of unique mood strings), plus the null embedding for CFG dropout.
    clap_emb_all = text_enc.encode(mood_texts).cpu()          # (N, clap_dim)
    null_clap_emb = text_enc.encode([""])[0].to(device)       # (clap_dim,)

    all_params = (list(dit.parameters()) +
                  list(melody_enc.parameters()) +
                  list(text_enc.parameters()))     # projection only; CLAP frozen
    optimizer = optim.AdamW(all_params, lr=cfg.diff_lr)

    # Resume from an interrupted run. latent_mean/std are recomputed above
    # (deterministic given the same AE + data), so the resumed run stays
    # consistent with the checkpoint's normalization.
    ckpt_path = os.path.join(cfg.output_dir, "diffusion_ckpt.pt")
    start_epoch = 1
    if getattr(cfg, "resume", True) and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        dit.load_state_dict(ck["dit"])
        melody_enc.load_state_dict(ck["melody_enc"])
        text_enc.load_state_dict(ck["text_enc"])
        optimizer.load_state_dict(ck["optimizer"])
        start_epoch = ck["epoch"] + 1
        print(f"[RESUME] Diffusion resumed from step {ck['epoch']} "
              f"-> continuing at {start_epoch} ({ckpt_path})")

    # Mood-balanced sampling: DEAM is heavily skewed (lots of happy/
    # energetic, few sad/dark clips), so uniform sampling would starve
    # the rare moods of gradient updates and bias the conditioning.
    # Weight each clip by the inverse of its mood's frequency so every
    # mood contributes ~equally to training batches.
    from collections import Counter
    mood_counts = Counter(mood_texts)
    sample_weights = torch.tensor(
        [1.0 / mood_counts[t] for t in mood_texts], dtype=torch.double)

    print(f"\n{'='*60}")
    print(f" Training Diffusion Model ({cfg.diff_epochs} steps, "
          f"{n_songs} clips, batch {cfg.batch_size})")
    print(f" DiT blocks: {cfg.n_dit_blocks} ({cfg.n_controlnet_blocks} w/ ControlNet)")
    print(f" d_model: {cfg.d_model}  heads: {cfg.n_heads}")
    print(f" Mood distribution (raw): {dict(mood_counts)}")
    print(f" Sampling: balanced — each of the {len(mood_counts)} moods "
          f"~{1.0 / len(mood_counts):.0%} of every batch")
    print(f"{'='*60}")

    dit.train()
    melody_enc.train()
    text_enc.train()

    ckpt_interval = getattr(cfg, "diff_ckpt_interval", cfg.diff_epochs)

    def _save_diff_ckpt(step):
        state = {
            "dit": dit.state_dict(),
            "melody_enc": melody_enc.state_dict(),
            "text_enc": text_enc.state_dict(),
            "latent_mean": latent_mean,
            "latent_std": latent_std,
        }
        # Inference-ready copy (what evaluate.py / edit mode load) ...
        _atomic_save(state, os.path.join(cfg.output_dir, "diffusion.pt"))
        # ... plus a resumable copy that also carries optimizer + step.
        _atomic_save({**state, "optimizer": optimizer.state_dict(),
                      "epoch": step}, ckpt_path)

    for epoch in tqdm(range(start_epoch, cfg.diff_epochs + 1), desc="Diffusion",
                      initial=start_epoch - 1, total=cfg.diff_epochs):
        optimizer.zero_grad()

        idx = torch.multinomial(sample_weights, cfg.batch_size,
                                replacement=True)
        z0 = z0_all[idx].to(device, non_blocking=True)
        t = torch.randint(0, cfg.num_train_timesteps,
                          (cfg.batch_size,), device=device)
        noise = torch.randn_like(z0)
        z_t = diffusion.q_sample(z0, t, noise)

        mel_emb = melody_enc(melody_all[idx].to(device, non_blocking=True), W_lat)

        clap_emb = clap_emb_all[idx].to(device)   # (B, clap_dim), a fresh copy
        drop = torch.rand(cfg.batch_size, device=device) < cfg.cfg_dropout
        clap_emb[drop] = null_clap_emb
        text_emb = text_enc(clap_emb)

        v_pred = dit(z_t, t, text_emb, mel_emb)
        v_tgt = diffusion.v_target(z0, noise, t)

        loss = F.mse_loss(v_pred, v_tgt)
        loss.backward()
        optimizer.step()

        if epoch % cfg.log_interval == 0 or epoch == 1:
            tqdm.write(f"  Epoch {epoch:4d} | Loss: {loss.item():.6f}")

        if epoch % ckpt_interval == 0 or epoch == cfg.diff_epochs:
            _save_diff_ckpt(epoch)

    dit.eval()
    melody_enc.eval()
    text_enc.eval()
    return dit, melody_enc, text_enc, diffusion, (latent_mean, latent_std)
