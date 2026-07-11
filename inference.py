"""Mood editing inference via SDEdit + classifier-free guidance."""

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import DiffusionConfig
from autoencoder import LatentAutoencoder
from melody import MelodyExtractor, MelodyEncoder
from text_encoder import TextEncoder
from dit import MoodDiT
from diffusion import GaussianDiffusion
from pipeline import bigvgan_mel_spectrogram, MelNormalizer, pad_spectrogram, unpad_spectrogram


@torch.no_grad()
def edit_mood(
    waveform: torch.Tensor,
    mood_text: str,
    ae: LatentAutoencoder,
    dit: MoodDiT,
    melody_enc: MelodyEncoder,
    text_enc: TextEncoder,
    diffusion: GaussianDiffusion,
    bigvgan_model,
    cfg: DiffusionConfig,
) -> torch.Tensor:
    """
    Edit the mood of an audio waveform using text conditioning.

    SDEdit approach (paper section IV-C):
      1. Encode input mel -> latent z0
      2. Add noise to z0 up to timestep t_start (controlled by edit_strength)
      3. Denoise from t_start -> 0 with text + melody conditioning
      4. CFG scale 7 on text only (melody unguided, per paper)
      5. Decode -> BigVGAN -> output waveform

    edit_strength=0: no change. edit_strength=1: full regen from noise.

    Returns:
        wav_out: (1, T_samples) output waveform tensor
    """
    device = cfg.device
    wav_np = waveform.squeeze().numpy()

    mel = bigvgan_mel_spectrogram(waveform, bigvgan_model)
    normalizer = MelNormalizer()
    mel_norm = normalizer.normalize(mel)
    mel_padded, orig_hw = pad_spectrogram(mel_norm.unsqueeze(0))
    mel_padded = mel_padded.to(device)

    z0 = ae.encoder(mel_padded)
    print(f"[EDIT] Latent z0: {z0.shape}")

    extractor = MelodyExtractor(
        sr=cfg.sample_rate, n_bins=cfg.cqt_bins,
        bins_per_octave=cfg.cqt_bins_per_octave,
        hop_length=cfg.cqt_hop, fmin=cfg.cqt_fmin,
        top_k=cfg.melody_top_k, highpass_cutoff=cfg.highpass_cutoff,
    )
    melody_indices = extractor.extract(wav_np)
    melody_tensor = torch.from_numpy(melody_indices).unsqueeze(0).to(device)
    W_lat = z0.shape[-1]
    melody_emb = melody_enc(melody_tensor, W_lat)

    tokens = TextEncoder.tokenize(mood_text, cfg.text_max_len).unsqueeze(0).to(device)
    text_emb = text_enc(tokens)
    null_tokens = TextEncoder.tokenize("", cfg.text_max_len).unsqueeze(0).to(device)
    null_text_emb = text_enc(null_tokens)

    # SDEdit: noise z0 up to t_start
    T = cfg.num_train_timesteps
    t_start = max(1, min(int(cfg.edit_strength * T), T - 1))

    step_ratio = T // cfg.num_inference_steps
    timesteps = list(range(t_start, 0, -step_ratio))
    if timesteps[-1] != 0:
        timesteps.append(0)

    noise = torch.randn_like(z0)
    t_tensor = torch.tensor([t_start], device=device)
    z_t = diffusion.q_sample(z0, t_tensor, noise)

    print(f"[EDIT] SDEdit from t={t_start} ({len(timesteps)} steps), "
          f"CFG scale={cfg.cfg_scale}")
    print(f"[EDIT] Mood text: \"{mood_text}\"")

    # DDIM denoising with CFG on text only (paper section IV-C)
    for i in tqdm(range(len(timesteps) - 1), desc="Denoising"):
        t_cur = torch.tensor([timesteps[i]], device=device)
        t_prev = torch.tensor([timesteps[i + 1]], device=device)

        v_cond = dit(z_t, t_cur, text_emb, melody_emb)
        v_uncond = dit(z_t, t_cur, null_text_emb, melody_emb)
        v_guided = v_uncond + cfg.cfg_scale * (v_cond - v_uncond)

        z_t = diffusion.ddim_step(z_t, v_guided, t_cur, t_prev)

    recon_mel_norm = ae.decoder(z_t)
    if recon_mel_norm.shape != mel_padded.shape:
        recon_mel_norm = F.interpolate(
            recon_mel_norm, size=mel_padded.shape[2:],
            mode="bilinear", align_corners=False
        )
    recon_mel_norm = unpad_spectrogram(recon_mel_norm, orig_hw)
    recon_mel = normalizer.denormalize(recon_mel_norm.squeeze(0).cpu())

    with torch.inference_mode():
        wav_out = bigvgan_model(recon_mel.to(device))
    return wav_out.squeeze(0).cpu()
