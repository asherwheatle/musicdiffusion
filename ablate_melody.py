"""Is melody control drowning out the mood edit?

sweep_conditioning.py showed the model reacts to the text (divergence scales
with cfg_scale) but moves audio in a mood-irrelevant direction (gain stays
negative, transfer pinned at chance). One candidate cause lives in dit.py:
the melody embedding is added to the latent stream at full magnitude at the
input of every ControlNet block, so the ControlNet may be pinning the output
to "reconstruct this exact clip" and leaving the text nothing to move.

This sweeps melody_scale -- the multiplier on the melody embedding -- holding
song and sampling noise fixed, and reports:

  clap_gain    does the edit move toward the correct mood? (signed)
  transfer%    does the target mood rank #1 of the moods on the edited audio?
  chroma       melody preservation (expected to FALL as melody_scale drops)
  clap_spread  how far apart the mood-edits of one song are

Interpretation:
  gain rises as melody_scale falls ... melody control WAS capping the edit;
                                       find the knee and retrain with a
                                       weaker ControlNet contribution.
  gain flat while chroma collapses ... melody is NOT the bottleneck. The
                                       text->audio mapping is what is broken;
                                       fixing it needs a retrain, not a knob.

It also prints the norm ratio ||melody_emb|| / ||input_proj(z)||, the direct
measure of how loud melody is relative to the latent it is added to.

Usage (GPU node):
  python ablate_melody.py \
      --ckpt_dir output/job_41279148 \
      --audio_dir /orange/ufdatastudios/asherwheatle/DEAM_audio/MEMD_audio \
      --annotations_dir /orange/ufdatastudios/asherwheatle/DEAM_audio/DEAM_Annotations \
      --clap_ckpt music_audioset_epoch_15_esc_90.14.pt \
      --n_songs 6
"""

import os
import csv
import argparse
import itertools

import numpy as np
import torch

from config import DiffusionConfig
from pipeline import load_bigvgan, pad_spectrogram, bigvgan_mel_spectrogram, FixedMelNormalizer
from melody import MelodyExtractor
from inference import edit_mood
from annotations import load_annotations, mood_from_va, song_id_from_filename
# reuse the evaluation helpers so the diagnostics stay consistent
from evaluate import (Clap, load_models, load_clip, chroma_similarity,
                      pick_annotated_songs, MOODS)

MELODY_SCALES = [0.0, 0.25, 0.5, 1.0]


def clap_spread(embs: list) -> float:
    """Mean pairwise cosine distance between embeddings."""
    d = [1.0 - float(a @ b) for a, b in itertools.combinations(embs, 2)]
    return float(np.mean(d)) if d else 0.0


@torch.no_grad()
def melody_vs_latent_norm(cfg, ae, dit, mel_enc, lat_m, lat_s,
                          wav: np.ndarray, bigvgan):
    """||melody_emb|| vs ||input_proj(z)||, both per-token means."""
    wt = torch.FloatTensor(wav).unsqueeze(0)
    mel = bigvgan_mel_spectrogram(wt, bigvgan)
    mel_norm = FixedMelNormalizer().normalize(mel)
    mel_padded, _ = pad_spectrogram(mel_norm.unsqueeze(0))
    z0 = ae.encoder(mel_padded.to(cfg.device))
    z0 = (z0 - lat_m) / lat_s

    B, C, H, W = z0.shape
    z_seq = dit.input_proj(z0.reshape(B, C, H * W).permute(0, 2, 1))

    extractor = MelodyExtractor(
        sr=cfg.sample_rate, n_bins=cfg.cqt_bins,
        bins_per_octave=cfg.cqt_bins_per_octave, hop_length=cfg.cqt_hop,
        fmin=cfg.cqt_fmin, top_k=cfg.melody_top_k,
        highpass_cutoff=cfg.highpass_cutoff)
    mel_idx = torch.from_numpy(extractor.extract(wav)).unsqueeze(0).to(cfg.device)
    melody_emb = mel_enc(mel_idx, W)

    return (float(melody_emb.norm(dim=-1).mean()),
            float(z_seq.norm(dim=-1).mean()))


def main():
    p = argparse.ArgumentParser(description="Melody-control ablation")
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--audio_dir", required=True)
    p.add_argument("--annotations_dir", required=True)
    p.add_argument("--clap_ckpt", default=None)
    p.add_argument("--n_songs", type=int, default=6)
    p.add_argument("--cfg_scale", type=float, default=5.0)
    p.add_argument("--edit_strength", type=float, default=0.6)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg = DiffusionConfig()
    cfg.cfg_scale = args.cfg_scale
    cfg.edit_strength = args.edit_strength
    va = load_annotations(args.annotations_dir, cfg.clip_start_seconds,
                          cfg.clip_seconds)

    print("[STEP] Loading BigVGAN...")
    bigvgan = load_bigvgan(cfg.device)
    sr = bigvgan.h.sampling_rate

    print("[STEP] Loading CLAP...")
    clap = Clap(args.clap_ckpt)

    files = pick_annotated_songs(args.audio_dir, va, args.n_songs)
    sample = load_clip(files[0], sr, cfg.clip_start_seconds, cfg.clip_seconds)
    print("[STEP] Loading mood-diffusion checkpoints...")
    ae, dit, mel_enc, txt_enc, diff, lat_m, lat_s = load_models(
        cfg, args.ckpt_dir, sample, bigvgan, clap=clap)

    songs = []
    for i, path in enumerate(files):
        wav = load_clip(path, sr, cfg.clip_start_seconds, cfg.clip_seconds)
        sid = song_id_from_filename(path)
        songs.append({"path": path, "sid": sid, "wav": wav,
                      "gt": mood_from_va(*va[sid]),
                      "orig_emb": clap.audio_embed(wav, sr),
                      "seed": args.seed + i})

    m_norm, z_norm = melody_vs_latent_norm(
        cfg, ae, dit, mel_enc, lat_m, lat_s, songs[0]["wav"], bigvgan)
    print(f"\n[NORMS] ||melody_emb|| = {m_norm:.3f}   "
          f"||input_proj(z)|| = {z_norm:.3f}   "
          f"ratio = {m_norm / (z_norm + 1e-8):.2f}x")
    print("[NORMS] dit.py adds these together at every ControlNet block; a "
          "ratio >> 1 means melody dominates the latent it is added to.")
    print(f"\n[GRID] cfg_scale={cfg.cfg_scale}  edit_strength={cfg.edit_strength}")

    rows = []
    for scale in MELODY_SCALES:
        gains, transfers, spreads, chromas = [], [], [], []
        for s in songs:
            edit_embs, orig_cos = [], clap.cos_to_moods(s["orig_emb"])
            for mood in MOODS:
                # SAME noise for every mood of this song => only text differs
                torch.manual_seed(s["seed"])
                wav_e = edit_mood(
                    torch.FloatTensor(s["wav"]).unsqueeze(0), mood,
                    ae, dit, mel_enc, txt_enc, diff, bigvgan, cfg,
                    lat_m, lat_s, melody_scale=scale,
                ).squeeze(0).numpy()
                emb = clap.audio_embed(wav_e, sr)
                edit_cos = clap.cos_to_moods(emb)
                ti = MOODS.index(mood)
                gains.append(float(edit_cos[ti] - orig_cos[ti]))
                transfers.append(int(MOODS[int(np.argmax(edit_cos))] == mood))
                chromas.append(chroma_similarity(s["wav"], wav_e, sr))
                edit_embs.append(emb)
            spreads.append(clap_spread(edit_embs))

        row = {
            "melody_scale": scale,
            "mean_clap_gain": round(float(np.mean(gains)), 4),
            "transfer_pct": round(100 * float(np.mean(transfers)), 1),
            "clap_spread": round(float(np.mean(spreads)), 4),
            "mean_chroma": round(float(np.mean(chromas)), 4),
        }
        rows.append(row)
        print(f"  melody_scale={scale:<5} | gain={row['mean_clap_gain']:+.4f} "
              f"transfer={row['transfer_pct']:>5.1f}% "
              f"| clap_spread={row['clap_spread']:.4f} "
              f"| chroma={row['mean_chroma']:.3f}")

    out_csv = os.path.join(args.ckpt_dir, "ablate_melody.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # ---- Verdict ----
    full = next(r for r in rows if r["melody_scale"] == 1.0)
    best = max(rows, key=lambda r: r["mean_clap_gain"])
    lift = best["mean_clap_gain"] - full["mean_clap_gain"]
    print(f"\n{'='*64}\n VERDICT\n{'='*64}")
    print(f" Best gain {best['mean_clap_gain']:+.4f} at melody_scale="
          f"{best['melody_scale']} vs {full['mean_clap_gain']:+.4f} at 1.0 "
          f"(lift {lift:+.4f}).")
    if lift > 0.02 and best["melody_scale"] < 1.0:
        print(f" => Melody control WAS capping the edit. chroma falls "
              f"{full['mean_chroma']:.3f} -> {best['mean_chroma']:.3f}; if that\n"
              f"    trade is acceptable, set cfg.melody_scale="
              f"{best['melody_scale']}, and consider retraining with a\n"
              f"    smaller ControlNet contribution.")
    elif best["mean_clap_gain"] <= 0:
        print(" => Melody is NOT the bottleneck: gain stays negative even with\n"
              "    melody removed entirely. The text->audio mapping itself is\n"
              "    wrong. Fix the text conditioning in training; no inference\n"
              "    knob will recover this.")
    else:
        print(" => Weakening melody helps only marginally. Look elsewhere.")
    print(f" Wrote {out_csv}")
    print("=" * 64)


if __name__ == "__main__":
    main()
