"""Does the text conditioning do ANYTHING? A controlled guidance sweep.

The decisive test: hold the song AND the sampling noise fixed, change only the
mood text, and see whether the output changes. edit_mood's sole source of
randomness is one torch.randn_like, so seeding right before each call makes all
5 mood-edits of a given song share identical noise. Then any divergence between
those 5 outputs is caused purely by the text — nothing else.

We sweep cfg_scale (guidance strength) x edit_strength and report, per combo:

  divergence   how different the 5 mood-edits of a song are from each other.
               ~0  => text does nothing (conditioning is dead / not learned).
               >0  => text changes the output. Anchored against the spread of 5
                      *different real songs* so you know what "a lot" means.
  clap_gain    does the edit move toward the *correct* mood (signed).
  transfer%    does the target mood rank #1 of 5 on the edited audio.
  chroma       melody preservation (sanity: should stay high).

Interpretation:
  divergence ~0 everywhere ............ conditioning is DEAD; retrain/rethink it.
  divergence grows w/ cfg but gain<=0 . model reacts to text but not in a
                                        mood-aligned way (bad text encoder).
  gain>0 at higher cfg ................ conditioning WORKS, was just too gentle.

Usage (GPU node):
  python sweep_conditioning.py \
      --ckpt_dir output/job_39423912 \
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
from pipeline import load_bigvgan
from inference import edit_mood
from annotations import load_annotations, mood_from_va, song_id_from_filename
# reuse the evaluation helpers so the two scripts stay consistent
from evaluate import (Clap, load_models, load_clip, chroma_similarity,
                      pick_annotated_songs, MOODS)

# Grid. cfg_scale=1.0 means "plain conditional, no guidance amplification";
# higher amplifies the (conditional - unconditional) difference.
CFG_SCALES = [1.0, 3.0, 5.0, 7.0]
EDIT_STRENGTHS = [0.6, 0.8]


def rms_divergence(wavs: list) -> float:
    """Mean pairwise RMS difference between same-song edits (model-free)."""
    diffs = []
    for a, b in itertools.combinations(wavs, 2):
        t = min(len(a), len(b))
        diffs.append(float(np.sqrt(np.mean((a[:t] - b[:t]) ** 2))))
    return float(np.mean(diffs)) if diffs else 0.0


def clap_spread(embs: list) -> float:
    """Mean pairwise cosine distance between embeddings (semantic divergence)."""
    d = []
    for a, b in itertools.combinations(embs, 2):
        d.append(1.0 - float(a @ b))
    return float(np.mean(d)) if d else 0.0


def main():
    p = argparse.ArgumentParser(description="Controlled text-conditioning sweep")
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--audio_dir", required=True)
    p.add_argument("--annotations_dir", required=True)
    p.add_argument("--clap_ckpt", default=None)
    p.add_argument("--n_songs", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg = DiffusionConfig()
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
        cfg, args.ckpt_dir, sample, bigvgan)

    # Preload each song's clip + original CLAP embedding once.
    songs = []
    for i, path in enumerate(files):
        wav = load_clip(path, sr, cfg.clip_start_seconds, cfg.clip_seconds)
        sid = song_id_from_filename(path)
        songs.append({
            "path": path, "sid": sid, "wav": wav,
            "gt": mood_from_va(*va[sid]),
            "orig_emb": clap.audio_embed(wav, sr),
            "seed": args.seed + i,       # fixed noise seed for THIS song
        })

    # Reference anchor: how far apart are DIFFERENT real songs in CLAP space?
    # This is the scale that "full divergence" looks like.
    ref_spread = clap_spread([s["orig_emb"] for s in songs])
    print(f"\n[ANCHOR] CLAP spread between {len(songs)} different real songs: "
          f"{ref_spread:.3f}  (this is what strong divergence looks like)")

    rows = []
    for cfg_scale, strength in itertools.product(CFG_SCALES, EDIT_STRENGTHS):
        cfg.cfg_scale = cfg_scale
        cfg.edit_strength = strength
        gains, transfers, divs, spreads, chromas = [], [], [], [], []

        for s in songs:
            wav_edits, edit_embs, orig_cos = [], [], clap.cos_to_moods(s["orig_emb"])
            for mood in MOODS:
                # SAME noise for every mood of this song => only text differs
                torch.manual_seed(s["seed"])
                wav_e = edit_mood(
                    torch.FloatTensor(s["wav"]).unsqueeze(0), mood,
                    ae, dit, mel_enc, txt_enc, diff, bigvgan, cfg, lat_m, lat_s,
                ).squeeze(0).numpy()
                emb = clap.audio_embed(wav_e, sr)
                edit_cos = clap.cos_to_moods(emb)
                ti = MOODS.index(mood)
                gains.append(float(edit_cos[ti] - orig_cos[ti]))
                transfers.append(int(MOODS[int(np.argmax(edit_cos))] == mood))
                chromas.append(chroma_similarity(s["wav"], wav_e, sr))
                wav_edits.append(wav_e); edit_embs.append(emb)
            divs.append(rms_divergence(wav_edits))
            spreads.append(clap_spread(edit_embs))

        row = {
            "cfg_scale": cfg_scale, "edit_strength": strength,
            "mean_clap_gain": round(float(np.mean(gains)), 4),
            "transfer_pct": round(100 * float(np.mean(transfers)), 1),
            "audio_divergence": round(float(np.mean(divs)), 5),
            "clap_spread": round(float(np.mean(spreads)), 4),
            "clap_spread_vs_ref": round(float(np.mean(spreads)) / (ref_spread + 1e-8), 3),
            "mean_chroma": round(float(np.mean(chromas)), 4),
        }
        rows.append(row)
        print(f"  cfg={cfg_scale:<4} strength={strength:<4} | "
              f"gain={row['mean_clap_gain']:+.4f} transfer={row['transfer_pct']:>5.1f}% "
              f"| divergence={row['audio_divergence']:.5f} "
              f"clap_spread={row['clap_spread']:.4f} "
              f"({row['clap_spread_vs_ref']*100:.0f}% of between-song) "
              f"| chroma={row['mean_chroma']:.3f}")

    out_csv = os.path.join(args.ckpt_dir, "sweep_conditioning.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # ---- Verdict ----
    max_spread_ratio = max(r["clap_spread_vs_ref"] for r in rows)
    best_gain = max(r["mean_clap_gain"] for r in rows)
    best = max(rows, key=lambda r: r["mean_clap_gain"])
    print(f"\n{'='*64}\n VERDICT\n{'='*64}")
    print(f" Max divergence reached: {max_spread_ratio*100:.0f}% of the "
          f"between-song spread.")
    if max_spread_ratio < 0.05:
        print(" => Text conditioning is effectively DEAD: changing the mood text\n"
              "    barely changes the output at any guidance scale. The model\n"
              "    ignores the text encoder. Fix training, not inference.")
    elif best_gain > 0.01:
        print(f" => Conditioning WORKS but was too gentle. Best combo: "
              f"cfg_scale={best['cfg_scale']}, edit_strength={best['edit_strength']}\n"
              f"    (gain={best['mean_clap_gain']:+.4f}, "
              f"transfer={best['transfer_pct']:.1f}%). Use these at inference.")
    else:
        print(" => The model REACTS to the text (output changes) but not toward\n"
              "    the correct mood (gain<=0). The text encoder learned to alter\n"
              "    audio without learning mood semantics. Rework the text encoder.")
    print(f" Wrote {out_csv}")
    print("="*64)


if __name__ == "__main__":
    main()
