"""Does the conditioning path work at all, independent of the modality gap?

Every diagnostic so far has fed the DiT a CLAP *text* embedding at inference
while training fed it CLAP *audio* embeddings (cfg.clap_cond_source="audio").
Those two live in offset cones — measured on this corpus, the audio and text
means sit at cosine ~0.21, while two *unrelated songs* sit at ~0.34. So the
model meets an out-of-distribution vector at inference and we cannot tell
whether a negative CLAP gain means "conditioning is broken" or merely
"conditioning was handed coordinates from the wrong cone".

This script separates those. Same checkpoint, same songs, same sampling noise
- only the conditioning VECTOR changes:

  text        the CLAP text embedding of the mood prompt. Reproduces current
              inference; the control arm.
  audio       the mean CLAP *audio* embedding of real clips whose ground-truth
              mood is the target. Same cone the model trained on, so the gap
              is removed entirely. THE DECISIVE ARM.
  text_shift  text embedding translated by (mean_audio - mean_text) and
              renormalized. The cheap "subtract the modality offset" fix
              (Liang et al. 2022), tested before paying to retrain.

Scoring is unchanged from evaluate.py - the judge is still CLAP cosine of the
edited audio against the mood *text* prompt - because that is the goal we
actually care about. Only the input side moves.

Read the result:
  audio gain > 0, text gain <= 0 ... conditioning WORKS; the gap is the bug.
                                     Invest in prompt augmentation / offset
                                     correction / a fitted text->audio map.
  audio gain <= 0 too ............... the gap is NOT the bug. The fault is
                                     inside the DiT cross-attention path and
                                     no embedding-space fix will touch it.
  text_shift ~ audio ................ the offset fix alone is enough.

Usage (GPU node):
  python probe_conditioning.py \
      --ckpt_dir output/job_41644276 \
      --audio_dir /orange/ufdatastudios/asherwheatle/DEAM_audio/MEMD_audio \
      --annotations_dir /orange/ufdatastudios/asherwheatle/DEAM_audio/DEAM_Annotations \
      --clap_ckpt music_audioset_epoch_15_esc_90.14.pt \
      --n_songs 10 --n_ref 24
"""

import os
import csv
import argparse

import numpy as np
import torch

from config import DiffusionConfig
from pipeline import load_bigvgan
from inference import edit_mood
from annotations import (load_annotations, mood_from_va,
                         song_id_from_filename)
# reuse the evaluation helpers so this script and evaluate.py agree on the judge
from evaluate import (Clap, load_models, load_clip, chroma_similarity,
                      pick_annotated_songs, MOODS, _l2)

ARMS = ["text", "audio", "text_shift"]


def build_reference_embeddings(clap, files, va, sr, start_s, dur_s, n_ref):
    """Mean CLAP *audio* embedding per mood, from real annotated clips.

    These are the vectors the model actually trained on (cfg.clap_cond_source
    ="audio" conditions each clip on its own audio embedding), so a centroid is
    an in-distribution stand-in for "what this mood looks like" in the cone the
    DiT learned. Returns (per_mood_centroid, mean_of_all_audio).
    """
    per_mood = {m: [] for m in MOODS}
    for path in files:
        gt = mood_from_va(*va[song_id_from_filename(path)])
        if gt is None or len(per_mood[gt]) >= n_ref:
            continue
        wav = load_clip(path, sr, start_s, dur_s)
        per_mood[gt].append(clap.audio_embed(wav, sr))
        if all(len(v) >= n_ref for v in per_mood.values()):
            break

    cents, pool = {}, []
    for m in MOODS:
        if not per_mood[m]:
            raise RuntimeError(f"No reference clips found for mood {m!r}. "
                               f"Raise --n_ref_pool.")
        arr = np.stack(per_mood[m])
        pool.append(arr)
        cents[m] = _l2(arr.mean(axis=0))
        print(f"[REF] {m:24s} centroid from {len(arr):3d} clips")
    mean_audio = _l2(np.concatenate(pool, axis=0).mean(axis=0))
    return cents, mean_audio


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--audio_dir", required=True)
    p.add_argument("--annotations_dir", required=True)
    p.add_argument("--clap_ckpt", default=None)
    p.add_argument("--n_songs", type=int, default=10,
                   help="Test songs to edit (each x every mood x every arm)")
    p.add_argument("--n_ref", type=int, default=24,
                   help="Clips per mood averaged into the audio centroid")
    p.add_argument("--n_ref_pool", type=int, default=160,
                   help="Candidate songs scanned to fill the reference set")
    p.add_argument("--cfg_scale", type=float, default=None,
                   help="Override cfg.cfg_scale (default: config value)")
    p.add_argument("--edit_strength", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg = DiffusionConfig()
    cfg.edit_strength = args.edit_strength
    if args.cfg_scale is not None:
        cfg.cfg_scale = args.cfg_scale
    if args.clap_ckpt:
        cfg.clap_ckpt = args.clap_ckpt
    sr, start_s, dur_s = cfg.sample_rate, cfg.clip_start_seconds, cfg.clip_seconds

    print(f"[PROBE] ckpt={args.ckpt_dir}  cfg_scale={cfg.cfg_scale}  "
          f"edit_strength={cfg.edit_strength}  moods={len(MOODS)}")

    va = load_annotations(args.annotations_dir, cfg.clip_start_seconds,
                          cfg.clip_seconds)
    clap = Clap(cfg.clap_ckpt)
    bigvgan = load_bigvgan(device=cfg.device)

    # Reference songs (for the audio centroids) must NOT overlap the test
    # songs, or the centroid has seen the very clip it will be used to edit.
    pool = pick_annotated_songs(args.audio_dir, va, args.n_ref_pool)
    test_files = pick_annotated_songs(args.audio_dir, va, args.n_songs)
    test_ids = {song_id_from_filename(f) for f in test_files}
    ref_files = [f for f in pool if song_id_from_filename(f) not in test_ids]
    print(f"[PROBE] {len(test_files)} test songs, {len(ref_files)} reference "
          f"candidates (disjoint)")

    cents, mean_audio = build_reference_embeddings(
        clap, ref_files, va, sr, start_s, dur_s, args.n_ref)

    # Modality offset: where the text cone sits relative to the audio cone.
    mean_text = _l2(clap.text_emb.mean(axis=0))
    offset = mean_audio - mean_text
    print(f"[GAP] cos(mean_audio, mean_text) = {float(mean_audio @ mean_text):+.4f}")
    for i, m in enumerate(MOODS):
        print(f"[GAP] cos(text[{m[:12]:12s}], centroid[{m[:12]:12s}]) = "
              f"{float(clap.text_emb[i] @ cents[m]):+.4f}")

    models = load_models(cfg, args.ckpt_dir,
                         load_clip(test_files[0], sr, start_s, dur_s),
                         bigvgan, clap=clap)
    ae, dit, melody_enc, text_enc, diffusion, lat_mean, lat_std = models

    def cond_for(arm, mood):
        """The CLAP vector fed to the DiT for this arm/mood (None = text)."""
        if arm == "text":
            return None
        if arm == "audio":
            return torch.from_numpy(cents[mood])
        if arm == "text_shift":
            i = MOODS.index(mood)
            return torch.from_numpy(_l2(clap.text_emb[i] + offset))
        raise ValueError(arm)

    rows = []
    for si, path in enumerate(test_files):
        sid = song_id_from_filename(path)
        gt = mood_from_va(*va[sid])
        wav = load_clip(path, sr, start_s, dur_s)
        cos_orig = clap.cos_to_moods(clap.audio_embed(wav, sr))
        wav_t = torch.FloatTensor(wav).unsqueeze(0)

        for mi, mood in enumerate(MOODS):
            for arm in ARMS:
                # Identical noise across arms AND moods for a given song, so
                # any difference is caused purely by the conditioning vector.
                torch.manual_seed(args.seed + si)
                wav_e = edit_mood(
                    wav_t, mood, ae, dit, melody_enc, text_enc, diffusion,
                    bigvgan, cfg, lat_mean, lat_std,
                    cond_emb=cond_for(arm, mood),
                ).squeeze().numpy()

                cos_edit = clap.cos_to_moods(clap.audio_embed(wav_e, sr))
                rows.append({
                    "song": os.path.basename(path), "song_id": sid,
                    "gt_mood": gt, "target_mood": mood, "arm": arm,
                    "clap_cos_original": round(float(cos_orig[mi]), 4),
                    "clap_cos_edited": round(float(cos_edit[mi]), 4),
                    "clap_gain": round(float(cos_edit[mi] - cos_orig[mi]), 4),
                    "clap_pred_edited": MOODS[int(np.argmax(cos_edit))],
                    "transfer_success": int(MOODS[int(np.argmax(cos_edit))] == mood),
                    "chroma_sim": round(chroma_similarity(wav, wav_e, sr), 4),
                })
        print(f"  {os.path.basename(path)} (gt={gt}) done  [{si+1}/{len(test_files)}]")

    out_csv = os.path.join(args.ckpt_dir, "probe_conditioning.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[PROBE] Wrote {out_csv}")

    # ---- summary -----------------------------------------------------------
    def agg(arm, key, mood=None):
        v = [r[key] for r in rows if r["arm"] == arm
             and (mood is None or r["target_mood"] == mood)]
        return float(np.mean(v)) if v else float("nan")

    print(f"\n{'='*64}\n PROBE SUMMARY  ({len(test_files)} songs x "
          f"{len(MOODS)} moods x {len(ARMS)} arms)\n{'='*64}")
    print(f" {'arm':<12}{'mean gain':>11}{'gain>0':>9}{'transfer%':>11}"
          f"{'chroma':>9}")
    for arm in ARMS:
        sub = [r for r in rows if r["arm"] == arm]
        pos = sum(1 for r in sub if r["clap_gain"] > 0)
        print(f" {arm:<12}{agg(arm,'clap_gain'):>+11.4f}{f'{pos}/{len(sub)}':>9}"
              f"{100*agg(arm,'transfer_success'):>10.1f}%{agg(arm,'chroma_sim'):>9.3f}")

    print(f"\n per-mood mean gain")
    print(f" {'arm':<12}" + "".join(f"{m[:16]:>18}" for m in MOODS))
    for arm in ARMS:
        print(f" {arm:<12}" + "".join(f"{agg(arm,'clap_gain',m):>+18.4f}"
                                      for m in MOODS))

    g_text, g_audio, g_shift = (agg(a, "clap_gain") for a in ARMS)
    print(f"\n{'='*64}\n VERDICT\n{'='*64}")
    if g_audio > 0 >= g_text:
        print(" Conditioning path WORKS. Feeding an in-distribution CLAP audio\n"
              " vector produces positive gain; only the text vector fails.\n"
              " => The modality gap IS the bug. Fix the input side: paraphrase\n"
              "    augmentation on the text path, offset correction, or a\n"
              "    fitted text->audio map. No DiT surgery needed.")
    elif g_audio <= 0:
        print(" Conditioning path is BROKEN independent of the modality gap.\n"
              " Even an in-distribution audio vector - the exact kind the model\n"
              " trained on - fails to move audio toward its mood.\n"
              " => Stop tuning embeddings. The fault is in how conditioning\n"
              "    reaches the latent (dit.py cross-attention / the trainable\n"
              "    projection), or the model never learned mood at all.")
    else:
        print(" Mixed: text already works, so the gap was not the blocker.\n"
              " Re-check the evaluation setup rather than the conditioning.")
    if g_shift > g_text:
        print(f"\n Offset correction helps: {g_text:+.4f} -> {g_shift:+.4f} "
              f"({g_shift - g_text:+.4f}).")
        if g_audio > 0 and g_shift >= 0.8 * g_audio:
            print(" It recovers most of the audio-arm gain — the cheap fix may"
                  " be enough on its own.")
    else:
        print(f"\n Offset correction does NOT help ({g_text:+.4f} -> "
              f"{g_shift:+.4f}); a constant translation is not the whole gap.")
    print("="*64)


if __name__ == "__main__":
    main()
