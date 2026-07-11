# musicdiffusion

Implementation of [Editing Music with Melody and Text: Using ControlNet for Diffusion Transformer](https://arxiv.org/abs/2410.05151) (Hou et al., 2024) on top of a VAE + BigVGAN audio pipeline.

## What it does

Takes an audio clip and a text mood description (e.g. `"dark and mysterious"`) and outputs a version of the audio with the mood altered while preserving the original melody structure. The ControlNet branch locks in pitch/melody via CQT features; the DiT is steered by the text prompt via cross-attention and classifier-free guidance.

## Usage

```bash
# Full pipeline: train autoencoder -> train diffusion -> edit
python mood_diffusion.py --mode full --audio_dir data/DEAM_audio/MEMD_audio --text "dark and mysterious"

# Train autoencoder only
python mood_diffusion.py --mode train_ae --audio_dir data/DEAM_audio/MEMD_audio

# Train diffusion model only (requires saved autoencoder.pt)
python mood_diffusion.py --mode train_diff --audio_dir data/DEAM_audio/MEMD_audio

# Edit mood of an existing WAV (requires saved autoencoder.pt + diffusion.pt)
python mood_diffusion.py --mode edit --input output/original.wav --text "happy and uplifting" --edit_strength 0.7

# Run the basic reconstruction pipeline (no mood editing)
python pipeline.py --audio_dir data/DEAM_audio/MEMD_audio
```

## File structure

```
mood_diffusion.py   CLI entry point and main() — reads args, orchestrates the three phases
pipeline.py         BigVGAN vocoder loading, DEAM data loading, mel spectrogram utils,
                    MelNormalizer, and the original VAE-GAN reconstruction pipeline

config.py           DiffusionConfig — all hyperparameters in one place (epochs, lr,
                    model dims, diffusion schedule, CFG scale, etc.)

autoencoder.py      LatentEncoder / LatentDecoder / LatentAutoencoder — convolutional
                    autoencoder that compresses mel spectrograms into a 2D spatial latent
                    suitable for diffusion (no flattening bottleneck)

dit.py              MoodDiT — the full Diffusion Transformer with ControlNet branch
                    (paper §III-A). Contains DiTBlock (self-attn + cross-attn + MLP
                    with AdaLayerNorm timestep conditioning), ZeroLinear (zero-init
                    ControlNet output gates), and SinusoidalTimestepEmbedding

diffusion.py        GaussianDiffusion — cosine noise schedule, v-prediction objective
                    (Salimans & Ho 2022), q_sample, v_target, DDIM sampling step

melody.py           MelodyExtractor — high-pass filter + CQT + top-k bin selection
                    (paper §III-B). MelodyEncoder — embeds pitch indices and downsamples
                    to match the latent temporal resolution for ControlNet injection

text_encoder.py     TextEncoder — lightweight character-level transformer encoder for
                    mood text prompts. Produces cross-attention keys/values for DiTBlock.
                    Swap in T5-base here for production quality

train.py            train_autoencoder() — MSE training loop for the latent AE
                    train_diffusion() — v-prediction training loop for DiT+ControlNet
                    with CFG dropout on text conditioning

inference.py        edit_mood() — SDEdit inference: encode -> add noise to t_start ->
                    DDIM denoise with CFG (text only, melody unguided) -> decode -> BigVGAN

visualize.py        plot_mood_edit() — side-by-side mel spectrogram comparison plot
                    (original vs. mood-edited)

musicmake.py        Standalone script using shikhr/music_maker to generate MIDI
```

## Outputs

All outputs go to `output/` by default:

| File | Description |
|---|---|
| `original.wav` | Source audio clip (15s) |
| `ae_reconstruction.wav` | AE encode→decode round-trip, to verify latent quality |
| `edited_<text>.wav` | Mood-edited output |
| `autoencoder.pt` | Saved AE weights |
| `diffusion.pt` | Saved DiT + MelodyEncoder + TextEncoder weights |
| `mood_edit_comparison.png` | Side-by-side spectrogram plot |

## Key hyperparameters (config.py)

| Parameter | Default | Effect |
|---|---|---|
| `ae_epochs` | 500 | More = better latent reconstruction |
| `diff_epochs` | 1000 | More = stronger mood conditioning |
| `edit_strength` | 0.7 | 0 = no change, 1 = full regeneration from noise |
| `cfg_scale` | 7.0 | Higher = stronger text adherence, less fidelity |
| `d_model` | 256 | DiT model width |
| `n_dit_blocks` | 8 | DiT depth |
| `clip_seconds` | 15 | Audio clip length fed into the pipeline |
