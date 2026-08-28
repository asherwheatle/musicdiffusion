import os

import torch


class DiffusionConfig:
    sample_rate = 44100
    clip_seconds = 15
    # Training clips start 15 s in: DEAM's dynamic annotations only cover
    # 15 s onward, so this aligns the audio with its mood label
    clip_start_seconds = 15
    n_mels = 128
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data
    annotations_dir = os.path.join("data", "DEAM_Annotations")
    cache_dir = "cache"

    # Autoencoder
    ae_channels = [1, 32, 64, 128, 32]
    ae_lr = 1e-3
    ae_epochs = 100

    # CQT / Melody (paper section III-B)
    cqt_bins = 128
    cqt_bins_per_octave = 12
    cqt_hop = 512
    cqt_fmin = 8.18
    melody_top_k = 4
    highpass_cutoff = 261.2

    # DiT + ControlNet
    d_model = 256
    n_heads = 4
    n_dit_blocks = 8
    n_controlnet_blocks = 4
    mlp_ratio = 4
    dropout = 0.0

    # Text encoder (Lever A: frozen CLAP text tower + trainable projection)
    text_max_len = 128           # legacy char encoder only; unused by CLAP
    clap_ckpt = "music_audioset_epoch_15_esc_90.14.pt"
    text_n_tokens = 4            # length of the projected conditioning sequence

    # Diffusion
    num_train_timesteps = 1000
    prediction_type = "v"

    # Training
    n_train_songs = None      # None = entire annotated dataset (~1800 songs)
    batch_size = 16
    diff_lr = 1e-4
    diff_epochs = 10000
    cfg_scale = 1.5
    cfg_dropout = 0.1

    # Inference
    num_inference_steps = 50
    edit_strength = 0.35

    log_interval = 50
    output_dir = "output"
