import os

import torch


class DiffusionConfig:
    sample_rate = 44100
    clip_seconds = 5
    # Training clips start 15 s in: DEAM's dynamic annotations only cover
    # 15 s onward, so this aligns the audio with its mood label
    clip_start_seconds = 15
    # 6 x 5 s covers 15-45 s, the whole annotated region of a DEAM
    # excerpt; each clip gets the mood of its own 5 s window
    clips_per_song = 6
    n_mels = 128

    # Mood balancing via waveform augmentation (see augment.py). DEAM has
    # far fewer dark clips than happy ones; we manufacture extra dark clips
    # by pitch-shifting / time-shifting / adding noise to the real ones.
    # Every mood except the majority "happy and uplifting" is boosted to
    # parity with it (~5.8k clips each); empty tuple disables augmentation.
    augment_moods = (
        "dark and mysterious",
        "sad and melancholic",
        "energetic and powerful",
        "calm and peaceful",
    )
    augment_target = None        # target clips/mood; None = match largest mood
    max_aug_per_clip = 12        # ceiling on variants per real clip
    aug_max_semitones = 2.0      # pitch shift range (+/-)
    aug_max_shift_frac = 0.2     # time shift range as fraction of clip
    aug_snr_db_range = (20.0, 35.0)  # gaussian-noise SNR range
    augment_seed = 0
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
    # Raised from 16: the old value badly underfilled a modern GPU, leaving it
    # copy-stalled rather than compute-bound.
    batch_size = 64
    num_workers = 4           # DataLoader workers for the autoencoder loop
    diff_lr = 1e-4
    diff_epochs = 10000
    cfg_scale = 1.5
    cfg_dropout = 0.1

    # Inference
    num_inference_steps = 50
    edit_strength = 0.35

    log_interval = 50
    output_dir = "output"

    # Checkpointing (eviction resilience — HiPerGator jobs can be killed
    # mid-run). Every interval we write a resumable *_ckpt.pt (model +
    # optimizer + step) and refresh the inference-ready autoencoder.pt /
    # diffusion.pt, so an interrupted run is both restartable and usable.
    ae_ckpt_interval = 25       # epochs between autoencoder checkpoints
    diff_ckpt_interval = 1000   # steps between diffusion checkpoints
    resume = True               # resume from *_ckpt.pt in output_dir if present
