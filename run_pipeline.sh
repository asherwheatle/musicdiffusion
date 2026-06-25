#!/bin/bash
# =============================================================================
# HiPerGator SLURM job script — musicdiffusion pipeline
# =============================================================================
# Submit with:  sbatch run_pipeline.sh
# Monitor with: squeue -u $USER
# =============================================================================

#SBATCH --job-name=musicdiffusion
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1          # request 1 A100 GPU (CUDA 12.4 capable)
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=04:00:00            # HH:MM:SS — adjust to your epoch count
#SBATCH --account=YOUR_GROUP       # <-- replace with your HiPerGator group/PI

# ---------------------------------------------------------------------------
# 1. Load system modules
# ---------------------------------------------------------------------------
module purge
module load cuda/12.4.1            # matches our cu124 PyTorch wheels

# ---------------------------------------------------------------------------
# 2. Install uv (user-local, no root needed) if not already present
# ---------------------------------------------------------------------------
if ! command -v uv &> /dev/null; then
    echo "[SETUP] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# ---------------------------------------------------------------------------
# 3. Create venv and sync all dependencies from pyproject.toml
#    uv will pull torch/torchaudio from the CUDA 12.4 index automatically.
#    Place the venv on $SCRATCH to avoid inode quota issues on $HOME.
# ---------------------------------------------------------------------------
export UV_PROJECT_ENVIRONMENT="$SCRATCH/.venvs/musicdiffusion"
mkdir -p "$SCRATCH/.venvs"

echo "[SETUP] Syncing dependencies with uv..."
uv sync

# ---------------------------------------------------------------------------
# 4. Activate the environment
# ---------------------------------------------------------------------------
source "$UV_PROJECT_ENVIRONMENT/bin/activate"

# ---------------------------------------------------------------------------
# 5. Create output and log directories
# ---------------------------------------------------------------------------
mkdir -p logs output

# ---------------------------------------------------------------------------
# 6. Clone BigVGAN from HuggingFace if not already present
#    (only downloads once; subsequent runs skip this)
# ---------------------------------------------------------------------------
if [ ! -d "bigvgan_v2_44khz_128band_512x" ]; then
    echo "[SETUP] Cloning BigVGAN model from HuggingFace..."
    module load git-lfs
    git lfs install
    git clone https://huggingface.co/nvidia/bigvgan_v2_44khz_128band_512x
fi

# ---------------------------------------------------------------------------
# 7. Run the pipeline
# ---------------------------------------------------------------------------
echo "[RUN] Starting pipeline on $(date)"
echo "[RUN] GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python pipeline.py \
    --audio_dir data/DEAM_audio/MEMD_audio \
    --output_dir output \
    --song_index 0 \
    --epochs_vae 300 \
    --epochs_gan 150 \
    --latent_dim 128

echo "[RUN] Done on $(date)"
