#!/bin/bash
# =============================================================================
# HiPerGator SLURM job script — musicdiffusion pipeline
# =============================================================================
# Submit with:  sbatch run_pipeline.sh
# Monitor with: squeue -u $USER
# =============================================================================

#SBATCH --job-name=user-model-pilot
#SBATCH --output=logs/pilotb200%j.out
#SBATCH --error=logs/pilotb200%j.err
#SBATCH --partition=hpg-b200
#SBATCH --account=ufdatastudios
#SBATCH --qos=ufdatastudios
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --mail-user=asherwheatle@ufl.edu
#SBATCH --mail-type=ALL

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
#    flock ensures only one concurrent job performs the clone — others wait.
# ---------------------------------------------------------------------------
(
  flock -x 200
  if [ ! -d "bigvgan_v2_44khz_128band_512x" ]; then
      echo "[SETUP] Cloning BigVGAN model from HuggingFace..."
      module load git-lfs
      git lfs install
      git clone https://huggingface.co/nvidia/bigvgan_v2_44khz_128band_512x
  else
      echo "[SETUP] BigVGAN already present, skipping clone."
  fi
) 200>/tmp/bigvgan_clone.lock

# ---------------------------------------------------------------------------
# 7. Run the pipeline
#    Each job gets its own output directory keyed by SLURM job ID,
#    so concurrent runs never clobber each other's results.
# ---------------------------------------------------------------------------
OUTPUT_DIR="output/job_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

echo "[RUN] Starting pipeline on $(date)"
echo "[RUN] Job ID: $SLURM_JOB_ID  ->  output dir: $OUTPUT_DIR"
echo "[RUN] GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python pipeline.py \
    --audio_dir data/DEAM_audio/MEMD_audio \
    --output_dir "$OUTPUT_DIR" \
    --song_index 0 \
    --epochs_vae 300 \
    --epochs_gan 150 \
    --latent_dim 128

echo "[RUN] Done on $(date)"
echo "[RUN] Results saved to: $OUTPUT_DIR"
