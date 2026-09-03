#!/bin/bash
# =============================================================================
# HiPerGator SLURM job script — musicdiffusion pipeline
# =============================================================================
# Submit with:  sbatch run_pipeline.sh
# Monitor with: squeue -u $USER
# =============================================================================

#SBATCH --job-name=mood-diffusion-full
#SBATCH --output=logs/mood_full_%j.out
#SBATCH --error=logs/mood_full_%j.err
#SBATCH --partition=hpg-turin
#SBATCH --account=ufdatastudios
#SBATCH --qos=ufdatastudios
# Single-GPU training. Extra CPUs feed the pinned-memory DataLoader workers
# (num_workers in config.py) plus the pin_memory thread.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --mail-user=asherwheatle@ufl.edu
#SBATCH --mail-type=ALL

# Fail loudly: abort on any unhandled error, unset variable, or broken pipe
# so setup problems don't silently fall through to a half-run pipeline.
set -euo pipefail

# ---------------------------------------------------------------------------
# 1. Load system modules
# ---------------------------------------------------------------------------
module purge
module load cuda/12.8.1            # B200 GPUs require CUDA >= 12.8

# ---------------------------------------------------------------------------
# 2. Install uv (user-local, no root needed) if not already present
# ---------------------------------------------------------------------------
# uv installs to $HOME/.local/bin on Linux
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv &> /dev/null; then
    echo "[SETUP] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # re-source in case installer updated PATH
    export PATH="$HOME/.local/bin:$PATH"
fi

# ---------------------------------------------------------------------------
# 3. Create venv and sync all dependencies from pyproject.toml
#    uv will pull torch/torchaudio from the CUDA 12.8 index automatically.
#    Venv lives in $HOME/.venvs — always writable, no root needed.
# ---------------------------------------------------------------------------
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/musicdiffusion"
mkdir -p "$HOME/.venvs"

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

DATA_ROOT="/orange/ufdatastudios/asherwheatle/DEAM_audio"

# Run training under its own error guard so we can report the REAL outcome.
# Without this, a killed/crashed run still fell through to "[RUN] Done"
# and the job showed up as COMPLETED even with no saved model.
set +e
python mood_diffusion.py \
    --mode full \
    --audio_dir "$DATA_ROOT/MEMD_audio" \
    --annotations_dir "$DATA_ROOT/DEAM_Annotations" \
    --output_dir "$OUTPUT_DIR"
status=$?
set -e

if [ "$status" -eq 0 ]; then
    echo "[RUN] Done on $(date)"
    echo "[RUN] Results saved to: $OUTPUT_DIR"
else
    echo "[RUN] FAILED (python exit $status) on $(date)" >&2
    echo "[RUN] Partial checkpoints (if any) in: $OUTPUT_DIR" >&2
fi
# Propagate the real exit code so SLURM records FAILED, not COMPLETED.
exit "$status"
