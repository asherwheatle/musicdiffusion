#!/bin/bash
# =============================================================================
# HiPerGator SLURM job — controlled text-conditioning sweep
# =============================================================================
# Submit with:  sbatch run_sweep.sh
# Monitor with: squeue -u $USER
# Result:       $CKPT_DIR/sweep_conditioning.csv  (+ verdict in the .out log)
# =============================================================================

#SBATCH --job-name=mood-sweep
#SBATCH --output=logs/sweep_%j.out
#SBATCH --error=logs/sweep_%j.err
#SBATCH --partition=hpg-turin
#SBATCH --account=ufdatastudios
#SBATCH --qos=ufdatastudios
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --mail-user=asherwheatle@ufl.edu
#SBATCH --mail-type=ALL

module purge
module load cuda/12.8.1

export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/musicdiffusion"
source "$UV_PROJECT_ENVIRONMENT/bin/activate"

mkdir -p logs

# --- Paths (edit if your checkpoint dir or data move) ---
CKPT_DIR="output/job_39423912"
DATA_ROOT="/orange/ufdatastudios/asherwheatle/DEAM_audio"
CLAP_CKPT="music_audioset_epoch_15_esc_90.14.pt"

# --- Make sure CLAP (and its torchvision dep) are installed ---
python -c "import laion_clap" 2>/dev/null || {
    echo "[SETUP] Installing laion-clap + torchvision..."
    uv pip install "torchvision==0.22.0" --index-url https://download.pytorch.org/whl/cu128
    uv pip install laion-clap
}
if [ ! -f "$CLAP_CKPT" ]; then
    echo "[SETUP] Downloading CLAP music checkpoint..."
    wget -q "https://huggingface.co/lukewys/laion_clap/resolve/main/$CLAP_CKPT"
fi

echo "[RUN] Starting conditioning sweep on $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python sweep_conditioning.py \
    --ckpt_dir "$CKPT_DIR" \
    --audio_dir "$DATA_ROOT/MEMD_audio" \
    --annotations_dir "$DATA_ROOT/DEAM_Annotations" \
    --clap_ckpt "$CLAP_CKPT" \
    --n_songs 6

echo "[RUN] Done on $(date)"
echo "[RUN] Result in: $CKPT_DIR/sweep_conditioning.csv"
