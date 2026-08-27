#!/bin/bash
# =============================================================================
# HiPerGator SLURM job — CLAP + chroma evaluation of the mood-editing model
# =============================================================================
# Submit with:  sbatch run_eval.sh
# Monitor with: squeue -u $USER
# Results land in $CKPT_DIR: eval_edits.csv, clap_validation.csv, eval_summary.txt
# =============================================================================

#SBATCH --job-name=mood-eval
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
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

# --- Paths (edit these if your checkpoint dir or data move) ---
CKPT_DIR="output/job_39423912"
DATA_ROOT="/orange/ufdatastudios/asherwheatle/DEAM_audio"
CLAP_CKPT="music_audioset_epoch_15_esc_90.14.pt"

# --- Make sure CLAP (and its torchvision dep) are installed ---
python -c "import laion_clap" 2>/dev/null || {
    echo "[SETUP] Installing laion-clap + torchvision..."
    # torchvision 0.22.0 matches torch 2.7.0/cu128; laion_clap imports it
    uv pip install "torchvision==0.22.0" --index-url https://download.pytorch.org/whl/cu128
    uv pip install laion-clap
}
if [ ! -f "$CLAP_CKPT" ]; then
    echo "[SETUP] Downloading CLAP music checkpoint..."
    wget -q "https://huggingface.co/lukewys/laion_clap/resolve/main/$CLAP_CKPT"
fi

echo "[RUN] Starting evaluation on $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python evaluate.py \
    --ckpt_dir "$CKPT_DIR" \
    --audio_dir "$DATA_ROOT/MEMD_audio" \
    --annotations_dir "$DATA_ROOT/DEAM_Annotations" \
    --clap_ckpt "$CLAP_CKPT" \
    --n_songs 20 --n_val 100 --edit_strength 0.5

echo "[RUN] Done on $(date)"
echo "[RUN] Results in: $CKPT_DIR"
