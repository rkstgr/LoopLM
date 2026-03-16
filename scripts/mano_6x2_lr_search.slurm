#!/usr/bin/env bash
#SBATCH --account=westai0047
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --job-name=mano_6x2_lr
#SBATCH --array=0-3
#SBATCH --output=/p/scratch/westai0047/looplm/logs/%x_%A_%a.out
#SBATCH --error=/p/scratch/westai0047/looplm/logs/%x_%A_%a.err

set -euo pipefail

module load uv

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_DIR="${PROJECT}/LoopLM"
LOG_DIR="${SCRATCH}/looplm/logs"
OUT_DIR="${SCRATCH}/looplm/mano_6x2_lr"
WANDB_DIR="${SCRATCH}/looplm/wandb"

mkdir -p "$LOG_DIR" "$OUT_DIR" "$WANDB_DIR"

# ── Offline mode (compute nodes have no internet) ───────────────────────────
export HF_HOME="${PROJECT}/hf_cache"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

export WANDB_MODE=offline
export WANDB_DIR="$WANDB_DIR"

# ── LR search for 6x2 config (paper LR candidates) ─────────────────────────
# Array task 0-3 maps to {5e-5, 1e-4, 2e-4, 5e-4}
LRS=(5e-5 1e-4 2e-4 5e-4)

LR="${LRS[$SLURM_ARRAY_TASK_ID]}"
TASK_OUT_DIR="${OUT_DIR}/lr_${LR}"

mkdir -p "$TASK_OUT_DIR"

echo "==> Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "==> Array task ${SLURM_ARRAY_TASK_ID}: config=6:2  lr=${LR}"
echo "==> Output dir: $TASK_OUT_DIR"

cd "$REPO_DIR"
uv run scripts/analyze.py mano \
    --max-ops            10                    \
    --n-train            500000                \
    --n-eval             1000                  \
    --model-preset       paper                 \
    --model-configs      "6:2"                 \
    --train-steps        2500                  \
    --batch-size         32                    \
    --accumulation-steps 4                     \
    --lr                 "$LR"                 \
    --num-seeds          1                     \
    --seed               42                    \
    --seq-len            1024                  \
    --log-every          100                   \
    --checkpoint-every   1000                  \
    --device             auto                  \
    --output-dir         "$TASK_OUT_DIR"       \
    --use-wandb                                \
    --wandb-project      looplm

echo "==> LR=${LR} complete. Results at ${TASK_OUT_DIR}/mano_results.csv"
