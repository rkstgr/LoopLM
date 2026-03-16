#!/usr/bin/env bash
#SBATCH --account=westai0047
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --job-name=mano_debug_ops3
#SBATCH --array=0-1
#SBATCH --output=/p/scratch/westai0047/looplm/logs/%x_%A_%a.out
#SBATCH --error=/p/scratch/westai0047/looplm/logs/%x_%A_%a.err

set -euo pipefail

module load uv

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_DIR="${PROJECT}/LoopLM"
LOG_DIR="${SCRATCH}/looplm/logs"
OUT_DIR="${SCRATCH}/looplm/mano_debug_ops3"
WANDB_DIR="${SCRATCH}/looplm/wandb"

mkdir -p "$LOG_DIR" "$OUT_DIR" "$WANDB_DIR"

# ── Offline mode (compute nodes have no internet) ───────────────────────────
export HF_HOME="${PROJECT}/hf_cache"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

export WANDB_MODE=offline
export WANDB_DIR="$WANDB_DIR"

# ── Debug run: max_ops=3, 20K steps, 12x1 baseline + 6x2 looped ─────────
CONFIGS=(       "12:1"  "6:2" )
BATCH_SIZES=(      64     32  )
ACCUM_STEPS=(       2      4  )

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
BATCH_SIZE="${BATCH_SIZES[$SLURM_ARRAY_TASK_ID]}"
ACCUMULATION="${ACCUM_STEPS[$SLURM_ARRAY_TASK_ID]}"
TASK_OUT_DIR="${OUT_DIR}/task_${SLURM_ARRAY_TASK_ID}"

mkdir -p "$TASK_OUT_DIR"

echo "==> Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "==> Array task ${SLURM_ARRAY_TASK_ID}: config=${CONFIG}  batch=${BATCH_SIZE}  accum=${ACCUMULATION}"
echo "==> Output dir: $TASK_OUT_DIR"

cd "$REPO_DIR"
uv run scripts/analyze.py mano \
    --max-ops            3                      \
    --n-train            500000                 \
    --n-eval             1000                   \
    --model-preset       paper                  \
    --model-configs      "$CONFIG"              \
    --train-steps        20000                  \
    --batch-size         "$BATCH_SIZE"          \
    --accumulation-steps "$ACCUMULATION"        \
    --lr                 1e-4                   \
    --num-seeds          1                      \
    --seed               42                     \
    --seq-len            1024                   \
    --log-every          200                    \
    --checkpoint-every   2000                   \
    --device             auto                   \
    --output-dir         "$TASK_OUT_DIR"        \
    --use-wandb                                 \
    --wandb-project      looplm

echo "==> Task ${SLURM_ARRAY_TASK_ID} complete. Results at ${TASK_OUT_DIR}/mano_results.csv"
