#!/usr/bin/env bash
#SBATCH --account=westai0047
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --job-name=arith_base
#SBATCH --output=/p/scratch/westai0047/looplm/logs/%x_%A_%a.out
#SBATCH --error=/p/scratch/westai0047/looplm/logs/%x_%A_%a.err
#SBATCH --array=0-5

# Step 1: Find where L=1, T=1 breaks (<80% accuracy).
# Sweep two-digit difficulty: max_ops = 2,3,4,5,6,8
# All with L=1, T=1, h=256, two-digit operands.

set -euo pipefail
module load uv

REPO_DIR="${PROJECT}/LoopLM"
OUTPUT_BASE="${SCRATCH}/looplm/arith_baseline"
WANDB_DIR="${SCRATCH}/looplm/wandb"

mkdir -p "$OUTPUT_BASE" "$WANDB_DIR"

export HF_HOME="${PROJECT}/hf_cache"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export WANDB_DIR="$WANDB_DIR"
export PYTHONUNBUFFERED=1

echo "==> Node: $(hostname)"
echo "==> GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "==> Task: ${SLURM_ARRAY_TASK_ID}"

cd "$REPO_DIR"

# Map task ID to max_ops
MAX_OPS_LIST=(2 3 4 5 6 8)
MAX_OPS=${MAX_OPS_LIST[$SLURM_ARRAY_TASK_ID]}

NAME="L1_T1_ops${MAX_OPS}"
echo "==> Experiment: ${NAME} (L=1, T=1, two-digit, max_ops=${MAX_OPS})"

uv run scripts/analyze.py arith \
    --max-ops       "$MAX_OPS"       \
    --num-recurrent-steps 1          \
    --num-layers    1                \
    --hidden-size   256              \
    --num-heads     4                \
    --intermediate-size 512          \
    --two-digit                      \
    --train-steps   5000             \
    --batch-size    128              \
    --seq-len       256              \
    --lr            1e-3             \
    --warmup-steps  500              \
    --beta-kl       0.0              \
    --n-eval        500              \
    --eval-every    1000             \
    --log-every     200              \
    --output-dir    "${OUTPUT_BASE}/${NAME}" \
    --wandb-run-name "arith_${NAME}" \
    --use-wandb --wandb-project looplm

echo "==> Done: ${NAME}"
