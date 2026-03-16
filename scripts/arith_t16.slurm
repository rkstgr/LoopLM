#!/usr/bin/env bash
#SBATCH --account=westai0047
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --job-name=arith_t16
#SBATCH --output=/p/scratch/westai0047/looplm/logs/%x_%j.out
#SBATCH --error=/p/scratch/westai0047/looplm/logs/%x_%j.err

# Train with T=16, eval at T={1,2,4,8,12,16,24,32}
# L=1, two-digit, max_ops=4

set -euo pipefail
module load uv

REPO_DIR="${PROJECT}/LoopLM"
OUTPUT_BASE="${SCRATCH}/looplm/arith_t16"
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

cd "$REPO_DIR"

uv run scripts/analyze.py arith \
    --max-ops       4                \
    --num-recurrent-steps 16         \
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
    --beta-kl       0.1              \
    --extrap-eval   1 2 4 8 12 16 24 32 \
    --n-eval        500              \
    --eval-every    5000             \
    --log-every     200              \
    --output-dir    "${OUTPUT_BASE}" \
    --wandb-run-name "arith_L1_T16_extrap" \
    --use-wandb --wandb-project looplm

echo "==> Done."
