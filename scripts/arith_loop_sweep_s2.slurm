#!/usr/bin/env bash
#SBATCH --account=westai0047
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --job-name=arith_loops2
#SBATCH --output=/p/scratch/westai0047/looplm/logs/%x_%A_%a.out
#SBATCH --error=/p/scratch/westai0047/looplm/logs/%x_%A_%a.err
#SBATCH --array=0-3

# Repeat loop sweep with seed=123 to check variance.
# L=1, two-digit, max_ops=4, T={1,2,4,8}

set -euo pipefail
module load uv

REPO_DIR="${PROJECT}/LoopLM"
OUTPUT_BASE="${SCRATCH}/looplm/arith_loops_s2"
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

T_LIST=(1 2 4 8)
T=${T_LIST[$SLURM_ARRAY_TASK_ID]}

# T=1 has no exit distribution, so use beta=0
BETA=0.1
if [ "$T" -eq 1 ]; then
    BETA=0.0
fi

NAME="L1_T${T}_ops4_2digit_s2"
echo "==> Experiment: ${NAME} (seed=123)"

uv run scripts/analyze.py arith \
    --max-ops       4                \
    --num-recurrent-steps "$T"       \
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
    --beta-kl       "$BETA"          \
    --n-eval        500              \
    --eval-every    1000             \
    --log-every     200              \
    --seed          123              \
    --output-dir    "${OUTPUT_BASE}/${NAME}" \
    --wandb-run-name "arith_${NAME}" \
    --use-wandb --wandb-project looplm

echo "==> Done: ${NAME}"
