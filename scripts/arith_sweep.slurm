#!/usr/bin/env bash
#SBATCH --account=westai0047
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --job-name=arith_sweep
#SBATCH --output=/p/scratch/westai0047/looplm/logs/%x_%A_%a.out
#SBATCH --error=/p/scratch/westai0047/looplm/logs/%x_%A_%a.err
#SBATCH --array=0-3

# Arithmetic exit gate sweep — 4 experiments, one variable changed each.
#
# Task 0: more_ops    — max_ops=12, L=4 (harder problems, same model)
# Task 1: mul         — max_ops=4 + multiplication (harder operators)
# Task 2: shallow     — max_ops=4, L=1 (force model to need loops)
# Task 3: two_digit   — max_ops=4, two-digit operands (harder per-op)
#
# Baseline (previous run): max_ops=4, L=4, +/- only, single-digit, beta=0.1

set -euo pipefail

module load uv

REPO_DIR="${PROJECT}/LoopLM"
OUTPUT_BASE="${SCRATCH}/looplm/arith_sweep"
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

# Common args
COMMON="--num-recurrent-steps 4 --hidden-size 256 --num-heads 4 --intermediate-size 512 \
        --train-steps 5000 --batch-size 128 --seq-len 256 --lr 1e-3 --warmup-steps 500 \
        --beta-kl 0.1 --n-eval 500 --eval-every 1000 --log-every 100 \
        --use-wandb --wandb-project looplm"

case ${SLURM_ARRAY_TASK_ID} in
    0)
        NAME="more_ops"
        echo "==> Experiment: more_ops (max_ops=12, L=4)"
        uv run scripts/analyze.py arith \
            --max-ops 12 --num-layers 4 \
            --output-dir "${OUTPUT_BASE}/${NAME}" \
            --wandb-run-name "arith_${NAME}" \
            $COMMON
        ;;
    1)
        NAME="mul"
        echo "==> Experiment: mul (max_ops=4, L=4, +multiplication)"
        uv run scripts/analyze.py arith \
            --max-ops 4 --num-layers 4 --use-mul \
            --output-dir "${OUTPUT_BASE}/${NAME}" \
            --wandb-run-name "arith_${NAME}" \
            $COMMON
        ;;
    2)
        NAME="shallow"
        echo "==> Experiment: shallow (max_ops=4, L=1 — force loops)"
        uv run scripts/analyze.py arith \
            --max-ops 4 --num-layers 1 \
            --output-dir "${OUTPUT_BASE}/${NAME}" \
            --wandb-run-name "arith_${NAME}" \
            $COMMON
        ;;
    3)
        NAME="two_digit"
        echo "==> Experiment: two_digit (max_ops=4, L=4, operands 10-99)"
        uv run scripts/analyze.py arith \
            --max-ops 4 --num-layers 4 --two-digit \
            --output-dir "${OUTPUT_BASE}/${NAME}" \
            --wandb-run-name "arith_${NAME}" \
            $COMMON
        ;;
esac

echo "==> Done: ${NAME}"
