#!/usr/bin/env bash
#SBATCH --account=westai0047
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --job-name=arith_exit
#SBATCH --output=/p/scratch/westai0047/looplm/logs/%x_%j.out
#SBATCH --error=/p/scratch/westai0047/looplm/logs/%x_%j.err

# Arithmetic exit gate experiment — single H100, ~5-10 min
# Sweeps over beta_kl values to find what prevents entropy collapse.
#
# Usage:
#   sbatch scripts/arith_exit_gate.slurm              # default beta=0.1
#   BETA_KL=1.0 sbatch scripts/arith_exit_gate.slurm  # override beta

set -euo pipefail

module load uv

REPO_DIR="${PROJECT}/LoopLM"
OUTPUT_DIR="${SCRATCH}/looplm/arith_exit"
WANDB_DIR="${SCRATCH}/looplm/wandb"
LOG_DIR="${SCRATCH}/looplm/logs"

mkdir -p "$OUTPUT_DIR" "$WANDB_DIR" "$LOG_DIR"

export HF_HOME="${PROJECT}/hf_cache"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export WANDB_DIR="$WANDB_DIR"
export PYTHONUNBUFFERED=1

# Configurable beta_kl (override via environment variable)
BETA_KL="${BETA_KL:-0.1}"

echo "==> Node: $(hostname)"
echo "==> GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "==> beta_kl: $BETA_KL"

cd "$REPO_DIR"

uv run scripts/analyze.py arith \
    --max-ops       4                \
    --num-recurrent-steps 4          \
    --hidden-size   256              \
    --num-layers    4                \
    --num-heads     4                \
    --intermediate-size 512          \
    --train-steps   5000             \
    --batch-size    128              \
    --seq-len       128              \
    --lr            1e-3             \
    --warmup-steps  500              \
    --beta-kl       "$BETA_KL"       \
    --n-eval        500              \
    --eval-every    1000             \
    --log-every     100              \
    --output-dir    "${OUTPUT_DIR}/beta_${BETA_KL}" \
    --use-wandb                      \
    --wandb-project looplm           \
    --wandb-run-name "arith_beta${BETA_KL}"

echo "==> Done."
