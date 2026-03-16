#!/usr/bin/env bash
#SBATCH --account=westai0047
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --job-name=mano_6x2_80k
#SBATCH --output=/p/scratch/westai0047/looplm/logs/%x_%j.out
#SBATCH --error=/p/scratch/westai0047/looplm/logs/%x_%j.err

set -euo pipefail

module load uv

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_DIR="${PROJECT}/LoopLM"
LOG_DIR="${SCRATCH}/looplm/logs"
OUT_DIR="${SCRATCH}/looplm/mano_6x2_80k"
WANDB_DIR="${SCRATCH}/looplm/wandb"

mkdir -p "$LOG_DIR" "$OUT_DIR" "$WANDB_DIR"

# ── Offline mode (compute nodes have no internet) ───────────────────────────
export HF_HOME="${PROJECT}/hf_cache"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

export WANDB_MODE=offline
export WANDB_DIR="$WANDB_DIR"

# ── 6x2 config, LR 2e-4, 80K steps (paper spec) ────────────────────────────
echo "==> Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "==> Config: 6:2  LR: 2e-4  Steps: 80000"
echo "==> Output dir: $OUT_DIR"

cd "$REPO_DIR"
uv run scripts/analyze.py mano \
    --max-ops            10                    \
    --n-train            500000                \
    --n-eval             1000                  \
    --model-preset       paper                 \
    --model-configs      "6:2"                 \
    --train-steps        80000                 \
    --batch-size         32                    \
    --accumulation-steps 4                     \
    --lr                 2e-4                  \
    --num-seeds          1                     \
    --seed               42                    \
    --seq-len            1024                  \
    --log-every          500                   \
    --checkpoint-every   5000                  \
    --device             auto                  \
    --output-dir         "$OUT_DIR"            \
    --use-wandb                                \
    --wandb-project      looplm

echo "==> Complete. Results at ${OUT_DIR}/mano_results.csv"
