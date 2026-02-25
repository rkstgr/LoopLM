#!/usr/bin/env bash
#SBATCH --account=FILL_IN_ACCOUNT        # <-- replace with your budget
#SBATCH --partition=westai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --job-name=looplm_small
#SBATCH --output=%SCRATCH%/looplm/logs/%x_%j.out
#SBATCH --error=%SCRATCH%/looplm/logs/%x_%j.err
set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_DIR="${SCRATCH}/looplm/checkpoints"
WANDB_DIR="${SCRATCH}/looplm/wandb"
LOG_DIR="${SCRATCH}/looplm/logs"

mkdir -p "$CKPT_DIR" "$WANDB_DIR" "$LOG_DIR"

# ── HuggingFace: use pre-staged cache, block network downloads ────────────────
export HF_HOME="${PROJECT}/hf_cache"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ── wandb: write offline runs to scratch, sync to cloud from login node ───────
export WANDB_MODE=offline
export WANDB_DIR="$WANDB_DIR"

# ── Run ───────────────────────────────────────────────────────────────────────
echo "==> Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "==> CKPT_DIR: $CKPT_DIR"
echo "==> WANDB_DIR: $WANDB_DIR"

cd "$REPO_DIR"
uv run scripts/train.py \
    --model-config    small       \
    --dataset         wikitext    \
    --dataset-config  wikitext-103-v1 \
    --max-steps       10000       \
    --batch-size      8           \
    --seq-len         512         \
    --lr              3e-4        \
    --beta-kl         0.1         \
    --log-every       100         \
    --save-every      2000        \
    --output-dir      "$CKPT_DIR" \
    --use-wandb                   \
    --wandb-project   looplm      \
    --run-name        "small_wikitext_10k_$(date +%Y%m%d_%H%M%S)"

echo "==> Training complete."
echo "==> To sync wandb: wandb sync ${WANDB_DIR}/offline-run-*"
