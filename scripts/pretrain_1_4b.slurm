#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --job-name=looplm_1_4b
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Note: Specify --account and --partition on the command line if required by your cluster:
#     sbatch --account=YOUR_ACCOUNT --partition=YOUR_PARTITION scripts/pretrain_1_4b.slurm
# Or configure scripts/cluster_env.sh (from scripts/cluster_env.sh.example).

# LoopLM 1.4B pretraining — 4x H100 DDP
# Stage 1a (T=8) + Stage 1b (T=4), ~10B tokens on SmolLM-Corpus
#
# Resubmit for multi-job runs:
#   The script auto-resumes from the latest checkpoint if one exists.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "${SCRIPT_DIR}/cluster_env.sh" ] && source "${SCRIPT_DIR}/cluster_env.sh" || true

command -v uv &>/dev/null || module load uv 2>/dev/null || true

NGPUS=4

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
SCRATCH="${SCRATCH:-${HOME}/scratch}"
PROJECT="${PROJECT:-${HOME}}"
DATA_DIR="${DATA_DIR:-${SCRATCH}/looplm/data/smollm_pretokenized}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH}/looplm/pretrain_1_4b}"
CKPT_DIR="${CKPT_DIR:-${OUTPUT_DIR}/checkpoints}"
WANDB_DIR="${WANDB_DIR:-${SCRATCH}/looplm/wandb}"
LOG_DIR="${LOG_DIR:-${SCRATCH}/looplm/logs}"

mkdir -p "$CKPT_DIR" "$WANDB_DIR" "$LOG_DIR" "${REPO_DIR}/logs"

# ── Offline environment ──────────────────────────────────────────────────────
export HF_HOME="${HF_HOME:-${PROJECT}/hf_cache}"
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export WANDB_MODE=offline
export WANDB_DIR="$WANDB_DIR"
export PYTHONUNBUFFERED=1

# ── Find latest checkpoint for resume ────────────────────────────────────────
RESUME_FLAG=""
LATEST_CKPT=$(ls -t "$CKPT_DIR"/step_*.pt 2>/dev/null | head -1 || true)
if [ -n "$LATEST_CKPT" ]; then
    echo "==> Resuming from checkpoint: $LATEST_CKPT"
    RESUME_FLAG="--resume-from $LATEST_CKPT"
fi

# ── Run info ─────────────────────────────────────────────────────────────────
echo "==> Node: $(hostname)"
echo "==> GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)x $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "==> DATA_DIR: $DATA_DIR"
echo "==> OUTPUT_DIR: $OUTPUT_DIR"
echo "==> Shards: $(ls "$DATA_DIR"/*.npy 2>/dev/null | wc -l) files"

cd "$REPO_DIR"

# ── Training ─────────────────────────────────────────────────────────────────
# 4 GPUs × micro_batch=2 × grad_accum=32 × seq_len=4096 = 1,048,576 tokens/step
# Gradient checkpointing required for T=8 with 1.4B model
# Stage 1a: 5000 steps (5B tokens), Stage 1b: 5000 steps (5B tokens)
uv run torchrun --nproc_per_node=$NGPUS scripts/train.py \
    --model-config    ouro_1_4b       \
    --data-dir        "$DATA_DIR"     \
    --batch-size      2               \
    --grad-accum-steps 32             \
    --seq-len         4096            \
    --gradient-checkpointing          \
    --stage1a-steps   5000            \
    --stage1b-steps   5000            \
    --lr              3e-4            \
    --beta-kl         0.1             \
    --log-every       10              \
    --save-every      500             \
    --output-dir      "$OUTPUT_DIR"   \
    --use-wandb                       \
    --wandb-project   looplm          \
    --run-name        "1_4b_smollm_pilot" \
    --eval-every      1000            \
    --eval-tasks      "hellaswag,arc_easy" \
    --eval-limit      200             \
    $RESUME_FLAG

echo "==> Training complete (or time limit reached)."
echo "==> To sync wandb: wandb sync ${WANDB_DIR}/offline-run-*"
