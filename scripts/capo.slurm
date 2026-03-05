#!/usr/bin/env bash
#SBATCH --account=westai0047
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --job-name=looplm_capo
#SBATCH --output=/p/scratch/westai0047/looplm/logs/%x_%j.out
#SBATCH --error=/p/scratch/westai0047/looplm/logs/%x_%j.err

set -euo pipefail

module load uv

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_DIR="${PROJECT}/LoopLM"
LOG_DIR="${SCRATCH}/looplm/logs"
OUT_DIR="${SCRATCH}/looplm/capo"

mkdir -p "$LOG_DIR" "$OUT_DIR"

# ── HuggingFace: use pre-staged cache, block network downloads ────────────────
export HF_HOME="${PROJECT}/hf_cache"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

# ── Run ───────────────────────────────────────────────────────────────────────
BATCH_SIZE=48
ACCUMULATION_STEPS=4

echo "==> Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "==> Output dir: $OUT_DIR"
echo "==> Batch size: $BATCH_SIZE, Accumulation steps: $ACCUMULATION_STEPS (Effective: $((BATCH_SIZE * ACCUMULATION_STEPS)))"

cd "$REPO_DIR"
uv run scripts/analyze.py capo \
    --n-individuals      85000    \
    --train-exposures    1000     \
    --model-sizes        micro    \
    --loop-counts        1 4      \
    --batch-size         "$BATCH_SIZE" \
    --accumulation-steps "$ACCUMULATION_STEPS" \
    --seq-len            512      \
    --log-every          20      \
    --device             auto     \
    --output-dir         "$OUT_DIR"

echo "==> Capo complete. Results at ${OUT_DIR}/capo_results.csv"
