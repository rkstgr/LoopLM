#!/usr/bin/env bash
#SBATCH --account=westai0047
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --job-name=looplm_mano
#SBATCH --output=/p/scratch/westai0047/looplm/logs/%x_%j.out
#SBATCH --error=/p/scratch/westai0047/looplm/logs/%x_%j.err

set -euo pipefail

module load uv

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_DIR="${PROJECT}/LoopLM"
LOG_DIR="${SCRATCH}/looplm/logs"
OUT_DIR="${SCRATCH}/looplm/mano"

mkdir -p "$LOG_DIR" "$OUT_DIR"

# ── No HF tokenizer needed for Mano, but keep offline mode in case ──────────
export HF_HOME="${PROJECT}/hf_cache"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

# ── Configuration ────────────────────────────────────────────────────────────
MAX_OPS=10
N_TRAIN=500000
N_EVAL=1000
MODEL_PRESET="small"            # hidden=128, 2 heads
MODEL_CONFIGS="4:1 2:2 1:4"    # iso-FLOP: 4 layer passes
TRAIN_STEPS=80000
BATCH_SIZE=128
ACCUMULATION_STEPS=1
NUM_SEEDS=3

echo "==> Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "==> Output dir: $OUT_DIR"
echo "==> max_ops=$MAX_OPS  preset=$MODEL_PRESET  configs=$MODEL_CONFIGS"
echo "==> train_steps=$TRAIN_STEPS  batch=$BATCH_SIZE  accum=$ACCUMULATION_STEPS  seeds=$NUM_SEEDS"

# ── Run ──────────────────────────────────────────────────────────────────────
cd "$REPO_DIR"
uv run scripts/analyze.py mano \
    --max-ops            "$MAX_OPS"            \
    --n-train            "$N_TRAIN"            \
    --n-eval             "$N_EVAL"             \
    --model-preset       "$MODEL_PRESET"       \
    --model-configs      $MODEL_CONFIGS        \
    --train-steps        "$TRAIN_STEPS"        \
    --batch-size         "$BATCH_SIZE"         \
    --accumulation-steps "$ACCUMULATION_STEPS" \
    --num-seeds          "$NUM_SEEDS"          \
    --seq-len            1024                  \
    --log-every          500                   \
    --device             auto                  \
    --output-dir         "$OUT_DIR"

echo "==> Mano complete. Results at ${OUT_DIR}/mano_results.csv"
