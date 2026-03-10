#!/usr/bin/env bash
#SBATCH --account=westai0047
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --job-name=looplm_mano
#SBATCH --array=0-62
#SBATCH --output=/p/scratch/westai0047/looplm/logs/%x_%A_%a.out
#SBATCH --error=/p/scratch/westai0047/looplm/logs/%x_%A_%a.err

set -euo pipefail

module load uv

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_DIR="${PROJECT}/LoopLM"
LOG_DIR="${SCRATCH}/looplm/logs"
OUT_DIR="${SCRATCH}/looplm/mano"
WANDB_DIR="${SCRATCH}/looplm/wandb"

mkdir -p "$LOG_DIR" "$OUT_DIR" "$WANDB_DIR"

# ── Offline mode (compute nodes have no internet) ───────────────────────────
export HF_HOME="${PROJECT}/hf_cache"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

export WANDB_MODE=offline
export WANDB_DIR="$WANDB_DIR"

# ── Paper configuration (Section 6.2, Appendix B.2) ─────────────────────────
# Models: baselines {2,3,6,12}×1, looped {2×6, 3×4, 6×2}  → 7 configs
# Seeds: 3 (paper reports best)
# LR: searched per config via --lr-search
# Difficulty: set via MAX_OPS (submit separate jobs for L=10,16,24)
#
# Total tasks per difficulty: 7 configs × 3 seeds × 3 LRs = 63 tasks
# (LR search is done within each run via --lr-search)
#
# To run all 3 difficulties, submit 3 times:
#   MAX_OPS=10 TRAIN_STEPS=80000  sbatch scripts/mano.slurm
#   MAX_OPS=16 TRAIN_STEPS=110000 sbatch scripts/mano.slurm
#   MAX_OPS=24 TRAIN_STEPS=200000 sbatch scripts/mano.slurm
#
# Or override via environment:
#   export MAX_OPS=16 TRAIN_STEPS=110000; sbatch scripts/mano.slurm

MAX_OPS="${MAX_OPS:-10}"
TRAIN_STEPS="${TRAIN_STEPS:-80000}"
N_TRAIN=500000
N_EVAL=1000
MODEL_PRESET="paper"           # hidden=1024, 16 heads
BATCH_SIZE=128
ACCUMULATION_STEPS=1

# Paper model configs: baselines + looped
CONFIGS=("2:1" "3:1" "6:1" "12:1" "2:6" "3:4" "6:2")
SEEDS=(42 43 44)

NUM_CONFIGS=${#CONFIGS[@]}
NUM_SEEDS=${#SEEDS[@]}

# ── Map SLURM_ARRAY_TASK_ID → (config, seed) ────────────────────────────────
TASK_ID=${SLURM_ARRAY_TASK_ID}
CONFIG_IDX=$(( TASK_ID / NUM_SEEDS ))
SEED_IDX=$(( TASK_ID % NUM_SEEDS ))

if (( CONFIG_IDX >= NUM_CONFIGS )); then
    echo "==> Task ${TASK_ID} exceeds config range (${NUM_CONFIGS} configs × ${NUM_SEEDS} seeds = $(( NUM_CONFIGS * NUM_SEEDS )) tasks), exiting."
    exit 0
fi

CONFIG="${CONFIGS[$CONFIG_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"
TASK_OUT_DIR="${OUT_DIR}/L${MAX_OPS}/task_${TASK_ID}"

mkdir -p "$TASK_OUT_DIR"

echo "==> Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "==> Array task ${TASK_ID}: config=${CONFIG}  seed=${SEED}  max_ops=${MAX_OPS}"
echo "==> Output dir: $TASK_OUT_DIR"
echo "==> preset=$MODEL_PRESET  train_steps=$TRAIN_STEPS  batch=$BATCH_SIZE"

# ── Run single (config, seed) pair with LR search ───────────────────────────
cd "$REPO_DIR"
uv run scripts/analyze.py mano \
    --max-ops            "$MAX_OPS"            \
    --n-train            "$N_TRAIN"            \
    --n-eval             "$N_EVAL"             \
    --model-preset       "$MODEL_PRESET"       \
    --model-configs      "$CONFIG"             \
    --train-steps        "$TRAIN_STEPS"        \
    --batch-size         "$BATCH_SIZE"         \
    --accumulation-steps "$ACCUMULATION_STEPS" \
    --lr-search                                \
    --num-seeds          1                     \
    --seed               "$SEED"               \
    --seq-len            1024                  \
    --log-every          500                   \
    --device             auto                  \
    --output-dir         "$TASK_OUT_DIR"        \
    --use-wandb                                \
    --wandb-project      looplm

echo "==> Task ${TASK_ID} complete. Results at ${TASK_OUT_DIR}/mano_results.csv"
echo "==> To sync wandb: wandb sync ${WANDB_DIR}/offline-run-*"
