#!/usr/bin/env bash
#SBATCH --account=westai0047
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --job-name=looplm_capo
#SBATCH --array=0-5
#SBATCH --output=/p/scratch/westai0047/looplm/logs/%x_%A_%a.out
#SBATCH --error=/p/scratch/westai0047/looplm/logs/%x_%A_%a.err

set -euo pipefail

module load uv

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_DIR="${PROJECT}/LoopLM"
LOG_DIR="${SCRATCH}/looplm/logs"
OUT_DIR="${SCRATCH}/looplm/capo"
WANDB_DIR="${SCRATCH}/looplm/wandb"

mkdir -p "$LOG_DIR" "$OUT_DIR" "$WANDB_DIR"

# ── Offline mode (compute nodes have no internet) ───────────────────────────
export HF_HOME="${PROJECT}/hf_cache"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

export WANDB_MODE=offline
export WANDB_DIR="$WANDB_DIR"

# ── Configuration ────────────────────────────────────────────────────────────
N_INDIVIDUALS=85000
TRAIN_EXPOSURES=1000
BATCH_SIZE=192
ACCUMULATION_STEPS=1
SEQ_LEN=512

# Model sizes and loop counts — edit these arrays to change the sweep
MODEL_SIZES=("micro")
LOOP_COUNTS=(1 4)
SEEDS=(42 43 44)

NUM_SIZES=${#MODEL_SIZES[@]}
NUM_LOOPS=${#LOOP_COUNTS[@]}
NUM_SEEDS=${#SEEDS[@]}
NUM_CONFIGS=$(( NUM_SIZES * NUM_LOOPS ))

# ── Map SLURM_ARRAY_TASK_ID → (model_size, loop_count, seed) ────────────────
TASK_ID=${SLURM_ARRAY_TASK_ID}
CONFIG_IDX=$(( TASK_ID / NUM_SEEDS ))
SEED_IDX=$(( TASK_ID % NUM_SEEDS ))

SIZE_IDX=$(( CONFIG_IDX / NUM_LOOPS ))
LOOP_IDX=$(( CONFIG_IDX % NUM_LOOPS ))

MODEL_SIZE="${MODEL_SIZES[$SIZE_IDX]}"
LOOP_COUNT="${LOOP_COUNTS[$LOOP_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"
TASK_OUT_DIR="${OUT_DIR}/task_${TASK_ID}"

mkdir -p "$TASK_OUT_DIR"

echo "==> Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "==> Array task ${TASK_ID}: model_size=${MODEL_SIZE}  loop=${LOOP_COUNT}  seed=${SEED}"
echo "==> Output dir: $TASK_OUT_DIR"
echo "==> N=${N_INDIVIDUALS}  exposures=${TRAIN_EXPOSURES}  batch=${BATCH_SIZE}"

# ── Run single (model_size, loop_count, seed) ───────────────────────────────
cd "$REPO_DIR"
uv run scripts/analyze.py capo \
    --n-individuals      "$N_INDIVIDUALS"      \
    --train-exposures    "$TRAIN_EXPOSURES"     \
    --model-sizes        "$MODEL_SIZE"          \
    --loop-counts        "$LOOP_COUNT"          \
    --batch-size         "$BATCH_SIZE"          \
    --accumulation-steps "$ACCUMULATION_STEPS"  \
    --seq-len            "$SEQ_LEN"             \
    --num-seeds          1                      \
    --seed               "$SEED"                \
    --log-every          20                     \
    --device             auto                   \
    --output-dir         "$TASK_OUT_DIR"         \
    --use-wandb                                 \
    --wandb-project      looplm

echo "==> Task ${TASK_ID} complete. Results at ${TASK_OUT_DIR}/capo_results.csv"
echo "==> To sync wandb: wandb sync ${WANDB_DIR}/offline-run-*"
