#!/bin/bash
#SBATCH --job-name=arith_detach
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --output=/p/scratch/westai0047/looplm/logs/arith_detach_%A_%a.out
#SBATCH --error=/p/scratch/westai0047/looplm/logs/arith_detach_%A_%a.err
#SBATCH --array=0-9
#SBATCH --account=westai0047

# Array layout:
#   0-4: BPTT     T={1,2,4,8,16}
#   5-9: Detach   T={1,2,4,8,16}

source /p/project1/westai0047/LoopLM/.venv/bin/activate
cd /p/project1/westai0047/LoopLM

T_VALUES=(1 2 4 8 16)
T_IDX=$((SLURM_ARRAY_TASK_ID % 5))
T=${T_VALUES[$T_IDX]}

if [ $SLURM_ARRAY_TASK_ID -lt 5 ]; then
    MODE="bptt"
    DETACH_FLAG=""
else
    MODE="detach"
    DETACH_FLAG="--detach-between-steps"
fi

OUTDIR="/p/scratch/westai0047/looplm/arith_detach/${MODE}_T${T}"

echo "=== Task $SLURM_ARRAY_TASK_ID: mode=$MODE  T=$T ==="
echo "Output: $OUTDIR"

export WANDB_MODE=offline
export WANDB_DIR=/p/scratch/westai0047/looplm/wandb

uv run scripts/analyze.py arith \
    --max-ops 4 \
    --num-recurrent-steps $T \
    --num-layers 1 \
    --hidden-size 256 \
    --num-heads 4 \
    --intermediate-size 512 \
    --two-digit \
    --train-steps 5000 \
    --batch-size 128 \
    --lr 1e-3 \
    --beta-kl 0.1 \
    --eval-every 1000 \
    --n-eval 500 \
    --output-dir "$OUTDIR" \
    --use-wandb \
    --wandb-run-name "arith_${MODE}_T${T}" \
    $DETACH_FLAG \
    --extrap-eval 1 2 4 8 16
