#!/bin/bash
#SBATCH --job-name=hrm_abl
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --output=/p/scratch/westai0047/looplm/logs/hrm_abl_%A_%a.out
#SBATCH --error=/p/scratch/westai0047/looplm/logs/hrm_abl_%A_%a.err
#SBATCH --array=0-29
#SBATCH --account=westai0047

# ═══════════════════════════════════════════════════════════════════════════
# HRM Ablation Sweep — all components tested on arithmetic testbed
#
# Array layout (30 tasks):
#
# --- Ablation 2: Deep Supervision (0-11) ---
#  0: deepsup M=1 T=2  (=detach only, control)
#  1: deepsup M=1 T=4
#  2: deepsup M=1 T=8
#  3: deepsup M=2 T=2  (total=4)
#  4: deepsup M=2 T=4  (total=8)
#  5: deepsup M=2 T=8  (total=16)
#  6: deepsup M=4 T=2  (total=8)
#  7: deepsup M=4 T=4  (total=16)
#  8: deepsup M=8 T=2  (total=16)
#  9: deepsup M=4 T=8  (total=32)
# 10: deepsup M=8 T=4  (total=32)
# 11: deepsup M=1 T=16 (=detach only T=16, control)
#
# --- Ablation 4: Stability Tweaks on best deepsup config M=2,T=4 (12-18) ---
# 12: deepsup M=2 T=4 + postnorm
# 13: deepsup M=2 T=4 + lecun_init
# 14: deepsup M=2 T=4 + adam_atan2
# 15: deepsup M=2 T=4 + postnorm + lecun_init
# 16: deepsup M=2 T=4 + postnorm + adam_atan2
# 17: deepsup M=2 T=4 + lecun_init + adam_atan2
# 18: deepsup M=2 T=4 + all three
#
# --- BPTT baselines for comparison (19-23) ---
# 19: BPTT T=2
# 20: BPTT T=4
# 21: BPTT T=8
# 22: BPTT T=16
# 23: BPTT T=4 + postnorm
#
# --- Ablation 4 stability on BPTT (24-26) ---
# 24: BPTT T=4 + lecun_init
# 25: BPTT T=4 + adam_atan2
# 26: BPTT T=4 + postnorm + lecun + atan2
#
# --- Phase 2: Best combos at higher compute (27-29) ---
# 27: deepsup M=4 T=4 + postnorm + lecun + atan2
# 28: deepsup M=8 T=4 + postnorm + lecun + atan2
# 29: deepsup M=2 T=8 + postnorm + lecun + atan2
# ═══════════════════════════════════════════════════════════════════════════

source /p/project1/westai0047/LoopLM/.venv/bin/activate
cd /p/project1/westai0047/LoopLM

export WANDB_MODE=offline
export WANDB_DIR=/p/scratch/westai0047/looplm/wandb

IDX=$SLURM_ARRAY_TASK_ID

# Common args
COMMON="--max-ops 4 --num-layers 1 --hidden-size 256 --num-heads 4 --intermediate-size 512 --two-digit --train-steps 5000 --batch-size 128 --lr 1e-3 --beta-kl 0.1 --eval-every 1000 --n-eval 500 --use-wandb --extrap-eval 1 2 4 8 16"

case $IDX in
    # --- Ablation 2: Deep Supervision ---
    0)  NAME="deepsup_M1_T2";  ARGS="--deep-supervision --num-segments 1 --num-recurrent-steps 2" ;;
    1)  NAME="deepsup_M1_T4";  ARGS="--deep-supervision --num-segments 1 --num-recurrent-steps 4" ;;
    2)  NAME="deepsup_M1_T8";  ARGS="--deep-supervision --num-segments 1 --num-recurrent-steps 8" ;;
    3)  NAME="deepsup_M2_T2";  ARGS="--deep-supervision --num-segments 2 --num-recurrent-steps 2" ;;
    4)  NAME="deepsup_M2_T4";  ARGS="--deep-supervision --num-segments 2 --num-recurrent-steps 4" ;;
    5)  NAME="deepsup_M2_T8";  ARGS="--deep-supervision --num-segments 2 --num-recurrent-steps 8" ;;
    6)  NAME="deepsup_M4_T2";  ARGS="--deep-supervision --num-segments 4 --num-recurrent-steps 2" ;;
    7)  NAME="deepsup_M4_T4";  ARGS="--deep-supervision --num-segments 4 --num-recurrent-steps 4" ;;
    8)  NAME="deepsup_M8_T2";  ARGS="--deep-supervision --num-segments 8 --num-recurrent-steps 2" ;;
    9)  NAME="deepsup_M4_T8";  ARGS="--deep-supervision --num-segments 4 --num-recurrent-steps 8" ;;
   10)  NAME="deepsup_M8_T4";  ARGS="--deep-supervision --num-segments 8 --num-recurrent-steps 4" ;;
   11)  NAME="deepsup_M1_T16"; ARGS="--deep-supervision --num-segments 1 --num-recurrent-steps 16" ;;

    # --- Ablation 4: Stability on deepsup M=2 T=4 ---
   12)  NAME="deepsup_M2T4_postnorm";     ARGS="--deep-supervision --num-segments 2 --num-recurrent-steps 4 --use-postnorm" ;;
   13)  NAME="deepsup_M2T4_lecun";        ARGS="--deep-supervision --num-segments 2 --num-recurrent-steps 4 --use-lecun-init" ;;
   14)  NAME="deepsup_M2T4_atan2";        ARGS="--deep-supervision --num-segments 2 --num-recurrent-steps 4 --use-adam-atan2" ;;
   15)  NAME="deepsup_M2T4_pn_lecun";     ARGS="--deep-supervision --num-segments 2 --num-recurrent-steps 4 --use-postnorm --use-lecun-init" ;;
   16)  NAME="deepsup_M2T4_pn_atan2";     ARGS="--deep-supervision --num-segments 2 --num-recurrent-steps 4 --use-postnorm --use-adam-atan2" ;;
   17)  NAME="deepsup_M2T4_lecun_atan2";  ARGS="--deep-supervision --num-segments 2 --num-recurrent-steps 4 --use-lecun-init --use-adam-atan2" ;;
   18)  NAME="deepsup_M2T4_all";          ARGS="--deep-supervision --num-segments 2 --num-recurrent-steps 4 --use-postnorm --use-lecun-init --use-adam-atan2" ;;

    # --- BPTT baselines ---
   19)  NAME="bptt_T2";  ARGS="--num-recurrent-steps 2" ;;
   20)  NAME="bptt_T4";  ARGS="--num-recurrent-steps 4" ;;
   21)  NAME="bptt_T8";  ARGS="--num-recurrent-steps 8" ;;
   22)  NAME="bptt_T16"; ARGS="--num-recurrent-steps 16" ;;
   23)  NAME="bptt_T4_postnorm"; ARGS="--num-recurrent-steps 4 --use-postnorm" ;;

    # --- Stability tweaks on BPTT ---
   24)  NAME="bptt_T4_lecun";  ARGS="--num-recurrent-steps 4 --use-lecun-init" ;;
   25)  NAME="bptt_T4_atan2";  ARGS="--num-recurrent-steps 4 --use-adam-atan2" ;;
   26)  NAME="bptt_T4_all";    ARGS="--num-recurrent-steps 4 --use-postnorm --use-lecun-init --use-adam-atan2" ;;

    # --- Phase 2: best combos at higher compute ---
   27)  NAME="deepsup_M4T4_all";  ARGS="--deep-supervision --num-segments 4 --num-recurrent-steps 4 --use-postnorm --use-lecun-init --use-adam-atan2" ;;
   28)  NAME="deepsup_M8T4_all";  ARGS="--deep-supervision --num-segments 8 --num-recurrent-steps 4 --use-postnorm --use-lecun-init --use-adam-atan2" ;;
   29)  NAME="deepsup_M2T8_all";  ARGS="--deep-supervision --num-segments 2 --num-recurrent-steps 8 --use-postnorm --use-lecun-init --use-adam-atan2" ;;
esac

OUTDIR="/p/scratch/westai0047/looplm/hrm_ablations/${NAME}"

echo "=== Task $IDX: $NAME ==="
echo "Output: $OUTDIR"
echo "Args: $ARGS"

uv run scripts/analyze.py arith \
    $COMMON \
    --output-dir "$OUTDIR" \
    --wandb-run-name "$NAME" \
    $ARGS
