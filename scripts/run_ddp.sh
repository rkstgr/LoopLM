#!/usr/bin/env bash
# scripts/run_ddp.sh — Universal multi-GPU DDP launcher for Cloud VMs & local servers.
# Works out of the box on RunPod, Lambda Labs, AWS EC2, GCP, Vast.ai, or bare-metal machines.
#
# Usage:
#   # Train small model on all available GPUs:
#   bash scripts/run_ddp.sh --model-config small --max-steps 10000
#
#   # Train Ouro-1.4B on 4 GPUs with gradient accumulation & checkpointing:
#   bash scripts/run_ddp.sh --gpus 4 --model-config ouro_1_4b --batch-size 2 \
#       --grad-accum-steps 32 --seq-len 4096 --gradient-checkpointing
#
#   # Dry run (prints command without executing):
#   bash scripts/run_ddp.sh --dry-run --model-config small

set -euo pipefail

# ── Help ──────────────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'EOF'
LoopLM Multi-GPU DDP Launcher

Options:
  --gpus N             Number of GPUs to use (default: auto-detected via nvidia-smi)
  --output-dir DIR     Directory to write checkpoints and logs (default: ./checkpoints)
  --dry-run            Print the torchrun command without launching
  -h, --help           Show this help message

All other flags are forwarded directly to scripts/train.py.
EOF
    exit 0
fi

# ── Detect Available GPUs ─────────────────────────────────────────────────────
DETECTED_GPUS=1
if command -v nvidia-smi &>/dev/null; then
    DETECTED_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)
    if [[ "$DETECTED_GPUS" -lt 1 ]]; then
        DETECTED_GPUS=1
    fi
fi

NGPUS="${NGPUS:-$DETECTED_GPUS}"
OUTPUT_DIR="checkpoints"
DRY_RUN=0
TRAIN_ARGS=()

# Parse wrapper-specific arguments and collect remaining for train.py
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)
            NGPUS="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            TRAIN_ARGS+=("--output-dir" "$2")
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        *)
            TRAIN_ARGS+=("$1")
            shift
            ;;
    esac
done

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR" "logs"

# ── Auto-Resume Detection ─────────────────────────────────────────────────────
# If user didn't specify --resume-from, check for existing checkpoints in OUTPUT_DIR
HAS_RESUME=0
for arg in "${TRAIN_ARGS[@]}"; do
    if [[ "$arg" == "--resume-from" ]]; then
        HAS_RESUME=1
        break
    fi
done

if [[ "$HAS_RESUME" -eq 0 && -d "$OUTPUT_DIR" ]]; then
    LATEST_CKPT=$(ls -t "$OUTPUT_DIR"/step_*.pt 2>/dev/null | head -1 || true)
    if [[ -n "$LATEST_CKPT" ]]; then
        echo "==> Found existing checkpoint: $LATEST_CKPT (auto-resuming)"
        TRAIN_ARGS+=("--resume-from" "$LATEST_CKPT")
    fi
fi

# ── Launch Torchrun ───────────────────────────────────────────────────────────
CMD=(uv run torchrun --nproc_per_node="$NGPUS" scripts/train.py "${TRAIN_ARGS[@]}")

echo "====================================================================="
echo " LoopLM Universal DDP Launcher"
echo "  GPUs:       $NGPUS"
echo "  Output:     $OUTPUT_DIR"
echo "  Command:    ${CMD[*]}"
echo "====================================================================="

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "==> Dry run requested. Exiting without execution."
    exit 0
fi

exec "${CMD[@]}"
