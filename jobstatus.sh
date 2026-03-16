#!/usr/bin/env bash
# looplm_status.sh — show training state of your SLURM jobs

LOG_DIR="${SCRATCH}/looplm/logs"
TAIL_LINES="${1:-5}"

# Get all running/pending job IDs for the current user
JOB_IDS=$(squeue --user="$USER" --noheader --format="%i" 2>/dev/null)

if [[ -z "$JOB_IDS" ]]; then
  echo "No active SLURM jobs found for $USER."
  exit 0
fi

for JOB_ID in $JOB_IDS; do
  # Match log file by job ID suffix: {name}_{jobid}.out
  LOG_FILE=$(ls "${LOG_DIR}"/*_${JOB_ID}.out 2>/dev/null | head -1)

  if [[ -n "$LOG_FILE" ]]; then
    NAME=$(basename "$LOG_FILE" .out)
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Job $JOB_ID  ($NAME)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    tail -n "$TAIL_LINES" "$LOG_FILE"
  else
    echo "  Job $JOB_ID"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  [no log file found matching *_${JOB_ID}.out in $LOG_DIR]"
  fi

  echo ""
done
