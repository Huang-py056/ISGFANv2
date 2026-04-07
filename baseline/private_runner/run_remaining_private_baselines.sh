#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/data2/ranran/chenghaoyue/ISGFAN"
PYTHON_BIN="/data2/ranran/chenghaoyue/anaconda3/bin/python"
RUNNER="$ROOT_DIR/baseline/private_runner/run_private_baseline.py"
LOG_DIR="$ROOT_DIR/baseline/logs"
OUT_ROOT="baseline/private_runs"

METHODS=(mmd jan entropy_min mcc)

GPU_ID=${1:-0}
EPOCHS=${2:-40}
BATCH_SIZE=${3:-32}
MAX_STEPS=${4:-80}

mkdir -p "$LOG_DIR"
cd "$ROOT_DIR"

echo "Using GPU $GPU_ID, epochs=$EPOCHS, batch_size=$BATCH_SIZE, max_steps=$MAX_STEPS"
for method in "${METHODS[@]}"; do
  LOG_FILE="$LOG_DIR/${method}.log"
  echo "=============================================================="
  echo "[START] $method"
  "$PYTHON_BIN" "$RUNNER" \
    --method "$method" \
    --gpu-id "$GPU_ID" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --max-steps-per-epoch "$MAX_STEPS" \
    --output-root "$OUT_ROOT" \
    --log-file "$LOG_FILE"
  echo "[DONE]  $method -> $LOG_FILE"
  echo "=============================================================="
  echo
 done

echo "Remaining baselines finished."
