#!/usr/bin/env bash
# launch.sh — run multiple conditions as background services
#
# Usage:
#   ./launch.sh                          # C0 + C5 on same GPU (sequential in each process)
#   ./launch.sh --gpu0 0 --gpu1 1        # C0 on GPU 0, C5 on GPU 1
#   ./launch.sh --conditions C0 C5       # explicit conditions (split: C0 on first, rest on second)
#   CONFIG=config_fast.yaml ./launch.sh  # use a different config
#
# Logs:
#   logs/C0.log, logs/C5.log  (created automatically)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-config.yaml}"
GPU0="${GPU0:-0}"   # GPU for C0 (grid search, CIFAR only — lightweight)
GPU1="${GPU1:-0}"   # GPU for C5 (GRPO — needs LLM + CIFAR VRAM)

# Parse --gpu0 / --gpu1 / --conditions overrides
CONDITIONS_C0="C0"
CONDITIONS_C1="C5"
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu0) GPU0="$2"; shift 2 ;;
        --gpu1) GPU1="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        *) echo "[warn] unknown arg: $1"; shift ;;
    esac
done

mkdir -p logs

export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "============================================"
echo "  CIFAR-10 AutoML — multi-condition launch"
echo "  Config  : $CONFIG"
echo "  C0 GPU  : $GPU0  →  logs/C0.log"
echo "  C5 GPU  : $GPU1  →  logs/C5.log"
echo "  Python  : $($PYTHON --version 2>&1)"
echo "============================================"

# --- Launch C0 (grid search) ---
echo "[launch] Starting C0 (grid search) ..."
CUDA_VISIBLE_DEVICES=$GPU0 $PYTHON experiments/run_all.py \
    --config "$CONFIG" \
    --conditions C0 \
    --no-parallel \
    > logs/C0.log 2>&1 &
PID_C0=$!
echo "  PID=$PID_C0   tail -f logs/C0.log"

# Small gap so both processes don't hit CIFAR download simultaneously
sleep 3

# --- Launch C5 (GRPO) ---
echo "[launch] Starting C5 (GRPO) ..."
CUDA_VISIBLE_DEVICES=$GPU1 $PYTHON experiments/run_all.py \
    --config "$CONFIG" \
    --conditions C5 \
    > logs/C5.log 2>&1 &
PID_C5=$!
echo "  PID=$PID_C5   tail -f logs/C5.log"

echo ""
echo "Both services running. To monitor:"
echo "  tail -f logs/C0.log"
echo "  tail -f logs/C5.log"
echo ""
echo "To stop both:"
echo "  kill $PID_C0 $PID_C5"
echo ""

# Wait for both and report exit codes
wait $PID_C0
EC0=$?
wait $PID_C5
EC1=$?

echo "============================================"
echo "  C0 exit code: $EC0"
echo "  C5 exit code: $EC1"
echo "============================================"
[[ $EC0 -eq 0 && $EC1 -eq 0 ]] && echo "All done." || echo "[warn] One or more jobs failed."
