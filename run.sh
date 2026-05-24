#!/usr/bin/env bash
# run.sh — launch a CIFAR-10 AutoML experiment from config.yaml
#
# Usage:
#   ./run.sh                          # use config.yaml
#   ./run.sh config_fast.yaml         # use a different config file
#   ./run.sh config.yaml --conditions C1 C2   # override specific args
#
# To use a specific Python interpreter:
#   PYTHON=/path/to/python ./run.sh
#
# Windows (Git Bash): bash run.sh
# Windows (PowerShell): bash run.sh  (requires Git Bash in PATH)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Python interpreter — override via env var if needed
# e.g. PYTHON="C:/Users/xxx/AppData/Local/Programs/Python/Python313/python.exe" ./run.sh
PYTHON="${PYTHON:-python}"

# Suppress libgomp warning; also best practice for GPU+multiprocessing
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# First positional arg = config file (default: config.yaml); rest forwarded to run_all.py
CONFIG="${1:-config.yaml}"
shift || true   # remaining args passed through

echo "============================================"
echo "  CIFAR-10 AutoML — starting experiment"
echo "  Config : $CONFIG"
echo "  Python : $($PYTHON --version 2>&1)"
echo "  Dir    : $SCRIPT_DIR"
echo "============================================"

$PYTHON experiments/run_all.py --config "$CONFIG" "$@"
