#!/usr/bin/env bash
# validate.sh — post-hoc stability validation for top configs
#
# Finds the C5 CSV in the most-recently-modified results subfolder
# (or an explicitly supplied folder), then runs validate_top_configs.py.
#
# Usage:
#   ./validate.sh                          # auto-find latest results folder
#   ./validate.sh results/conds-05-steps-60-...  # explicit folder
#   ./validate.sh results/conds-05-steps-60-... --top-k 3 --seeds 0 1 2

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python}"
export OMP_NUM_THREADS=1

# ── Resolve results folder ───────────────────────────────────────────────────
if [[ $# -ge 1 && -d "$1" ]]; then
    RESULTS_DIR="$1"
    shift
else
    # Find the most-recently modified subfolder under results/
    RESULTS_DIR="$(ls -dt results/*/ 2>/dev/null | head -1)"
    if [[ -z "$RESULTS_DIR" ]]; then
        echo "[error] No results subfolder found under results/. Run an experiment first."
        exit 1
    fi
    echo "[validate] Auto-selected: $RESULTS_DIR"
fi

# ── Find C5 CSV ──────────────────────────────────────────────────────────────
CSV="$(ls "${RESULTS_DIR}"C5_grpo_G4_steps.csv 2>/dev/null | head -1)"
if [[ -z "$CSV" ]]; then
    echo "[error] No C5_grpo_G4_steps.csv found in $RESULTS_DIR"
    echo "        Available files:"
    ls "$RESULTS_DIR"
    exit 1
fi

echo "============================================"
echo "  Validating top configs"
echo "  Results folder : $RESULTS_DIR"
echo "  C5 CSV         : $CSV"
echo "  Extra args     : $*"
echo "============================================"

$PYTHON experiments/validate_top_configs.py \
    --csv "$CSV" \
    --data-root ./data \
    "$@"
