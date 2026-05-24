#!/usr/bin/env bash
# plot.sh — generate comparison plot from latest (or specified) results folder
#
# Usage:
#   ./plot.sh                              # auto-find latest results subfolder
#   ./plot.sh results/conds-05-steps-60-...  # explicit folder
#   ./plot.sh results/... --no-band        # individual run lines instead of band

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python}"

if [[ $# -ge 1 && -d "$1" ]]; then
    DIR_ARG="--results-dir $1"
    shift
else
    DIR_ARG=""
fi

$PYTHON experiments/plot_results.py $DIR_ARG "$@"
