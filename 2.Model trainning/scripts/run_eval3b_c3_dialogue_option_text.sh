#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

python3 scripts/eval3b_c3_dialogue_option_text.py "$@"

echo
echo "summary.csv"
cat eval_results/eval3b_c3_dialogue_option_text/summary.csv
