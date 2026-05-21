#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

python3 scripts/build_nonhomophone_control_v2.py "$@"
python3 scripts/eval_nonhomophone_control_v2.py

echo
echo "summary_by_model_and_scoring.csv"
cat eval_results/eval2/nonhomophone_control_v2/summary_by_model_and_scoring.csv

echo
echo "homophone_vs_control_gap.csv"
cat eval_results/eval2/nonhomophone_control_v2/homophone_vs_control_gap.csv
