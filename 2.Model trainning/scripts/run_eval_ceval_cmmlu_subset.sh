#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

python3 scripts/eval_ceval_cmmlu_subset.py "$@"

echo
echo "summary_overall.csv"
cat eval_results/eval3/mcq_subset/summary_overall.csv

echo
echo "label_bias_diagnostics.csv"
cat eval_results/eval3/mcq_subset/label_bias_diagnostics.csv

echo
echo "chinese_vs_diacritic_comparison.csv"
cat eval_results/eval3/mcq_subset/chinese_vs_diacritic_comparison.csv
