#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

OUTPUT_DIR="eval_results/eval5_chid_idiom_cloze"

python3 scripts/eval5_chid_idiom_cloze.py "$@"

echo
echo "Eval5 summary.csv"
cat "${OUTPUT_DIR}/summary.csv"

echo
echo "Eval5 report"
cat "${OUTPUT_DIR}/eval5_chid_report.md"
