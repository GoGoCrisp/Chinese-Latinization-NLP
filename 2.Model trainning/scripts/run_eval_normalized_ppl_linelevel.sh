#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-eval_results/eval1/normalized_ppl_4epoch_linelevel}"
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}/run_eval_normalized_ppl_linelevel.log}"

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}"

python3 scripts/eval_normalized_ppl_linelevel.py \
  --output-dir "${OUTPUT_DIR}" \
  "$@" 2>&1 | tee "${LOG_FILE}"

echo
echo "summary.csv"
cat "${OUTPUT_DIR}/summary.csv"
