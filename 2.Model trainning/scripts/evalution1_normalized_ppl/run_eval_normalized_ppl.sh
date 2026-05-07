#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-eval_results/normalized_ppl_4epoch}"
BATCH_SIZE="${BATCH_SIZE:-8}"

cd "${PROJECT_ROOT}"

python3 scripts/evalution1_normalized_ppl/eval_normalized_ppl.py \
  --run all \
  --batch-size "${BATCH_SIZE}" \
  --output-dir "${OUTPUT_DIR}" \
  "$@"

echo
echo "summary.csv"
cat "${OUTPUT_DIR}/summary.csv"
