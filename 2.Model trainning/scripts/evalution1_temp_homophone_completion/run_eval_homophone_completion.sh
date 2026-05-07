#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-/private/tmp/evalution2_homophone_completion}"
MAX_PAIRS="${MAX_PAIRS:-50}"
CONTEXTS_PER_PAIR="${CONTEXTS_PER_PAIR:-1}"

cd "${PROJECT_ROOT}"

python3 scripts/evalution2_homophone_completion/eval_homophone_completion.py \
  --max-pairs "${MAX_PAIRS}" \
  --contexts-per-pair "${CONTEXTS_PER_PAIR}" \
  --output-dir "${OUTPUT_DIR}" \
  "$@"

echo
echo "summary.csv"
cat "${OUTPUT_DIR}/summary.csv"
