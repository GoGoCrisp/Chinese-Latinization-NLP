#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

DATA_DIR="eval_data/eval4_chinese_blimp_style"
DATASET="${DATA_DIR}/eval4_chinese_blimp_style.jsonl"
SMOKE_DIR="eval_results/eval4_chinese_blimp_style/smoke_500"
FORMAL_DIR="eval_results/eval4_chinese_blimp_style"

python3 scripts/build_eval4_chinese_blimp_style.py \
  --all-items \
  --prefer-zhoblimp \
  --output-dir "${DATA_DIR}"

echo
echo "Eval 4 smoke run: 500 items"
python3 scripts/eval4_chinese_blimp_style.py \
  --dataset "${DATASET}" \
  --output-dir "${SMOKE_DIR}" \
  --max-items 500 \
  --print-random-examples 20 \
  --print-contrast-examples 10

echo
echo "Smoke summary_overall.csv"
cat "${SMOKE_DIR}/summary_overall.csv"

echo
echo "Eval 4 formal run: full ZhoBLiMP dataset"
python3 scripts/eval4_chinese_blimp_style.py \
  --dataset "${DATASET}" \
  --output-dir "${FORMAL_DIR}" \
  --print-random-examples 20 \
  --print-contrast-examples 10

echo
echo "Formal summary_overall.csv"
cat "${FORMAL_DIR}/summary_overall.csv"

echo
echo "Formal model_comparison.csv"
cat "${FORMAL_DIR}/model_comparison.csv"
