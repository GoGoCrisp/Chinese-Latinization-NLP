#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MANIFEST="${MANIFEST:-configs/robustness/eval_runs_matched_data_diacritic_seed43_44.json}"
OUTPUT_BASE="${OUTPUT_BASE:-eval_results/robustness_matched_data_diacritic_seed43_44_eval}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-10000}"
DRY_RUN="${DRY_RUN:-1}"

cd "${PROJECT_ROOT}"

commands=()
commands+=("python3 scripts/eval_normalized_ppl_linelevel.py --model-runs-json ${MANIFEST} --output-dir ${OUTPUT_BASE}/eval1_ppl")
commands+=("python3 scripts/eval_homophone_probe_v2.py --model-runs-json ${MANIFEST} --output-dir ${OUTPUT_BASE}/eval2_homophone")
commands+=("python3 scripts/eval_nonhomophone_control_v2.py --model-runs-json ${MANIFEST} --output-dir ${OUTPUT_BASE}/eval2_hard_control --homophone-matched-summary ${OUTPUT_BASE}/eval2_homophone/summary_matched_subsets.csv")
commands+=("python3 scripts/eval_easy_random_control_v2.py --model-runs-json ${MANIFEST} --output-dir ${OUTPUT_BASE}/eval2_easy_control --homophone-matched-summary ${OUTPUT_BASE}/eval2_homophone/summary_matched_subsets.csv --hard-control-summary ${OUTPUT_BASE}/eval2_hard_control/summary_by_model_and_scoring.csv")
commands+=("python3 scripts/eval4_chinese_blimp_style.py --model-runs-json ${MANIFEST} --output-dir ${OUTPUT_BASE}/eval4_zhoblimp --print-random-examples 0 --print-contrast-examples 0")
commands+=("python3 scripts/robustness/summarize_robustness_eval.py --ppl-summary ${OUTPUT_BASE}/eval1_ppl/summary.csv --homophone ${OUTPUT_BASE}/eval2_homophone/item_scores.csv --hard-control ${OUTPUT_BASE}/eval2_hard_control/item_scores.csv --easy-control ${OUTPUT_BASE}/eval2_easy_control/item_scores.csv --zhoblimp ${OUTPUT_BASE}/eval4_zhoblimp/item_scores.csv --out-dir ${OUTPUT_BASE}/summary --bootstrap-samples ${BOOTSTRAP_SAMPLES}")

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'Dry run only. Commands prepared:\n'
  for cmd in "${commands[@]}"; do
    printf '  %s\n' "${cmd}"
  done
  exit 0
fi

for cmd in "${commands[@]}"; do
  printf '\n==> %s\n' "${cmd}"
  eval "${cmd}"
done
