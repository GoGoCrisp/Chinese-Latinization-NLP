#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MANIFEST="${MANIFEST:-configs/robustness/eval_runs_robustness.json}"
OUTPUT_BASE="${OUTPUT_BASE:-eval_results/robustness_134m_eval}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-10000}"
DRY_RUN="${DRY_RUN:-1}"
EXTRACT="${EXTRACT:-0}"

cd "${PROJECT_ROOT}"

commands=()
prepare_cmd=(python3 scripts/robustness/prepare_robustness_eval.py --write-manifest)
if [[ "${EXTRACT}" == "1" ]]; then
  prepare_cmd+=(--extract)
fi
commands+=("$(printf '%q ' "${prepare_cmd[@]}")")
commands+=("python3 scripts/eval_normalized_ppl_linelevel.py --model-runs-json ${MANIFEST} --output-dir ${OUTPUT_BASE}/eval1_ppl")
commands+=("python3 scripts/eval_homophone_probe_v2.py --model-runs-json ${MANIFEST} --output-dir ${OUTPUT_BASE}/eval2_homophone")
commands+=("python3 scripts/eval_nonhomophone_control_v2.py --model-runs-json ${MANIFEST} --output-dir ${OUTPUT_BASE}/eval2_hard_control --homophone-matched-summary ${OUTPUT_BASE}/eval2_homophone/summary_matched_subsets.csv")
commands+=("python3 scripts/eval_easy_random_control_v2.py --model-runs-json ${MANIFEST} --output-dir ${OUTPUT_BASE}/eval2_easy_control --homophone-matched-summary ${OUTPUT_BASE}/eval2_homophone/summary_matched_subsets.csv --hard-control-summary ${OUTPUT_BASE}/eval2_hard_control/summary_by_model_and_scoring.csv")
commands+=("python3 scripts/eval4_chinese_blimp_style.py --model-runs-json ${MANIFEST} --output-dir ${OUTPUT_BASE}/eval4_zhoblimp --print-random-examples 0 --print-contrast-examples 0")
commands+=("python3 scripts/robustness/summarize_robustness_eval.py --ppl-summary ${OUTPUT_BASE}/eval1_ppl/summary.csv --homophone ${OUTPUT_BASE}/eval2_homophone/item_scores.csv --hard-control ${OUTPUT_BASE}/eval2_hard_control/item_scores.csv --easy-control ${OUTPUT_BASE}/eval2_easy_control/item_scores.csv --zhoblimp ${OUTPUT_BASE}/eval4_zhoblimp/item_scores.csv --out-dir ${OUTPUT_BASE}/summary --bootstrap-samples ${BOOTSTRAP_SAMPLES}")
commands+=("python3 scripts/robustness/eval_bootstrap_and_significance.py --homophone ${OUTPUT_BASE}/eval2_homophone/item_scores.csv --hard-control ${OUTPUT_BASE}/eval2_hard_control/item_scores.csv --easy-control ${OUTPUT_BASE}/eval2_easy_control/item_scores.csv --zhoblimp ${OUTPUT_BASE}/eval4_zhoblimp/item_scores.csv --out-dir ${OUTPUT_BASE}/paired_ci_seed43 --chinese-model chinese_125m_b1024_matched_token_seed43 --pinyin-model diacritic_125m_b1024_matched_token_seed43 --bootstrap-samples ${BOOTSTRAP_SAMPLES}")
commands+=("python3 scripts/robustness/eval_bootstrap_and_significance.py --homophone ${OUTPUT_BASE}/eval2_homophone/item_scores.csv --hard-control ${OUTPUT_BASE}/eval2_hard_control/item_scores.csv --easy-control ${OUTPUT_BASE}/eval2_easy_control/item_scores.csv --zhoblimp ${OUTPUT_BASE}/eval4_zhoblimp/item_scores.csv --out-dir ${OUTPUT_BASE}/paired_ci_seed44 --chinese-model chinese_125m_b1024_matched_token_seed44 --pinyin-model diacritic_125m_b1024_matched_token_seed44 --bootstrap-samples ${BOOTSTRAP_SAMPLES}")

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
