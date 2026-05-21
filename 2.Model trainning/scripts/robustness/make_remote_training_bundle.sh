#!/usr/bin/env bash
# Build a remote training bundle for robustness runs without checkpoints/eval outputs.
set -euo pipefail

OVERWRITE=0
if [[ "${1:-}" == "--overwrite" ]]; then
  OVERWRITE=1
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--overwrite]" >&2
  exit 2
fi

BASE_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$BASE_DIR"

ARCHIVE="robustness_training_plan/robustness_training_bundle.tar.gz"
BUNDLE_NAME="robustness_training_bundle"

if [[ -e "$ARCHIVE" && "$OVERWRITE" -ne 1 ]]; then
  echo "Archive already exists: $ARCHIVE" >&2
  echo "Use --overwrite to replace it." >&2
  exit 2
fi

require_path() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    echo "Missing required path: $path" >&2
    exit 2
  fi
}

required_paths=(
  "step5_train_lm_formal.py"
  "requirements.txt"
  "configs/robustness/diacritic_125m_b1024_matched_data_4epoch_seed43.yaml"
  "configs/robustness/diacritic_125m_b1024_matched_data_4epoch_seed44.yaml"
  "scripts/robustness/make_remote_training_bundle.sh"
  "scripts/robustness/remote_preflight_check.sh"
  "scripts/robustness/validate_robustness_configs.py"
  "scripts/robustness/run_one_robustness_training.sh"
  "scripts/robustness/run_batch_robustness_training.sh"
  "scripts/robustness/run_batch_matched_data_extra_seeds.sh"
  "robustness_training_plan/run_manifest.csv"
  "robustness_training_plan/run_manifest_matched_data_extra_seeds.csv"
  "robustness_training_plan/matched_data_extra_setup.md"
  "robustness_training_plan/REMOTE_STEPS_AFTER_SSH.md"
  "robustness_training_plan/expected_cost_and_time.md"
  "tokenizers/pinyin_diacritic_32k_eos"
  "data/tokenized/diacritic_train_full_eos_1024"
  "data/tokenized/diacritic_valid_full_eos_1024"
)

for path in "${required_paths[@]}"; do
  require_path "$path"
done

tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/robustness_bundle.XXXXXX")"
cleanup() {
  rm -rf "$tmp_dir"
}
trap cleanup EXIT

staging="$tmp_dir/$BUNDLE_NAME"
mkdir -p "$staging/configs" "$staging/scripts" "$staging/robustness_training_plan"
mkdir -p "$staging/tokenizers" "$staging/data/tokenized" "$staging/outputs/logs"
mkdir -p "$staging/configs/robustness"

cp "step5_train_lm_formal.py" "$staging/"
cp "requirements.txt" "$staging/"
cp "configs/robustness/diacritic_125m_b1024_matched_data_4epoch_seed43.yaml" "$staging/configs/robustness/"
cp "configs/robustness/diacritic_125m_b1024_matched_data_4epoch_seed44.yaml" "$staging/configs/robustness/"
mkdir -p "$staging/scripts/robustness"
cp "scripts/robustness/make_remote_training_bundle.sh" "$staging/scripts/robustness/"
cp "scripts/robustness/remote_preflight_check.sh" "$staging/scripts/robustness/"
cp "scripts/robustness/validate_robustness_configs.py" "$staging/scripts/robustness/"
cp "scripts/robustness/run_one_robustness_training.sh" "$staging/scripts/robustness/"
cp "scripts/robustness/run_batch_robustness_training.sh" "$staging/scripts/robustness/"
cp "scripts/robustness/run_batch_matched_data_extra_seeds.sh" "$staging/scripts/robustness/"
cp "robustness_training_plan/run_manifest.csv" "$staging/robustness_training_plan/"
cp "robustness_training_plan/run_manifest_matched_data_extra_seeds.csv" "$staging/robustness_training_plan/"
cp "robustness_training_plan/matched_data_extra_setup.md" "$staging/robustness_training_plan/"
cp "robustness_training_plan/REMOTE_STEPS_AFTER_SSH.md" "$staging/robustness_training_plan/"
cp "robustness_training_plan/expected_cost_and_time.md" "$staging/robustness_training_plan/"
cp -R "tokenizers/pinyin_diacritic_32k_eos" "$staging/tokenizers/"
cp -R "data/tokenized/diacritic_train_full_eos_1024" "$staging/data/tokenized/"
cp -R "data/tokenized/diacritic_valid_full_eos_1024" "$staging/data/tokenized/"

chmod +x "$staging/scripts/robustness/"*.sh

(
  cd "$staging"
  while IFS= read -r -d '' file; do
    size="$(wc -c < "$file" | tr -d ' ')"
    printf '%s %s\n' "$size" "$file"
  done < <(find . -type f -print0 | sort -z) > MANIFEST.txt
)

mkdir -p "$(dirname "$ARCHIVE")"
if [[ "$OVERWRITE" -eq 1 ]]; then
  rm -f "$ARCHIVE"
fi
tar -czf "$ARCHIVE" -C "$tmp_dir" "$BUNDLE_NAME"

echo "Created archive: $ARCHIVE"
du -sh "$ARCHIVE"
