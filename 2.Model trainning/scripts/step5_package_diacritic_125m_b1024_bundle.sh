#!/usr/bin/env bash
# Step 5 helper: package the Pinyin-Diacritic 125M block_size=1024 server bundle.
set -euo pipefail

OVERWRITE=0
if [[ "${1:-}" == "--overwrite" ]]; then
  OVERWRITE=1
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--overwrite]" >&2
  exit 2
fi

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BASE_DIR"

BUNDLE_NAME="server_bundle_exp2_diacritic_125m_b1024"
STAGING_DIR="$BASE_DIR/$BUNDLE_NAME"
ARCHIVE="$BASE_DIR/$BUNDLE_NAME.tar.gz"

if [[ -e "$ARCHIVE" && "$OVERWRITE" -ne 1 ]]; then
  echo "Archive already exists: $ARCHIVE" >&2
  echo "Use --overwrite to replace it." >&2
  exit 2
fi
if [[ -e "$STAGING_DIR" && "$OVERWRITE" -ne 1 ]]; then
  echo "Staging directory already exists: $STAGING_DIR" >&2
  echo "Use --overwrite to replace it." >&2
  exit 2
fi

rm -rf "$STAGING_DIR"
if [[ "$OVERWRITE" -eq 1 ]]; then
  rm -f "$ARCHIVE"
fi

require_path() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    echo "Missing required path: $path" >&2
    exit 2
  fi
}

require_path "tokenizers/pinyin_diacritic_32k_eos"
require_path "data/tokenized/diacritic_train_full_eos_1024"
require_path "data/tokenized/diacritic_valid_full_eos_1024"
require_path "step5_train_lm_formal.py"
require_path "configs/step5_diacritic_125m_b1024_matched_token.yaml"
require_path "configs/step5_diacritic_125m_b1024_matched_token_4epoch.yaml"
require_path "configs/step5_diacritic_125m_b1024_matched_content.yaml"
require_path "requirements.txt"
require_path "README_server_run_diacritic.md"

mkdir -p "$STAGING_DIR/tokenizers" "$STAGING_DIR/data/tokenized" "$STAGING_DIR/scripts" "$STAGING_DIR/configs"

cp -R "tokenizers/pinyin_diacritic_32k_eos" "$STAGING_DIR/tokenizers/"
cp -R "data/tokenized/diacritic_train_full_eos_1024" "$STAGING_DIR/data/tokenized/"
cp -R "data/tokenized/diacritic_valid_full_eos_1024" "$STAGING_DIR/data/tokenized/"
if [[ -d "data/tokenized/diacritic_test_full_eos_1024" ]]; then
  cp -R "data/tokenized/diacritic_test_full_eos_1024" "$STAGING_DIR/data/tokenized/"
fi

cp "step5_train_lm_formal.py" "$STAGING_DIR/"
cp "requirements.txt" "$STAGING_DIR/"
cp "README_server_run_diacritic.md" "$STAGING_DIR/README_server_run.md"
cp "configs/step5_diacritic_125m_b1024_matched_token.yaml" "$STAGING_DIR/configs/"
cp "configs/step5_diacritic_125m_b1024_matched_token_4epoch.yaml" "$STAGING_DIR/configs/"
cp "configs/step5_diacritic_125m_b1024_matched_content.yaml" "$STAGING_DIR/configs/"
cp "scripts/step5_run_diacritic_125m_smoke.sh" "$STAGING_DIR/scripts/"
cp "scripts/step5_run_diacritic_125m_matched_token.sh" "$STAGING_DIR/scripts/"
cp "scripts/step5_run_diacritic_125m_matched_token_4epoch.sh" "$STAGING_DIR/scripts/"
cp "scripts/step5_run_diacritic_125m_matched_content.sh" "$STAGING_DIR/scripts/"
cp "scripts/step5_wait_for_next_checkpoint_then_exit.sh" "$STAGING_DIR/scripts/"
cp "scripts/step5_verify_diacritic_server_bundle.sh" "$STAGING_DIR/scripts/"

chmod +x "$STAGING_DIR/scripts/"*.sh

(
  cd "$STAGING_DIR"
  while IFS= read -r -d '' file; do
    size="$(wc -c < "$file" | tr -d ' ')"
    printf '%s %s\n' "$size" "$file"
  done < <(find . -type f -print0 | sort -z) > MANIFEST.txt
)

tar -czf "$ARCHIVE" -C "$BASE_DIR" "$BUNDLE_NAME"

echo "Created archive: $ARCHIVE"
echo "Archive size:"
du -sh "$ARCHIVE"
echo "Top-level bundle contents:"
find "$STAGING_DIR" -maxdepth 1 -mindepth 1 -print | sort
