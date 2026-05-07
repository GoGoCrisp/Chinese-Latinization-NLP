#!/usr/bin/env bash
# Step 5 helper: run Pinyin-Diacritic 125M with Chinese matched-token budget.
set -euo pipefail

cd "$(dirname "$0")/.."

python3 step5_train_lm_formal.py \
  --config configs/step5_diacritic_125m_b1024_matched_token.yaml \
  --output_dir outputs/diacritic_125m_b1024_matched_token_seed42 \
  --max_steps 6794
