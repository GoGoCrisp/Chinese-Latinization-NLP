#!/usr/bin/env bash
# Step 5 helper: run Pinyin-Diacritic 125M with Diacritic matched-content budget.
set -euo pipefail

cd "$(dirname "$0")/.."

python3 step5_train_lm_formal.py \
  --config configs/step5_diacritic_125m_b1024_matched_content.yaml \
  --output_dir outputs/diacritic_125m_b1024_matched_content_seed42 \
  --max_steps 7441
