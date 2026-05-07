#!/usr/bin/env bash
# Step 5 helper: run Pinyin-Diacritic 12-layer model with Chinese 4epoch matched-token budget.
set -euo pipefail

cd "$(dirname "$0")/.."

python3 step5_train_lm_formal.py \
  --config configs/step5_diacritic_125m_b1024_matched_token_4epoch.yaml \
  --output_dir outputs/diacritic_125m_b1024_matched_token_4epoch_seed42 \
  --max_steps 27176
