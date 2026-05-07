#!/usr/bin/env bash
# Step 5 helper: run a short Pinyin-Diacritic 12-layer smoke test on A100.
set -euo pipefail

cd "$(dirname "$0")/.."

python3 step5_train_lm_formal.py \
  --config configs/step5_diacritic_125m_b1024_matched_token_4epoch.yaml \
  --output_dir outputs/diacritic_125m_b1024_smoke \
  --max_steps 20 \
  --eval_steps 10 \
  --save_steps 10
