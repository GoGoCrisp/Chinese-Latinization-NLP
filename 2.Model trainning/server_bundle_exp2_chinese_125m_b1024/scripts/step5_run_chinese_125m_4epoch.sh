#!/usr/bin/env bash
# Step 5 helper: run Chinese-Origin 12-layer model for strict 4x one-epoch token budget.
set -euo pipefail

cd "$(dirname "$0")/.."

python3 step5_train_lm_formal.py \
  --config configs/step5_chinese_125m_b1024_4epoch.yaml \
  --output_dir outputs/chinese_125m_b1024_4epoch_seed42 \
  --max_steps 27176
