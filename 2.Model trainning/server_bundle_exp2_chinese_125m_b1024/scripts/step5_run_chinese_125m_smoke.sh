#!/usr/bin/env bash
# Step 5 helper: run a 20-step Chinese-Origin 12-layer smoke test on the server.
set -euo pipefail

cd "$(dirname "$0")/.."

python3 step5_train_lm_formal.py \
  --config configs/step5_chinese_125m_b1024_4epoch.yaml \
  --output_dir outputs/chinese_125m_b1024_smoke \
  --max_steps 20 \
  --eval_steps 10 \
  --save_steps 10
