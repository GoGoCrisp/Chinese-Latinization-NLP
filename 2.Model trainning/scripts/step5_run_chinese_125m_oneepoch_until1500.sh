#!/usr/bin/env bash
# Step 5 helper: start the one-epoch Chinese-Origin run and stop after checkpoint-1500.
set -euo pipefail

cd "$(dirname "$0")/.."

python3 step5_train_lm_formal.py \
  --config configs/step5_chinese_125m_b1024.yaml \
  --output_dir outputs/chinese_125m_b1024_oneepoch_seed42 \
  --max_steps 6794 \
  --save_steps 500 \
  --eval_steps 500 \
  --logging_steps 10 \
  --save_total_limit 3 \
  --stop_after_checkpoint_step 1500
