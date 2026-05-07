#!/usr/bin/env bash
# Step 5 helper: run an approximately one-epoch Chinese-Origin 125M job.
set -euo pipefail

cd "$(dirname "$0")/.."

python3 step5_train_lm_formal.py \
  --config configs/step5_chinese_125m_b1024.yaml \
  --output_dir outputs/chinese_125m_b1024_oneepoch_seed42 \
  --max_steps 6794
