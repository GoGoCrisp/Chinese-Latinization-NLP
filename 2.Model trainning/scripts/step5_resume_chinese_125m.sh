#!/usr/bin/env bash
# Step 5 helper: resume Chinese-Origin 125M training from a checkpoint.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <checkpoint_dir> <max_steps> [output_dir]" >&2
  exit 2
fi

CHECKPOINT_DIR="$1"
MAX_STEPS="$2"
OUTPUT_DIR="${3:-outputs/chinese_125m_b1024_100m_seed42}"

cd "$(dirname "$0")/.."

python3 step5_train_lm_formal.py \
  --config configs/step5_chinese_125m_b1024.yaml \
  --output_dir "$OUTPUT_DIR" \
  --resume_from_checkpoint "$CHECKPOINT_DIR" \
  --max_steps "$MAX_STEPS"
