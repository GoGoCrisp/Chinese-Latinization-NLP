#!/usr/bin/env bash
# Step 5 helper: deprecated. Use the one-epoch run so the scheduler is initialized
# with max_steps=6794 from the beginning.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "This script is deprecated for the formal run." >&2
echo "Use scripts/step5_run_chinese_125m_oneepoch.sh, optionally with --stop_after_checkpoint_step 1500." >&2
exit 2

python3 step5_train_lm_formal.py \
  --config configs/step5_chinese_125m_b1024.yaml \
  --output_dir outputs/chinese_125m_b1024_oneepoch_seed42 \
  --max_steps 6794
