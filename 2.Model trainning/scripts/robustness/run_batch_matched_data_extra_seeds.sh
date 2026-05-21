#!/usr/bin/env bash
# Server-side sequential runner for the two new Pinyin-Diacritic matched-data extra seeds.
set -euo pipefail

export ROBUSTNESS_BATCH_SET=matched_data_extra
export MIN_FREE_DISK_GB="${MIN_FREE_DISK_GB:-10}"

BASE_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$BASE_DIR"

exec bash scripts/robustness/run_batch_robustness_training.sh
