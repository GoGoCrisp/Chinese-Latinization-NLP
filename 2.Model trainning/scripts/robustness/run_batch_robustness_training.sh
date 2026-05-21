#!/usr/bin/env bash
# Server-side sequential robustness runner.
# Default behavior is intentionally one-at-a-time:
#   run exactly one pending model, archive/hash it, then stop for local download
#   and sha256 verification before another full model can start.
set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$BASE_DIR"

ROBUSTNESS_BATCH_SET="${ROBUSTNESS_BATCH_SET:-matched_data_extra}"
case "$ROBUSTNESS_BATCH_SET" in
  matched_data_extra)
    CONFIGS=(
      "configs/robustness/diacritic_125m_b1024_matched_data_4epoch_seed43.yaml"
      "configs/robustness/diacritic_125m_b1024_matched_data_4epoch_seed44.yaml"
    )
    ;;
  *)
    echo "Unknown ROBUSTNESS_BATCH_SET=$ROBUSTNESS_BATCH_SET" >&2
    echo "Use: matched_data_extra" >&2
    exit 2
    ;;
esac

AUTO_DELETE_VERIFIED="${AUTO_DELETE_VERIFIED:-0}"
RUN_CONTINUOUS_AFTER_VERIFIED="${RUN_CONTINUOUS_AFTER_VERIFIED:-0}"
MIN_FREE_DISK_GB="${MIN_FREE_DISK_GB:-10}"
COMPLETED_DIR="${COMPLETED_DIR:-outputs/completed_runs}"
ALLOW_VERIFIED_UNCOMPRESSED_OUTPUTS="${ALLOW_VERIFIED_UNCOMPRESSED_OUTPUTS:-1}"

bytes_free() {
  df -Pk . | awk 'NR==2 {print $4 * 1024}'
}

path_bytes() {
  local path="$1"
  if [[ -e "$path" ]]; then
    du -sk "$path" | awk '{print $1 * 1024}'
  else
    echo 0
  fi
}

gb_to_bytes() {
  python3 - "$1" <<'PY'
import sys
print(int(float(sys.argv[1]) * 1024 ** 3))
PY
}

human_bytes() {
  python3 - "$1" <<'PY'
import sys
n = float(sys.argv[1])
for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
    if abs(n) < 1024 or unit == "TiB":
        print(f"{n:.2f} {unit}")
        break
    n /= 1024
PY
}

read_config_field() {
  local config_path="$1"
  local field="$2"
  python3 - "$config_path" "$field" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
field = sys.argv[2]
try:
    import yaml
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
except ImportError:
    data = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        clean = line.split("#", 1)[0].strip()
        if not clean or ":" not in clean:
            continue
        key, value = clean.split(":", 1)
        data[key.strip()] = value.strip().strip("\"'")
print(data.get(field, ""))
PY
}

marker_exists() {
  local tarball="$1"
  [[ -f "${tarball}.local_verified" || -f "${tarball}.downloaded" ]]
}

completed_marker() {
  local run_id="$1"
  echo "${COMPLETED_DIR}/${run_id}.done"
}

cleanup_verified_artifacts() {
  if [[ "$AUTO_DELETE_VERIFIED" != "1" ]]; then
    return 0
  fi
  local config run_id output_dir tarball marker
  mkdir -p "$COMPLETED_DIR"
  for config in "${CONFIGS[@]}"; do
    run_id="$(basename "$config" .yaml)"
    output_dir="$(read_config_field "$config" output_dir)"
    tarball="outputs/${run_id}.tar.gz"
    if marker_exists "$tarball"; then
      echo "Deleting locally verified remote artifacts for $run_id"
      touch "$(completed_marker "$run_id")"
      [[ -n "$output_dir" && -d "$output_dir" ]] && rm -rf "$output_dir"
      rm -f "$tarball" "${tarball}.sha256"
      for marker in "${tarball}.local_verified" "${tarball}.downloaded"; do
        [[ -f "$marker" ]] && rm -f "$marker"
      done
    fi
  done
}

stop_if_unverified_completed_run_exists() {
  local config run_id output_dir tarball
  for config in "${CONFIGS[@]}"; do
    run_id="$(basename "$config" .yaml)"
    output_dir="$(read_config_field "$config" output_dir)"
    tarball="outputs/${run_id}.tar.gz"
    if [[ -f "$(completed_marker "$run_id")" ]]; then
      continue
    fi
    if [[ -f "$tarball" || -f "${tarball}.sha256" || -d "$output_dir" ]]; then
      if marker_exists "$tarball"; then
        continue
      fi
      cat >&2 <<EOF
Completed or partial remote artifacts exist for $run_id and are not marked locally verified.

Remote output_dir: $output_dir
Remote tarball: $tarball

Do not start another full model yet. First:
  1. Download $tarball and ${tarball}.sha256 to local storage.
  2. Verify sha256 locally.
  3. Mark the server artifact:
       touch ${tarball}.local_verified
  4. Rerun this batch script. It will delete verified remote artifacts before continuing.

This protects the 40GB-disk workflow by avoiding two uncompressed model outputs.
EOF
      exit 3
    fi
  done
}

ensure_no_other_uncompressed_outputs() {
  local current_config="$1"
  local current_output
  current_output="$(read_config_field "$current_config" output_dir)"
  local config run_id output_dir
  for config in "${CONFIGS[@]}"; do
    output_dir="$(read_config_field "$config" output_dir)"
    run_id="$(basename "$config" .yaml)"
    if [[ "$output_dir" != "$current_output" && -d "$output_dir" ]]; then
      if [[ "$ALLOW_VERIFIED_UNCOMPRESSED_OUTPUTS" == "1" && -f "$(completed_marker "$run_id")" ]]; then
        echo "Keeping verified previous output for $run_id: $output_dir"
        continue
      fi
      echo "ERROR: uncompressed output already exists for $run_id: $output_dir" >&2
      echo "Download/verify/mark/delete it before starting another model." >&2
      exit 3
    fi
  done
}

ensure_min_disk() {
  local free_now min_free
  free_now="$(bytes_free)"
  min_free="$(gb_to_bytes "$MIN_FREE_DISK_GB")"
  echo "Free disk before run: $(human_bytes "$free_now")"
  if (( free_now < min_free )); then
    echo "ERROR: free disk below MIN_FREE_DISK_GB=${MIN_FREE_DISK_GB}: $(human_bytes "$free_now")" >&2
    exit 3
  fi
}

bash scripts/robustness/remote_preflight_check.sh
cleanup_verified_artifacts
stop_if_unverified_completed_run_exists

for config in "${CONFIGS[@]}"; do
  run_id="$(basename "$config" .yaml)"
  output_dir="$(read_config_field "$config" output_dir)"
  tarball="outputs/${run_id}.tar.gz"

  if [[ -f "$(completed_marker "$run_id")" ]]; then
    echo "Skipping already completed and locally verified run: $run_id"
    continue
  fi

  if [[ -f "$tarball" && -f "${tarball}.sha256" ]]; then
    echo "Found completed archive for $run_id."
    stop_if_unverified_completed_run_exists
  fi

  if [[ -d "$output_dir" ]]; then
    echo "Found existing output directory for $run_id."
    stop_if_unverified_completed_run_exists
  fi

  ensure_no_other_uncompressed_outputs "$config"
  ensure_min_disk

  echo "Starting one pending robustness run: $run_id"
  free_before="$(bytes_free)"
  bash scripts/robustness/run_one_robustness_training.sh "$config"
  free_after="$(bytes_free)"

  output_bytes="$(path_bytes "$output_dir")"
  tarball_bytes="$(path_bytes "$tarball")"
  echo "Completed $run_id"
  echo "  output_dir_size: $(human_bytes "$output_bytes")"
  echo "  tarball_size: $(human_bytes "$tarball_bytes")"
  echo "  free_before: $(human_bytes "$free_before")"
  echo "  free_after: $(human_bytes "$free_after")"
  echo "  disk_delta: $(human_bytes "$((free_before - free_after))")"

  cat <<EOF

Stopping after one full model by default.

Download and verify before the next model:
  rsync/scp outputs/${run_id}.tar.gz and outputs/${run_id}.tar.gz.sha256 to local storage
  sha256sum -c ${run_id}.tar.gz.sha256
  touch outputs/${run_id}.tar.gz.local_verified
  bash scripts/robustness/run_batch_robustness_training.sh

Set RUN_CONTINUOUS_AFTER_VERIFIED=1 only if you are actively marking verified artifacts between runs.
EOF

  if [[ "$RUN_CONTINUOUS_AFTER_VERIFIED" != "1" ]]; then
    exit 0
  fi
done

echo "No pending robustness configs remain."
