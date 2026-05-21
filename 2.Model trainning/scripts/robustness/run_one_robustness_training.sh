#!/usr/bin/env bash
# Server-side helper: run one robustness training config and archive the result.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash scripts/robustness/run_one_robustness_training.sh <config_path>" >&2
  exit 2
fi

CONFIG_PATH="$1"
BASE_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$BASE_DIR"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config not found: $CONFIG_PATH" >&2
  exit 2
fi

read_config_field() {
  local field="$1"
  python3 - "$CONFIG_PATH" "$field" <<'PY'
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

RUN_ID="$(basename "$CONFIG_PATH" .yaml)"
OUTPUT_DIR="$(read_config_field output_dir)"
MAX_STEPS="$(read_config_field max_steps)"
if [[ -z "$OUTPUT_DIR" ]]; then
  echo "Could not read output_dir from $CONFIG_PATH" >&2
  exit 2
fi

mkdir -p outputs/logs
LOG_PATH="outputs/logs/${RUN_ID}.log"
TARBALL="outputs/${RUN_ID}.tar.gz"
SHA_PATH="${TARBALL}.sha256"
META_PATH="${OUTPUT_DIR}/robustness_run_metadata.json"
SMOKE_OUTPUT_DIR="outputs/smoke_${RUN_ID}"
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

sha256_file() {
  local path="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$path"
  else
    shasum -a 256 "$path"
  fi
}

if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Output directory already exists: $OUTPUT_DIR" >&2
  echo "Refusing to overwrite. Move it aside or choose a new output_dir." >&2
  exit 2
fi
if [[ -e "$TARBALL" || -e "$SHA_PATH" ]]; then
  echo "Archive already exists: $TARBALL or $SHA_PATH" >&2
  echo "Refusing to overwrite." >&2
  exit 2
fi
if [[ -e "$SMOKE_OUTPUT_DIR" ]]; then
  echo "Smoke output directory already exists: $SMOKE_OUTPUT_DIR" >&2
  echo "Delete or inspect it before running." >&2
  exit 2
fi

bash scripts/robustness/remote_preflight_check.sh "$CONFIG_PATH"

python3 - "$CONFIG_PATH" "$COMPLETED_DIR" "$ALLOW_VERIFIED_UNCOMPRESSED_OUTPUTS" <<'PY'
import sys
from pathlib import Path

selected = Path(sys.argv[1])
completed_dir = Path(sys.argv[2])
allow_verified = sys.argv[3] == "1"
try:
    import yaml
except ImportError as exc:
    raise SystemExit(f"PyYAML missing after preflight: {exc}")

selected_cfg = yaml.safe_load(selected.read_text(encoding="utf-8")) or {}
selected_output = Path(selected_cfg["output_dir"])
for config_path in sorted(Path("configs/robustness").glob("*.yaml")):
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    output_dir = Path(cfg.get("output_dir", ""))
    if output_dir and output_dir != selected_output and output_dir.exists():
        marker = completed_dir / f"{config_path.stem}.done"
        if allow_verified and marker.exists():
            print(f"Keeping verified previous output for {config_path.stem}: {output_dir}")
            continue
        raise SystemExit(
            "Another uncompressed robustness output exists: "
            f"{output_dir}. Download/verify/mark/delete it before starting a new model."
        )
print("No other uncompressed robustness model outputs found.")
PY

echo "2-step smoke training"
echo "---------------------"
python3 step5_train_lm_formal.py \
  --config "$CONFIG_PATH" \
  --output_dir "$SMOKE_OUTPUT_DIR" \
  --max_steps 2 \
  --eval_steps 1 \
  --save_steps 1 \
  --logging_steps 1 \
  --save_total_limit 1

python3 - "$SMOKE_OUTPUT_DIR" <<'PY'
import json
import math
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
summary_path = output_dir / "run_summary.json"
log_path = output_dir / "train_log.jsonl"
checkpoint = output_dir / "checkpoint-2"
if not summary_path.exists():
    raise SystemExit(f"missing smoke run summary: {summary_path}")
if not checkpoint.exists():
    raise SystemExit(f"missing smoke checkpoint: {checkpoint}")
summary = json.loads(summary_path.read_text(encoding="utf-8"))
if summary.get("final_step") != 2:
    raise SystemExit(f"smoke final_step={summary.get('final_step')}, expected 2")
if summary.get("parameter_count") != 134107392:
    raise SystemExit(f"smoke parameter_count={summary.get('parameter_count')}, expected 134107392")
lines = [line for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
if not lines:
    raise SystemExit("smoke train_log.jsonl is empty")
last = json.loads(lines[-1])
for key in ["train_loss", "eval_loss"]:
    value = last.get(key)
    if value is None or not math.isfinite(float(value)):
        raise SystemExit(f"smoke {key} is not finite: {value}")
print("Smoke training passed")
print(f"  final_step: {summary.get('final_step')}")
print(f"  tokens_seen: {summary.get('tokens_seen')}")
print(f"  train_loss: {last.get('train_loss')}")
print(f"  eval_loss: {last.get('eval_loss')}")
PY
rm -rf "$SMOKE_OUTPUT_DIR"
echo "Deleted smoke output: $SMOKE_OUTPUT_DIR"

FREE_BEFORE="$(bytes_free)"
echo "Run: $RUN_ID"
echo "Config: $CONFIG_PATH"
echo "Output: $OUTPUT_DIR"
echo "Max steps: $MAX_STEPS"
echo "Free before: $(human_bytes "$FREE_BEFORE")"
echo "Disk usage before:"
df -h .
echo "GPU info:"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi not found"
fi

set +e
python3 step5_train_lm_formal.py --config "$CONFIG_PATH" 2>&1 | tee "$LOG_PATH"
train_status=${PIPESTATUS[0]}
set -e
if [[ "$train_status" -ne 0 ]]; then
  echo "Training failed with status $train_status. Log: $LOG_PATH" >&2
  exit "$train_status"
fi

python3 - "$OUTPUT_DIR" "$MAX_STEPS" <<'PY'
import json
import math
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
expected_steps = int(sys.argv[2])
expected_tokens = expected_steps * 65536
summary = json.loads((output_dir / "run_summary.json").read_text(encoding="utf-8"))
log_path = output_dir / "train_log.jsonl"
lines = [line for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
if not lines:
    raise SystemExit("train_log.jsonl is empty after full training")
last = json.loads(lines[-1])
checks = {
    "final_step": (summary.get("final_step"), expected_steps),
    "tokens_seen": (summary.get("tokens_seen"), expected_tokens),
    "parameter_count": (summary.get("parameter_count"), 134107392),
}
for label, (actual, expected) in checks.items():
    if actual != expected:
        raise SystemExit(f"{label}={actual}, expected {expected}")
for label in ["train_loss", "eval_loss"]:
    value = last.get(label)
    if value is None or not math.isfinite(float(value)):
        raise SystemExit(f"{label} is not finite: {value}")
print("Full-run metadata checks passed")
print(f"  final_step: {summary.get('final_step')}")
print(f"  tokens_seen: {summary.get('tokens_seen')}")
print(f"  parameter_count: {summary.get('parameter_count')}")
print(f"  final_train_loss: {last.get('train_loss')}")
print(f"  final_eval_loss: {last.get('eval_loss')}")
PY

FREE_AFTER_TRAIN="$(bytes_free)"
OUTPUT_BYTES="$(path_bytes "$OUTPUT_DIR")"
DELTA_AFTER_TRAIN=$((FREE_BEFORE - FREE_AFTER_TRAIN))

python3 - "$OUTPUT_DIR" "$CONFIG_PATH" "$RUN_ID" "$FREE_BEFORE" "$FREE_AFTER_TRAIN" "$OUTPUT_BYTES" <<'PY'
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
config_path = sys.argv[2]
run_id = sys.argv[3]
free_before = int(float(sys.argv[4]))
free_after_train = int(float(sys.argv[5]))
output_bytes = int(float(sys.argv[6]))
summary_path = output_dir / "run_summary.json"
log_path = output_dir / "train_log.jsonl"
summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
last_log = {}
if log_path.exists():
    lines = [line for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if lines:
        last_log = json.loads(lines[-1])
payload = {
    "run_id": run_id,
    "config_path": config_path,
    "output_dir": str(output_dir),
    "free_before_bytes": free_before,
    "free_after_train_bytes": free_after_train,
    "disk_delta_after_train_bytes": free_before - free_after_train,
    "output_dir_bytes": output_bytes,
    "run_summary": summary,
    "last_train_log": last_log,
}
(output_dir / "robustness_run_metadata.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY

echo "Archiving output directory..."
tar -czf "$TARBALL" "$OUTPUT_DIR"
sha256_file "$TARBALL" > "$SHA_PATH"
TARBALL_BYTES="$(path_bytes "$TARBALL")"
FREE_AFTER_TAR="$(bytes_free)"
DELTA_AFTER_TAR=$((FREE_BEFORE - FREE_AFTER_TAR))

python3 - "$META_PATH" "$FREE_AFTER_TAR" "$TARBALL_BYTES" "$DELTA_AFTER_TAR" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
payload["free_after_tar_bytes"] = int(float(sys.argv[2]))
payload["tarball_bytes"] = int(float(sys.argv[3]))
payload["disk_delta_after_tar_bytes"] = int(float(sys.argv[4]))
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

echo "Disk accounting:"
echo "  output_dir_size: $(human_bytes "$OUTPUT_BYTES")"
echo "  tarball_size: $(human_bytes "$TARBALL_BYTES")"
echo "  free_after_train: $(human_bytes "$FREE_AFTER_TRAIN")"
echo "  free_after_tar: $(human_bytes "$FREE_AFTER_TAR")"
echo "  disk_delta_after_train: $(human_bytes "$DELTA_AFTER_TRAIN")"
echo "  disk_delta_after_tar: $(human_bytes "$DELTA_AFTER_TAR")"
echo "Archive: $TARBALL"
echo "SHA256:"
cat "$SHA_PATH"

python3 - "$OUTPUT_DIR" <<'PY'
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
summary = json.loads((output_dir / "run_summary.json").read_text(encoding="utf-8"))
last = {}
log_path = output_dir / "train_log.jsonl"
if log_path.exists():
    lines = [line for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if lines:
        last = json.loads(lines[-1])
print("Final metrics:")
print(f"  final_step: {summary.get('final_step')}")
print(f"  tokens_seen: {summary.get('tokens_seen')}")
print(f"  train_loss: {last.get('train_loss')}")
print(f"  eval_loss: {last.get('eval_loss')}")
PY

cat <<EOF
Next required step before deleting remote artifacts or starting another full model:
  1. Download $TARBALL and $SHA_PATH to local storage.
  2. Verify the sha256 locally.
  3. Mark this server artifact as locally verified:
       touch ${TARBALL}.local_verified
  4. Then rerun the batch script or manually delete remote artifacts if needed.

The script intentionally keeps the unpacked output and tarball until local verification is marked.
EOF
