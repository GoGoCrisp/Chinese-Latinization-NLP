#!/usr/bin/env bash
# Step 5 helper: wait until the next checkpoint appears complete.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <output_dir>" >&2
  exit 2
fi

OUTPUT_DIR="$1"
if [[ ! -d "$OUTPUT_DIR" ]]; then
  echo "Output directory does not exist: $OUTPUT_DIR" >&2
  exit 2
fi

latest_step() {
  find "$OUTPUT_DIR" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null \
    | sed -E 's/.*checkpoint-([0-9]+)$/\1/' \
    | sort -n \
    | tail -n 1
}

checkpoint_complete() {
  local ckpt="$1"
  [[ -f "$ckpt/model.safetensors" || -f "$ckpt/pytorch_model.bin" ]] || return 1
  [[ -f "$ckpt/optimizer.pt" ]] || return 1
  [[ -f "$ckpt/scheduler.pt" ]] || return 1
  [[ -f "$ckpt/trainer_state.json" ]] || return 1
  [[ -f "$ckpt/training_args.bin" || -f "$ckpt/training_args.json" || -f "$ckpt/run_config_resolved.json" ]] || return 1
  return 0
}

INITIAL_STEP="$(latest_step || true)"
if [[ -z "${INITIAL_STEP:-}" ]]; then
  INITIAL_STEP=0
fi

echo "Monitoring $OUTPUT_DIR"
echo "Current latest checkpoint step: $INITIAL_STEP"
echo "Waiting for the next complete checkpoint. This script will not stop training."

while true; do
  STEP="$(latest_step || true)"
  if [[ -n "${STEP:-}" && "$STEP" -gt "$INITIAL_STEP" ]]; then
    CKPT="$OUTPUT_DIR/checkpoint-$STEP"
    if checkpoint_complete "$CKPT"; then
      sleep 5
      if checkpoint_complete "$CKPT"; then
        echo "Checkpoint appears complete: $CKPT"
        if compgen -G "$CKPT/rng_state*.pth" >/dev/null; then
          echo "RNG state file found."
        else
          echo "RNG state file not found; continuing because it is optional for this check."
        fi
        echo "Safe to interrupt training manually if you want to stop at this checkpoint."
        exit 0
      fi
    fi
  fi
  sleep 15
done
