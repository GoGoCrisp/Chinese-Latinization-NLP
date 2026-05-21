#!/usr/bin/env bash
# Remote preflight for robustness training. Does not start full training.
set -euo pipefail

CONFIG_PATH="${1:-}"
BASE_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$BASE_DIR"

MIN_VRAM_GB="${MIN_VRAM_GB:-35}"
MIN_FREE_DISK_GB="${MIN_FREE_DISK_GB:-10}"
INSTALL_MISSING_DEPS="${INSTALL_MISSING_DEPS:-1}"

gb_to_bytes() {
  python3 - "$1" <<'PY'
import sys
print(int(float(sys.argv[1]) * 1024 ** 3))
PY
}

bytes_free() {
  df -Pk . | awk 'NR==2 {print $4 * 1024}'
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

echo "System checks"
echo "-------------"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found; GPU is not visible in this environment." >&2
  exit 2
fi
nvidia-smi
df -h .
if command -v free >/dev/null 2>&1; then
  free -h
else
  echo "free command not found"
fi
if command -v lscpu >/dev/null 2>&1; then
  lscpu | head
else
  echo "lscpu command not found"
fi

free_bytes="$(bytes_free)"
min_free_bytes="$(gb_to_bytes "$MIN_FREE_DISK_GB")"
echo "Free disk: $(human_bytes "$free_bytes")"
echo "Minimum free disk required by preflight: $(human_bytes "$min_free_bytes")"
if (( free_bytes < min_free_bytes )); then
  echo "ERROR: free disk is below MIN_FREE_DISK_GB=${MIN_FREE_DISK_GB}." >&2
  exit 2
fi

echo
echo "Dependency check"
echo "----------------"
python3 - <<'PY'
import importlib
import subprocess
import sys

required = [
    ("torch", "torch", False),
    ("transformers", "transformers", True),
    ("tokenizers", "tokenizers", True),
    ("datasets", "datasets", True),
    ("yaml", "pyyaml", True),
    ("numpy", "numpy", True),
    ("pandas", "pandas", True),
    ("tqdm", "tqdm", True),
    ("pypinyin", "pypinyin", True),
    ("jieba", "jieba", True),
]
missing_installable = []
missing_noninstallable = []
for module_name, package_name, installable in required:
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "version_unknown")
        print(f"  OK {module_name}: {version}")
    except ImportError:
        print(f"  MISSING {module_name}")
        if installable:
            missing_installable.append(package_name)
        else:
            missing_noninstallable.append(package_name)

if missing_noninstallable:
    raise SystemExit(
        "Missing non-installable/core dependency in this preflight: "
        + ", ".join(missing_noninstallable)
        + ". Do not blindly reinstall torch; choose a CUDA-compatible image/wheel first."
    )

if missing_installable:
    print("  Missing non-torch packages:", ", ".join(missing_installable))
    if __import__("os").environ.get("INSTALL_MISSING_DEPS", "1") == "1":
        cmd = [sys.executable, "-m", "pip", "install", *missing_installable]
        print("  Installing missing non-torch packages:", " ".join(cmd))
        subprocess.check_call(cmd)
    else:
        raise SystemExit("Missing dependencies and INSTALL_MISSING_DEPS is not 1.")
PY

echo
echo "PyTorch/CUDA/bf16 check"
echo "----------------------"
python3 - "$MIN_VRAM_GB" <<'PY'
import sys
import torch

min_vram_gb = float(sys.argv[1])
print("  torch:", torch.__version__)
print("  cuda available:", torch.cuda.is_available())
print("  torch cuda:", torch.version.cuda)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available; refusing to train.")

device_index = torch.cuda.current_device()
props = torch.cuda.get_device_properties(device_index)
total_gb = props.total_memory / 1024**3
print("  device:", torch.cuda.get_device_name(device_index))
print(f"  total VRAM GiB: {total_gb:.2f}")
print("  bf16 supported:", torch.cuda.is_bf16_supported())
if total_gb < min_vram_gb:
    raise SystemExit(f"GPU VRAM {total_gb:.2f} GiB is below MIN_VRAM_GB={min_vram_gb}.")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("bf16 is not supported; this does not match the previous successful setup.")

a = torch.randn((1024, 1024), device="cuda", dtype=torch.bfloat16)
b = torch.randn((1024, 1024), device="cuda", dtype=torch.bfloat16)
c = a @ b
torch.cuda.synchronize()
if not torch.isfinite(c.float()).all():
    raise SystemExit("CUDA bf16 matrix multiplication produced non-finite values.")
print("  bf16 matmul smoke: OK")
PY

echo
echo "File/path and config validation"
echo "-------------------------------"
required_paths=(
  "tokenizers/pinyin_diacritic_32k_eos"
  "data/tokenized/diacritic_train_full_eos_1024"
  "data/tokenized/diacritic_valid_full_eos_1024"
  "configs/robustness"
  "step5_train_lm_formal.py"
)
for path in "${required_paths[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "ERROR: missing required path: $path" >&2
    exit 2
  fi
  echo "  OK $path"
done

python3 scripts/robustness/validate_robustness_configs.py

echo
echo "Tokenizer, dataset, and parameter-count validation"
echo "--------------------------------------------------"
python3 - <<'PY'
from pathlib import Path

from datasets import load_from_disk
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

tokenizers = [("Pinyin-Diacritic", Path("tokenizers/pinyin_diacritic_32k_eos"))]
for label, path in tokenizers:
    tok = PreTrainedTokenizerFast.from_pretrained(str(path), local_files_only=True)
    print(f"  {label} tokenizer loaded: {path}")
    print(f"    vocab_size: {len(tok)}")
    print(f"    eos_token_id: {tok.eos_token_id}")
    print(f"    pad_token_id: {tok.pad_token_id}")
    if len(tok) != 32001 or tok.eos_token_id != 32000 or tok.pad_token_id != 32000:
        raise SystemExit(f"{label} tokenizer ids do not match 32001/32000/32000")

datasets = [
    ("Pinyin-Diacritic train", Path("data/tokenized/diacritic_train_full_eos_1024")),
    ("Pinyin-Diacritic valid", Path("data/tokenized/diacritic_valid_full_eos_1024")),
]
for label, path in datasets:
    ds = load_from_disk(str(path))
    if "input_ids" not in ds.column_names:
        raise SystemExit(f"{label} missing input_ids column")
    row_len = len(ds[0]["input_ids"])
    print(f"  {label}: rows={len(ds)} row0_len={row_len}")
    if row_len != 1024:
        raise SystemExit(f"{label} row length is {row_len}, expected 1024")

config = LlamaConfig(
    vocab_size=32001,
    hidden_size=768,
    intermediate_size=2048,
    num_hidden_layers=12,
    num_attention_heads=12,
    max_position_embeddings=1024,
    rms_norm_eps=1.0e-5,
    hidden_act="silu",
    rope_theta=10000.0,
    pad_token_id=32000,
    eos_token_id=32000,
    tie_word_embeddings=False,
)
model = LlamaForCausalLM(config)
param_count = sum(p.numel() for p in model.parameters())
print(f"  parameter_count: {param_count}")
if param_count != 134107392:
    raise SystemExit(f"parameter_count={param_count}, expected 134107392")
del model
PY

if [[ -n "$CONFIG_PATH" ]]; then
  if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "ERROR: config does not exist: $CONFIG_PATH" >&2
    exit 2
  fi
  python3 - "$CONFIG_PATH" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    import yaml
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
except ImportError:
    cfg = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        clean = line.split("#", 1)[0].strip()
        if not clean or ":" not in clean:
            continue
        key, value = clean.split(":", 1)
        cfg[key.strip()] = value.strip().strip("\"'")
out = cfg.get("output_dir")
if not out:
    raise SystemExit(f"{path} has no output_dir")
if Path(out).exists():
    raise SystemExit(f"output_dir already exists and would be overwritten: {out}")
print(f"  OK selected config output_dir is new: {out}")
PY
fi

echo
echo "Remote preflight passed."
