#!/usr/bin/env bash
# Step 5 helper: verify the extracted Pinyin-Diacritic 125M server bundle.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "Bundle disk usage:"
du -sh .

python3 - <<'PY'
import json
from pathlib import Path

from datasets import load_from_disk
from transformers import PreTrainedTokenizerFast

root = Path(".")
tokenizer_dir = root / "tokenizers" / "pinyin_diacritic_32k_eos"
train_dir = root / "data" / "tokenized" / "diacritic_train_full_eos_1024"
valid_dir = root / "data" / "tokenized" / "diacritic_valid_full_eos_1024"
configs = [
    root / "configs" / "step5_diacritic_125m_b1024_matched_token.yaml",
    root / "configs" / "step5_diacritic_125m_b1024_matched_token_4epoch.yaml",
    root / "configs" / "step5_diacritic_125m_b1024_matched_content.yaml",
]
train_script = root / "step5_train_lm_formal.py"
requirements = root / "requirements.txt"

required_paths = [
    ("tokenizer directory", tokenizer_dir),
    ("train dataset", train_dir),
    ("valid dataset", valid_dir),
    ("training script", train_script),
    ("requirements.txt", requirements),
]
required_paths.extend((f"config {path.name}", path) for path in configs)

for label, path in required_paths:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")

tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_dir), local_files_only=True)
print("Tokenizer")
print(f"  class: {tokenizer.__class__.__name__}")
print(f"  vocab_size: {len(tokenizer)}")
print(f"  eos_token_id: {tokenizer.eos_token_id}")
print(f"  pad_token_id: {tokenizer.pad_token_id}")
if len(tokenizer) != 32001:
    raise ValueError(f"Expected vocab_size=32001; got {len(tokenizer)}")
if tokenizer.eos_token_id != 32000:
    raise ValueError(f"Expected eos_token_id=32000; got {tokenizer.eos_token_id}")
if tokenizer.pad_token_id != 32000:
    raise ValueError(f"Expected pad_token_id=32000; got {tokenizer.pad_token_id}")

for label, dataset_dir in [("train", train_dir), ("valid", valid_dir)]:
    ds = load_from_disk(str(dataset_dir))
    if "input_ids" not in ds.column_names:
        raise ValueError(f"{label} dataset missing input_ids: {ds.column_names}")
    first_len = len(ds[0]["input_ids"])
    meta_path = dataset_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata for {label}: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if first_len != 1024:
        raise ValueError(f"{label} row length must be 1024; got {first_len}")
    if meta.get("block_size") != 1024:
        raise ValueError(f"{label} metadata block_size must be 1024; got {meta.get('block_size')}")
    if meta.get("vocab_size") != 32001:
        raise ValueError(f"{label} metadata vocab_size must be 32001; got {meta.get('vocab_size')}")
    if meta.get("eos_token_id") != 32000:
        raise ValueError(f"{label} metadata eos_token_id must be 32000; got {meta.get('eos_token_id')}")
    print(f"{label} dataset")
    print(f"  rows: {len(ds)}")
    print(f"  row0_len: {first_len}")
    print(f"  lines: {meta.get('num_lines_read')}")
    print(f"  tokens: {meta.get('num_tokens')}")
    print(f"  blocks: {meta.get('num_blocks')}")
    print(f"  empty_lines: {meta.get('empty_lines')}")

test_dir = root / "data" / "tokenized" / "diacritic_test_full_eos_1024"
if test_dir.exists():
    ds = load_from_disk(str(test_dir))
    meta = json.loads((test_dir / "metadata.json").read_text(encoding="utf-8"))
    print("test dataset")
    print(f"  rows: {len(ds)}")
    print(f"  row0_len: {len(ds[0]['input_ids'])}")
    print(f"  lines: {meta.get('num_lines_read')}")
    print(f"  tokens: {meta.get('num_tokens')}")
    print(f"  blocks: {meta.get('num_blocks')}")

print("Diacritic server bundle verification passed.")
PY
