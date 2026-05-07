#!/usr/bin/env python3
# Step 3 cleanup program: add a shared EOS/PAD token to debug tokenizers.
"""Create EOS-enabled copies of the 32K SuperBPE tokenizers."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from transformers import AutoTokenizer, PreTrainedTokenizerFast


EOS_TOKEN = "<|endoftext|>"
ALLOWED_VOCAB_SIZES = {32000, 32001}


TOKENIZER_JOBS = [
    (
        Path("tokenizers/chinese_origin_32k"),
        Path("tokenizers/chinese_origin_32k_eos"),
    ),
    (
        Path("tokenizers/pinyin_diacritic_32k"),
        Path("tokenizers/pinyin_diacritic_32k_eos"),
    ),
]


def tokenizer_json_vocab_size(tokenizer_dir: Path) -> int | None:
    tokenizer_json = tokenizer_dir / "tokenizer.json"
    if not tokenizer_json.exists():
        return None
    with tokenizer_json.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    model_vocab = payload.get("model", {}).get("vocab", {})
    added_tokens = payload.get("added_tokens", [])
    return len(model_vocab) + len(added_tokens)


def load_tokenizer(tokenizer_dir: Path):
    expected_size = tokenizer_json_vocab_size(tokenizer_dir)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir,
            use_fast=True,
            local_files_only=True,
        )
        if expected_size is None or len(tokenizer) == expected_size:
            return tokenizer, "AutoTokenizer"
        print(
            "  warning: AutoTokenizer loaded an unexpected vocab size "
            f"({len(tokenizer)} vs expected {expected_size}); using tokenizer.json."
        )
    except Exception as exc:
        print(f"  warning: AutoTokenizer failed ({exc}); using tokenizer.json.")

    tokenizer_json = tokenizer_dir / "tokenizer.json"
    if not tokenizer_json.exists():
        raise FileNotFoundError(f"Missing tokenizer.json in {tokenizer_dir}")
    return PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_json)), "tokenizer.json"


def assert_vocab_size(label: str, vocab_size: int, allowed: set[int]) -> None:
    if vocab_size not in allowed:
        raise ValueError(
            f"{label} vocab size must be one of {sorted(allowed)}; got {vocab_size}"
        )


def print_tokenizer_state(label: str, tokenizer) -> None:
    print(f"  {label}_class: {tokenizer.__class__.__name__}")
    print(f"  {label}_vocab_size: {len(tokenizer)}")
    print(f"  {label}_eos_token: {tokenizer.eos_token}")
    print(f"  {label}_eos_token_id: {tokenizer.eos_token_id}")
    print(f"  {label}_pad_token: {tokenizer.pad_token}")
    print(f"  {label}_pad_token_id: {tokenizer.pad_token_id}")


def update_tokenizer(input_dir: Path, output_dir: Path) -> dict[str, object]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input tokenizer directory does not exist: {input_dir}")
    if output_dir.exists():
        shutil.rmtree(output_dir)

    tokenizer, load_method = load_tokenizer(input_dir)
    print_tokenizer_state("loaded", tokenizer)
    old_vocab_size = len(tokenizer)
    assert_vocab_size("Loaded tokenizer", old_vocab_size, ALLOWED_VOCAB_SIZES)

    if tokenizer.eos_token_id is None:
        tokenizer.add_special_tokens({"eos_token": EOS_TOKEN})
        expected_new_vocab_size = old_vocab_size + 1
    else:
        expected_new_vocab_size = old_vocab_size

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    new_vocab_size = len(tokenizer)
    print_tokenizer_state("updated", tokenizer)
    assert_vocab_size("Updated tokenizer", new_vocab_size, ALLOWED_VOCAB_SIZES)
    if old_vocab_size == 32000 and new_vocab_size != 32001:
        raise ValueError(
            "Adding EOS should change vocab size from 32000 to 32001; "
            f"got {new_vocab_size}"
        )
    if new_vocab_size != expected_new_vocab_size:
        raise ValueError(
            f"Unexpected vocab growth: old={old_vocab_size}, "
            f"expected_new={expected_new_vocab_size}, actual_new={new_vocab_size}"
        )
    if tokenizer.eos_token != EOS_TOKEN:
        raise ValueError(f"Expected eos_token {EOS_TOKEN!r}; got {tokenizer.eos_token!r}")
    if tokenizer.pad_token != tokenizer.eos_token:
        raise ValueError("pad_token must be set to eos_token")

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_dir)

    reloaded, reload_method = load_tokenizer(output_dir)
    print_tokenizer_state("reloaded", reloaded)
    if len(reloaded) != 32001:
        raise ValueError(
            f"Saved tokenizer must reload with vocab size 32001; got {len(reloaded)}"
        )
    if reloaded.eos_token_id is None:
        raise ValueError("Saved tokenizer reloaded without eos_token_id")
    if reloaded.pad_token_id != reloaded.eos_token_id:
        raise ValueError("Saved tokenizer pad_token_id must equal eos_token_id")

    source_meta = input_dir / "meta.json"
    if source_meta.exists():
        shutil.copy2(source_meta, output_dir / "source_meta.json")

    report = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "load_method": load_method,
        "reload_method": reload_method,
        "old_vocab_size": old_vocab_size,
        "new_vocab_size": new_vocab_size,
        "reloaded_vocab_size": len(reloaded),
        "eos_token": reloaded.eos_token,
        "eos_token_id": reloaded.eos_token_id,
        "pad_token": reloaded.pad_token,
        "pad_token_id": reloaded.pad_token_id,
    }
    (output_dir / "eos_update_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    base_dir = Path(__file__).resolve().parent
    print("Adding EOS/PAD tokens")
    for input_rel, output_rel in TOKENIZER_JOBS:
        print(f"\n{input_rel} -> {output_rel}")
        report = update_tokenizer(base_dir / input_rel, base_dir / output_rel)
        print(f"  load_method: {report['load_method']}")
        print(f"  old_vocab_size: {report['old_vocab_size']}")
        print(f"  new_vocab_size: {report['new_vocab_size']}")
        print(f"  eos_token: {report['eos_token']}")
        print(f"  eos_token_id: {report['eos_token_id']}")
        print(f"  pad_token: {report['pad_token']}")
        print(f"  pad_token_id: {report['pad_token_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
