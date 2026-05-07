#!/usr/bin/env python3
"""Held-out normalized perplexity evaluation for Experiment 2 causal LMs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


EXPECTED_VOCAB_SIZE = 32001
EXPECTED_EOS_ID = 32000
EXPECTED_PAD_ID = 32000
EXPECTED_BLOCK_SIZE = 1024
EXPECTED_ALIGNED_CHINESE_LINES = 13375

DEFAULT_OUTPUT_DIR = Path("eval_results/normalized_ppl_4epoch")

CSV_COLUMNS = [
    "run_name",
    "script",
    "checkpoint",
    "tokenizer",
    "test_dataset",
    "raw_text_for_script",
    "aligned_chinese_text_for_normalization",
    "final_train_steps",
    "train_tokens_seen",
    "eval_total_nll",
    "eval_token_nll",
    "eval_token_ppl",
    "eval_raw_tokens",
    "eval_effective_loss_tokens",
    "raw_line_count",
    "aligned_chinese_chars",
    "aligned_chinese_utf8_bytes",
    "nll_per_chinese_char",
    "ppl_per_chinese_char",
    "nll_per_utf8_byte",
    "ppl_per_utf8_byte",
    "device",
    "dtype",
    "batch_size",
    "notes",
]


@dataclass(frozen=True)
class EvalRun:
    run_name: str
    script: str
    checkpoint: str
    tokenizer: str
    test_dataset: str
    raw_text_for_script: str
    aligned_chinese_text_for_normalization: str
    final_train_steps: int
    train_tokens_seen: int
    output_json: str
    note: str


RUNS: dict[str, EvalRun] = {
    "chinese_4epoch": EvalRun(
        run_name="chinese_4epoch",
        script="chinese_origin",
        checkpoint=(
            "server_outputs/4epoch/outputs/"
            "chinese_125m_b1024_4epoch_seed42/checkpoint-27176"
        ),
        tokenizer="tokenizers/chinese_origin_32k_eos",
        test_dataset="data/tokenized/chinese_test_full_eos_1024",
        raw_text_for_script="data/raw/test.zh.txt",
        aligned_chinese_text_for_normalization="data/raw/test.zh.txt",
        final_train_steps=27176,
        train_tokens_seen=1781006336,
        output_json="chinese_4epoch.json",
        note=(
            "Chinese-Origin 4epoch held-out LM; char/byte normalization uses the same "
            "Chinese test text."
        ),
    ),
    "diacritic_matched_token_4epoch": EvalRun(
        run_name="diacritic_matched_token_4epoch",
        script="pinyin_diacritic",
        checkpoint=(
            "server_outputs/4epoch/diacritic_125m_b1024_matched_token_4epoch_seed42_outputs/"
            "outputs/diacritic_125m_b1024_matched_token_4epoch_seed42/checkpoint-27176"
        ),
        tokenizer="tokenizers/pinyin_diacritic_32k_eos",
        test_dataset="data/tokenized/diacritic_test_full_eos_1024",
        raw_text_for_script="data/raw/test.diacritic.txt",
        aligned_chinese_text_for_normalization="data/raw/test.zh.txt",
        final_train_steps=27176,
        train_tokens_seen=1781006336,
        output_json="diacritic_matched_token_4epoch.json",
        note=(
            "Pinyin-Diacritic matched-token 4epoch held-out LM; char/byte normalization uses aligned "
            "original Chinese test text."
        ),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Experiment 2 causal LMs on held-out LM loss."
    )
    parser.add_argument(
        "--run",
        choices=["all", *RUNS.keys()],
        default="all",
        help="Run one configured evaluation or all three.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Per-device eval batch size. Default: 8.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for per-run JSON files and summary.csv.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker count. Default: 0 for portability.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Debug only: evaluate at most this many batches.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse existing per-run JSON files instead of recomputing them.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    return parser.parse_args()


def choose_device_and_dtype() -> tuple[torch.device, torch.dtype, str]:
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            return torch.device("cuda"), torch.bfloat16, "bf16"
        return torch.device("cuda"), torch.float32, "fp32"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), torch.float32, "fp32"
    return torch.device("cpu"), torch.float32, "fp32"


def as_project_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def safe_exp(value: float) -> float:
    try:
        return math.exp(value)
    except OverflowError:
        return float("inf")


def count_raw_text(path: Path) -> dict[str, int]:
    line_count = 0
    char_count = 0
    byte_count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.rstrip("\n")
            line_count += 1
            char_count += len(text)
            byte_count += len(text.encode("utf-8"))
    return {
        "raw_line_count": line_count,
        "aligned_chinese_chars": char_count,
        "aligned_chinese_utf8_bytes": byte_count,
    }


def collate_lm(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    input_ids = torch.tensor([example["input_ids"] for example in batch], dtype=torch.long)
    return {"input_ids": input_ids, "labels": input_ids.clone()}


def require_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def validate_no_train_leak(run: EvalRun) -> None:
    dataset_name = Path(run.test_dataset).name.lower()
    raw_name = Path(run.raw_text_for_script).name.lower()
    if "test" not in dataset_name or "test" not in raw_name:
        raise ValueError(
            f"{run.run_name}: expected test paths, got dataset={run.test_dataset}, "
            f"raw={run.raw_text_for_script}"
        )
    if "train" in dataset_name or "valid" in dataset_name:
        raise ValueError(f"{run.run_name}: test_dataset appears unsafe: {run.test_dataset}")


def validate_tokenizer_and_config(checkpoint: Path, tokenizer_path: Path) -> tuple[Any, int]:
    config = AutoConfig.from_pretrained(str(checkpoint), local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)

    if config.vocab_size != len(tokenizer):
        raise ValueError(
            f"Model config vocab_size ({config.vocab_size}) != tokenizer size ({len(tokenizer)})"
        )
    if config.vocab_size != EXPECTED_VOCAB_SIZE:
        raise ValueError(
            f"Expected vocab size {EXPECTED_VOCAB_SIZE}; got config.vocab_size={config.vocab_size}"
        )
    if tokenizer.eos_token_id != EXPECTED_EOS_ID:
        raise ValueError(
            f"Expected tokenizer eos_token_id={EXPECTED_EOS_ID}; got {tokenizer.eos_token_id}"
        )
    if tokenizer.pad_token_id != EXPECTED_PAD_ID:
        raise ValueError(
            f"Expected tokenizer pad_token_id={EXPECTED_PAD_ID}; got {tokenizer.pad_token_id}"
        )
    return tokenizer, int(config.vocab_size)


def load_model(checkpoint: Path, dtype: torch.dtype, device: torch.device):
    kwargs = {"local_files_only": True}
    if device.type == "cuda":
        kwargs["dtype"] = dtype
    try:
        model = AutoModelForCausalLM.from_pretrained(str(checkpoint), **kwargs)
    except TypeError:
        if "dtype" in kwargs:
            kwargs["torch_dtype"] = kwargs.pop("dtype")
        model = AutoModelForCausalLM.from_pretrained(str(checkpoint), **kwargs)
    model.to(device)
    model.eval()
    return model


def validate_dataset(dataset, run: EvalRun) -> int:
    if "input_ids" not in dataset.column_names:
        raise ValueError(f"{run.run_name}: dataset has no input_ids column: {dataset.column_names}")
    if len(dataset) == 0:
        raise ValueError(f"{run.run_name}: dataset is empty")
    first_len = len(dataset[0]["input_ids"])
    if first_len != EXPECTED_BLOCK_SIZE:
        raise ValueError(
            f"{run.run_name}: expected block length {EXPECTED_BLOCK_SIZE}; got {first_len}"
        )
    return first_len


def evaluate_run(
    root: Path,
    run: EvalRun,
    output_dir: Path,
    batch_size: int,
    num_workers: int,
    max_batches: int | None,
    show_progress: bool,
    skip_existing: bool,
) -> dict[str, Any]:
    existing_json_path = output_dir / run.output_json
    if skip_existing and max_batches is None and existing_json_path.exists():
        print(f"\n== {run.run_name} ==")
        print(f"reusing existing result: {existing_json_path}")
        return json.loads(existing_json_path.read_text(encoding="utf-8"))

    checkpoint = as_project_path(root, run.checkpoint)
    tokenizer_path = as_project_path(root, run.tokenizer)
    test_dataset_path = as_project_path(root, run.test_dataset)
    raw_text_path = as_project_path(root, run.raw_text_for_script)
    aligned_chinese_path = as_project_path(root, run.aligned_chinese_text_for_normalization)

    validate_no_train_leak(run)
    require_exists(checkpoint, "model checkpoint")
    require_exists(tokenizer_path, "tokenizer")
    require_exists(test_dataset_path, "tokenized test dataset")
    require_exists(raw_text_path, "raw held-out text")
    require_exists(aligned_chinese_path, "aligned Chinese normalization text")

    device, dtype, dtype_name = choose_device_and_dtype()
    print(f"\n== {run.run_name} ==")
    print(f"checkpoint: {checkpoint}")
    print(f"tokenizer: {tokenizer_path}")
    print(f"test_dataset: {test_dataset_path}")
    print(f"device: {device.type}, dtype: {dtype_name}, batch_size: {batch_size}")

    tokenizer, model_config_vocab_size = validate_tokenizer_and_config(checkpoint, tokenizer_path)
    print(
        "validated vocab/tokenizer: "
        f"vocab_size={len(tokenizer)}, eos={tokenizer.eos_token_id}, pad={tokenizer.pad_token_id}"
    )

    dataset = load_from_disk(str(test_dataset_path))
    block_len = validate_dataset(dataset, run)
    print(f"tokenized test rows: {len(dataset)}")
    print(f"block length: {block_len}")

    raw_counts = count_raw_text(aligned_chinese_path)
    raw_line_count = raw_counts["raw_line_count"]
    print(f"aligned Chinese raw lines: {raw_line_count}")
    if raw_line_count != EXPECTED_ALIGNED_CHINESE_LINES:
        raise ValueError(
            f"Expected aligned Chinese raw line count {EXPECTED_ALIGNED_CHINESE_LINES}; "
            f"got {raw_line_count}"
        )

    model = load_model(checkpoint, dtype, device)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_lm,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    total_nll = 0.0
    raw_tokens = 0
    effective_loss_tokens = 0
    finite_loss_batches = 0
    total_batches = len(loader) if max_batches is None else min(len(loader), max_batches)
    iterator = tqdm(
        loader,
        total=total_batches,
        desc=run.run_name,
        disable=not show_progress,
    )

    with torch.inference_mode():
        for batch_idx, batch in enumerate(iterator):
            if max_batches is not None and batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(device, non_blocking=(device.type == "cuda"))
            labels = batch["labels"].to(device, non_blocking=(device.type == "cuda"))
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            loss_value = float(loss.detach().cpu().item())
            if not math.isfinite(loss_value):
                raise ValueError(f"{run.run_name}: non-finite loss at batch {batch_idx}: {loss_value}")

            batch_raw_tokens = int(labels.numel())
            if labels.ndim != 2:
                raise ValueError(f"{run.run_name}: expected labels shape [batch, seq], got {labels.shape}")
            batch_effective_tokens = int(labels[:, 1:].numel())

            total_nll += loss_value * batch_effective_tokens
            raw_tokens += batch_raw_tokens
            effective_loss_tokens += batch_effective_tokens
            finite_loss_batches += 1

    if effective_loss_tokens <= 0:
        raise ValueError(f"{run.run_name}: no effective loss tokens were evaluated")

    token_nll = total_nll / effective_loss_tokens
    nll_per_char = total_nll / raw_counts["aligned_chinese_chars"]
    nll_per_byte = total_nll / raw_counts["aligned_chinese_utf8_bytes"]

    notes = [
        run.note,
        "HF CausalLM loss shifts labels internally, so total NLL uses batch_size * (seq_len - 1).",
        "No generation or downstream benchmark evaluation was run.",
    ]
    if max_batches is not None:
        notes.append(f"DEBUG PARTIAL EVAL: max_batches={max_batches}.")

    result = {
        "run_name": run.run_name,
        "script": run.script,
        "checkpoint": run.checkpoint,
        "tokenizer": run.tokenizer,
        "test_dataset": run.test_dataset,
        "raw_text_for_script": run.raw_text_for_script,
        "aligned_chinese_text_for_normalization": run.aligned_chinese_text_for_normalization,
        "final_train_steps": run.final_train_steps,
        "train_tokens_seen": run.train_tokens_seen,
        "eval_total_nll": total_nll,
        "eval_token_nll": token_nll,
        "eval_token_ppl": safe_exp(token_nll),
        "eval_raw_tokens": raw_tokens,
        "eval_effective_loss_tokens": effective_loss_tokens,
        "raw_line_count": raw_line_count,
        "aligned_chinese_chars": raw_counts["aligned_chinese_chars"],
        "aligned_chinese_utf8_bytes": raw_counts["aligned_chinese_utf8_bytes"],
        "nll_per_chinese_char": nll_per_char,
        "ppl_per_chinese_char": safe_exp(nll_per_char),
        "nll_per_utf8_byte": nll_per_byte,
        "ppl_per_utf8_byte": safe_exp(nll_per_byte),
        "device": device.type,
        "dtype": dtype_name,
        "batch_size": batch_size,
        "notes": " ".join(notes),
        "validation": {
            "checkpoint_exists": True,
            "model_config_vocab_size": model_config_vocab_size,
            "tokenizer_vocab_size": len(tokenizer),
            "tokenizer_eos_token_id": tokenizer.eos_token_id,
            "tokenizer_pad_token_id": tokenizer.pad_token_id,
            "tokenized_test_rows": len(dataset),
            "block_length": block_len,
            "aligned_chinese_raw_line_count_expected": EXPECTED_ALIGNED_CHINESE_LINES,
            "all_losses_finite": True,
            "finite_loss_batches": finite_loss_batches,
            "no_train_set_used": True,
            "loss_token_accounting": {
                "raw_tokens": raw_tokens,
                "effective_loss_tokens": effective_loss_tokens,
                "causal_shift_ignored_first_token_per_sequence": True,
            },
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / run.output_json
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote: {json_path}")
    return result


def write_summary_csv(output_dir: Path, rows: list[dict[str, Any]]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in CSV_COLUMNS})
    return summary_path


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    root = Path.cwd()
    output_dir = as_project_path(root, args.output_dir)

    selected_runs = list(RUNS.values()) if args.run == "all" else [RUNS[args.run]]
    results = [
        evaluate_run(
            root=root,
            run=run,
            output_dir=output_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_batches=args.max_batches,
            show_progress=not args.no_progress,
            skip_existing=args.skip_existing,
        )
        for run in selected_runs
    ]
    summary_path = write_summary_csv(output_dir, results)
    print(f"\nwrote summary: {summary_path}")


if __name__ == "__main__":
    main()
