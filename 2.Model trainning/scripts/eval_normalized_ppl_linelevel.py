#!/usr/bin/env python3
"""Line-level normalized PPL evaluation for 4epoch causal LMs.

This Eval 1 variant scores aligned held-out raw lines independently and reports
source-Chinese-character-normalized NLL/PPL as the primary metric.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


EXPECTED_VOCAB_SIZE = 32001
EXPECTED_EOS_ID = 32000
EXPECTED_PAD_ID = 32000
DEFAULT_OUTPUT_DIR = Path("eval_results/eval1/normalized_ppl_4epoch_linelevel")
OLD_BLOCK_SUMMARY = Path("eval_results/eval1/normalized_ppl_4epoch/summary.csv")

PER_LINE_COLUMNS = [
    "line_id",
    "zh_text",
    "diacritic_text",
    "zh_num_chars",
    "zh_num_utf8_bytes",
    "chinese_total_nll",
    "chinese_num_model_tokens",
    "chinese_num_chunks",
    "chinese_line_char_nll",
    "diacritic_total_nll",
    "diacritic_num_model_tokens",
    "diacritic_num_chunks",
    "diacritic_line_char_nll",
]

SUMMARY_COLUMNS = [
    "model",
    "total_lines",
    "scored_lines",
    "skipped_lines",
    "total_source_chars",
    "total_nll",
    "char_nll",
    "char_ppl",
    "diagnostic_model_tokens",
    "diagnostic_token_nll",
    "diagnostic_token_ppl",
    "lines_over_1024",
    "total_chunks",
]


@dataclass(frozen=True)
class EvalRun:
    model: str
    checkpoint: str
    tokenizer: str
    text_key: str
    output_json: str


RUNS: dict[str, EvalRun] = {
    "chinese_4epoch": EvalRun(
        model="chinese_4epoch",
        checkpoint=(
            "server_outputs/4epoch/outputs/"
            "chinese_125m_b1024_4epoch_seed42/checkpoint-27176"
        ),
        tokenizer="tokenizers/chinese_origin_32k_eos",
        text_key="zh_text",
        output_json="chinese_4epoch.json",
    ),
    "diacritic_matched_token_4epoch": EvalRun(
        model="diacritic_matched_token_4epoch",
        checkpoint=(
            "server_outputs/4epoch/diacritic_125m_b1024_matched_token_4epoch_seed42_outputs/"
            "outputs/diacritic_125m_b1024_matched_token_4epoch_seed42/checkpoint-27176"
        ),
        tokenizer="tokenizers/pinyin_diacritic_32k_eos",
        text_key="diacritic_text",
        output_json="diacritic_matched_token_4epoch.json",
    ),
}


def load_eval_runs(path_text: str | None, root: Path) -> list[EvalRun]:
    if path_text is None:
        return list(RUNS.values())
    path = as_project_path(root, path_text)
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("model_runs", payload) if isinstance(payload, dict) else payload
    runs: list[EvalRun] = []
    for record in records:
        model = record.get("model") or record.get("run_name")
        text_key = record.get("ppl_text_key") or record.get("text_key")
        if text_key in {"zh", "zh_text"}:
            text_key = "zh_text"
        elif text_key in {"diacritic", "diacritic_text"}:
            text_key = "diacritic_text"
        elif record.get("script") == "chinese_origin":
            text_key = "zh_text"
        else:
            text_key = "diacritic_text"
        runs.append(
            EvalRun(
                model=model,
                checkpoint=record["checkpoint"],
                tokenizer=record["tokenizer"],
                text_key=text_key,
                output_json=record.get("output_json") or f"{model}.json",
            )
        )
    if not runs:
        raise ValueError(f"No Eval 1 runs found in {path}")
    return runs


def per_line_prefix(run: EvalRun, legacy_default_pair: bool) -> str:
    if legacy_default_pair:
        return "chinese" if run.text_key == "zh_text" else "diacritic"
    return re.sub(r"[^0-9A-Za-z_]+", "_", run.model).strip("_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate 4epoch Chinese and Diacritic LMs on aligned raw held-out "
            "lines with source-character-normalized PPL."
        )
    )
    parser.add_argument("--zh-text", default="data/raw/test.zh.txt")
    parser.add_argument("--diacritic-text", default="data/raw/test.diacritic.txt")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--old-summary", default=str(OLD_BLOCK_SUMMARY))
    parser.add_argument(
        "--model-runs-json",
        default=None,
        help="Optional JSON manifest with model runs to evaluate. Defaults to the built-in seed42 pair.",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=None,
        help="Smoke/debug only: evaluate the first N aligned raw lines.",
    )
    parser.add_argument(
        "--target-chunk-size",
        type=int,
        default=None,
        help=(
            "Number of target tokens scored per forward pass. Default is "
            "max_position_embeddings - 1 so an EOS context prefix fits."
        ),
    )
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def as_project_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def require_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def safe_exp(value: float) -> float:
    try:
        return math.exp(value)
    except OverflowError:
        return float("inf")


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


def read_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.rstrip("\n").rstrip("\r") for line in handle]


def validate_aligned_lines(
    zh_lines: list[str],
    diacritic_lines: list[str],
    max_lines: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(zh_lines) != len(diacritic_lines):
        raise ValueError(
            "Aligned raw files have different line counts: "
            f"zh={len(zh_lines)}, diacritic={len(diacritic_lines)}"
        )
    if max_lines is not None:
        if max_lines <= 0:
            raise ValueError(f"--max-lines must be positive when set; got {max_lines}")
        zh_lines = zh_lines[:max_lines]
        diacritic_lines = diacritic_lines[:max_lines]

    empty_line_ids = [
        idx + 1
        for idx, (zh_text, diacritic_text) in enumerate(zip(zh_lines, diacritic_lines))
        if zh_text == "" or diacritic_text == ""
    ]
    rows: list[dict[str, Any]] = []
    for idx, (zh_text, diacritic_text) in enumerate(zip(zh_lines, diacritic_lines), start=1):
        zh_num_chars = len(zh_text)
        rows.append(
            {
                "line_id": idx,
                "zh_text": zh_text,
                "diacritic_text": diacritic_text,
                "zh_num_chars": zh_num_chars,
                "zh_num_utf8_bytes": len(zh_text.encode("utf-8")),
                "skip": zh_text == "" or diacritic_text == "",
                "skip_reason": "empty raw aligned line"
                if zh_text == "" or diacritic_text == ""
                else "",
            }
        )

    diagnostics = {
        "evaluated_lines_checked": len(zh_lines),
        "empty_line_count": len(empty_line_ids),
        "empty_line_ids_preview": empty_line_ids[:50],
        "primary_source_char_count_definition": (
            "All non-newline Unicode characters in data/raw/test.zh.txt are counted, "
            "including spaces, punctuation, digits, Latin letters, and CJK characters."
        ),
    }
    return rows, diagnostics


def count_cjk_chars(texts: list[str]) -> int:
    total = 0
    for text in texts:
        for char in text:
            codepoint = ord(char)
            if (
                0x3400 <= codepoint <= 0x4DBF
                or 0x4E00 <= codepoint <= 0x9FFF
                or 0xF900 <= codepoint <= 0xFAFF
                or 0x20000 <= codepoint <= 0x2A6DF
                or 0x2A700 <= codepoint <= 0x2B73F
                or 0x2B740 <= codepoint <= 0x2B81F
                or 0x2B820 <= codepoint <= 0x2CEAF
                or 0x30000 <= codepoint <= 0x3134F
            ):
                total += 1
    return total


def validate_tokenizer_and_config(checkpoint: Path, tokenizer_path: Path) -> tuple[Any, int]:
    config = AutoConfig.from_pretrained(str(checkpoint), local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        use_fast=True,
        local_files_only=True,
    )
    if config.vocab_size != len(tokenizer):
        raise ValueError(
            f"Model config vocab_size ({config.vocab_size}) != tokenizer size ({len(tokenizer)})"
        )
    if config.vocab_size != EXPECTED_VOCAB_SIZE:
        raise ValueError(
            f"Expected vocab_size={EXPECTED_VOCAB_SIZE}; got config.vocab_size={config.vocab_size}"
        )
    if tokenizer.eos_token_id != EXPECTED_EOS_ID:
        raise ValueError(
            f"Expected eos_token_id={EXPECTED_EOS_ID}; got {tokenizer.eos_token_id}"
        )
    if tokenizer.pad_token_id != EXPECTED_PAD_ID:
        raise ValueError(
            f"Expected pad_token_id={EXPECTED_PAD_ID}; got {tokenizer.pad_token_id}"
        )
    if int(config.max_position_embeddings) != 1024:
        raise ValueError(
            "Expected model max_position_embeddings=1024; "
            f"got {config.max_position_embeddings}"
        )
    return tokenizer, int(config.max_position_embeddings)


def load_model(checkpoint: Path, dtype: torch.dtype, device: torch.device):
    kwargs: dict[str, Any] = {"local_files_only": True}
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


def score_target_tokens(
    model,
    target_ids: list[int],
    eos_token_id: int,
    max_position_embeddings: int,
    target_chunk_size: int,
    device: torch.device,
) -> tuple[float, int, int]:
    if not target_ids:
        return 0.0, 0, 0
    if target_chunk_size <= 0:
        raise ValueError(f"target_chunk_size must be positive; got {target_chunk_size}")
    if target_chunk_size > max_position_embeddings - 1:
        raise ValueError(
            "target_chunk_size must leave room for one context token; "
            f"got {target_chunk_size} for max_position_embeddings={max_position_embeddings}"
        )

    total_nll = 0.0
    total_scored_tokens = 0
    num_chunks = 0
    full_ids = [eos_token_id, *target_ids]

    with torch.inference_mode():
        for start in range(0, len(target_ids), target_chunk_size):
            end = min(len(target_ids), start + target_chunk_size)
            input_end = end + 1
            input_start = max(0, input_end - max_position_embeddings)
            score_start = start + 1
            score_end = end + 1
            input_chunk = full_ids[input_start:input_end]
            label_ids = list(input_chunk)
            for label_pos, full_pos in enumerate(range(input_start, input_end)):
                if full_pos < score_start or full_pos >= score_end:
                    label_ids[label_pos] = -100

            input_ids = torch.tensor(
                [input_chunk],
                dtype=torch.long,
                device=device,
            )
            labels = torch.tensor([label_ids], dtype=torch.long, device=device)
            outputs = model(input_ids=input_ids, labels=labels)
            loss_value = float(outputs.loss.detach().cpu().item())
            if not math.isfinite(loss_value):
                raise ValueError(f"non-finite loss while scoring chunk: {loss_value}")

            num_scored = end - start
            total_nll += loss_value * num_scored
            total_scored_tokens += num_scored
            num_chunks += 1

    return total_nll, total_scored_tokens, num_chunks


def evaluate_model(
    root: Path,
    run: EvalRun,
    rows: list[dict[str, Any]],
    output_dir: Path,
    total_source_chars: int,
    total_source_bytes: int,
    cjk_source_chars: int,
    requested_target_chunk_size: int | None,
    show_progress: bool,
    line_prefix: str,
) -> dict[str, Any]:
    checkpoint = as_project_path(root, run.checkpoint)
    tokenizer_path = as_project_path(root, run.tokenizer)
    require_path(checkpoint, "model checkpoint")
    require_path(tokenizer_path, "tokenizer")

    device, dtype, dtype_name = choose_device_and_dtype()
    tokenizer, max_position_embeddings = validate_tokenizer_and_config(
        checkpoint, tokenizer_path
    )
    target_chunk_size = requested_target_chunk_size or (max_position_embeddings - 1)
    if target_chunk_size > max_position_embeddings - 1:
        raise ValueError(
            "The line-level scorer prepends one EOS/context token, so "
            f"--target-chunk-size must be <= {max_position_embeddings - 1}"
        )

    print(f"\n== {run.model} ==")
    print(f"checkpoint: {checkpoint}")
    print(f"tokenizer: {tokenizer_path}")
    print(
        "validated tokenizer/config: "
        f"vocab_size={len(tokenizer)}, eos={tokenizer.eos_token_id}, "
        f"pad={tokenizer.pad_token_id}, max_position_embeddings={max_position_embeddings}"
    )
    print(
        f"device: {device.type}, dtype: {dtype_name}, "
        f"target_chunk_size={target_chunk_size}"
    )

    model = load_model(checkpoint, dtype, device)
    total_nll = 0.0
    diagnostic_model_tokens = 0
    scored_lines = 0
    skipped_lines = 0
    lines_over_1024 = 0
    total_chunks = 0

    iterator = tqdm(
        rows,
        total=len(rows),
        desc=run.model,
        disable=not show_progress,
        unit="line",
    )
    for row in iterator:
        if row["skip"]:
            skipped_lines += 1
            row[f"{line_prefix}_total_nll"] = ""
            continue

        text = row[run.text_key]
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        target_ids = [*token_ids, tokenizer.eos_token_id]
        if len(target_ids) > max_position_embeddings:
            lines_over_1024 += 1

        line_nll, line_model_tokens, line_chunks = score_target_tokens(
            model=model,
            target_ids=target_ids,
            eos_token_id=tokenizer.eos_token_id,
            max_position_embeddings=max_position_embeddings,
            target_chunk_size=target_chunk_size,
            device=device,
        )
        row[f"{line_prefix}_total_nll"] = line_nll
        row[f"{line_prefix}_num_model_tokens"] = line_model_tokens
        row[f"{line_prefix}_num_chunks"] = line_chunks
        row[f"{line_prefix}_line_char_nll"] = (
            line_nll / row["zh_num_chars"] if row["zh_num_chars"] > 0 else ""
        )

        total_nll += line_nll
        diagnostic_model_tokens += line_model_tokens
        total_chunks += line_chunks
        scored_lines += 1

    if scored_lines <= 0:
        raise ValueError(f"{run.model}: no non-empty aligned lines were scored")
    if total_source_chars <= 0:
        raise ValueError("total_source_chars must be positive")
    if diagnostic_model_tokens <= 0:
        raise ValueError(f"{run.model}: no model tokens were scored")

    char_nll = total_nll / total_source_chars
    token_nll = total_nll / diagnostic_model_tokens
    byte_nll = total_nll / total_source_bytes if total_source_bytes > 0 else float("nan")

    result: dict[str, Any] = {
        "model": run.model,
        "checkpoint": run.checkpoint,
        "tokenizer": run.tokenizer,
        "raw_text_for_model": "data/raw/test.zh.txt"
        if run.text_key == "zh_text"
        else "data/raw/test.diacritic.txt",
        "aligned_chinese_text_for_normalization": "data/raw/test.zh.txt",
        "total_lines": len(rows),
        "scored_lines": scored_lines,
        "skipped_lines": skipped_lines,
        "total_source_chars": total_source_chars,
        "total_source_utf8_bytes": total_source_bytes,
        "diagnostic_cjk_source_chars": cjk_source_chars,
        "total_nll": total_nll,
        "char_nll": char_nll,
        "char_ppl": safe_exp(char_nll),
        "diagnostic_model_tokens": diagnostic_model_tokens,
        "diagnostic_token_nll": token_nll,
        "diagnostic_token_ppl": safe_exp(token_nll),
        "diagnostic_byte_nll": byte_nll,
        "diagnostic_byte_ppl": safe_exp(byte_nll),
        "lines_over_1024": lines_over_1024,
        "total_chunks": total_chunks,
        "device": device.type,
        "dtype": dtype_name,
        "vocab_size": len(tokenizer),
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "max_position_embeddings": max_position_embeddings,
        "target_chunk_size": target_chunk_size,
        "per_line_prefix": line_prefix,
        "scoring_notes": (
            "Each original aligned raw line is scored independently. Text is tokenized "
            "with add_special_tokens=False, EOS is appended manually, and an EOS/context "
            "prefix is prepended only for likelihood scoring so every target token, "
            "including the line-initial token and appended EOS, is scored exactly once. "
            "No context is carried across raw lines. Token- and byte-normalized metrics "
            "are diagnostics only; source-character-normalized PPL is primary."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / run.output_json
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote: {json_path}")

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()

    return result


def write_per_line_csv(output_dir: Path, rows: list[dict[str, Any]], line_prefixes: list[str]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "per_line_scores.csv"
    fields = [
        "line_id",
        "zh_text",
        "diacritic_text",
        "zh_num_chars",
        "zh_num_utf8_bytes",
    ]
    for prefix in line_prefixes:
        fields.extend(
            [
                f"{prefix}_total_nll",
                f"{prefix}_num_model_tokens",
                f"{prefix}_num_chunks",
                f"{prefix}_line_char_nll",
            ]
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in fields})
    return path


def write_summary_csv(output_dir: Path, results: list[dict[str, Any]]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "summary.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for result in results:
            writer.writerow({column: result[column] for column in SUMMARY_COLUMNS})
    return path


def print_old_new_comparison(root: Path, old_summary_arg: str, new_results: list[dict[str, Any]]) -> None:
    old_summary_path = as_project_path(root, old_summary_arg)
    if not old_summary_path.exists():
        print(f"\nold block summary not found: {old_summary_path}")
        return

    old_rows: dict[str, dict[str, str]] = {}
    with old_summary_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            old_rows[row["run_name"]] = row

    new_rows = {row["model"]: row for row in new_results}
    chinese_old = old_rows.get("chinese_4epoch")
    diacritic_old = old_rows.get("diacritic_matched_token_4epoch")
    chinese_new = new_rows.get("chinese_4epoch")
    diacritic_new = new_rows.get("diacritic_matched_token_4epoch")

    print("\nold block eval vs new line-level eval")
    for model in ("chinese_4epoch", "diacritic_matched_token_4epoch"):
        old = old_rows.get(model)
        new = new_rows.get(model)
        if old is None or new is None:
            continue
        print(
            f"{model}: old char_nll={float(old['nll_per_chinese_char']):.6f}, "
            f"old char_ppl={float(old['ppl_per_chinese_char']):.6f}; "
            f"new char_nll={new['char_nll']:.6f}, new char_ppl={new['char_ppl']:.6f}"
        )

    if chinese_old and diacritic_old and chinese_new and diacritic_new:
        old_direction = (
            float(chinese_old["nll_per_chinese_char"])
            < float(diacritic_old["nll_per_chinese_char"])
        )
        new_direction = chinese_new["char_nll"] < diacritic_new["char_nll"]
        print(
            "Chinese < Diacritic direction: "
            f"old={old_direction}, new={new_direction}, remains_same={old_direction == new_direction}"
        )


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    root = Path.cwd()
    output_dir = as_project_path(root, args.output_dir)
    zh_path = as_project_path(root, args.zh_text)
    diacritic_path = as_project_path(root, args.diacritic_text)
    eval_runs = load_eval_runs(args.model_runs_json, root)
    legacy_default_pair = args.model_runs_json is None
    require_path(zh_path, "Chinese raw held-out text")
    require_path(diacritic_path, "Diacritic raw held-out text")

    zh_lines_full = read_lines(zh_path)
    diacritic_lines_full = read_lines(diacritic_path)
    if len(zh_lines_full) != len(diacritic_lines_full):
        raise ValueError(
            "Raw aligned files have different full line counts: "
            f"zh={len(zh_lines_full)}, diacritic={len(diacritic_lines_full)}"
        )

    rows, alignment_diagnostics = validate_aligned_lines(
        zh_lines_full,
        diacritic_lines_full,
        args.max_lines,
    )
    total_source_chars = sum(row["zh_num_chars"] for row in rows if not row["skip"])
    total_source_bytes = sum(row["zh_num_utf8_bytes"] for row in rows if not row["skip"])
    cjk_source_chars = count_cjk_chars([row["zh_text"] for row in rows if not row["skip"]])

    print("line-level Eval 1")
    print(f"Chinese raw text: {zh_path}")
    print(f"Diacritic raw text: {diacritic_path}")
    print(f"full aligned raw lines: {len(zh_lines_full)}")
    print(f"evaluated aligned raw lines: {len(rows)}")
    print(f"empty/skipped aligned lines: {alignment_diagnostics['empty_line_count']}")
    print(f"primary total_source_chars: {total_source_chars}")
    print(f"diagnostic total_source_utf8_bytes: {total_source_bytes}")
    print(f"diagnostic CJK-only source chars: {cjk_source_chars}")
    if args.max_lines is not None:
        print(f"DEBUG/SMOKE PARTIAL EVAL: max_lines={args.max_lines}")

    results = []
    line_prefixes = []
    print(f"model runs: {', '.join(run.model for run in eval_runs)}")
    for run in eval_runs:
        prefix = per_line_prefix(run, legacy_default_pair)
        line_prefixes.append(prefix)
        results.append(
            evaluate_model(
                root=root,
                run=run,
                rows=rows,
                output_dir=output_dir,
                total_source_chars=total_source_chars,
                total_source_bytes=total_source_bytes,
                cjk_source_chars=cjk_source_chars,
                requested_target_chunk_size=args.target_chunk_size,
                show_progress=not args.no_progress,
                line_prefix=prefix,
            )
        )

    per_line_path = write_per_line_csv(output_dir, rows, line_prefixes)
    summary_path = write_summary_csv(output_dir, results)
    diagnostics_path = output_dir / "alignment_diagnostics.json"
    diagnostics_payload = {
        **alignment_diagnostics,
        "full_zh_line_count": len(zh_lines_full),
        "full_diacritic_line_count": len(diacritic_lines_full),
        "evaluated_line_count": len(rows),
        "max_lines": args.max_lines,
        "total_source_chars": total_source_chars,
        "total_source_utf8_bytes": total_source_bytes,
        "diagnostic_cjk_source_chars": cjk_source_chars,
    }
    diagnostics_path.write_text(
        json.dumps(diagnostics_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\nwrote per-line scores: {per_line_path}")
    print(f"wrote summary: {summary_path}")
    print(f"wrote alignment diagnostics: {diagnostics_path}")
    print_old_new_comparison(root, args.old_summary, results)


if __name__ == "__main__":
    main()
