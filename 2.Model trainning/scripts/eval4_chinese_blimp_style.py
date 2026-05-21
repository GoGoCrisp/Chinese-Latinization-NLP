#!/usr/bin/env python3
"""Eval 4: Chinese BLiMP-style minimal-pair evaluation."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import torch
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


EXPECTED_VOCAB_SIZE = 32001
EXPECTED_EOS_ID = 32000
EXPECTED_PAD_ID = 32000
EXPECTED_PARAM_COUNT = 134_107_392
MAX_SEQ_LEN = 1024
DEFAULT_DATASET = "eval_data/eval4_chinese_blimp_style/eval4_chinese_blimp_style.jsonl"
DEFAULT_OUTPUT_DIR = "eval_results/eval4_chinese_blimp_style"


@dataclass(frozen=True)
class ModelRun:
    run_name: str
    script: str
    checkpoint: str
    tokenizer: str
    text_key: str
    output_json: str


MODEL_RUNS = [
    ModelRun(
        "chinese_4epoch",
        "chinese_origin",
        "server_outputs/4epoch/outputs/chinese_125m_b1024_4epoch_seed42/checkpoint-27176",
        "tokenizers/chinese_origin_32k_eos",
        "zh",
        "chinese_4epoch.json",
    ),
    ModelRun(
        "diacritic_matched_token_4epoch",
        "pinyin_diacritic",
        "server_outputs/4epoch/diacritic_125m_b1024_matched_token_4epoch_seed42_outputs/outputs/"
        "diacritic_125m_b1024_matched_token_4epoch_seed42/checkpoint-27176",
        "tokenizers/pinyin_diacritic_32k_eos",
        "diacritic",
        "diacritic_matched_token_4epoch.json",
    ),
]


def load_model_runs_json(path_text: str | None, root: Path) -> list[ModelRun]:
    if path_text is None:
        return list(MODEL_RUNS)
    path = project_path(root, path_text)
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("model_runs", payload) if isinstance(payload, dict) else payload
    runs: list[ModelRun] = []
    for record in records:
        run_name = record.get("run_name") or record.get("model")
        script = record.get("script")
        text_key = record.get("eval4_text_key") or record.get("text_key")
        if text_key in {"zh_text", "zh"}:
            text_key = "zh"
        elif text_key in {"diacritic_text", "diacritic"}:
            text_key = "diacritic"
        elif script == "chinese_origin":
            text_key = "zh"
        else:
            text_key = "diacritic"
        if script is None:
            script = "chinese_origin" if text_key == "zh" else "pinyin_diacritic"
        runs.append(
            ModelRun(
                run_name=run_name,
                script=script,
                checkpoint=record["checkpoint"],
                tokenizer=record["tokenizer"],
                text_key=text_key,
                output_json=record.get("output_json") or f"{run_name}.json",
            )
        )
    if not runs:
        raise ValueError(f"No Eval 4 model runs found in {path}")
    return runs

ITEM_SCORE_FIELDS = [
    "id",
    "phenomenon",
    "subtype_if_any",
    "model_run",
    "script",
    "good_sentence",
    "bad_sentence",
    "correct",
    "tie",
    "margin",
    "good_mean_logprob",
    "bad_mean_logprob",
    "good_total_logprob",
    "bad_total_logprob",
    "good_token_count",
    "bad_token_count",
    "good_num_chunks",
    "bad_num_chunks",
    "non_finite",
    "data_source",
    "generation_method",
    "quality_flags",
]

SUMMARY_OVERALL_FIELDS = [
    "dataset_name",
    "model_run",
    "script",
    "n_items",
    "accuracy",
    "baseline",
    "mean_margin",
    "median_margin",
    "accuracy_ci_low",
    "accuracy_ci_high",
    "mean_margin_ci_low",
    "mean_margin_ci_high",
    "ties",
    "non_finite",
    "device",
    "dtype",
    "checkpoint",
    "tokenizer",
    "notes",
]

SUMMARY_BY_PHENOMENON_FIELDS = [
    "dataset_name",
    "model_run",
    "script",
    "phenomenon",
    "n_items",
    "accuracy",
    "mean_margin",
    "median_margin",
    "ties",
    "tie_rate",
    "collapsed_count",
    "collapsed_rate",
    "noncollapsed_n_items",
    "noncollapsed_accuracy",
    "non_finite",
]

COMPARISON_FIELDS = [
    "dataset_name",
    "n_items",
    "chinese_accuracy",
    "diacritic_accuracy",
    "accuracy_gap_chinese_minus_diacritic",
    "chinese_mean_margin",
    "diacritic_mean_margin",
    "mean_margin_gap_chinese_minus_diacritic",
    "largest_gap_phenomena",
    "both_above_random_baseline",
    "interpretation",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Eval 4 Chinese BLiMP-style minimal pairs.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--model-runs-json",
        default=None,
        help="Optional JSON manifest with model runs to evaluate. Defaults to the built-in seed42 pair.",
    )
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--tie-epsilon",
        type=float,
        default=1e-5,
        help=(
            "Absolute mean-logprob margin at or below which a pair is treated as a tie. "
            "Identical model-input good/bad strings are always ties."
        ),
    )
    parser.add_argument("--print-random-examples", type=int, default=20)
    parser.add_argument("--print-contrast-examples", type=int, default=10)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def project_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def require_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def stratified_cap(items: list[dict[str, Any]], max_items: int, seed: int) -> list[dict[str, Any]]:
    if len(items) <= max_items:
        return items
    rng = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        groups[item["phenomenon"]].append(item)
    for rows in groups.values():
        rng.shuffle(rows)
    phenomena = sorted(groups)
    base = max_items // len(phenomena)
    remainder = max_items % len(phenomena)
    sampled = []
    leftovers = []
    for idx, phenomenon in enumerate(phenomena):
        quota = base + (1 if idx < remainder else 0)
        rows = groups[phenomenon]
        sampled.extend(rows[:quota])
        leftovers.extend(rows[quota:])
    if len(sampled) < max_items:
        rng.shuffle(leftovers)
        sampled.extend(leftovers[: max_items - len(sampled)])
    rng.shuffle(sampled)
    return sampled[:max_items]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


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


def validate_checkpoint_and_tokenizer(root: Path, run: ModelRun):
    checkpoint = project_path(root, run.checkpoint)
    tokenizer_path = project_path(root, run.tokenizer)
    require_path(checkpoint, "checkpoint")
    require_path(tokenizer_path, "tokenizer")
    config = AutoConfig.from_pretrained(str(checkpoint), local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), use_fast=True, local_files_only=True)
    if int(config.vocab_size) != EXPECTED_VOCAB_SIZE or len(tokenizer) != EXPECTED_VOCAB_SIZE:
        raise ValueError(f"{run.run_name}: expected vocab_size=32001; config={config.vocab_size}, tokenizer={len(tokenizer)}")
    if tokenizer.eos_token_id != EXPECTED_EOS_ID:
        raise ValueError(f"{run.run_name}: expected eos_token_id=32000; got {tokenizer.eos_token_id}")
    if tokenizer.pad_token_id != EXPECTED_PAD_ID:
        raise ValueError(f"{run.run_name}: expected pad_token_id=32000; got {tokenizer.pad_token_id}")
    if int(config.max_position_embeddings) != MAX_SEQ_LEN:
        raise ValueError(f"{run.run_name}: expected max_position_embeddings=1024; got {config.max_position_embeddings}")
    return checkpoint, tokenizer


def load_model(checkpoint: Path, device: torch.device, dtype: torch.dtype):
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


def count_params(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def encode_for_scoring(tokenizer, text: str) -> list[int]:
    return tokenizer(str(text), add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]


def score_long_ids_chunked(
    model,
    target_ids: list[int],
    eos_token_id: int,
    device: torch.device,
    target_chunk_size: int = MAX_SEQ_LEN - 1,
) -> dict[str, Any]:
    full_ids = [eos_token_id, *target_ids]
    total_logprob = 0.0
    token_count = 0
    chunks = 0
    with torch.inference_mode():
        for start in range(0, len(target_ids), target_chunk_size):
            end = min(len(target_ids), start + target_chunk_size)
            input_end = end + 1
            input_start = max(0, input_end - MAX_SEQ_LEN)
            input_chunk = full_ids[input_start:input_end]
            input_ids = torch.tensor([input_chunk], dtype=torch.long, device=device)
            logits = model(input_ids=input_ids).logits[0]
            log_probs = torch.log_softmax(logits, dim=-1)
            score_start = start + 1
            score_end = end + 1
            for full_pos in range(score_start, score_end):
                local_pos = full_pos - input_start
                token_id = full_ids[full_pos]
                value = float(log_probs[local_pos - 1, token_id].detach().cpu().item())
                total_logprob += value
                token_count += 1
            chunks += 1
    mean_logprob = total_logprob / token_count if token_count else float("nan")
    return {
        "mean_logprob": mean_logprob,
        "total_logprob": total_logprob,
        "token_count": token_count,
        "num_chunks": chunks,
    }


def score_texts_batched(
    model,
    tokenizer,
    texts: list[str],
    device: torch.device,
    batch_size: int,
    show_progress: bool,
    desc: str,
) -> list[dict[str, Any]]:
    encoded = [encode_for_scoring(tokenizer, text) for text in texts]
    results: list[dict[str, Any] | None] = [None] * len(texts)
    short_indices = [idx for idx, ids in enumerate(encoded) if len(ids) + 1 <= MAX_SEQ_LEN]
    long_indices = [idx for idx, ids in enumerate(encoded) if len(ids) + 1 > MAX_SEQ_LEN]

    iterator = tqdm(
        range(0, len(short_indices), batch_size),
        total=math.ceil(len(short_indices) / batch_size) if short_indices else 0,
        desc=desc,
        disable=not show_progress,
        unit="batch",
    )
    with torch.inference_mode():
        for offset in iterator:
            batch_indices = short_indices[offset : offset + batch_size]
            batch_targets = [encoded[idx] for idx in batch_indices]
            batch_inputs = [[tokenizer.eos_token_id, *ids] for ids in batch_targets]
            max_len = max(len(ids) for ids in batch_inputs)
            input_rows = []
            attention_rows = []
            for ids in batch_inputs:
                pad = max_len - len(ids)
                input_rows.append(ids + [tokenizer.pad_token_id] * pad)
                attention_rows.append([1] * len(ids) + [0] * pad)
            input_tensor = torch.tensor(input_rows, dtype=torch.long, device=device)
            attention_tensor = torch.tensor(attention_rows, dtype=torch.long, device=device)
            logits = model(input_ids=input_tensor, attention_mask=attention_tensor).logits
            log_probs = torch.log_softmax(logits, dim=-1)
            for row_idx, item_idx in enumerate(batch_indices):
                target_ids = batch_targets[row_idx]
                values = []
                for target_pos, token_id in enumerate(target_ids, start=1):
                    values.append(float(log_probs[row_idx, target_pos - 1, token_id].detach().cpu().item()))
                total = sum(values)
                results[item_idx] = {
                    "mean_logprob": total / len(values),
                    "total_logprob": total,
                    "token_count": len(values),
                    "num_chunks": 1,
                }

    for item_idx in tqdm(long_indices, desc=f"{desc}_long", disable=not show_progress, unit="sent"):
        results[item_idx] = score_long_ids_chunked(
            model=model,
            target_ids=encoded[item_idx],
            eos_token_id=tokenizer.eos_token_id,
            device=device,
        )

    final = [row for row in results if row is not None]
    if len(final) != len(texts):
        raise RuntimeError(f"Internal scoring error: expected {len(texts)} scores, got {len(final)}")
    return final


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    index = (len(values) - 1) * pct
    low = int(math.floor(index))
    high = int(math.ceil(index))
    if low == high:
        return values[low]
    return values[low] * (high - index) + values[high] * (index - low)


def bootstrap_ci(values: list[float], statistic, samples: int, seed: int) -> tuple[float | str, float | str]:
    if not values:
        return "", ""
    rng = random.Random(seed)
    stats = []
    for _ in range(samples):
        sample = [values[rng.randrange(len(values))] for _ in values]
        stats.append(float(statistic(sample)))
    return percentile(stats, 0.025), percentile(stats, 0.975)


def length_stats(values: list[int]) -> dict[str, float | int | str]:
    if not values:
        return {"min": "", "mean": "", "median": "", "max": ""}
    return {
        "min": min(values),
        "mean": sum(values) / len(values),
        "median": median(values),
        "max": max(values),
    }


def evaluate_run(
    root: Path,
    run: ModelRun,
    items: list[dict[str, Any]],
    output_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    dtype_name: str,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    checkpoint, tokenizer = validate_checkpoint_and_tokenizer(root, run)
    print(f"\n== {run.run_name} ==")
    print(f"checkpoint exists: {checkpoint}")
    print(
        "tokenizer/config validated: "
        f"vocab_size={len(tokenizer)}, eos_token_id={tokenizer.eos_token_id}, pad_token_id={tokenizer.pad_token_id}"
    )
    model = load_model(checkpoint, device, dtype)
    param_count = count_params(model)
    print(f"model parameter count: {param_count}")
    if param_count != EXPECTED_PARAM_COUNT:
        raise ValueError(f"{run.run_name}: expected parameter count={EXPECTED_PARAM_COUNT}; got {param_count}")

    if run.text_key == "zh":
        good_texts = [item["good_sentence_zh"] for item in items]
        bad_texts = [item["bad_sentence_zh"] for item in items]
    else:
        good_texts = [item["good_sentence_diacritic"] for item in items]
        bad_texts = [item["bad_sentence_diacritic"] for item in items]

    all_scores = score_texts_batched(
        model=model,
        tokenizer=tokenizer,
        texts=[*good_texts, *bad_texts],
        device=device,
        batch_size=args.batch_size,
        show_progress=not args.no_progress,
        desc=run.run_name,
    )
    good_scores = all_scores[: len(items)]
    bad_scores = all_scores[len(items) :]

    rows = []
    for item, good, bad, good_text, bad_text in zip(items, good_scores, bad_scores, good_texts, bad_texts):
        margin = float(good["mean_logprob"]) - float(bad["mean_logprob"])
        non_finite = int(
            not all(
                math.isfinite(float(value))
                for value in [
                    good["mean_logprob"],
                    bad["mean_logprob"],
                    good["total_logprob"],
                    bad["total_logprob"],
                    margin,
                ]
            )
        )
        identical_model_text = good_text == bad_text
        tie = int((identical_model_text or abs(margin) <= args.tie_epsilon) and not non_finite)
        rows.append(
            {
                "id": item["id"],
                "phenomenon": item["phenomenon"],
                "subtype_if_any": item.get("subtype_if_any", ""),
                "model_run": run.run_name,
                "script": run.script,
                "good_sentence": good_text,
                "bad_sentence": bad_text,
                "correct": int(margin > args.tie_epsilon and not identical_model_text and not non_finite),
                "tie": tie,
                "margin": margin,
                "good_mean_logprob": good["mean_logprob"],
                "bad_mean_logprob": bad["mean_logprob"],
                "good_total_logprob": good["total_logprob"],
                "bad_total_logprob": bad["total_logprob"],
                "good_token_count": good["token_count"],
                "bad_token_count": bad["token_count"],
                "good_num_chunks": good["num_chunks"],
                "bad_num_chunks": bad["num_chunks"],
                "non_finite": non_finite,
                "data_source": item.get("data_source", ""),
                "generation_method": item.get("generation_method", ""),
                "quality_flags": json.dumps(item.get("quality_flags", []), ensure_ascii=False),
            }
        )

    correct_values = [float(row["correct"]) for row in rows if not row["non_finite"]]
    margins = [float(row["margin"]) for row in rows if not row["non_finite"]]
    acc_low, acc_high = bootstrap_ci(correct_values, lambda sample: sum(sample) / len(sample), args.bootstrap_samples, args.seed + 11)
    margin_low, margin_high = bootstrap_ci(margins, lambda sample: sum(sample) / len(sample), args.bootstrap_samples, args.seed + 29)
    summary = {
        "dataset_name": "eval4_chinese_blimp_style",
        "model_run": run.run_name,
        "script": run.script,
        "n_items": len(rows),
        "accuracy": sum(correct_values) / len(correct_values) if correct_values else "",
        "baseline": 0.5,
        "mean_margin": sum(margins) / len(margins) if margins else "",
        "median_margin": median(margins) if margins else "",
        "accuracy_ci_low": acc_low,
        "accuracy_ci_high": acc_high,
        "mean_margin_ci_low": margin_low,
        "mean_margin_ci_high": margin_high,
        "ties": sum(int(row["tie"]) for row in rows),
        "non_finite": sum(int(row["non_finite"]) for row in rows),
        "device": device.type,
        "dtype": dtype_name,
        "checkpoint": str(checkpoint),
        "tokenizer": run.tokenizer,
        "notes": "Mean token logprob over sentence tokens with add_special_tokens=False and manually appended EOS. An EOS context prefix is used only to score the initial sentence token; every target token including appended EOS is scored exactly once.",
        "diagnostics": {
            "good_token_lengths": length_stats([int(row["good_token_count"]) for row in rows]),
            "bad_token_lengths": length_stats([int(row["bad_token_count"]) for row in rows]),
            "items_over_1024_good": sum(int(row["good_num_chunks"]) > 1 for row in rows),
            "items_over_1024_bad": sum(int(row["bad_num_chunks"]) > 1 for row in rows),
        },
    }
    (output_dir / run.output_json).write_text(
        json.dumps({"run": run.__dict__, "summary": summary}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    return rows, summary


def summarize_by_phenomenon(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model_run"], row["script"], row["phenomenon"])].append(row)
    summaries = []
    for (model_run, script, phenomenon), group_rows in sorted(grouped.items()):
        finite = [row for row in group_rows if not int(row["non_finite"])]
        correct = [float(row["correct"]) for row in finite]
        margins = [float(row["margin"]) for row in finite]
        collapsed_rows = [
            row
            for row in group_rows
            if "identical_diacritic" in json.loads(row.get("quality_flags") or "[]")
        ]
        noncollapsed_finite = [
            row
            for row in finite
            if "identical_diacritic" not in json.loads(row.get("quality_flags") or "[]")
        ]
        noncollapsed_correct = [float(row["correct"]) for row in noncollapsed_finite]
        ties = sum(int(row["tie"]) for row in group_rows)
        summaries.append(
            {
                "dataset_name": "eval4_chinese_blimp_style",
                "model_run": model_run,
                "script": script,
                "phenomenon": phenomenon,
                "n_items": len(group_rows),
                "accuracy": sum(correct) / len(correct) if correct else "",
                "mean_margin": sum(margins) / len(margins) if margins else "",
                "median_margin": median(margins) if margins else "",
                "ties": ties,
                "tie_rate": ties / len(group_rows) if group_rows else "",
                "collapsed_count": len(collapsed_rows),
                "collapsed_rate": len(collapsed_rows) / len(group_rows) if group_rows else "",
                "noncollapsed_n_items": len(noncollapsed_finite),
                "noncollapsed_accuracy": sum(noncollapsed_correct) / len(noncollapsed_correct)
                if noncollapsed_correct
                else "",
                "non_finite": sum(int(row["non_finite"]) for row in group_rows),
            }
        )
    return summaries


def diacritic_near_identical_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    exact = 0
    normalized = 0
    bag_near = 0
    for item in items:
        good = item["good_sentence_diacritic"]
        bad = item["bad_sentence_diacritic"]
        if good == bad:
            exact += 1
        if re.sub(r"\s+", "", good) == re.sub(r"\s+", "", bad):
            normalized += 1
        good_tokens = good.split()
        bad_tokens = bad.split()
        if good_tokens and bad_tokens:
            union = set(good_tokens) | set(bad_tokens)
            overlap = len(set(good_tokens) & set(bad_tokens)) / len(union)
            if overlap >= 0.95:
                bag_near += 1
    return {
        "exact_identical": exact,
        "whitespace_normalized_identical": normalized,
        "token_bag_overlap_ge_0.95": bag_near,
    }


def model_comparison(
    overall: list[dict[str, Any]],
    by_phenomenon: list[dict[str, Any]],
    chinese_model: str = "chinese_4epoch",
    diacritic_model: str = "diacritic_matched_token_4epoch",
) -> list[dict[str, Any]]:
    lookup = {row["model_run"]: row for row in overall}
    chinese = lookup.get(chinese_model)
    diacritic = lookup.get(diacritic_model)
    if not chinese or not diacritic:
        return []
    ph_lookup = {(row["model_run"], row["phenomenon"]): row for row in by_phenomenon}
    gaps = []
    for phenomenon in sorted({row["phenomenon"] for row in by_phenomenon}):
        ch = ph_lookup.get((chinese_model, phenomenon))
        di = ph_lookup.get((diacritic_model, phenomenon))
        if ch and di and ch["accuracy"] != "" and di["accuracy"] != "":
            gaps.append((phenomenon, float(ch["accuracy"]) - float(di["accuracy"])))
    gaps.sort(key=lambda pair: abs(pair[1]), reverse=True)
    ch_acc = float(chinese["accuracy"])
    di_acc = float(diacritic["accuracy"])
    both_above = ch_acc > 0.5 and di_acc > 0.5
    gap = ch_acc - di_acc
    interpretation = (
        "Chinese model is higher on this general linguistic minimal-pair evaluation."
        if gap > 0
        else "Diacritic model is higher on this general linguistic minimal-pair evaluation."
        if gap < 0
        else "The two models tie on this general linguistic minimal-pair evaluation."
    )
    return [
        {
            "dataset_name": "eval4_chinese_blimp_style",
            "n_items": chinese["n_items"],
            "chinese_accuracy": chinese["accuracy"],
            "diacritic_accuracy": diacritic["accuracy"],
            "accuracy_gap_chinese_minus_diacritic": gap,
            "chinese_mean_margin": chinese["mean_margin"],
            "diacritic_mean_margin": diacritic["mean_margin"],
            "mean_margin_gap_chinese_minus_diacritic": float(chinese["mean_margin"]) - float(diacritic["mean_margin"]),
            "largest_gap_phenomena": json.dumps(gaps[:8], ensure_ascii=False),
            "both_above_random_baseline": both_above,
            "interpretation": interpretation,
        }
    ]


def print_examples(items: list[dict[str, Any]], rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    if args.print_random_examples <= 0 and args.print_contrast_examples <= 0:
        return
    rng = random.Random(args.seed)
    by_key = {(row["id"], row["model_run"]): row for row in rows}
    default_models_present = all(
        (items[0]["id"], model) in by_key
        for model in ("chinese_4epoch", "diacritic_matched_token_4epoch")
    ) if items else False
    if not default_models_present:
        print("\nrandom/contrast examples skipped: default seed42 comparison pair is absent")
        return
    print(f"\nrandom scored examples ({min(args.print_random_examples, len(items))})")
    for item in rng.sample(items, min(args.print_random_examples, len(items))):
        ch = by_key[(item["id"], "chinese_4epoch")]
        di = by_key[(item["id"], "diacritic_matched_token_4epoch")]
        print(
            f"[{item['id']}] {item['phenomenon']} | zh_correct={ch['correct']} margin={float(ch['margin']):.4f} "
            f"| dia_correct={di['correct']} margin={float(di['margin']):.4f}"
        )
        print(f"  good: {item['good_sentence_zh']}")
        print(f"  bad:  {item['bad_sentence_zh']}")

    ch_correct_di_wrong = []
    di_correct_ch_wrong = []
    for item in items:
        ch = by_key[(item["id"], "chinese_4epoch")]
        di = by_key[(item["id"], "diacritic_matched_token_4epoch")]
        if int(ch["correct"]) == 1 and int(di["correct"]) == 0:
            ch_correct_di_wrong.append((item, ch, di))
        if int(di["correct"]) == 1 and int(ch["correct"]) == 0:
            di_correct_ch_wrong.append((item, ch, di))

    print(f"\nChinese correct and Diacritic wrong ({min(args.print_contrast_examples, len(ch_correct_di_wrong))})")
    for item, ch, di in ch_correct_di_wrong[: args.print_contrast_examples]:
        print(f"[{item['id']}] {item['phenomenon']} zh_margin={float(ch['margin']):.4f} dia_margin={float(di['margin']):.4f}")
        print(f"  good: {item['good_sentence_zh']}")
        print(f"  bad:  {item['bad_sentence_zh']}")

    print(f"\nDiacritic correct and Chinese wrong ({min(args.print_contrast_examples, len(di_correct_ch_wrong))})")
    for item, ch, di in di_correct_ch_wrong[: args.print_contrast_examples]:
        print(f"[{item['id']}] {item['phenomenon']} zh_margin={float(ch['margin']):.4f} dia_margin={float(di['margin']):.4f}")
        print(f"  good: {item['good_sentence_zh']}")
        print(f"  bad:  {item['bad_sentence_zh']}")


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    root = Path.cwd()
    dataset_path = project_path(root, args.dataset)
    output_dir = project_path(root, args.output_dir)
    model_runs = load_model_runs_json(args.model_runs_json, root)
    output_dir.mkdir(parents=True, exist_ok=True)
    require_path(dataset_path, "Eval 4 dataset")
    items = read_jsonl(dataset_path)
    if args.max_items is not None:
        if args.max_items <= 0:
            raise ValueError("--max-items must be positive when set")
        items = stratified_cap(items, args.max_items, args.seed)
    if not items:
        raise ValueError(f"No Eval 4 items loaded from {dataset_path}")

    phenomena = Counter(item["phenomenon"] for item in items)
    print("Eval 4: Chinese BLiMP-style general linguistic minimal-pair evaluation")
    print(f"dataset: {dataset_path}")
    print(f"items: {len(items)}")
    print(f"phenomena: {json.dumps(dict(phenomena), ensure_ascii=False, sort_keys=True)}")
    if args.max_items is not None:
        print(f"SMOKE/PARTIAL EVAL: max_items={args.max_items}")

    diacritic_diag = diacritic_near_identical_counts(items)
    print(f"diacritic near-identical diagnostics: {json.dumps(diacritic_diag, ensure_ascii=False, sort_keys=True)}")

    print(f"model runs: {', '.join(run.run_name for run in model_runs)}")
    for run in model_runs:
        checkpoint, tokenizer = validate_checkpoint_and_tokenizer(root, run)
        print(
            f"preflight {run.run_name}: checkpoint={checkpoint}, "
            f"vocab_size={len(tokenizer)}, eos_token_id={tokenizer.eos_token_id}, pad_token_id={tokenizer.pad_token_id}"
        )

    device, dtype, dtype_name = choose_device_and_dtype()
    print(f"device: {device.type}, dtype: {dtype_name}, batch_size={args.batch_size}")
    all_rows: list[dict[str, Any]] = []
    overall: list[dict[str, Any]] = []
    for run in model_runs:
        rows, summary = evaluate_run(root, run, items, output_dir, device, dtype, dtype_name, args)
        all_rows.extend(rows)
        overall.append({key: summary.get(key, "") for key in SUMMARY_OVERALL_FIELDS})

    by_phenomenon = summarize_by_phenomenon(all_rows)
    comparison = model_comparison(overall, by_phenomenon)

    write_csv(output_dir / "item_scores.csv", all_rows, ITEM_SCORE_FIELDS)
    write_csv(output_dir / "summary_overall.csv", overall, SUMMARY_OVERALL_FIELDS)
    write_csv(output_dir / "summary_by_phenomenon.csv", by_phenomenon, SUMMARY_BY_PHENOMENON_FIELDS)
    write_csv(output_dir / "model_comparison.csv", comparison, COMPARISON_FIELDS)
    diagnostics = {
        "dataset": str(dataset_path),
        "n_items": len(items),
        "phenomena": dict(phenomena),
        "diacritic_near_identical": diacritic_diag,
        "model_token_length_stats": {
            row["model_run"]: {
                "good": length_stats([int(score["good_token_count"]) for score in all_rows if score["model_run"] == row["model_run"]]),
                "bad": length_stats([int(score["bad_token_count"]) for score in all_rows if score["model_run"] == row["model_run"]]),
            }
            for row in overall
        },
    }
    (output_dir / "diagnostics.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote: {output_dir / 'item_scores.csv'}")
    print(f"wrote: {output_dir / 'summary_overall.csv'}")
    print(f"wrote: {output_dir / 'summary_by_phenomenon.csv'}")
    print(f"wrote: {output_dir / 'model_comparison.csv'}")
    print(f"wrote: {output_dir / 'diagnostics.json'}")

    print_examples(items, all_rows, args)

    if comparison:
        row = comparison[0]
        print("\nEval 4 summary")
        print(f"dataset size: {row['n_items']}")
        print(f"phenomena included: {', '.join(sorted(phenomena))}")
        print(f"Chinese overall accuracy: {float(row['chinese_accuracy']):.6f}")
        print(f"Diacritic overall accuracy: {float(row['diacritic_accuracy']):.6f}")
        print(f"gap (Chinese - Diacritic): {float(row['accuracy_gap_chinese_minus_diacritic']):.6f}")
        print(f"largest phenomenon gaps: {row['largest_gap_phenomena']}")
        print(f"both above 50% random baseline: {row['both_above_random_baseline']}")
        print(f"general linguistic ability interpretation: {row['interpretation']}")


if __name__ == "__main__":
    main()
