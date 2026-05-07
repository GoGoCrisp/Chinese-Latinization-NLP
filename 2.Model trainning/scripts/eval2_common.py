#!/usr/bin/env python3
"""Shared model scoring utilities for eval2 probes."""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


EXPECTED_VOCAB_SIZE = 32001
EXPECTED_EOS_ID = 32000
EXPECTED_PAD_ID = 32000
MAX_SEQ_LEN = 1024
SCORING_MODES = ["candidate_only", "candidate_plus_suffix"]


@dataclass(frozen=True)
class ModelRun:
    run_name: str
    script: str
    checkpoint: str
    tokenizer: str


MODEL_RUNS = [
    ModelRun(
        "chinese_4epoch",
        "chinese_origin",
        "server_outputs/4epoch/outputs/chinese_125m_b1024_4epoch_seed42/checkpoint-27176",
        "tokenizers/chinese_origin_32k_eos",
    ),
    ModelRun(
        "diacritic_matched_token_4epoch",
        "pinyin_diacritic",
        "server_outputs/4epoch/diacritic_125m_b1024_matched_token_4epoch_seed42_outputs/outputs/"
        "diacritic_125m_b1024_matched_token_4epoch_seed42/checkpoint-27176",
        "tokenizers/pinyin_diacritic_32k_eos",
    ),
]


def project_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def choose_device_and_dtype() -> tuple[torch.device, torch.dtype, str]:
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.device("cuda"), torch.bfloat16, "bf16"
        return torch.device("cuda"), torch.float32, "fp32"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), torch.float32, "fp32"
    return torch.device("cpu"), torch.float32, "fp32"


def validate_checkpoint_and_tokenizer(run: ModelRun, root: Path):
    checkpoint = project_path(root, run.checkpoint)
    tokenizer_path = project_path(root, run.tokenizer)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Missing tokenizer: {tokenizer_path}")
    config = AutoConfig.from_pretrained(str(checkpoint), local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    if config.vocab_size != EXPECTED_VOCAB_SIZE or len(tokenizer) != EXPECTED_VOCAB_SIZE:
        raise ValueError(
            f"Vocab mismatch for {run.run_name}: config={config.vocab_size}, tokenizer={len(tokenizer)}"
        )
    if tokenizer.eos_token_id != EXPECTED_EOS_ID or tokenizer.pad_token_id != EXPECTED_PAD_ID:
        raise ValueError(
            f"EOS/PAD mismatch for {run.run_name}: eos={tokenizer.eos_token_id}, pad={tokenizer.pad_token_id}"
        )
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


def completion_parts(item: dict[str, Any], run: ModelRun, mode: str) -> tuple[str, str, str]:
    if run.script == "chinese_origin":
        prefix = item["prefix_zh"]
        gold = item["gold_zh"]
        distractor = item["distractor_zh"]
        suffix = item["suffix_zh"]
    else:
        prefix = item["prefix_diacritic"]
        gold = item["gold_pinyin_diacritic"]
        distractor = item["distractor_pinyin_diacritic"]
        suffix = item["suffix_diacritic"]
    if mode == "candidate_only":
        return prefix, gold, distractor
    if run.script == "chinese_origin":
        return prefix, gold + suffix, distractor + suffix
    return prefix, f"{gold} {suffix}".strip(), f"{distractor} {suffix}".strip()


def score_completion(model, tokenizer, device: torch.device, prefix: str, completion: str) -> tuple[float, int]:
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    if not completion_ids:
        raise ValueError(f"Empty completion: {completion!r}")
    max_prefix_len = MAX_SEQ_LEN - len(completion_ids)
    if max_prefix_len < 1:
        completion_ids = completion_ids[: MAX_SEQ_LEN - 1]
        max_prefix_len = 1
    prefix_ids = prefix_ids[-max_prefix_len:]
    input_ids = prefix_ids + completion_ids
    target_start = len(prefix_ids)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        logits = model(input_ids=input_tensor).logits[0]
        log_probs = torch.log_softmax(logits, dim=-1)
    values: list[float] = []
    for token_index in range(max(1, target_start), len(input_ids)):
        token_id = input_ids[token_index]
        values.append(float(log_probs[token_index - 1, token_id].detach().cpu().item()))
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("Non-finite completion score")
    return sum(values) / len(values), len(values)


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


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
