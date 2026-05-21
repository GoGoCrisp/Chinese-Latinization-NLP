#!/usr/bin/env python3
"""Standalone Eval3b pilot: C3 dialogue MCQ with option-text likelihood scoring."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import traceback
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import torch
from pypinyin import Style, lazy_pinyin
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


EXPECTED_VOCAB_SIZE = 32001
EXPECTED_EOS_ID = 32000
EXPECTED_PAD_ID = 32000
EXPECTED_PARAM_COUNT = 134_107_392
MAX_SEQ_LEN = 1024
RANDOM_BASELINE = 0.25
OUTPUT_DIR = "eval_results/eval3b_c3_dialogue_option_text"
LABELS = ["A", "B", "C", "D"]


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


ITEM_SCORE_FIELDS = [
    "item_id",
    "split",
    "context",
    "question",
    "options_json",
    "gold_index",
    "gold_text",
    "model",
    "pred_index",
    "pred_text",
    "correct",
    "gold_score",
    "pred_score",
    "option_scores_json",
    "margin_gold_minus_best_wrong",
    "prompt_len_tokens",
    "option_lens_tokens_json",
]

SUMMARY_FIELDS = [
    "model",
    "n_items",
    "accuracy",
    "random_baseline",
    "gap_vs_baseline",
    "mean_margin_gold_minus_best_wrong",
    "median_margin_gold_minus_best_wrong",
    "n_skipped",
    "bootstrap_acc_ci_low",
    "bootstrap_acc_ci_high",
]

REVIEW_FIELDS = [
    "item_id",
    "split",
    "context",
    "question",
    "options_json",
    "gold_index",
    "gold_text",
    "chinese_pred_index",
    "chinese_pred_text",
    "chinese_correct",
    "chinese_margin",
    "diacritic_pred_index",
    "diacritic_pred_text",
    "diacritic_correct",
    "diacritic_margin",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--max-items", type=int, default=1000)
    parser.add_argument("--smoke-items", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--print-examples", type=int, default=5)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    return parser.parse_args()


def project_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def to_diacritic(text: str) -> str:
    parts = lazy_pinyin(
        str(text),
        style=Style.TONE,
        neutral_tone_with_five=False,
        errors=lambda chunk: list(chunk),
    )
    return " ".join(part.strip() for part in parts if part.strip()).strip()


def make_prompt(context: str, question: str) -> str:
    return f"阅读材料：\n{context}\n\n问题：\n{question}\n\n答案："


def cjk_len(text: str) -> int:
    count = sum(1 for char in str(text) if "\u4e00" <= char <= "\u9fff")
    return count if count else len(str(text))


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        parts = [normalize_text(item) for item in value]
        return "\n".join(part for part in parts if part).strip()
    if isinstance(value, dict):
        for key in ("text", "utterance", "content", "sentence"):
            if key in value:
                return normalize_text(value[key])
        return "\n".join(normalize_text(value[key]) for key in sorted(value)).strip()
    return str(value).strip()


def normalize_options(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        if all(label in value for label in LABELS):
            return [normalize_text(value[label]) for label in LABELS]
        if all(label.lower() in value for label in LABELS):
            return [normalize_text(value[label.lower()]) for label in LABELS]
        return [normalize_text(value[key]) for key in sorted(value)]
    if isinstance(value, (list, tuple)):
        return [normalize_text(item) for item in value]
    return []


def answer_to_index(answer: Any, options: list[str]) -> int | None:
    if answer is None:
        return None
    if isinstance(answer, int) and 0 <= answer < len(options):
        return answer
    text = normalize_text(answer)
    upper = text.upper()
    if upper in LABELS:
        index = LABELS.index(upper)
        return index if index < len(options) else None
    if text.isdigit():
        value = int(text)
        if 0 <= value < len(options):
            return value
        if 1 <= value <= len(options):
            return value - 1
    for index, option in enumerate(options):
        if text == option:
            return index
    return None


def first_present(raw: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if name in raw and raw[name] not in (None, ""):
            return raw[name]
    return None


def features_to_string(features: Any) -> str:
    try:
        return repr(features)
    except Exception:
        return str(features)


def dataset_schema_report(dataset: Any) -> tuple[str, int]:
    lines: list[str] = []
    raw_count = 0
    if hasattr(dataset, "items"):
        for split, data in dataset.items():
            count = len(data)
            raw_count += count
            lines.append(f"split={split} count={count} features={features_to_string(getattr(data, 'features', ''))}")
    else:
        raw_count = len(dataset)
        lines.append(f"split=<dataset> count={raw_count} features={features_to_string(getattr(dataset, 'features', ''))}")
    return "\n".join(lines), raw_count


def load_dataset_exact(path: str, name: str | None) -> Any:
    from datasets import load_dataset

    if name is None:
        return load_dataset(path)
    return load_dataset(path, name)


def try_load_dataset_option(path: str, name: str | None) -> tuple[Any | None, str, list[str]]:
    errors: list[str] = []
    display = f'datasets.load_dataset("{path}")' if name is None else f'datasets.load_dataset("{path}", "{name}")'
    try:
        return load_dataset_exact(path, name), display, errors
    except Exception as exc:
        errors.append(f"{display}\n{traceback.format_exc()}")

    try:
        from datasets import get_dataset_config_names, load_dataset

        configs = get_dataset_config_names(path)
        dialogue_configs = [
            config
            for config in configs
            if config.lower() in {"dialog", "dialogue", "d", "c3-d", "c3_dialog", "c3_dialogue"}
        ]
        for config in dialogue_configs:
            display_config = f'datasets.load_dataset("{path}", "{config}")'
            try:
                return load_dataset(path, config), display_config, errors
            except Exception:
                errors.append(f"{display_config}\n{traceback.format_exc()}")
    except Exception:
        errors.append(f"get_dataset_config_names({path!r})\n{traceback.format_exc()}")

    return None, display, errors


def load_c3_dataset() -> tuple[Any, str, list[str]]:
    candidates = [("c3", None), ("clue", "c3"), ("liweili/c3", None)]
    all_errors: list[str] = []
    for path, name in candidates:
        dataset, display, errors = try_load_dataset_option(path, name)
        all_errors.extend(errors)
        if dataset is not None:
            return dataset, display, all_errors
    print("Failed to load C3 from the requested sources. Exact errors:")
    for error in all_errors:
        print("\n---")
        print(error)
    raise RuntimeError("Could not load C3 dataset from c3, clue/c3, or liweili/c3.")


def split_name_is_dialogue(name: str) -> bool:
    lower = name.lower()
    return lower in {"dialog", "dialogue", "d", "c3-d"} or "dialog" in lower or lower.endswith("_d")


def value_mentions_dialogue(value: Any) -> bool:
    text = normalize_text(value).lower()
    return text in {"dialog", "dialogue", "d", "c3-d"} or "dialog" in text or "对话" in text


def context_looks_like_dialogue(context_value: Any, context_text: str) -> bool:
    if isinstance(context_value, (list, tuple)) and len(context_value) >= 2:
        return True
    markers = ["男：", "女：", "甲：", "乙：", "a：", "b：", "A：", "B：", "speaker", "对话"]
    return any(marker in context_text for marker in markers)


def split_iterables(dataset: Any) -> list[tuple[str, Any]]:
    if hasattr(dataset, "items"):
        return list(dataset.items())
    return [("dataset", dataset)]


def select_dialogue_rows(dataset: Any, dataset_source: str) -> tuple[list[tuple[str, dict[str, Any]]], str]:
    splits = split_iterables(dataset)
    if "dialog" in dataset_source.lower():
        rows = [(name, dict(raw)) for name, data in splits for raw in data]
        split_names = ",".join(name for name, _ in splits)
        return rows, f"loaded_dialogue_config_all_splits({split_names})"

    dialogue_splits = [(name, data) for name, data in splits if split_name_is_dialogue(name)]
    if dialogue_splits:
        rows = [(name, dict(raw)) for name, data in dialogue_splits for raw in data]
        return rows, ",".join(name for name, _ in dialogue_splits)

    metadata_rows: list[tuple[str, dict[str, Any]]] = []
    for split, data in splits:
        for raw in data:
            row = dict(raw)
            for field in ("type", "source", "domain", "subset", "category", "task", "data_type"):
                if field in row and value_mentions_dialogue(row[field]):
                    metadata_rows.append((split, row))
                    break
    if metadata_rows:
        return metadata_rows, "metadata_dialogue_filter"

    inferred_rows: list[tuple[str, dict[str, Any]]] = []
    for split, data in splits:
        for raw in data:
            row = dict(raw)
            context_value = first_present(row, ("context", "contexts", "article", "passage", "document", "documents", "story"))
            context_text = normalize_text(context_value)
            if context_looks_like_dialogue(context_value, context_text):
                inferred_rows.append((split, row))
    if inferred_rows:
        return inferred_rows, "conservative_context_dialogue_inference"

    rows = [(name, dict(raw)) for name, data in splits for raw in data]
    return rows, "no_dialogue_marker_found_all_loaded_rows"


def flatten_c3_rows(rows: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for raw_index, (split, raw) in enumerate(rows):
        context_value = first_present(raw, ("context", "contexts", "article", "passage", "document", "documents", "story"))
        context = normalize_text(context_value)
        question_groups = first_present(raw, ("questions", "qas", "qa", "problems"))
        if isinstance(question_groups, list) and question_groups and isinstance(question_groups[0], dict):
            for q_index, question_raw in enumerate(question_groups):
                item = normalize_single_question(split, raw_index, q_index, context, question_raw)
                if item:
                    items.append(item)
            continue
        if isinstance(question_groups, dict):
            questions = question_groups.get("question") or question_groups.get("questions") or []
            answers = question_groups.get("answer") or question_groups.get("answers") or []
            choices = question_groups.get("choice") or question_groups.get("choices") or question_groups.get("options") or []
            if isinstance(questions, list) and isinstance(answers, list) and isinstance(choices, list):
                for q_index, (question, answer, choice) in enumerate(zip(questions, answers, choices)):
                    question_raw = {"question": question, "answer": answer, "choice": choice}
                    item = normalize_single_question(split, raw_index, q_index, context, question_raw)
                    if item:
                        items.append(item)
                continue
        item = normalize_single_question(split, raw_index, 0, context, raw)
        if item:
            items.append(item)
    return items


def normalize_single_question(
    split: str,
    raw_index: int,
    question_index: int,
    context: str,
    raw: dict[str, Any],
) -> dict[str, Any] | None:
    question = normalize_text(first_present(raw, ("question", "query", "stem", "problem")))
    options = normalize_options(first_present(raw, ("choice", "choices", "option", "options", "candidates")))
    if not options:
        options = normalize_options({label: raw.get(label) or raw.get(label.lower()) for label in LABELS})
    gold_index = answer_to_index(first_present(raw, ("answer", "label", "gold", "correct", "target")), options)
    if gold_index is None:
        return None
    item_id = normalize_text(first_present(raw, ("id", "idx", "qid", "question_id")))
    if not item_id:
        item_id = f"{split}_{raw_index:06d}_{question_index:02d}"
    return {
        "item_id": f"c3_dialogue_{item_id}",
        "split": split,
        "context": context,
        "question": question,
        "options": options,
        "gold_index": gold_index,
        "gold_text": options[gold_index] if 0 <= gold_index < len(options) else "",
    }


def filter_items(items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    report: dict[str, Any] = {"normalized_question_items": len(items)}
    exact4 = [item for item in items if len(item["options"]) == 4]
    report["after_exactly_4_options"] = len(exact4)
    nonempty = [
        item
        for item in exact4
        if item["context"].strip()
        and item["question"].strip()
        and item["gold_text"].strip()
        and all(option.strip() for option in item["options"])
    ]
    report["after_nonempty_fields"] = len(nonempty)
    short_question = [item for item in nonempty if cjk_len(item["question"]) <= 50]
    report["after_question_len_le_50"] = len(short_question)
    short_options = [item for item in short_question if all(cjk_len(option) <= 40 for option in item["options"])]
    report["after_option_len_le_40"] = len(short_options)

    selected: list[dict[str, Any]] = []
    context_reports = []
    for limit in (250, 400, 600):
        selected = [item for item in short_options if cjk_len(item["context"]) <= limit]
        context_reports.append({"context_len_limit": limit, "count": len(selected)})
        if len(selected) >= 300:
            break
    report["context_length_filter_steps"] = context_reports
    report["final_filtered"] = len(selected)
    return selected, report


def choose_device_and_dtype() -> tuple[torch.device, torch.dtype, str]:
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.device("cuda"), torch.bfloat16, "bf16"
        return torch.device("cuda"), torch.float32, "fp32"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), torch.float32, "fp32"
    return torch.device("cpu"), torch.float32, "fp32"


def validate_checkpoint_and_tokenizer(root: Path, run: ModelRun):
    checkpoint = project_path(root, run.checkpoint)
    tokenizer_path = project_path(root, run.tokenizer)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint for {run.run_name}: {checkpoint}")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Missing tokenizer for {run.run_name}: {tokenizer_path}")
    config = AutoConfig.from_pretrained(str(checkpoint), local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    if config.vocab_size != EXPECTED_VOCAB_SIZE or len(tokenizer) != EXPECTED_VOCAB_SIZE:
        raise ValueError(f"Vocab mismatch for {run.run_name}: config={config.vocab_size}, tokenizer={len(tokenizer)}")
    if tokenizer.eos_token_id != EXPECTED_EOS_ID or tokenizer.pad_token_id != EXPECTED_PAD_ID:
        raise ValueError(f"EOS/PAD mismatch for {run.run_name}: eos={tokenizer.eos_token_id}, pad={tokenizer.pad_token_id}")
    return checkpoint, tokenizer, config


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


def encode_prompt_and_options(item: dict[str, Any], run: ModelRun) -> tuple[str, list[str]]:
    prompt = make_prompt(item["context"], item["question"])
    options = list(item["options"])
    if run.script == "pinyin_diacritic":
        prompt = to_diacritic(prompt)
        options = [to_diacritic(option) for option in options]
    return prompt, options


def score_options(
    model,
    tokenizer,
    device: torch.device,
    prompt: str,
    options: list[str],
    max_seq_len: int,
) -> tuple[list[float], int, list[int]]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    option_ids = [tokenizer(option, add_special_tokens=False)["input_ids"] for option in options]
    if any(len(ids) == 0 for ids in option_ids):
        raise ValueError("zero_option_tokens")
    if any(len(prompt_ids) + len(ids) > max_seq_len for ids in option_ids):
        raise ValueError("prompt_plus_option_exceeds_1024")

    sequences = [prompt_ids + ids for ids in option_ids]
    max_len = max(len(sequence) for sequence in sequences)
    padded = [sequence + [tokenizer.pad_token_id] * (max_len - len(sequence)) for sequence in sequences]
    attention = [[1] * len(sequence) + [0] * (max_len - len(sequence)) for sequence in sequences]
    input_tensor = torch.tensor(padded, dtype=torch.long, device=device)
    attention_tensor = torch.tensor(attention, dtype=torch.long, device=device)

    with torch.inference_mode():
        logits = model(input_ids=input_tensor, attention_mask=attention_tensor).logits
        log_probs = torch.log_softmax(logits, dim=-1)

    scores: list[float] = []
    target_start = len(prompt_ids)
    for row_index, ids in enumerate(option_ids):
        values: list[float] = []
        for offset, token_id in enumerate(ids):
            token_index = target_start + offset
            values.append(float(log_probs[row_index, token_index - 1, token_id].detach().cpu().item()))
        if not values or not all(math.isfinite(value) for value in values):
            raise ValueError("non_finite_score")
        scores.append(sum(values) / len(values))
    return scores, len(prompt_ids), [len(ids) for ids in option_ids]


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


def bootstrap_acc_ci(correct: list[float], samples: int, seed: int) -> tuple[float | str, float | str]:
    if not correct:
        return "", ""
    rng = random.Random(seed)
    stats = []
    for _ in range(samples):
        sample = [correct[rng.randrange(len(correct))] for _ in correct]
        stats.append(sum(sample) / len(sample))
    return percentile(stats, 0.025), percentile(stats, 0.975)


def evaluate_loaded_model(
    run: ModelRun,
    model,
    tokenizer,
    items: list[dict[str, Any]],
    device: torch.device,
    args: argparse.Namespace,
    desc: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    max_seq_len = int(getattr(model.config, "max_position_embeddings", MAX_SEQ_LEN) or MAX_SEQ_LEN)
    if max_seq_len != MAX_SEQ_LEN:
        print(f"WARNING: {run.run_name} max_position_embeddings={max_seq_len}; expected {MAX_SEQ_LEN}")
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    iterator = tqdm(items, desc=desc, disable=args.no_progress)
    for index, item in enumerate(iterator, start=1):
        if args.progress_every and index % args.progress_every == 0:
            print(f"{desc}: scored {index}/{len(items)} candidate items")
        prompt, options_for_model = encode_prompt_and_options(item, run)
        try:
            scores, prompt_len, option_lens = score_options(model, tokenizer, device, prompt, options_for_model, max_seq_len)
        except ValueError as exc:
            skipped.append({"item_id": item["item_id"], "model": run.run_name, "reason": str(exc)})
            continue
        pred_index = max(range(4), key=lambda option_index: scores[option_index])
        gold_index = int(item["gold_index"])
        wrong_scores = [score for option_index, score in enumerate(scores) if option_index != gold_index]
        margin = scores[gold_index] - max(wrong_scores)
        if not math.isfinite(margin):
            raise ValueError(f"Non-finite margin for {run.run_name}/{item['item_id']}")
        rows.append(
            {
                "item_id": item["item_id"],
                "split": item["split"],
                "context": item["context"],
                "question": item["question"],
                "options_json": json.dumps(item["options"], ensure_ascii=False),
                "gold_index": gold_index,
                "gold_text": item["gold_text"],
                "model": run.run_name,
                "pred_index": pred_index,
                "pred_text": item["options"][pred_index],
                "correct": int(pred_index == gold_index),
                "gold_score": scores[gold_index],
                "pred_score": scores[pred_index],
                "option_scores_json": json.dumps(scores, ensure_ascii=False),
                "margin_gold_minus_best_wrong": margin,
                "prompt_len_tokens": prompt_len,
                "option_lens_tokens_json": json.dumps(option_lens, ensure_ascii=False),
            }
        )
    return rows, skipped


def summarize_model(run: ModelRun, rows: list[dict[str, Any]], skipped: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    correct = [float(row["correct"]) for row in rows]
    margins = [float(row["margin_gold_minus_best_wrong"]) for row in rows]
    accuracy = sum(correct) / len(correct) if correct else ""
    low, high = bootstrap_acc_ci(correct, args.bootstrap_samples, args.seed + 17)
    return {
        "model": run.run_name,
        "n_items": len(rows),
        "accuracy": accuracy,
        "random_baseline": RANDOM_BASELINE,
        "gap_vs_baseline": accuracy - RANDOM_BASELINE if correct else "",
        "mean_margin_gold_minus_best_wrong": sum(margins) / len(margins) if margins else "",
        "median_margin_gold_minus_best_wrong": median(margins) if margins else "",
        "n_skipped": len(skipped),
        "bootstrap_acc_ci_low": low,
        "bootstrap_acc_ci_high": high,
    }


def load_score_unload(
    root: Path,
    run: ModelRun,
    items: list[dict[str, Any]],
    device: torch.device,
    dtype: torch.dtype,
    dtype_name: str,
    args: argparse.Namespace,
    desc: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    checkpoint, tokenizer, _config = validate_checkpoint_and_tokenizer(root, run)
    print(f"\n== {run.run_name} ==")
    print(f"checkpoint exists: {checkpoint}")
    print(f"tokenizer exists: {project_path(root, run.tokenizer)}")
    print(f"tokenizer vocab_size/eos/pad: {len(tokenizer)}/{tokenizer.eos_token_id}/{tokenizer.pad_token_id}")
    model = load_model(checkpoint, device, dtype)
    param_count = count_params(model)
    print(f"model parameter_count: {param_count}")
    if param_count != EXPECTED_PARAM_COUNT:
        raise ValueError(f"Unexpected parameter count for {run.run_name}: {param_count}")
    rows, skipped = evaluate_loaded_model(run, model, tokenizer, items, device, args, desc)
    summary = summarize_model(run, rows, skipped, args)
    summary_extra = {
        "run": {
            "run_name": run.run_name,
            "script": run.script,
            "checkpoint": str(checkpoint),
            "tokenizer": run.tokenizer,
            "device": device.type,
            "dtype": dtype_name,
            "parameter_count": param_count,
            "scoring": "option_text_scoring_only_raw_option_text_no_eos",
        },
        "summary": summary,
        "skipped": skipped,
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows, skipped, summary_extra


def print_filter_report(report: dict[str, Any]) -> None:
    print(f"normalized question items: {report['normalized_question_items']}")
    print(f"after exactly 4 options: {report['after_exactly_4_options']}")
    print(f"after non-empty fields: {report['after_nonempty_fields']}")
    print(f"after question length <= 50 Chinese chars: {report['after_question_len_le_50']}")
    print(f"after option length <= 40 Chinese chars: {report['after_option_len_le_40']}")
    for step in report["context_length_filter_steps"]:
        print(f"after context length <= {step['context_len_limit']} Chinese chars: {step['count']}")
    if report["context_length_filter_steps"][0]["count"] < 300:
        relaxed = [step["context_len_limit"] for step in report["context_length_filter_steps"][1:]]
        print(f"context length relaxed because <300 examples remained: {relaxed}")
    print(f"final filtered count: {report['final_filtered']}")


def print_examples(items: list[dict[str, Any]], n: int, seed: int) -> None:
    sample = random.Random(seed).sample(items, min(n, len(items)))
    print(f"\nFiltered examples ({len(sample)}):")
    for item in sample:
        print(f"\n[{item['item_id']}] split={item['split']} answer={item['gold_index']} {item['gold_text']}")
        print(f"context: {item['context']}")
        print(f"question: {item['question']}")
        print(f"options: {json.dumps(item['options'], ensure_ascii=False)}")


def build_review_rows(all_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_item: dict[str, dict[str, dict[str, Any]]] = {}
    for row in all_rows:
        by_item.setdefault(row["item_id"], {})[row["model"]] = row
    buckets = {
        "examples_correct_both.csv": [],
        "examples_wrong_both.csv": [],
        "examples_chinese_correct_diacritic_wrong.csv": [],
        "examples_diacritic_correct_chinese_wrong.csv": [],
    }
    for item_id, model_rows in sorted(by_item.items()):
        ch = model_rows.get("chinese_4epoch")
        di = model_rows.get("diacritic_matched_token_4epoch")
        if not ch or not di:
            continue
        review = {
            "item_id": item_id,
            "split": ch["split"],
            "context": ch["context"],
            "question": ch["question"],
            "options_json": ch["options_json"],
            "gold_index": ch["gold_index"],
            "gold_text": ch["gold_text"],
            "chinese_pred_index": ch["pred_index"],
            "chinese_pred_text": ch["pred_text"],
            "chinese_correct": ch["correct"],
            "chinese_margin": ch["margin_gold_minus_best_wrong"],
            "diacritic_pred_index": di["pred_index"],
            "diacritic_pred_text": di["pred_text"],
            "diacritic_correct": di["correct"],
            "diacritic_margin": di["margin_gold_minus_best_wrong"],
        }
        if int(ch["correct"]) and int(di["correct"]):
            buckets["examples_correct_both.csv"].append(review)
        elif not int(ch["correct"]) and not int(di["correct"]):
            buckets["examples_wrong_both.csv"].append(review)
        elif int(ch["correct"]) and not int(di["correct"]):
            buckets["examples_chinese_correct_diacritic_wrong.csv"].append(review)
        elif not int(ch["correct"]) and int(di["correct"]):
            buckets["examples_diacritic_correct_chinese_wrong.csv"].append(review)
    return buckets


def concise_report(
    dataset_source: str,
    split_used: str,
    raw_count: int,
    filtered_count: int,
    evaluated_count: int,
    summaries: list[dict[str, Any]],
) -> None:
    summary_by_model = {row["model"]: row for row in summaries}
    ch = summary_by_model["chinese_4epoch"]
    di = summary_by_model["diacritic_matched_token_4epoch"]
    ch_acc = float(ch["accuracy"])
    di_acc = float(di["accuracy"])
    ch_margin = float(ch["mean_margin_gold_minus_best_wrong"])
    di_margin = float(di["mean_margin_gold_minus_best_wrong"])
    print("\nFinal report")
    print(f"dataset source: {dataset_source}")
    print(f"split/subset used: {split_used}")
    print(f"raw count: {raw_count}")
    print(f"filtered count: {filtered_count}")
    print(f"evaluated count: {evaluated_count}")
    print(f"Chinese accuracy: {ch_acc:.4f}")
    print(f"Diacritic accuracy: {di_acc:.4f}")
    print("random baseline: 25.00%")
    print(f"gap Chinese - Diacritic: {ch_acc - di_acc:.4f}")
    print(f"gap vs baseline Chinese: {ch_acc - RANDOM_BASELINE:.4f}")
    print(f"gap vs baseline Diacritic: {di_acc - RANDOM_BASELINE:.4f}")
    print(f"mean margin Chinese: {ch_margin:.6f}")
    print(f"mean margin Diacritic: {di_margin:.6f}")
    above = []
    if ch_acc > RANDOM_BASELINE + 0.05:
        above.append("Chinese")
    if di_acc > RANDOM_BASELINE + 0.05:
        above.append("Diacritic")
    if above:
        print(f"above random: {', '.join(above)}")
        print("Interpretation: C3-dialogue is a better simple MCQ diagnostic than C-Eval/CMMLU for this pilot.")
    else:
        print("above random: neither model is meaningfully above 25%")
        print(
            "Interpretation: this suggests even C3-dialogue MCQ is too hard or option-text scoring is not effective "
            "for these pure pretrained 134M models."
        )
    if ch_acc > di_acc:
        print(f"Pilot gap: Chinese > Diacritic by {ch_acc - di_acc:.4f}; treat this as provisional until review.")


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    root = Path.cwd()
    output_dir = project_path(root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Validation: checkpoints and tokenizers")
    tokenizer_by_script = {}
    for run in MODEL_RUNS:
        checkpoint, tokenizer, config = validate_checkpoint_and_tokenizer(root, run)
        tokenizer_by_script[run.script] = tokenizer
        print(f"{run.run_name}: checkpoint exists: {checkpoint}")
        print(
            f"{run.run_name}: tokenizer vocab_size={len(tokenizer)}, eos_token_id={tokenizer.eos_token_id}, "
            f"pad_token_id={tokenizer.pad_token_id}, config_vocab_size={config.vocab_size}"
        )

    dataset, dataset_source, load_errors = load_c3_dataset()
    schema, raw_count = dataset_schema_report(dataset)
    print("\nDataset schema")
    print(schema)
    print(f"raw examples: {raw_count}")
    if load_errors:
        print(f"non-fatal load attempts before selected source: {len(load_errors)}")

    dialogue_rows, split_used = select_dialogue_rows(dataset, dataset_source)
    print(f"dialogue subset selection: {split_used}")
    print(f"raw rows in selected subset: {len(dialogue_rows)}")
    normalized = flatten_c3_rows(dialogue_rows)
    filtered, filter_report = filter_items(normalized)
    print_filter_report(filter_report)
    if not filtered:
        raise RuntimeError("No valid C3 dialogue examples left after filtering.")

    for index, item in enumerate(filtered):
        item["item_id"] = f"c3_dialogue_seed{args.seed}_{index:05d}"
    write_jsonl(output_dir / "c3_dialogue_filtered_examples.jsonl", filtered)
    print(f"wrote filtered examples: {output_dir / 'c3_dialogue_filtered_examples.jsonl'}")
    print_examples(filtered, args.print_examples, args.seed)

    rng = random.Random(args.seed)
    eval_items = list(filtered)
    rng.shuffle(eval_items)
    eval_items = eval_items[: args.max_items]
    eval_items.sort(key=lambda row: row["item_id"])
    smoke_items = eval_items[: min(args.smoke_items, len(eval_items))]

    device, dtype, dtype_name = choose_device_and_dtype()
    print(f"\ndevice: {device.type}, dtype: {dtype_name}")

    if not args.skip_smoke:
        print(f"\nSmoke test on {len(smoke_items)} examples")
        smoke_summaries = []
        for run in MODEL_RUNS:
            rows, skipped, _extra = load_score_unload(
                root,
                run,
                smoke_items,
                device,
                dtype,
                dtype_name,
                args,
                f"smoke_{run.run_name}",
            )
            if skipped:
                print(f"smoke skipped for {run.run_name}: {len(skipped)}")
            if any(not math.isfinite(float(row["gold_score"])) for row in rows):
                raise ValueError(f"Non-finite smoke score for {run.run_name}")
            summary = summarize_model(run, rows, skipped, args)
            smoke_summaries.append(summary)
            print(f"smoke {run.run_name}: n={summary['n_items']} accuracy={summary['accuracy']}")
        if any(summary["n_items"] == 0 for summary in smoke_summaries):
            raise RuntimeError("Smoke test produced zero scored items for at least one model.")
        print("Smoke test complete: all scored gold scores are finite.")

    print(f"\nFull pilot on {len(eval_items)} examples")
    all_rows: list[dict[str, Any]] = []
    all_skipped: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    json_summaries: dict[str, dict[str, Any]] = {}
    for run in MODEL_RUNS:
        rows, skipped, extra = load_score_unload(
            root,
            run,
            eval_items,
            device,
            dtype,
            dtype_name,
            args,
            f"full_{run.run_name}",
        )
        all_rows.extend(rows)
        all_skipped.extend(skipped)
        summaries.append(extra["summary"])
        json_summaries[run.run_name] = extra

    write_csv(output_dir / "item_scores.csv", all_rows, ITEM_SCORE_FIELDS)
    write_csv(output_dir / "summary.csv", summaries, SUMMARY_FIELDS)
    write_jsonl(output_dir / "skipped_items.jsonl", all_skipped)
    for run in MODEL_RUNS:
        payload = {
            **json_summaries[run.run_name],
            "dataset": {
                "source": dataset_source,
                "split_used": split_used,
                "raw_count": raw_count,
                "filtered_count": len(filtered),
                "evaluated_count": len(eval_items),
                "filter_report": filter_report,
            },
        }
        (output_dir / f"{run.run_name}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    review_buckets = build_review_rows(all_rows)
    for filename, rows in review_buckets.items():
        write_csv(output_dir / filename, rows, REVIEW_FIELDS)

    print(f"wrote: {output_dir / 'item_scores.csv'}")
    print(f"wrote: {output_dir / 'summary.csv'}")
    print(f"wrote: {output_dir / 'skipped_items.jsonl'}")
    print(f"wrote: {output_dir / 'chinese_4epoch.json'}")
    print(f"wrote: {output_dir / 'diacritic_matched_token_4epoch.json'}")
    for filename in review_buckets:
        print(f"wrote: {output_dir / filename}")

    concise_report(dataset_source, split_used, raw_count, len(filtered), len(eval_items), summaries)


if __name__ == "__main__":
    main()
