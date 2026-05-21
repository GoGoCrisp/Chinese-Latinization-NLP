#!/usr/bin/env python3
"""Eval5: CHID idiom cloze pilot with option-text candidate scoring."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
import re
import shutil
import subprocess
import traceback
import urllib.request
import zipfile
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
DEFAULT_OUTPUT_DIR = "eval_results/eval5_chid_idiom_cloze"
PLACEHOLDER_RE = re.compile(r"#idiom\d+#")
BLANK_MARKER = "____"
SCORING_MODE = "option_text_scoring"


@dataclass(frozen=True)
class ModelRun:
    run_name: str
    script: str
    checkpoint: str
    tokenizer: str
    output_json: str


MODEL_RUNS = [
    ModelRun(
        "chinese_4epoch",
        "chinese_origin",
        "server_outputs/4epoch/outputs/chinese_125m_b1024_4epoch_seed42/checkpoint-27176",
        "tokenizers/chinese_origin_32k_eos",
        "chinese_4epoch.json",
    ),
    ModelRun(
        "diacritic_matched_token_4epoch",
        "pinyin_diacritic",
        "server_outputs/4epoch/diacritic_125m_b1024_matched_token_4epoch_seed42_outputs/outputs/"
        "diacritic_125m_b1024_matched_token_4epoch_seed42/checkpoint-27176",
        "tokenizers/pinyin_diacritic_32k_eos",
        "diacritic_matched_token_4epoch.json",
    ),
]


ITEM_SCORE_FIELDS = [
    "item_id",
    "split",
    "context_with_blank",
    "gold_idiom",
    "candidates_json",
    "gold_index",
    "model",
    "scoring_mode",
    "pred_index",
    "pred_idiom",
    "correct",
    "gold_score",
    "pred_score",
    "option_scores_json",
    "margin_gold_minus_best_wrong",
    "collapse_affected",
    "num_unique_diacritic_candidates",
    "prompt_len_tokens",
    "candidate_lens_tokens_json",
]


SUMMARY_FIELDS = [
    "model",
    "scoring_mode",
    "n_items",
    "accuracy",
    "random_baseline",
    "gap_vs_baseline",
    "mean_margin_gold_minus_best_wrong",
    "median_margin_gold_minus_best_wrong",
    "n_collapse_affected",
    "collapse_affected_rate",
    "accuracy_noncollapsed",
    "accuracy_collapse_affected",
    "n_skipped",
    "bootstrap_acc_ci_low",
    "bootstrap_acc_ci_high",
]


REVIEW_FIELDS = [
    "item_id",
    "split",
    "context_with_blank",
    "gold_idiom",
    "candidates_json",
    "gold_index",
    "collapse_affected",
    "num_unique_diacritic_candidates",
    "chinese_pred_index",
    "chinese_pred_idiom",
    "chinese_correct",
    "chinese_margin",
    "diacritic_pred_index",
    "diacritic_pred_idiom",
    "diacritic_correct",
    "diacritic_margin",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", default="validation", help="Preferred CHID split; falls back to another answered split.")
    parser.add_argument("--max-items", type=int, default=1000)
    parser.add_argument("--smoke-items", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--print-examples", type=int, default=5)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--only-smoke", action="store_true")
    return parser.parse_args()


def project_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def make_prompt(context_with_blank: str) -> str:
    return (
        f"阅读材料：\n{context_with_blank}\n\n"
        "请从以下候选成语中选择最合适的一项填入空格。\n"
        "答案："
    )


def cjk_len(text: str) -> int:
    count = sum(1 for char in str(text) if "\u4e00" <= char <= "\u9fff")
    return count if count else len(str(text))


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return "\n".join(part for part in (normalize_text(item) for item in value) if part).strip()
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).strip()


def normalize_string_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [normalize_text(item) for item in value if normalize_text(item)]
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    return []


def unique_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    output = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            output.append(value)
    return output


def features_to_string(features: Any) -> str:
    try:
        return repr(features)
    except Exception:
        return str(features)


def split_iterables(dataset: Any) -> list[tuple[str, Any]]:
    if hasattr(dataset, "items"):
        return list(dataset.items())
    return [("dataset", dataset)]


def dataset_schema_report(dataset: Any) -> tuple[str, dict[str, int], int]:
    lines: list[str] = []
    counts: dict[str, int] = {}
    total = 0
    for split, data in split_iterables(dataset):
        count = len(data)
        counts[split] = count
        total += count
        features = getattr(data, "features", "list[dict]")
        lines.append(f"split={split} count={count} features={features_to_string(features)}")
    return "\n".join(lines), counts, total


def print_raw_examples(dataset: Any, n: int, preferred_split: str) -> None:
    print(f"\nRaw CHID examples ({n} from split={preferred_split} when available):")
    splits = dict(split_iterables(dataset))
    ordered_splits = []
    if preferred_split in splits:
        ordered_splits.append(preferred_split)
    ordered_splits.extend(split for split in splits if split not in ordered_splits)
    for split in ordered_splits:
        data = splits[split]
        if len(data) == 0:
            continue
        for index in range(min(n, len(data))):
            print(f"\n[{split} raw {index}]")
            print(json.dumps(dict(data[index]), ensure_ascii=False)[:5000])
        return


def print_chid_schema_identification() -> None:
    print("\nCHID field identification")
    print("passage/context field: content (list of passages/snippets)")
    print("blank placeholder format: #idiomNNNNNN#")
    print("candidate idioms field: candidates (row-level candidate list)")
    print("answer/gold field: answers.text and answers.candidate_id")
    print("candidate set per blank: shared row-level candidate set; this pilot samples 3 distractors per blank")
    print("multiple blanks per passage: supported; one item is created per resolved blank")
    print("other blanks in the same passage: filled with their mapped gold idioms when unambiguous")


def try_load_hf(path: str, name: str | None) -> tuple[Any | None, str, list[str]]:
    from datasets import load_dataset

    display = f'datasets.load_dataset("{path}")' if name is None else f'datasets.load_dataset("{path}", "{name}")'
    try:
        dataset = load_dataset(path) if name is None else load_dataset(path, name)
        return dataset, display, []
    except Exception:
        return None, display, [f"{display}\n{traceback.format_exc()}"]


def read_json_or_jsonl(path: Path) -> list[Any]:
    rows: list[Any] = []
    try:
        if path.suffix.lower() == ".jsonl":
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        rows.append(json.loads(line))
            return rows
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("data", "train", "validation", "valid", "dev", "test"):
                if isinstance(payload.get(key), list):
                    return payload[key]
            return [payload]
    except Exception:
        return []
    return rows


def infer_split_from_path(path: Path) -> str:
    lower = path.name.lower()
    if "valid" in lower or "dev" in lower:
        return "validation"
    if "test" in lower:
        return "test"
    if "train" in lower:
        return "train"
    return "dataset"


def load_official_github(output_dir: Path) -> tuple[dict[str, list[dict[str, Any]]] | None, str, list[str]]:
    errors: list[str] = []
    cache_dir = output_dir / "_official_chid_github_cache"
    repo_dir = cache_dir / "ChID-Dataset"
    url = "https://github.com/chujiezheng/ChID-Dataset"
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        if not repo_dir.exists():
            if shutil.which("git"):
                command = ["git", "clone", "--depth", "1", url, str(repo_dir)]
                completed = subprocess.run(command, text=True, capture_output=True, check=False)
                if completed.returncode != 0:
                    errors.append(
                        "official GitHub git clone\n"
                        f"command: {' '.join(command)}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
                    )
            if not repo_dir.exists():
                zip_path = cache_dir / "ChID-Dataset.zip"
                try:
                    urllib.request.urlretrieve(f"{url}/archive/refs/heads/master.zip", zip_path)
                    with zipfile.ZipFile(zip_path) as archive:
                        archive.extractall(cache_dir)
                    extracted = cache_dir / "ChID-Dataset-master"
                    if extracted.exists():
                        extracted.rename(repo_dir)
                except Exception:
                    errors.append(f"official GitHub zip download\n{traceback.format_exc()}")
        if not repo_dir.exists():
            return None, "official GitHub ChID-Dataset", errors

        rows_by_split: dict[str, list[dict[str, Any]]] = {}
        for path in repo_dir.rglob("*"):
            if path.suffix.lower() not in {".json", ".jsonl"}:
                continue
            loaded = read_json_or_jsonl(path)
            normalized = [row for row in loaded if isinstance(row, dict) and "content" in row and "candidates" in row]
            if not normalized:
                continue
            rows_by_split.setdefault(infer_split_from_path(path), []).extend(normalized)
        if rows_by_split:
            return rows_by_split, "official GitHub ChID-Dataset", errors
        errors.append(f"official GitHub scan\nNo JSON/JSONL files with CHID content+candidates found under {repo_dir}")
    except Exception:
        errors.append(f"official GitHub fallback\n{traceback.format_exc()}")
    return None, "official GitHub ChID-Dataset", errors


def load_chid_dataset(output_dir: Path) -> tuple[Any, str, list[str]]:
    all_errors: list[str] = []
    for path, name in [("clue", "chid"), ("chid", None)]:
        dataset, display, errors = try_load_hf(path, name)
        all_errors.extend(errors)
        if dataset is not None:
            return dataset, display, all_errors
    dataset, display, errors = load_official_github(output_dir)
    all_errors.extend(errors)
    if dataset is not None:
        return dataset, display, all_errors
    print("Failed to load CHID from the requested sources. Exact errors:")
    for error in all_errors:
        print("\n---")
        print(error)
    raise RuntimeError("Could not load CHID from clue/chid, chid, or official GitHub ChID-Dataset.")


def row_answers(row: dict[str, Any]) -> tuple[list[str], list[int | None]]:
    answers = row.get("answers") or row.get("answer") or {}
    answer_texts: list[str] = []
    candidate_ids: list[int | None] = []
    if isinstance(answers, dict):
        answer_texts = normalize_string_list(answers.get("text") or answers.get("texts") or answers.get("answer"))
        raw_ids = answers.get("candidate_id") or answers.get("candidate_ids") or answers.get("label") or []
        if isinstance(raw_ids, (list, tuple)):
            for value in raw_ids:
                try:
                    candidate_ids.append(int(value))
                except Exception:
                    candidate_ids.append(None)
        elif raw_ids not in (None, ""):
            try:
                candidate_ids = [int(raw_ids)]
            except Exception:
                candidate_ids = [None]
    else:
        answer_texts = normalize_string_list(answers)
    return answer_texts, candidate_ids


def resolve_gold(
    answer_text: str,
    candidate_id: int | None,
    candidates: list[str],
) -> tuple[str, int | None]:
    if candidate_id is not None and 0 <= candidate_id < len(candidates):
        candidate = candidates[candidate_id]
        if not answer_text or answer_text == candidate or answer_text in candidates:
            return candidate, candidate_id
    if answer_text and answer_text in candidates:
        return answer_text, candidates.index(answer_text)
    return "", None


def normalize_blank_items_from_split(split: str, rows: Any) -> tuple[list[dict[str, Any]], Counter]:
    items: list[dict[str, Any]] = []
    stats: Counter = Counter()
    for raw_index, raw_any in enumerate(rows):
        row = dict(raw_any)
        candidates = unique_preserve_order(normalize_string_list(row.get("candidates") or row.get("candidate")))
        contents = normalize_string_list(row.get("content") or row.get("contents") or row.get("passage") or row.get("context"))
        answer_texts, candidate_ids = row_answers(row)
        placeholders: list[tuple[int, str]] = []
        for content_index, content in enumerate(contents):
            for match in PLACEHOLDER_RE.finditer(content):
                placeholders.append((content_index, match.group(0)))
        stats["raw_rows"] += 1
        stats["raw_placeholders"] += len(placeholders)
        if not candidates or not contents or not answer_texts:
            stats["missing_candidates_content_or_answers"] += 1
            continue
        if len(placeholders) == 0:
            stats["missing_placeholders"] += 1
            continue
        if len(answer_texts) != len(placeholders):
            stats["answer_placeholder_count_mismatch"] += 1
        gold_by_placeholder: dict[str, str] = {}
        gold_candidate_id_by_placeholder: dict[str, int] = {}
        for answer_index, (_content_index, placeholder) in enumerate(placeholders):
            if answer_index >= len(answer_texts):
                continue
            candidate_id = candidate_ids[answer_index] if answer_index < len(candidate_ids) else None
            gold, resolved_id = resolve_gold(answer_texts[answer_index], candidate_id, candidates)
            if not gold or resolved_id is None:
                stats["unresolved_gold"] += 1
                continue
            gold_by_placeholder[placeholder] = gold
            gold_candidate_id_by_placeholder[placeholder] = resolved_id

        for content_index, target_placeholder in placeholders:
            if target_placeholder not in gold_by_placeholder:
                continue
            content = contents[content_index]
            unresolved_other_blank = False

            def replace_placeholder(match: re.Match[str]) -> str:
                nonlocal unresolved_other_blank
                placeholder = match.group(0)
                if placeholder == target_placeholder:
                    return BLANK_MARKER
                if placeholder in gold_by_placeholder:
                    return gold_by_placeholder[placeholder]
                unresolved_other_blank = True
                return placeholder

            context_with_blank = PLACEHOLDER_RE.sub(replace_placeholder, content).strip()
            if unresolved_other_blank:
                stats["ambiguous_multi_blank_context"] += 1
                continue
            row_id = normalize_text(row.get("idx") or row.get("id") or raw_index)
            placeholder_id = target_placeholder.strip("#")
            items.append(
                {
                    "item_id": f"chid_{split}_{row_id}_{placeholder_id}",
                    "split": split,
                    "source_row_index": raw_index,
                    "source_content_index": content_index,
                    "target_placeholder": target_placeholder,
                    "context_with_blank": context_with_blank,
                    "gold_idiom": gold_by_placeholder[target_placeholder],
                    "gold_candidate_id": gold_candidate_id_by_placeholder[target_placeholder],
                    "provided_candidates": candidates,
                    "num_placeholders_in_content": len(PLACEHOLDER_RE.findall(content)),
                    "num_placeholders_in_row": len(placeholders),
                }
            )
    stats["valid_blank_level_items"] = len(items)
    return items, stats


def answered_split_names(dataset: Any) -> list[str]:
    names = []
    for split, rows in split_iterables(dataset):
        has_answers = False
        for index in range(min(10, len(rows))):
            row = dict(rows[index])
            answer_texts, _candidate_ids = row_answers(row)
            if answer_texts:
                has_answers = True
                break
        if has_answers:
            names.append(split)
    return names


def select_split(dataset: Any, preferred: str) -> tuple[str, Any]:
    splits = dict(split_iterables(dataset))
    if preferred in splits:
        rows = splits[preferred]
        if rows and any(row_answers(dict(rows[index]))[0] for index in range(min(10, len(rows)))):
            return preferred, rows
    for fallback in ("validation", "dev", "train", "dataset"):
        if fallback in splits:
            rows = splits[fallback]
            if rows and any(row_answers(dict(rows[index]))[0] for index in range(min(10, len(rows)))):
                return fallback, rows
    for split, rows in splits.items():
        if rows and any(row_answers(dict(rows[index]))[0] for index in range(min(10, len(rows)))):
            return split, rows
    raise RuntimeError("No CHID split with answer/gold fields was found. Test split is not usable for this pilot.")


def add_diacritic_collapse_metadata(item: dict[str, Any]) -> None:
    diacritic_candidates = [to_diacritic(candidate) for candidate in item["candidates"]]
    gold_diacritic = diacritic_candidates[int(item["gold_index"])]
    item["diacritic_candidates"] = diacritic_candidates
    item["num_unique_diacritic_candidates"] = len(set(diacritic_candidates))
    item["collapse_affected"] = int(any(idx != int(item["gold_index"]) and value == gold_diacritic for idx, value in enumerate(diacritic_candidates)))


def make_4choice_items(blank_items: list[dict[str, Any]], seed: int) -> tuple[list[dict[str, Any]], Counter]:
    rng = random.Random(seed)
    output: list[dict[str, Any]] = []
    stats: Counter = Counter()
    for item in blank_items:
        candidates = unique_preserve_order(list(item["provided_candidates"]))
        gold = item["gold_idiom"]
        if not gold or gold not in candidates:
            stats["gold_not_in_candidates"] += 1
            continue
        if len(candidates) < 4:
            stats["fewer_than_4_candidates"] += 1
            continue
        distractors = [candidate for candidate in candidates if candidate != gold]
        if len(distractors) < 3:
            stats["fewer_than_3_distractors"] += 1
            continue
        sampled = rng.sample(distractors, 3)
        options = [gold, *sampled]
        rng.shuffle(options)
        new_item = dict(item)
        new_item["candidates"] = options
        new_item["gold_index"] = options.index(gold)
        new_item["context_cjk_len"] = cjk_len(new_item["context_with_blank"])
        add_diacritic_collapse_metadata(new_item)
        output.append(new_item)
    stats["after_4choice_filtering"] = len(output)
    return output, stats


def apply_context_length_filter(items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, int]]]:
    steps = []
    selected: list[dict[str, Any]] = []
    for limit in (300, 500, 800):
        selected = [item for item in items if int(item["context_cjk_len"]) <= limit]
        steps.append({"context_len_limit": limit, "count": len(selected)})
        if len(selected) >= 500:
            return selected, steps
    return selected, steps


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
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint for {run.run_name}: {checkpoint}")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Missing tokenizer for {run.run_name}: {tokenizer_path}")
    config = AutoConfig.from_pretrained(str(checkpoint), local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), use_fast=True, local_files_only=True)
    if int(config.vocab_size) != EXPECTED_VOCAB_SIZE or len(tokenizer) != EXPECTED_VOCAB_SIZE:
        raise ValueError(f"{run.run_name}: expected vocab_size=32001; config={config.vocab_size}, tokenizer={len(tokenizer)}")
    if tokenizer.eos_token_id != EXPECTED_EOS_ID:
        raise ValueError(f"{run.run_name}: expected eos_token_id=32000; got {tokenizer.eos_token_id}")
    if tokenizer.pad_token_id != EXPECTED_PAD_ID:
        raise ValueError(f"{run.run_name}: expected pad_token_id=32000; got {tokenizer.pad_token_id}")
    if int(getattr(config, "max_position_embeddings", MAX_SEQ_LEN)) != MAX_SEQ_LEN:
        raise ValueError(
            f"{run.run_name}: expected max_position_embeddings=1024; got {config.max_position_embeddings}"
        )
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


def encode_prompt_and_candidates(item: dict[str, Any], run: ModelRun) -> tuple[str, list[str]]:
    prompt = make_prompt(item["context_with_blank"])
    candidates = list(item["candidates"])
    if run.script == "pinyin_diacritic":
        prompt = to_diacritic(prompt)
        candidates = [to_diacritic(candidate) for candidate in candidates]
    return prompt, candidates


def score_options(
    model,
    tokenizer,
    device: torch.device,
    prompt: str,
    candidates: list[str],
    max_seq_len: int,
) -> tuple[list[float], int, list[int]]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    candidate_ids = [tokenizer(candidate, add_special_tokens=False)["input_ids"] for candidate in candidates]
    if any(len(ids) == 0 for ids in candidate_ids):
        raise ValueError("zero_candidate_tokens")
    if any(len(prompt_ids) + len(ids) > max_seq_len for ids in candidate_ids):
        raise ValueError("prompt_plus_candidate_exceeds_1024")

    sequences = [prompt_ids + ids for ids in candidate_ids]
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
    for row_index, ids in enumerate(candidate_ids):
        values: list[float] = []
        for offset, token_id in enumerate(ids):
            token_index = target_start + offset
            values.append(float(log_probs[row_index, token_index - 1, token_id].detach().cpu().item()))
        if not values or not all(math.isfinite(value) for value in values):
            raise ValueError("non_finite_score")
        scores.append(sum(values) / len(values))
    return scores, len(prompt_ids), [len(ids) for ids in candidate_ids]


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
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    iterator = tqdm(items, desc=desc, disable=args.no_progress)
    for index, item in enumerate(iterator, start=1):
        if args.progress_every and index % args.progress_every == 0:
            print(f"{desc}: scored {index}/{len(items)} candidate items")
        prompt, candidates_for_model = encode_prompt_and_candidates(item, run)
        try:
            scores, prompt_len, candidate_lens = score_options(
                model, tokenizer, device, prompt, candidates_for_model, max_seq_len
            )
        except ValueError as exc:
            skipped.append({"item_id": item["item_id"], "model": run.run_name, "reason": str(exc)})
            continue
        pred_index = max(range(len(scores)), key=lambda option_index: scores[option_index])
        gold_index = int(item["gold_index"])
        wrong_scores = [score for option_index, score in enumerate(scores) if option_index != gold_index]
        margin = scores[gold_index] - max(wrong_scores)
        if not all(math.isfinite(value) for value in [scores[gold_index], scores[pred_index], margin]):
            raise ValueError(f"Non-finite score for {run.run_name}/{item['item_id']}")
        rows.append(
            {
                "item_id": item["item_id"],
                "split": item["split"],
                "context_with_blank": item["context_with_blank"],
                "gold_idiom": item["gold_idiom"],
                "candidates_json": json.dumps(item["candidates"], ensure_ascii=False),
                "gold_index": gold_index,
                "model": run.run_name,
                "scoring_mode": SCORING_MODE,
                "pred_index": pred_index,
                "pred_idiom": item["candidates"][pred_index],
                "correct": int(pred_index == gold_index),
                "gold_score": scores[gold_index],
                "pred_score": scores[pred_index],
                "option_scores_json": json.dumps(scores, ensure_ascii=False),
                "margin_gold_minus_best_wrong": margin,
                "collapse_affected": int(item["collapse_affected"]),
                "num_unique_diacritic_candidates": int(item["num_unique_diacritic_candidates"]),
                "prompt_len_tokens": prompt_len,
                "candidate_lens_tokens_json": json.dumps(candidate_lens, ensure_ascii=False),
            }
        )
    return rows, skipped


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


def summarize_model(run: ModelRun, rows: list[dict[str, Any]], skipped: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    correct = [float(row["correct"]) for row in rows]
    margins = [float(row["margin_gold_minus_best_wrong"]) for row in rows]
    collapse_rows = [row for row in rows if int(row["collapse_affected"])]
    noncollapse_rows = [row for row in rows if not int(row["collapse_affected"])]
    low, high = bootstrap_acc_ci(correct, args.bootstrap_samples, args.seed + 23)
    accuracy = sum(correct) / len(correct) if correct else ""
    return {
        "model": run.run_name,
        "scoring_mode": SCORING_MODE,
        "n_items": len(rows),
        "accuracy": accuracy,
        "random_baseline": RANDOM_BASELINE,
        "gap_vs_baseline": accuracy - RANDOM_BASELINE if correct else "",
        "mean_margin_gold_minus_best_wrong": sum(margins) / len(margins) if margins else "",
        "median_margin_gold_minus_best_wrong": median(margins) if margins else "",
        "n_collapse_affected": len(collapse_rows),
        "collapse_affected_rate": len(collapse_rows) / len(rows) if rows else "",
        "accuracy_noncollapsed": (
            sum(float(row["correct"]) for row in noncollapse_rows) / len(noncollapse_rows) if noncollapse_rows else ""
        ),
        "accuracy_collapse_affected": (
            sum(float(row["correct"]) for row in collapse_rows) / len(collapse_rows) if collapse_rows else ""
        ),
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
    checkpoint, tokenizer, config = validate_checkpoint_and_tokenizer(root, run)
    print(f"\n== {run.run_name} ==")
    print(f"checkpoint exists: {checkpoint}")
    print(f"tokenizer exists: {project_path(root, run.tokenizer)}")
    print(
        f"tokenizer vocab_size={len(tokenizer)}, eos_token_id={tokenizer.eos_token_id}, "
        f"pad_token_id={tokenizer.pad_token_id}, config_vocab_size={config.vocab_size}"
    )
    model = load_model(checkpoint, device, dtype)
    param_count = count_params(model)
    print(f"model parameter_count={param_count}")
    if param_count != EXPECTED_PARAM_COUNT:
        raise ValueError(f"Unexpected parameter count for {run.run_name}: {param_count}")
    rows, skipped = evaluate_loaded_model(run, model, tokenizer, items, device, args, desc)
    summary = summarize_model(run, rows, skipped, args)
    payload = {
        "run": {
            "run_name": run.run_name,
            "script": run.script,
            "checkpoint": str(checkpoint),
            "tokenizer": run.tokenizer,
            "device": device.type,
            "dtype": dtype_name,
            "parameter_count": param_count,
            "scoring": "option_text_scoring_no_labels_no_eos_add_special_tokens_false",
        },
        "summary": summary,
        "skipped": skipped,
    }
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    return rows, skipped, payload


def print_filter_report(
    split_name: str,
    selected_raw_count: int,
    blank_stats: Counter,
    choice_stats: Counter,
    context_steps: list[dict[str, int]],
    final_count: int,
) -> None:
    print(f"\nSelected split: {split_name}")
    print(f"raw rows in selected split: {selected_raw_count}")
    print(f"raw placeholders in selected split: {blank_stats['raw_placeholders']}")
    print(f"valid blank-level items: {blank_stats['valid_blank_level_items']}")
    print(f"after 4-choice filtering: {choice_stats['after_4choice_filtering']}")
    for step in context_steps:
        print(f"after context length <= {step['context_len_limit']} Chinese chars: {step['count']}")
    if context_steps and context_steps[0]["count"] < 500:
        print("context length relaxed because fewer than 500 valid items remained at <=300 Chinese chars")
    print(f"final 4-choice item count before max-items sampling: {final_count}")
    for key in sorted(blank_stats):
        if key not in {"raw_rows", "raw_placeholders", "valid_blank_level_items"} and blank_stats[key]:
            print(f"blank normalization note: {key}={blank_stats[key]}")
    for key in sorted(choice_stats):
        if key != "after_4choice_filtering" and choice_stats[key]:
            print(f"4-choice filtering note: {key}={choice_stats[key]}")


def print_final_examples(items: list[dict[str, Any]], n: int, seed: int) -> None:
    sample = random.Random(seed).sample(items, min(n, len(items)))
    print(f"\nFinal Eval5 examples ({len(sample)}):")
    for item in sample:
        print(f"\n[{item['item_id']}] split={item['split']} gold={item['gold_index']} {item['gold_idiom']}")
        print(f"context: {item['context_with_blank']}")
        print(f"candidates: {json.dumps(item['candidates'], ensure_ascii=False)}")
        print(
            "diacritic collapse: "
            f"collapse_affected={item['collapse_affected']} "
            f"num_unique={item['num_unique_diacritic_candidates']}"
        )


def build_review_rows(all_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_item: dict[str, dict[str, dict[str, Any]]] = {}
    for row in all_rows:
        by_item.setdefault(row["item_id"], {})[row["model"]] = row
    buckets = {
        "examples_correct_both.csv": [],
        "examples_wrong_both.csv": [],
        "examples_chinese_correct_diacritic_wrong.csv": [],
        "examples_diacritic_correct_chinese_wrong.csv": [],
        "examples_collapse_affected.csv": [],
    }
    for item_id, model_rows in sorted(by_item.items()):
        ch = model_rows.get("chinese_4epoch")
        di = model_rows.get("diacritic_matched_token_4epoch")
        if not ch or not di:
            continue
        review = {
            "item_id": item_id,
            "split": ch["split"],
            "context_with_blank": ch["context_with_blank"],
            "gold_idiom": ch["gold_idiom"],
            "candidates_json": ch["candidates_json"],
            "gold_index": ch["gold_index"],
            "collapse_affected": ch["collapse_affected"],
            "num_unique_diacritic_candidates": ch["num_unique_diacritic_candidates"],
            "chinese_pred_index": ch["pred_index"],
            "chinese_pred_idiom": ch["pred_idiom"],
            "chinese_correct": ch["correct"],
            "chinese_margin": ch["margin_gold_minus_best_wrong"],
            "diacritic_pred_index": di["pred_index"],
            "diacritic_pred_idiom": di["pred_idiom"],
            "diacritic_correct": di["correct"],
            "diacritic_margin": di["margin_gold_minus_best_wrong"],
        }
        if int(ch["collapse_affected"]):
            buckets["examples_collapse_affected.csv"].append(review)
        if int(ch["correct"]) and int(di["correct"]):
            buckets["examples_correct_both.csv"].append(review)
        elif not int(ch["correct"]) and not int(di["correct"]):
            buckets["examples_wrong_both.csv"].append(review)
        elif int(ch["correct"]) and not int(di["correct"]):
            buckets["examples_chinese_correct_diacritic_wrong.csv"].append(review)
        elif not int(ch["correct"]) and int(di["correct"]):
            buckets["examples_diacritic_correct_chinese_wrong.csv"].append(review)
    return buckets


def near_random(accuracy: float) -> bool:
    return abs(accuracy - RANDOM_BASELINE) <= 0.05


def interpretation_text(ch_acc: float, di_acc: float, collapse_rate: float) -> str:
    if near_random(ch_acc) and near_random(di_acc):
        return "Both models are near 25%; CHID is too hard or option-text scoring is not effective for these 134M pure pretrained models."
    if ch_acc > RANDOM_BASELINE + 0.05 and near_random(di_acc):
        return "Chinese is above random while Pinyin-Diacritic is near random; CHID may indicate idiom identity is more accessible in character form."
    if ch_acc > RANDOM_BASELINE + 0.05 and di_acc > RANDOM_BASELINE + 0.05 and ch_acc > di_acc:
        return "Both models are above random and Chinese is higher; this is a potential idiom-specific script effect."
    if collapse_rate > 0.05:
        return "Results are diagnostic because Pinyin-Diacritic candidate collapse affects a non-trivial share of items; do not overclaim."
    return "Treat this as a standalone diagnostic pilot; do not overclaim without additional scoring modes or replications."


def write_report(
    path: Path,
    dataset_source: str,
    split_name: str,
    raw_count: int,
    raw_total: int,
    blank_count: int,
    final_count: int,
    evaluated_count: int,
    summaries: list[dict[str, Any]],
    context_steps: list[dict[str, int]],
    load_errors: list[str],
) -> None:
    by_model = {row["model"]: row for row in summaries}
    ch = by_model["chinese_4epoch"]
    di = by_model["diacritic_matched_token_4epoch"]
    ch_acc = float(ch["accuracy"])
    di_acc = float(di["accuracy"])
    collapse_rate = float(di["collapse_affected_rate"]) if di["collapse_affected_rate"] != "" else 0.0
    interpretation = interpretation_text(ch_acc, di_acc, collapse_rate)
    lines = [
        "# Eval5: CHID Idiom Cloze Pilot",
        "",
        "## Dataset",
        f"- Source used: `{dataset_source}`",
        f"- Split used: `{split_name}`",
        "- Passage/context field: `content`",
        "- Blank placeholder format: `#idiomNNNNNN#`",
        "- Candidate idioms field: `candidates`",
        "- Answer/gold field: `answers.text` and `answers.candidate_id`",
        "- Candidate set per blank: shared row-level candidate set; the pilot samples 3 distractors per blank",
        "- Multiple blanks per passage: supported when placeholder-order mapping is unambiguous",
        f"- Raw count in selected split: {raw_count}",
        f"- Raw count across loaded splits: {raw_total}",
        f"- Valid blank-level item count: {blank_count}",
        f"- Final 4-choice item count before max-items sampling: {final_count}",
        f"- Evaluated item count: {evaluated_count}",
        f"- Context filtering steps: `{json.dumps(context_steps, ensure_ascii=False)}`",
        "",
        "## Scoring",
        f"- Primary scoring mode: `{SCORING_MODE}`",
        "- Candidate labels A/B/C/D are not scored.",
        "- Completion score is mean logprob over candidate idiom tokens conditioned on the shared prompt.",
        "- `add_special_tokens=False`; no EOS token is appended for option-text scoring, matching the prior Eval3 option-text pilot.",
        "- Secondary `candidate_plus_suffix` scoring was not run because this standalone pilot uses the primary option-text definition.",
        "",
        "## Results",
        f"- Random baseline: {RANDOM_BASELINE:.2%}",
        f"- Chinese accuracy: {ch_acc:.2%}",
        f"- Pinyin-Diacritic accuracy: {di_acc:.2%}",
        f"- Gap Chinese - Pinyin-Diacritic: {ch_acc - di_acc:.2%}",
        f"- Chinese gap vs baseline: {ch_acc - RANDOM_BASELINE:.2%}",
        f"- Pinyin-Diacritic gap vs baseline: {di_acc - RANDOM_BASELINE:.2%}",
        f"- Pinyin-Diacritic candidate collapse count/rate: {di['n_collapse_affected']} / {collapse_rate:.2%}",
        f"- Chinese meaningfully above random: {ch_acc > RANDOM_BASELINE + 0.05}",
        f"- Pinyin-Diacritic meaningfully above random: {di_acc > RANDOM_BASELINE + 0.05}",
        "",
        "## Interpretation",
        interpretation,
        "",
        "This Eval5 result is a standalone pilot only. It does not modify Eval1, Eval2, Eval3, or Eval4 results, and no models were retrained.",
    ]
    if load_errors:
        lines.extend(
            [
                "",
                "## Non-Fatal Dataset Loading Attempts",
                f"{len(load_errors)} earlier loading attempt(s) failed before the selected source succeeded. Full tracebacks were printed to console.",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def concise_report(dataset_source: str, n_items: int, summaries: list[dict[str, Any]]) -> None:
    by_model = {row["model"]: row for row in summaries}
    ch = by_model["chinese_4epoch"]
    di = by_model["diacritic_matched_token_4epoch"]
    ch_acc = float(ch["accuracy"])
    di_acc = float(di["accuracy"])
    collapse_rate = float(di["collapse_affected_rate"]) if di["collapse_affected_rate"] != "" else 0.0
    print("\nFinal Eval5 report")
    print(f"CHID source used: {dataset_source}")
    print(f"evaluated n_items: {n_items}")
    print(f"Chinese accuracy: {ch_acc:.4f}")
    print(f"Pinyin-Diacritic accuracy: {di_acc:.4f}")
    print(f"random baseline: {RANDOM_BASELINE:.4f}")
    print(f"Chinese minus baseline: {ch_acc - RANDOM_BASELINE:.4f}")
    print(f"Pinyin-Diacritic minus baseline: {di_acc - RANDOM_BASELINE:.4f}")
    print(f"Chinese - Pinyin-Diacritic gap: {ch_acc - di_acc:.4f}")
    print(f"Pinyin-Diacritic collapse rate: {collapse_rate:.4f}")
    print(interpretation_text(ch_acc, di_acc, collapse_rate))


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    root = Path.cwd()
    output_dir = project_path(root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Eval5 validation: checkpoints and tokenizers")
    for run in MODEL_RUNS:
        checkpoint, tokenizer, config = validate_checkpoint_and_tokenizer(root, run)
        print(f"{run.run_name}: checkpoint exists: {checkpoint}")
        print(
            f"{run.run_name}: tokenizer vocab_size={len(tokenizer)}, eos_token_id={tokenizer.eos_token_id}, "
            f"pad_token_id={tokenizer.pad_token_id}, config_vocab_size={config.vocab_size}"
        )

    dataset, dataset_source, load_errors = load_chid_dataset(output_dir)
    schema, raw_counts, raw_total = dataset_schema_report(dataset)
    print("\nCHID dataset schema")
    print(schema)
    print(f"split names: {', '.join(raw_counts)}")
    print(f"raw examples across loaded splits: {raw_total}")
    print(f"answered splits: {', '.join(answered_split_names(dataset))}")
    split_name, selected_rows = select_split(dataset, args.split)
    print_raw_examples(dataset, 3, split_name)
    print_chid_schema_identification()
    if load_errors:
        print(f"\nNon-fatal load attempts before selected source: {len(load_errors)}")

    blank_items, blank_stats = normalize_blank_items_from_split(split_name, selected_rows)
    choice_items, choice_stats = make_4choice_items(blank_items, args.seed)
    length_filtered, context_steps = apply_context_length_filter(choice_items)
    if not length_filtered:
        raise RuntimeError("No valid CHID examples left after blank, 4-choice, and context-length filtering.")

    rng = random.Random(args.seed)
    eval_items = list(length_filtered)
    rng.shuffle(eval_items)
    eval_items = eval_items[: args.max_items]
    eval_items.sort(key=lambda row: row["item_id"])
    for new_index, item in enumerate(eval_items):
        item["eval5_index"] = new_index

    write_jsonl(output_dir / "eval5_chid_filtered_4choice.jsonl", eval_items)
    print_filter_report(
        split_name,
        raw_counts.get(split_name, len(selected_rows)),
        blank_stats,
        choice_stats,
        context_steps,
        len(length_filtered),
    )
    print(f"wrote filtered 4-choice items: {output_dir / 'eval5_chid_filtered_4choice.jsonl'}")
    print_final_examples(eval_items, args.print_examples, args.seed)

    smoke_items = eval_items[: min(args.smoke_items, len(eval_items))]
    device, dtype, dtype_name = choose_device_and_dtype()
    print(f"\ndevice: {device.type}, dtype: {dtype_name}")

    if not args.skip_smoke:
        print(f"\nEval5 smoke test on {len(smoke_items)} items")
        for run in MODEL_RUNS:
            rows, skipped, _payload = load_score_unload(
                root,
                run,
                smoke_items,
                device,
                dtype,
                dtype_name,
                args,
                f"smoke_eval5_{run.run_name}",
            )
            if any(not math.isfinite(float(row["gold_score"])) for row in rows):
                raise ValueError(f"Non-finite smoke score for {run.run_name}")
            summary = summarize_model(run, rows, skipped, args)
            print(
                f"smoke {run.run_name}: n={summary['n_items']} accuracy={summary['accuracy']} "
                f"skipped={summary['n_skipped']}"
            )
            if run.script == "pinyin_diacritic":
                print(
                    "smoke Pinyin-Diacritic collapse count: "
                    f"{summary['n_collapse_affected']} / {summary['n_items']}"
                )
            if summary["n_items"] == 0:
                raise RuntimeError(f"Smoke test produced zero scored items for {run.run_name}.")
        print("Eval5 smoke test complete: all scored gold scores are finite.")

    if args.only_smoke:
        print("Stopping after smoke test because --only-smoke was set.")
        return

    print(f"\nEval5 full pilot on {len(eval_items)} items")
    all_rows: list[dict[str, Any]] = []
    all_skipped: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    json_payloads: dict[str, dict[str, Any]] = {}
    for run in MODEL_RUNS:
        rows, skipped, payload = load_score_unload(
            root,
            run,
            eval_items,
            device,
            dtype,
            dtype_name,
            args,
            f"full_eval5_{run.run_name}",
        )
        all_rows.extend(rows)
        all_skipped.extend(skipped)
        summaries.append(payload["summary"])
        json_payloads[run.run_name] = payload

    write_csv(output_dir / "item_scores.csv", all_rows, ITEM_SCORE_FIELDS)
    write_csv(output_dir / "summary.csv", summaries, SUMMARY_FIELDS)
    write_jsonl(output_dir / "skipped_items.jsonl", all_skipped)
    for run in MODEL_RUNS:
        payload = {
            **json_payloads[run.run_name],
            "dataset": {
                "source": dataset_source,
                "split_used": split_name,
                "raw_count_selected_split": raw_counts.get(split_name, len(selected_rows)),
                "raw_count_all_loaded_splits": raw_total,
                "valid_blank_level_item_count": blank_stats["valid_blank_level_items"],
                "after_4choice_filtering": choice_stats["after_4choice_filtering"],
                "final_4choice_item_count_before_max_items": len(length_filtered),
                "evaluated_count": len(eval_items),
                "context_length_filter_steps": context_steps,
                "blank_normalization_stats": dict(blank_stats),
                "choice_filter_stats": dict(choice_stats),
            },
        }
        (output_dir / run.output_json).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    review_buckets = build_review_rows(all_rows)
    for filename, rows in review_buckets.items():
        write_csv(output_dir / filename, rows, REVIEW_FIELDS)

    write_report(
        output_dir / "eval5_chid_report.md",
        dataset_source,
        split_name,
        raw_counts.get(split_name, len(selected_rows)),
        raw_total,
        blank_stats["valid_blank_level_items"],
        len(length_filtered),
        len(eval_items),
        summaries,
        context_steps,
        load_errors,
    )

    print(f"wrote: {output_dir / 'item_scores.csv'}")
    print(f"wrote: {output_dir / 'summary.csv'}")
    print(f"wrote: {output_dir / 'skipped_items.jsonl'}")
    print(f"wrote: {output_dir / 'chinese_4epoch.json'}")
    print(f"wrote: {output_dir / 'diacritic_matched_token_4epoch.json'}")
    for filename in review_buckets:
        print(f"wrote: {output_dir / filename}")
    print(f"wrote: {output_dir / 'eval5_chid_report.md'}")

    concise_report(dataset_source, len(eval_items), summaries)


if __name__ == "__main__":
    main()
