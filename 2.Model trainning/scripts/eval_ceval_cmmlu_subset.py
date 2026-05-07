#!/usr/bin/env python3
"""Zero-shot likelihood evaluation on a C-Eval/CMMLU-style MCQ subset."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from collections import Counter, defaultdict
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
LABELS = ["A", "B", "C", "D"]
SCORING_MODES = ["label_scoring", "option_text_scoring"]
DEFAULT_OUTPUT_DATA = "eval_data/mcq_subset"
DEFAULT_OUTPUT_RESULTS = "eval_results/mcq_subset"


@dataclass(frozen=True)
class ModelRun:
    run_name: str
    script: str
    checkpoint_candidates: tuple[str, ...]
    tokenizer: str


MODEL_RUNS = [
    ModelRun(
        "chinese_4epoch",
        "chinese_origin",
        (
            "server_outputs/4epoch/outputs/chinese_125m_b1024_4epoch_seed42/checkpoint-27176",
            "server_outputs/chinese_125m_b1024_4epoch_seed42_outputs/outputs/"
            "chinese_125m_b1024_4epoch_seed42/checkpoint-27176",
        ),
        "tokenizers/chinese_origin_32k_eos",
    ),
    ModelRun(
        "diacritic_matched_token_4epoch",
        "pinyin_diacritic",
        (
            "server_outputs/4epoch/diacritic_125m_b1024_matched_token_4epoch_seed42_outputs/outputs/"
            "diacritic_125m_b1024_matched_token_4epoch_seed42/checkpoint-27176",
            "server_outputs/diacritic_125m_b1024_matched_token_4epoch_seed42_outputs/outputs/"
            "diacritic_125m_b1024_matched_token_4epoch_seed42/checkpoint-27176",
        ),
        "tokenizers/pinyin_diacritic_32k_eos",
    ),
]

ITEM_FIELDS = [
    "id",
    "dataset_name",
    "subject",
    "category_if_available",
    "level_if_available",
    "question",
    "A",
    "B",
    "C",
    "D",
    "answer",
    "prompt_zh",
    "prompt_diacritic",
    "option_texts_zh",
    "option_texts_diacritic",
]

ITEM_SCORE_FIELDS = [
    "id",
    "dataset_name",
    "subject",
    "category_if_available",
    "level_if_available",
    "model_run",
    "script",
    "scoring_mode",
    "answer",
    "prediction",
    "correct",
    "margin",
    "gold_mean_logprob",
    "best_wrong_mean_logprob",
    "A_mean_logprob",
    "B_mean_logprob",
    "C_mean_logprob",
    "D_mean_logprob",
    "A_total_logprob",
    "B_total_logprob",
    "C_total_logprob",
    "D_total_logprob",
    "A_token_count",
    "B_token_count",
    "C_token_count",
    "D_token_count",
]

SUMMARY_OVERALL_FIELDS = [
    "dataset_name",
    "subset_name",
    "model_run",
    "script",
    "scoring_mode",
    "n_items",
    "accuracy",
    "baseline",
    "mean_margin",
    "median_margin",
    "accuracy_ci_low",
    "accuracy_ci_high",
    "mean_margin_ci_low",
    "mean_margin_ci_high",
    "device",
    "dtype",
    "notes",
]

SUMMARY_BY_SUBJECT_FIELDS = [
    "dataset_name",
    "subset_name",
    "model_run",
    "script",
    "scoring_mode",
    "subject",
    "n_items",
    "accuracy",
    "mean_margin",
]

SUMMARY_BY_CATEGORY_FIELDS = [
    "dataset_name",
    "subset_name",
    "model_run",
    "script",
    "scoring_mode",
    "category",
    "level",
    "n_items",
    "accuracy",
]

LABEL_BIAS_FIELDS = [
    "dataset_name",
    "subset_name",
    "model_run",
    "script",
    "scoring_mode",
    "predicted_A_count",
    "predicted_B_count",
    "predicted_C_count",
    "predicted_D_count",
    "gold_A_count",
    "gold_B_count",
    "gold_C_count",
    "gold_D_count",
]

COMPARISON_FIELDS = [
    "dataset_name",
    "subset_name",
    "scoring_mode",
    "n_items",
    "chinese_accuracy",
    "diacritic_accuracy",
    "accuracy_gap",
    "chinese_mean_margin",
    "diacritic_mean_margin",
    "margin_gap",
]


CEVAL_CATEGORIES = {
    "computer_network": "STEM",
    "operating_system": "STEM",
    "computer_architecture": "STEM",
    "college_programming": "STEM",
    "college_physics": "STEM",
    "college_chemistry": "STEM",
    "advanced_mathematics": "STEM",
    "probability_and_statistics": "STEM",
    "discrete_mathematics": "STEM",
    "electrical_engineer": "STEM",
    "metrology_engineer": "STEM",
    "high_school_mathematics": "STEM",
    "high_school_physics": "STEM",
    "high_school_chemistry": "STEM",
    "high_school_biology": "STEM",
    "middle_school_mathematics": "STEM",
    "middle_school_biology": "STEM",
    "middle_school_physics": "STEM",
    "middle_school_chemistry": "STEM",
    "veterinary_medicine": "STEM",
    "college_economics": "Social Science",
    "business_administration": "Social Science",
    "marxism": "Social Science",
    "mao_zedong_thought": "Social Science",
    "education_science": "Social Science",
    "teacher_qualification": "Social Science",
    "high_school_politics": "Social Science",
    "high_school_geography": "Social Science",
    "middle_school_politics": "Social Science",
    "middle_school_geography": "Social Science",
    "modern_chinese_history": "Social Science",
    "ideological_and_moral_cultivation": "Social Science",
    "logic": "Social Science",
    "law": "Social Science",
    "chinese_language_and_literature": "Humanities",
    "art_studies": "Humanities",
    "professional_tour_guide": "Humanities",
    "legal_professional": "Other",
    "high_school_chinese": "Humanities",
    "high_school_history": "Humanities",
    "middle_school_history": "Humanities",
    "civil_servant": "Other",
    "sports_science": "Other",
    "plant_protection": "Other",
    "basic_medicine": "Other",
    "clinical_medicine": "Other",
    "urban_and_rural_planner": "Other",
    "accountant": "Other",
    "fire_engineer": "Other",
    "environmental_impact_assessment_engineer": "Other",
    "tax_accountant": "Other",
    "physician": "Other",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate C-Eval/CMMLU MCQ subset with likelihood scoring.")
    parser.add_argument("--dataset", choices=["ceval", "cmmlu"], default="ceval")
    parser.add_argument("--hf-dataset", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--local-data-dir", action="append", default=[])
    parser.add_argument("--eval-data-dir", default=DEFAULT_OUTPUT_DATA)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_RESULTS)
    parser.add_argument("--subset-size", type=int, default=1000)
    parser.add_argument("--max-items", type=int, default=None, help="Smoke-test cap after stratified sampling.")
    parser.add_argument("--max-items-per-subject", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--print-examples", type=int, default=5)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--reuse-subset", action="store_true")
    parser.add_argument(
        "--offline-cache",
        action="store_true",
        help="Use already cached HuggingFace dataset files without Hub metadata checks when possible.",
    )
    return parser.parse_args()


def project_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def to_diacritic(text: str) -> str:
    parts = lazy_pinyin(
        str(text),
        style=Style.TONE,
        neutral_tone_with_five=False,
        errors=lambda chunk: list(chunk),
    )
    return " ".join(part.strip() for part in parts if part.strip()).strip()


def make_prompt_zh(question: str, options: dict[str, str]) -> str:
    return (
        f"问题：{question}\n"
        f"A. {options['A']}\n"
        f"B. {options['B']}\n"
        f"C. {options['C']}\n"
        f"D. {options['D']}\n"
        "答案："
    )


def make_prompt_diacritic(question: str, options: dict[str, str]) -> str:
    return (
        f"wèn tí: {to_diacritic(question)}\n"
        f"A. {to_diacritic(options['A'])}\n"
        f"B. {to_diacritic(options['B'])}\n"
        f"C. {to_diacritic(options['C'])}\n"
        f"D. {to_diacritic(options['D'])}\n"
        "dá àn:"
    )


def normalize_answer(value: Any) -> str | None:
    text = str(value).strip().upper()
    if text in LABELS:
        return text
    return None


def normalize_record(raw: dict[str, Any], dataset_name: str, subject: str, index: int) -> dict[str, Any] | None:
    question = raw.get("question") or raw.get("Question") or raw.get("stem")
    options = {label: raw.get(label) or raw.get(label.lower()) for label in LABELS}
    answer = normalize_answer(raw.get("answer") or raw.get("Answer") or raw.get("label"))
    if not question or not answer or any(options[label] in (None, "") for label in LABELS):
        return None
    category = raw.get("category") or raw.get("category_if_available") or CEVAL_CATEGORIES.get(subject, "")
    level = raw.get("level") or raw.get("level_if_available") or ""
    options = {label: str(options[label]).strip() for label in LABELS}
    item_id = f"{dataset_name}_{subject}_{index:05d}"
    prompt_zh = make_prompt_zh(str(question).strip(), options)
    option_texts_diacritic = {label: to_diacritic(options[label]) for label in LABELS}
    return {
        "id": item_id,
        "dataset_name": dataset_name,
        "subject": subject,
        "category_if_available": category,
        "level_if_available": level,
        "question": str(question).strip(),
        "A": options["A"],
        "B": options["B"],
        "C": options["C"],
        "D": options["D"],
        "answer": answer,
        "prompt_zh": prompt_zh,
        "prompt_diacritic": make_prompt_diacritic(str(question).strip(), options),
        "option_texts_zh": json.dumps(options, ensure_ascii=False),
        "option_texts_diacritic": json.dumps(option_texts_diacritic, ensure_ascii=False),
    }


def read_local_csv(path: Path, dataset_name: str, subject: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, raw in enumerate(reader):
            item = normalize_record(raw, dataset_name, subject, index)
            if item:
                rows.append(item)
    return rows


def discover_local_items(root: Path, dataset: str, local_dirs: list[str], split: str | None) -> list[dict[str, Any]]:
    dirs = [project_path(root, value) for value in local_dirs]
    default_candidates = [root / "data", root / "eval_data", root.parent, root.parent / "1.Tokenization"]
    dirs.extend(path for path in default_candidates if path.exists())
    split_terms = [split] if split else ["val", "validation", "dev"]
    dataset_terms = ["ceval", "c-eval"] if dataset == "ceval" else ["cmmlu"]
    seen: set[Path] = set()
    items: list[dict[str, Any]] = []
    for base in dirs:
        if not base.exists():
            continue
        for path in base.rglob("*.csv"):
            resolved = path.resolve()
            if resolved in seen:
                continue
            lower = str(path).lower()
            if not any(term in lower for term in dataset_terms):
                continue
            if not any(term in lower for term in split_terms):
                continue
            if "train" in lower:
                continue
            seen.add(resolved)
            subject = path.stem
            for term in ["_val", "_validation", "_dev", "-val", "-dev"]:
                subject = subject.replace(term, "")
            subject = subject.replace(" ", "_")
            items.extend(read_local_csv(path, dataset, subject))
    return items


def load_hf_items(dataset: str, hf_dataset: str | None, split: str | None, offline_cache: bool = False) -> list[dict[str, Any]]:
    from datasets import DownloadConfig, get_dataset_config_names, load_dataset

    dataset_path = hf_dataset or ("ceval/ceval-exam" if dataset == "ceval" else "haonan-li/cmmlu")
    split_candidates = [split] if split else ["val", "validation", "dev"]
    if offline_cache and dataset == "ceval" and hf_dataset is None:
        configs = sorted(CEVAL_CATEGORIES)
    else:
        try:
            configs = get_dataset_config_names(dataset_path, trust_remote_code=True)
        except TypeError:
            configs = get_dataset_config_names(dataset_path)
        except Exception:
            if dataset == "ceval" and hf_dataset is None:
                configs = sorted(CEVAL_CATEGORIES)
            else:
                raise
    download_config = DownloadConfig(local_files_only=True) if offline_cache else None
    items: list[dict[str, Any]] = []
    for subject in configs:
        loaded = None
        used_split = None
        for split_name in split_candidates:
            try:
                try:
                    loaded = load_dataset(
                        dataset_path,
                        subject,
                        split=split_name,
                        trust_remote_code=True,
                        download_config=download_config,
                    )
                except TypeError:
                    loaded = load_dataset(dataset_path, subject, split=split_name, download_config=download_config)
                used_split = split_name
                break
            except Exception:
                continue
        if loaded is None:
            continue
        for index, raw in enumerate(loaded):
            item = normalize_record(dict(raw), dataset, subject, index)
            if item:
                item["id"] = f"{dataset}_{used_split}_{subject}_{index:05d}"
                items.append(item)
    return items


def stratified_sample(items: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_subject[item["subject"]].append(item)
    rng = random.Random(args.seed)
    sampled: list[dict[str, Any]] = []
    for subject in sorted(by_subject):
        rows = list(by_subject[subject])
        rng.shuffle(rows)
        sampled.extend(rows[: args.max_items_per_subject])
    rng.shuffle(sampled)
    cap = args.subset_size
    if args.max_items is not None:
        cap = min(cap, args.max_items)
    sampled = sampled[:cap]
    sampled.sort(key=lambda row: (row["subject"], row["id"]))
    for index, item in enumerate(sampled):
        item["id"] = f"{item['dataset_name']}_subset_seed{args.seed}_{index:05d}"
    return sampled


def write_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps({field: item[field] for field in ITEM_FIELDS}, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def subset_paths(eval_data_dir: Path, dataset: str, subset_size: int, seed: int, max_items: int | None) -> tuple[Path, Path, str]:
    size_label = max_items if max_items is not None else subset_size
    subset_name = f"evalution3_{dataset}_subset_{size_label}_seed{seed}"
    return (
        eval_data_dir / f"{subset_name}.jsonl",
        eval_data_dir / f"{subset_name}_review.csv",
        subset_name,
    )


def build_or_load_subset(root: Path, args: argparse.Namespace) -> tuple[list[dict[str, Any]], str]:
    eval_data_dir = project_path(root, args.eval_data_dir)
    eval_data_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path, review_path, subset_name = subset_paths(
        eval_data_dir, args.dataset, args.subset_size, args.seed, args.max_items
    )
    if args.reuse_subset and jsonl_path.exists():
        items = read_jsonl(jsonl_path)
        print(f"reusing subset: {jsonl_path}")
        return items, subset_name

    items = discover_local_items(root, args.dataset, args.local_data_dir, args.split)
    source = "local"
    if not items:
        source = "huggingface"
        items = load_hf_items(args.dataset, args.hf_dataset, args.split, offline_cache=args.offline_cache)
    if not items:
        raise RuntimeError(f"No labeled {args.dataset} validation/dev items found locally or via HuggingFace.")
    sampled = stratified_sample(items, args)
    write_jsonl(jsonl_path, sampled)
    write_csv(review_path, sampled, ITEM_FIELDS)
    meta = {
        "dataset": args.dataset,
        "source": source,
        "n_loaded_labeled_items": len(items),
        "n_sampled_items": len(sampled),
        "subset_name": subset_name,
        "max_items_per_subject": args.max_items_per_subject,
        "seed": args.seed,
        "subjects": dict(Counter(item["subject"] for item in sampled)),
    }
    (eval_data_dir / f"{subset_name}_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"dataset source: {source}")
    print(f"loaded labeled items: {len(items)}")
    print(f"wrote subset: {jsonl_path}")
    print(f"wrote review: {review_path}")
    return sampled, subset_name


def choose_device_and_dtype() -> tuple[torch.device, torch.dtype, str]:
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.device("cuda"), torch.bfloat16, "bf16"
        return torch.device("cuda"), torch.float32, "fp32"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), torch.float32, "fp32"
    return torch.device("cpu"), torch.float32, "fp32"


def resolve_checkpoint(root: Path, run: ModelRun) -> Path:
    for candidate in run.checkpoint_candidates:
        path = project_path(root, candidate)
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing checkpoint for {run.run_name}: {run.checkpoint_candidates}")


def validate_checkpoint_and_tokenizer(root: Path, run: ModelRun):
    checkpoint = resolve_checkpoint(root, run)
    tokenizer_path = project_path(root, run.tokenizer)
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Missing tokenizer: {tokenizer_path}")
    config = AutoConfig.from_pretrained(str(checkpoint), local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    if config.vocab_size != EXPECTED_VOCAB_SIZE or len(tokenizer) != EXPECTED_VOCAB_SIZE:
        raise ValueError(f"Vocab mismatch for {run.run_name}: config={config.vocab_size}, tokenizer={len(tokenizer)}")
    if tokenizer.eos_token_id != EXPECTED_EOS_ID or tokenizer.pad_token_id != EXPECTED_PAD_ID:
        raise ValueError(f"EOS/PAD mismatch for {run.run_name}: eos={tokenizer.eos_token_id}, pad={tokenizer.pad_token_id}")
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


def score_completion(model, tokenizer, device: torch.device, prompt: str, completion: str) -> tuple[float, float, int]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    if not completion_ids:
        raise ValueError(f"Empty completion: {completion!r}")
    max_prompt_len = MAX_SEQ_LEN - len(completion_ids)
    if max_prompt_len < 1:
        completion_ids = completion_ids[: MAX_SEQ_LEN - 1]
        max_prompt_len = 1
    prompt_ids = prompt_ids[-max_prompt_len:]
    input_ids = prompt_ids + completion_ids
    target_start = len(prompt_ids)
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
    return sum(values) / len(values), sum(values), len(values)


def completion_candidates(item: dict[str, Any], run: ModelRun, mode: str) -> tuple[str, dict[str, str]]:
    if run.script == "chinese_origin":
        prompt = item["prompt_zh"]
        options = json.loads(item["option_texts_zh"])
    else:
        prompt = item["prompt_diacritic"]
        options = json.loads(item["option_texts_diacritic"])
    if mode == "label_scoring":
        return prompt, {label: label for label in LABELS}
    return prompt, {label: str(options[label]) for label in LABELS}


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


def evaluate_run(
    root: Path,
    run: ModelRun,
    items: list[dict[str, Any]],
    output_dir: Path,
    subset_name: str,
    device: torch.device,
    dtype: torch.dtype,
    dtype_name: str,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    checkpoint, tokenizer = validate_checkpoint_and_tokenizer(root, run)
    print(f"\n== {run.run_name} ==")
    print(f"checkpoint: {checkpoint}")
    print(f"tokenizer vocab/eos/pad validated: {len(tokenizer)}/{tokenizer.eos_token_id}/{tokenizer.pad_token_id}")
    model = load_model(checkpoint, device, dtype)
    param_count = count_params(model)
    print(f"model parameter count: {param_count}")
    if param_count != EXPECTED_PARAM_COUNT:
        raise ValueError(f"Unexpected parameter count for {run.run_name}: {param_count}")
    rows: list[dict[str, Any]] = []
    iterator = tqdm(items, desc=run.run_name, disable=args.no_progress)
    for index, item in enumerate(iterator, start=1):
        if args.progress_every and index % args.progress_every == 0:
            print(f"{run.run_name}: scored {index}/{len(items)} items")
        for mode in SCORING_MODES:
            prompt, candidates = completion_candidates(item, run, mode)
            score_rows = {}
            for label in LABELS:
                mean_lp, total_lp, token_count = score_completion(model, tokenizer, device, prompt, candidates[label])
                score_rows[label] = {"mean": mean_lp, "total": total_lp, "tokens": token_count}
            prediction = max(LABELS, key=lambda label: score_rows[label]["mean"])
            answer = item["answer"]
            wrong_scores = [score_rows[label]["mean"] for label in LABELS if label != answer]
            margin = score_rows[answer]["mean"] - max(wrong_scores)
            if not math.isfinite(margin):
                raise ValueError(f"Non-finite margin for {run.run_name}/{mode}/{item['id']}")
            rows.append(
                {
                    "id": item["id"],
                    "dataset_name": item["dataset_name"],
                    "subject": item["subject"],
                    "category_if_available": item["category_if_available"],
                    "level_if_available": item["level_if_available"],
                    "model_run": run.run_name,
                    "script": run.script,
                    "scoring_mode": mode,
                    "answer": answer,
                    "prediction": prediction,
                    "correct": int(prediction == answer),
                    "margin": margin,
                    "gold_mean_logprob": score_rows[answer]["mean"],
                    "best_wrong_mean_logprob": max(wrong_scores),
                    **{f"{label}_mean_logprob": score_rows[label]["mean"] for label in LABELS},
                    **{f"{label}_total_logprob": score_rows[label]["total"] for label in LABELS},
                    **{f"{label}_token_count": score_rows[label]["tokens"] for label in LABELS},
                }
            )
    summaries = summarize_overall(rows, subset_name, run, device.type, dtype_name, args)
    (output_dir / f"{run.run_name}.json").write_text(
        json.dumps(
            {"run": {"run_name": run.run_name, "script": run.script, "checkpoint": str(checkpoint), "tokenizer": run.tokenizer}, "summary": summaries},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows, summaries


def summarize_overall(rows: list[dict[str, Any]], subset_name: str, run: ModelRun, device: str, dtype_name: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    summaries = []
    for mode in SCORING_MODES:
        mode_rows = [row for row in rows if row["scoring_mode"] == mode]
        correct = [float(row["correct"]) for row in mode_rows]
        margins = [float(row["margin"]) for row in mode_rows]
        acc = sum(correct) / len(correct) if correct else ""
        acc_low, acc_high = bootstrap_ci(correct, lambda sample: sum(sample) / len(sample), args.bootstrap_samples, args.seed + 17)
        margin_low, margin_high = bootstrap_ci(margins, lambda sample: sum(sample) / len(sample), args.bootstrap_samples, args.seed + 31)
        summaries.append(
            {
                "dataset_name": mode_rows[0]["dataset_name"] if mode_rows else args.dataset,
                "subset_name": subset_name,
                "model_run": run.run_name,
                "script": run.script,
                "scoring_mode": mode,
                "n_items": len(mode_rows),
                "accuracy": acc,
                "baseline": 0.25,
                "mean_margin": sum(margins) / len(margins) if margins else "",
                "median_margin": median(margins) if margins else "",
                "accuracy_ci_low": acc_low,
                "accuracy_ci_high": acc_high,
                "mean_margin_ci_low": margin_low,
                "mean_margin_ci_high": margin_high,
                "device": device,
                "dtype": dtype_name,
                "notes": "Zero-shot likelihood scoring. Same prompt for label and option-text scoring; only scored completion differs.",
            }
        )
    return summaries


def summarize_by_subject(rows: list[dict[str, Any]], subset_name: str) -> list[dict[str, Any]]:
    output = []
    groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["dataset_name"], row["model_run"], row["script"], row["scoring_mode"], row["subject"])].append(row)
    for (dataset_name, model_run, script, mode, subject), group_rows in sorted(groups.items()):
        correct = [float(row["correct"]) for row in group_rows]
        margins = [float(row["margin"]) for row in group_rows]
        output.append(
            {
                "dataset_name": dataset_name,
                "subset_name": subset_name,
                "model_run": model_run,
                "script": script,
                "scoring_mode": mode,
                "subject": subject,
                "n_items": len(group_rows),
                "accuracy": sum(correct) / len(correct) if correct else "",
                "mean_margin": sum(margins) / len(margins) if margins else "",
            }
        )
    return output


def summarize_by_category(rows: list[dict[str, Any]], subset_name: str) -> list[dict[str, Any]]:
    output = []
    groups: dict[tuple[str, str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                row["dataset_name"],
                row["model_run"],
                row["script"],
                row["scoring_mode"],
                row.get("category_if_available") or "",
                row.get("level_if_available") or "",
            )
        ].append(row)
    for (dataset_name, model_run, script, mode, category, level), group_rows in sorted(groups.items()):
        correct = [float(row["correct"]) for row in group_rows]
        output.append(
            {
                "dataset_name": dataset_name,
                "subset_name": subset_name,
                "model_run": model_run,
                "script": script,
                "scoring_mode": mode,
                "category": category,
                "level": level,
                "n_items": len(group_rows),
                "accuracy": sum(correct) / len(correct) if correct else "",
            }
        )
    return output


def label_bias(rows: list[dict[str, Any]], subset_name: str) -> list[dict[str, Any]]:
    output = []
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["dataset_name"], row["model_run"], row["script"], row["scoring_mode"])].append(row)
    for (dataset_name, model_run, script, mode), group_rows in sorted(groups.items()):
        pred = Counter(row["prediction"] for row in group_rows)
        gold = Counter(row["answer"] for row in group_rows)
        output.append(
            {
                "dataset_name": dataset_name,
                "subset_name": subset_name,
                "model_run": model_run,
                "script": script,
                "scoring_mode": mode,
                **{f"predicted_{label}_count": pred[label] for label in LABELS},
                **{f"gold_{label}_count": gold[label] for label in LABELS},
            }
        )
    return output


def comparison_table(summaries: list[dict[str, Any]], subset_name: str) -> list[dict[str, Any]]:
    lookup = {(row["scoring_mode"], row["model_run"]): row for row in summaries}
    rows = []
    for mode in SCORING_MODES:
        ch = lookup.get((mode, "chinese_4epoch"))
        di = lookup.get((mode, "diacritic_matched_token_4epoch"))
        if not ch or not di:
            continue
        rows.append(
            {
                "dataset_name": ch["dataset_name"],
                "subset_name": subset_name,
                "scoring_mode": mode,
                "n_items": ch["n_items"],
                "chinese_accuracy": ch["accuracy"],
                "diacritic_accuracy": di["accuracy"],
                "accuracy_gap": float(ch["accuracy"]) - float(di["accuracy"]),
                "chinese_mean_margin": ch["mean_margin"],
                "diacritic_mean_margin": di["mean_margin"],
                "margin_gap": float(ch["mean_margin"]) - float(di["mean_margin"]),
            }
        )
    return rows


def print_examples(items: list[dict[str, Any]], tokenizer_by_script: dict[str, Any], args: argparse.Namespace) -> None:
    sample = random.Random(args.seed).sample(items, min(args.print_examples, len(items)))
    print(f"random prompt examples ({len(sample)}):")
    for item in sample:
        print(f"\n[{item['id']}] answer={item['answer']} subject={item['subject']}")
        print(item["prompt_zh"])
        print(item["prompt_diacritic"])
        zh_prompt_len = len(tokenizer_by_script["chinese_origin"](item["prompt_zh"], add_special_tokens=False)["input_ids"])
        py_prompt_len = len(tokenizer_by_script["pinyin_diacritic"](item["prompt_diacritic"], add_special_tokens=False)["input_ids"])
        zh_options = json.loads(item["option_texts_zh"])
        py_options = json.loads(item["option_texts_diacritic"])
        print(f"tokenized prompt lengths: zh={zh_prompt_len}, diacritic={py_prompt_len}")
        for label in LABELS:
            zh_len = len(tokenizer_by_script["chinese_origin"](zh_options[label], add_special_tokens=False)["input_ids"])
            py_len = len(tokenizer_by_script["pinyin_diacritic"](py_options[label], add_special_tokens=False)["input_ids"])
            label_zh_len = len(tokenizer_by_script["chinese_origin"](label, add_special_tokens=False)["input_ids"])
            label_py_len = len(tokenizer_by_script["pinyin_diacritic"](label, add_special_tokens=False)["input_ids"])
            print(f"  {label}: label_tokens zh/py={label_zh_len}/{label_py_len}, option_tokens zh/py={zh_len}/{py_len}")


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    root = Path.cwd()
    output_dir = project_path(root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    items, subset_name = build_or_load_subset(root, args)
    print(f"subset name: {subset_name}")
    print(f"subset items: {len(items)}")
    print(f"subjects: {len(set(item['subject'] for item in items))}")

    tokenizer_by_script = {}
    for run in MODEL_RUNS:
        checkpoint, tokenizer = validate_checkpoint_and_tokenizer(root, run)
        tokenizer_by_script[run.script] = tokenizer
        print(f"validated {run.run_name}: checkpoint={checkpoint}, tokenizer={len(tokenizer)}/{tokenizer.eos_token_id}/{tokenizer.pad_token_id}")
    print_examples(items, tokenizer_by_script, args)

    device, dtype, dtype_name = choose_device_and_dtype()
    print(f"device: {device.type}, dtype: {dtype_name}")
    all_rows: list[dict[str, Any]] = []
    all_summaries: list[dict[str, Any]] = []
    for run in MODEL_RUNS:
        rows, summaries = evaluate_run(root, run, items, output_dir, subset_name, device, dtype, dtype_name, args)
        all_rows.extend(rows)
        all_summaries.extend(summaries)

    write_csv(output_dir / "item_scores.csv", all_rows, ITEM_SCORE_FIELDS)
    write_csv(output_dir / "summary_overall.csv", all_summaries, SUMMARY_OVERALL_FIELDS)
    write_csv(output_dir / "summary_by_subject.csv", summarize_by_subject(all_rows, subset_name), SUMMARY_BY_SUBJECT_FIELDS)
    write_csv(output_dir / "summary_by_category_or_level.csv", summarize_by_category(all_rows, subset_name), SUMMARY_BY_CATEGORY_FIELDS)
    write_csv(output_dir / "label_bias_diagnostics.csv", label_bias(all_rows, subset_name), LABEL_BIAS_FIELDS)
    comparison = comparison_table(all_summaries, subset_name)
    write_csv(output_dir / "chinese_vs_diacritic_comparison.csv", comparison, COMPARISON_FIELDS)
    print(f"wrote: {output_dir / 'item_scores.csv'}")
    print(f"wrote: {output_dir / 'summary_overall.csv'}")
    print(f"wrote: {output_dir / 'summary_by_subject.csv'}")
    print(f"wrote: {output_dir / 'summary_by_category_or_level.csv'}")
    print(f"wrote: {output_dir / 'label_bias_diagnostics.csv'}")
    print(f"wrote: {output_dir / 'chinese_vs_diacritic_comparison.csv'}")

    print("\nComparison")
    for row in comparison:
        print(
            f"{row['scoring_mode']}: chinese={row['chinese_accuracy']} "
            f"diacritic={row['diacritic_accuracy']} gap={row['accuracy_gap']} "
            f"margin_gap={row['margin_gap']}"
        )


if __name__ == "__main__":
    main()
