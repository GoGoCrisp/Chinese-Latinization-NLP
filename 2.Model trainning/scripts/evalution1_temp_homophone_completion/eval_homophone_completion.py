#!/usr/bin/env python3
"""Quick homophone/collision completion probe for Experiment 2 models."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from pypinyin import Style, lazy_pinyin
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_OUTPUT_DIR = Path("/private/tmp/evalution2_homophone_completion")
EXPECTED_VOCAB_SIZE = 32001
EXPECTED_EOS_ID = 32000
EXPECTED_PAD_ID = 32000
MAX_SEQ_LEN = 1024
DEFAULT_EXCLUDED_CHARS = "的地得了着过之"


@dataclass(frozen=True)
class ModelRun:
    run_name: str
    script: str
    checkpoint: str
    tokenizer: str


MODEL_RUNS = [
    ModelRun(
        run_name="chinese_oneepoch",
        script="chinese_origin",
        checkpoint=(
            "server_outputs/chinese_125m_b1024_oneepoch_seed42_outputs/"
            "outputs/chinese_125m_b1024_oneepoch_seed42/checkpoint-6794"
        ),
        tokenizer="tokenizers/chinese_origin_32k_eos",
    ),
    ModelRun(
        run_name="diacritic_matched_token",
        script="pinyin_diacritic",
        checkpoint=(
            "server_outputs/diacritic_125m_b1024_matched_token_seed42_outputs/"
            "outputs/diacritic_125m_b1024_matched_token_seed42/checkpoint-6794"
        ),
        tokenizer="tokenizers/pinyin_diacritic_32k_eos",
    ),
    ModelRun(
        run_name="diacritic_matched_content",
        script="pinyin_diacritic",
        checkpoint=(
            "server_outputs/diacritic_125m_b1024_matched_content_seed42_outputs/"
            "outputs/diacritic_125m_b1024_matched_content_seed42/checkpoint-7441"
        ),
        tokenizer="tokenizers/pinyin_diacritic_32k_eos",
    ),
]


ITEM_COLUMNS = [
    "item_id",
    "pair_id",
    "source_group",
    "toneless_key",
    "candidate_a",
    "candidate_b",
    "answer",
    "distractor",
    "answer_pinyin",
    "distractor_pinyin",
    "diacritic_collapsed",
    "context_source",
    "line_index",
    "prefix_chars",
    "sentence",
]

QUESTION_COLUMNS = [
    "item_id",
    "pair_id",
    "question",
    "option_a",
    "option_b",
    "answer",
    "answer_pinyin",
    "distractor_pinyin",
    "diacritic_collapsed",
    "context_source",
    "line_index",
]

PRED_COLUMNS = [
    "item_id",
    "pair_id",
    "model_run",
    "script",
    "status",
    "prediction",
    "correct",
    "answer_score",
    "distractor_score",
    "margin",
    "answer_token_count",
    "distractor_token_count",
]

SUMMARY_COLUMNS = [
    "model_run",
    "script",
    "total_items",
    "collapsed_items",
    "collapse_rate",
    "resolvable_items",
    "resolvable_accuracy",
    "mean_margin",
    "median_margin",
    "device",
    "dtype",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and run a quick two-choice homophone completion probe."
    )
    parser.add_argument("--max-pairs", type=int, default=50)
    parser.add_argument("--min-pairs", type=int, default=30)
    parser.add_argument("--contexts-per-pair", type=int, default=1)
    parser.add_argument("--min-prefix-chars", type=int, default=8)
    parser.add_argument("--max-prefix-chars", type=int, default=220)
    parser.add_argument("--min-cjk-chars", type=int, default=2)
    parser.add_argument("--max-cjk-chars", type=int, default=4)
    parser.add_argument(
        "--exclude-chars",
        default=DEFAULT_EXCLUDED_CHARS,
        help="Drop candidate tokens containing these characters. Empty string disables this filter.",
    )
    parser.add_argument(
        "--collision-csv",
        default=(
            "../1.Tokenization/decoded_superTokenizers_2048_subset100k/table2/"
            "table2_ab_overlap_superBPE_outputs/table2_ab_overlap_superBPE_details.csv"
        ),
        help="AB Chinese-to-toneless collision details CSV.",
    )
    parser.add_argument(
        "--context-corpus",
        action="append",
        default=None,
        help="Chinese corpus path to search for real contexts. Can be repeated.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--print-questions",
        action="store_true",
        help="Print every constructed two-choice question to stdout.",
    )
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def choose_device_and_dtype() -> tuple[torch.device, torch.dtype, str]:
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.device("cuda"), torch.bfloat16, "bf16"
        return torch.device("cuda"), torch.float32, "fp32"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), torch.float32, "fp32"
    return torch.device("cpu"), torch.float32, "fp32"


def project_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def is_cjk_token(text: str, min_cjk_chars: int, max_cjk_chars: int, exclude_chars: str) -> bool:
    if re.search(r"[A-Za-z0-9]", text):
        return False
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    if cjk < min_cjk_chars or cjk > max_cjk_chars or cjk != len(text):
        return False
    return not any(ch in text for ch in exclude_chars)


def to_diacritic_pinyin(text: str) -> str:
    parts = lazy_pinyin(
        text,
        style=Style.TONE,
        neutral_tone_with_five=False,
        errors=lambda chunk: list(chunk),
    )
    return " ".join(part.strip() for part in parts if part.strip()).strip()


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


class TokenTrie:
    def __init__(self, tokens: set[str]) -> None:
        self.root: dict[str, Any] = {}
        self.max_len = 0
        for token in tokens:
            node = self.root
            for ch in token:
                node = node.setdefault(ch, {})
            node.setdefault("_tokens", []).append(token)
            self.max_len = max(self.max_len, len(token))

    def find(self, text: str) -> list[tuple[int, str]]:
        hits: list[tuple[int, str]] = []
        for start in range(len(text)):
            node = self.root
            for offset, ch in enumerate(text[start : start + self.max_len]):
                node = node.get(ch)
                if node is None:
                    break
                for token in node.get("_tokens", []):
                    hits.append((start, token))
        return hits


def load_collision_pairs(
    collision_csv: Path,
    min_cjk_chars: int,
    max_cjk_chars: int,
    exclude_chars: str,
) -> tuple[list[dict[str, Any]], set[str]]:
    rows: list[dict[str, Any]] = []
    tokens: set[str] = set()
    with collision_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_tokens = json.loads(row["a_tokens_json"])
            clean_tokens = [
                token.strip()
                for token in raw_tokens
                if is_cjk_token(token.strip(), min_cjk_chars, max_cjk_chars, exclude_chars)
            ]
            clean_tokens = sorted(set(clean_tokens))
            if len(clean_tokens) < 2:
                continue
            rows.append(
                {
                    "toneless_key": row["b_token"],
                    "source_group": row["a_tokens_json"],
                    "tokens": clean_tokens,
                }
            )
            tokens.update(clean_tokens)
    return rows, tokens


def load_context_lines(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_index, line in enumerate(handle):
                text = line.rstrip("\n")
                if text:
                    records.append(
                        {
                            "source": str(path),
                            "line_index": line_index,
                            "text": text,
                        }
                    )
    return records


def scan_token_counts_and_contexts(
    lines: list[dict[str, Any]],
    tokens: set[str],
    max_contexts_per_token: int,
    min_prefix_chars: int,
    max_prefix_chars: int,
    show_progress: bool,
) -> tuple[Counter[str], dict[str, list[dict[str, Any]]]]:
    trie = TokenTrie(tokens)
    counts: Counter[str] = Counter()
    contexts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    iterator = tqdm(lines, desc="scan contexts", disable=not show_progress)

    for record in iterator:
        text = record["text"]
        for start, token in trie.find(text):
            counts[token] += 1
            if len(contexts[token]) >= max_contexts_per_token:
                continue
            prefix = text[:start]
            if len(prefix) < min_prefix_chars:
                continue
            contexts[token].append(
                {
                    "context_source": record["source"],
                    "line_index": record["line_index"],
                    "prefix": prefix[-max_prefix_chars:],
                    "sentence": text,
                }
            )
    return counts, contexts


def build_probe_items(
    collision_rows: list[dict[str, Any]],
    counts: Counter[str],
    contexts_by_token: dict[str, list[dict[str, Any]]],
    max_pairs: int,
    min_pairs: int,
    contexts_per_pair: int,
) -> list[dict[str, Any]]:
    pair_candidates: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()

    for row in collision_rows:
        for candidate_a, candidate_b in itertools.combinations(row["tokens"], 2):
            pair_key = tuple(sorted((candidate_a, candidate_b)))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            if not contexts_by_token.get(candidate_a) and not contexts_by_token.get(candidate_b):
                continue
            freq_a = counts[candidate_a]
            freq_b = counts[candidate_b]
            if freq_a + freq_b <= 0:
                continue
            pair_candidates.append(
                {
                    "candidate_a": candidate_a,
                    "candidate_b": candidate_b,
                    "toneless_key": row["toneless_key"],
                    "source_group": row["source_group"],
                    "freq_sum": freq_a + freq_b,
                    "freq_min": min(freq_a, freq_b),
                }
            )

    pair_candidates.sort(key=lambda row: (row["freq_min"], row["freq_sum"]), reverse=True)
    selected = pair_candidates[:max_pairs]
    if len(selected) < min_pairs:
        raise ValueError(f"Only found {len(selected)} usable pairs; requested at least {min_pairs}")

    items: list[dict[str, Any]] = []
    item_id = 0
    for pair_id, pair in enumerate(selected):
        per_pair_contexts: list[tuple[str, str, dict[str, Any]]] = []
        a = pair["candidate_a"]
        b = pair["candidate_b"]
        for context in contexts_by_token.get(a, []):
            per_pair_contexts.append((a, b, context))
        for context in contexts_by_token.get(b, []):
            per_pair_contexts.append((b, a, context))
        per_pair_contexts = per_pair_contexts[:contexts_per_pair]
        for answer, distractor, context in per_pair_contexts:
            answer_pinyin = normalize_space(to_diacritic_pinyin(answer))
            distractor_pinyin = normalize_space(to_diacritic_pinyin(distractor))
            items.append(
                {
                    "item_id": item_id,
                    "pair_id": pair_id,
                    "source_group": pair["source_group"],
                    "toneless_key": pair["toneless_key"],
                    "candidate_a": a,
                    "candidate_b": b,
                    "answer": answer,
                    "distractor": distractor,
                    "answer_pinyin": answer_pinyin,
                    "distractor_pinyin": distractor_pinyin,
                    "diacritic_collapsed": answer_pinyin == distractor_pinyin,
                    "context_source": context["context_source"],
                    "line_index": context["line_index"],
                    "prefix_chars": len(context["prefix"]),
                    "prefix": context["prefix"],
                    "sentence": context["sentence"],
                }
            )
            item_id += 1
    return items


def load_tokenizer(tokenizer_path: Path):
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    if len(tokenizer) != EXPECTED_VOCAB_SIZE:
        raise ValueError(f"Tokenizer vocab size mismatch for {tokenizer_path}: {len(tokenizer)}")
    if tokenizer.eos_token_id != EXPECTED_EOS_ID or tokenizer.pad_token_id != EXPECTED_PAD_ID:
        raise ValueError(
            f"Tokenizer eos/pad mismatch for {tokenizer_path}: "
            f"eos={tokenizer.eos_token_id}, pad={tokenizer.pad_token_id}"
        )
    return tokenizer


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


def score_completion(
    model,
    tokenizer,
    device: torch.device,
    prefix: str,
    completion: str,
) -> tuple[float, int]:
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    if not completion_ids:
        raise ValueError(f"Empty completion after tokenization: {completion!r}")
    max_prefix_len = MAX_SEQ_LEN - len(completion_ids)
    if max_prefix_len < 1:
        raise ValueError(f"Completion too long for max sequence length: {completion!r}")
    prefix_ids = prefix_ids[-max_prefix_len:]
    input_ids = prefix_ids + completion_ids
    if len(input_ids) < 2:
        raise ValueError("Need at least two tokens to score a completion")

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
        raise ValueError("Non-finite or empty completion score")
    return sum(values) / len(values), len(values)


def evaluate_model_run(
    root: Path,
    run: ModelRun,
    items: list[dict[str, Any]],
    device: torch.device,
    dtype: torch.dtype,
    dtype_name: str,
    show_progress: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    checkpoint = project_path(root, run.checkpoint)
    tokenizer_path = project_path(root, run.tokenizer)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    tokenizer = load_tokenizer(tokenizer_path)
    model = load_model(checkpoint, device, dtype)

    predictions: list[dict[str, Any]] = []
    margins: list[float] = []
    correct_count = 0
    resolvable_count = 0
    collapsed_count = 0

    iterator = tqdm(items, desc=run.run_name, disable=not show_progress)
    for item in iterator:
        if run.script == "pinyin_diacritic":
            if item["diacritic_collapsed"]:
                collapsed_count += 1
                predictions.append(
                    {
                        "item_id": item["item_id"],
                        "pair_id": item["pair_id"],
                        "model_run": run.run_name,
                        "script": run.script,
                        "status": "collapsed",
                        "prediction": "",
                        "correct": "",
                        "answer_score": "",
                        "distractor_score": "",
                        "margin": "",
                        "answer_token_count": "",
                        "distractor_token_count": "",
                    }
                )
                continue
            prefix = normalize_space(to_diacritic_pinyin(item["prefix"]))
            answer = item["answer_pinyin"]
            distractor = item["distractor_pinyin"]
        else:
            prefix = item["prefix"]
            answer = item["answer"]
            distractor = item["distractor"]

        answer_score, answer_token_count = score_completion(model, tokenizer, device, prefix, answer)
        distractor_score, distractor_token_count = score_completion(
            model, tokenizer, device, prefix, distractor
        )
        margin = answer_score - distractor_score
        prediction = item["answer"] if margin > 0 else item["distractor"]
        correct = margin > 0
        resolvable_count += 1
        correct_count += int(correct)
        margins.append(margin)
        predictions.append(
            {
                "item_id": item["item_id"],
                "pair_id": item["pair_id"],
                "model_run": run.run_name,
                "script": run.script,
                "status": "scored",
                "prediction": prediction,
                "correct": int(correct),
                "answer_score": answer_score,
                "distractor_score": distractor_score,
                "margin": margin,
                "answer_token_count": answer_token_count,
                "distractor_token_count": distractor_token_count,
            }
        )

    margins_sorted = sorted(margins)
    median_margin = ""
    if margins_sorted:
        mid = len(margins_sorted) // 2
        if len(margins_sorted) % 2:
            median_margin = margins_sorted[mid]
        else:
            median_margin = (margins_sorted[mid - 1] + margins_sorted[mid]) / 2

    summary = {
        "model_run": run.run_name,
        "script": run.script,
        "total_items": len(items),
        "collapsed_items": collapsed_count,
        "collapse_rate": collapsed_count / len(items) if items else "",
        "resolvable_items": resolvable_count,
        "resolvable_accuracy": correct_count / resolvable_count if resolvable_count else "",
        "mean_margin": sum(margins) / len(margins) if margins else "",
        "median_margin": median_margin,
        "device": device.type,
        "dtype": dtype_name,
        "notes": (
            "Scores are mean log-probability per completion token. "
            "Diacritic collapsed items are not scored because both candidates map to the same pinyin string."
        ),
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return predictions, summary


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def question_rows(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in items:
        rows.append(
            {
                "item_id": item["item_id"],
                "pair_id": item["pair_id"],
                "question": f"{item['prefix']} ____",
                "option_a": item["candidate_a"],
                "option_b": item["candidate_b"],
                "answer": item["answer"],
                "answer_pinyin": item["answer_pinyin"],
                "distractor_pinyin": item["distractor_pinyin"],
                "diacritic_collapsed": item["diacritic_collapsed"],
                "context_source": item["context_source"],
                "line_index": item["line_index"],
            }
        )
    return rows


def write_questions_txt(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(f"Q{int(row['item_id']):03d} pair={row['pair_id']}\n")
            handle.write(f"Context: {row['question']}\n")
            handle.write(f"A. {row['option_a']}\n")
            handle.write(f"B. {row['option_b']}\n")
            handle.write(f"Answer: {row['answer']}\n")
            handle.write(
                "Pinyin: "
                f"answer={row['answer_pinyin']} | distractor={row['distractor_pinyin']} | "
                f"collapsed={row['diacritic_collapsed']}\n"
            )
            handle.write(f"Source: {row['context_source']}:{row['line_index']}\n\n")


def print_questions(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        print(f"Q{int(row['item_id']):03d} pair={row['pair_id']}")
        print(f"Context: {row['question']}")
        print(f"A. {row['option_a']}")
        print(f"B. {row['option_b']}")
        print(f"Answer: {row['answer']}")
        print(
            "Pinyin: "
            f"answer={row['answer_pinyin']} | distractor={row['distractor_pinyin']} | "
            f"collapsed={row['diacritic_collapsed']}"
        )
        print(f"Source: {row['context_source']}:{row['line_index']}")
        print()


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    root = Path.cwd()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    context_corpora = args.context_corpus or [
        "data/raw/test.zh.txt",
        "../1.Tokenization/corpora/chinese_origin_中国_test10.txt",
    ]
    context_paths = [project_path(root, path) for path in context_corpora]
    collision_csv = project_path(root, args.collision_csv)

    print(f"collision_csv: {collision_csv}")
    print("context corpora:")
    for path in context_paths:
        print(f"  {path}")
        if not path.exists():
            raise FileNotFoundError(f"Missing context corpus: {path}")

    collision_rows, candidate_tokens = load_collision_pairs(
        collision_csv,
        args.min_cjk_chars,
        args.max_cjk_chars,
        args.exclude_chars,
    )
    print(f"collision groups after filtering: {len(collision_rows)}")
    print(f"candidate Chinese tokens: {len(candidate_tokens)}")

    context_lines = load_context_lines(context_paths)
    counts, contexts_by_token = scan_token_counts_and_contexts(
        lines=context_lines,
        tokens=candidate_tokens,
        max_contexts_per_token=max(3, args.contexts_per_pair),
        min_prefix_chars=args.min_prefix_chars,
        max_prefix_chars=args.max_prefix_chars,
        show_progress=not args.no_progress,
    )
    items = build_probe_items(
        collision_rows=collision_rows,
        counts=counts,
        contexts_by_token=contexts_by_token,
        max_pairs=args.max_pairs,
        min_pairs=args.min_pairs,
        contexts_per_pair=args.contexts_per_pair,
    )
    print(f"probe pairs: {len(set(item['pair_id'] for item in items))}")
    print(f"probe items: {len(items)}")
    print(f"diacritic-collapsed items: {sum(1 for item in items if item['diacritic_collapsed'])}")

    item_rows = [{column: item.get(column, "") for column in ITEM_COLUMNS} for item in items]
    write_csv(output_dir / "items.csv", item_rows, ITEM_COLUMNS)
    with (output_dir / "items.jsonl").open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    questions = question_rows(items)
    write_csv(output_dir / "questions.csv", questions, QUESTION_COLUMNS)
    write_questions_txt(output_dir / "questions.txt", questions)
    if args.print_questions:
        print_questions(questions)

    device, dtype, dtype_name = choose_device_and_dtype()
    print(f"device: {device.type}, dtype: {dtype_name}")
    all_predictions: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for run in MODEL_RUNS:
        predictions, summary = evaluate_model_run(
            root=root,
            run=run,
            items=items,
            device=device,
            dtype=dtype,
            dtype_name=dtype_name,
            show_progress=not args.no_progress,
        )
        all_predictions.extend(predictions)
        summaries.append(summary)
        print(
            f"{run.run_name}: collapse_rate={summary['collapse_rate']}, "
            f"resolvable_accuracy={summary['resolvable_accuracy']}, "
            f"mean_margin={summary['mean_margin']}"
        )

    write_csv(output_dir / "predictions.csv", all_predictions, PRED_COLUMNS)
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in all_predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_csv(output_dir / "summary.csv", summaries, SUMMARY_COLUMNS)
    (output_dir / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote outputs to: {output_dir}")


if __name__ == "__main__":
    main()
