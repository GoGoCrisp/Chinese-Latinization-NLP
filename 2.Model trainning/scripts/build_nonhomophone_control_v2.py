#!/usr/bin/env python3
"""Build a matched non-homophone control probe from Homophone Probe v2 contexts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from probe_build_common_v2 import (
    DEFAULT_COLLISION_CSV,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EXCLUDE_CHARS,
    DEFAULT_EXCLUDE_TOKENS,
    DEFAULT_RAW_TEST,
    TokenTrie,
    cosine_distance,
    join_diacritic,
    load_embeddings,
    path_looks_like_train_split,
    project_path,
    summary_stats as distance_stats,
    to_diacritic,
    to_toneless,
    token_is_clean_lexical,
)

DEFAULT_SOURCE_JSONL = "eval_data/homophone_probe_v2/probe_v2.jsonl"
DEFAULT_OUTPUT_DIR = "eval_data/nonhomophone_control_v2"

JSONL_FIELDS = [
    "id",
    "source_homophone_item_id",
    "source_file",
    "source_line",
    "gold_zh",
    "distractor_zh",
    "gold_pinyin_diacritic",
    "distractor_pinyin_diacritic",
    "gold_toneless_pinyin",
    "distractor_toneless_pinyin",
    "nonhomophone_verified",
    "prefix_zh",
    "suffix_zh",
    "context_zh_with_blank",
    "full_gold_sentence_zh",
    "full_distractor_sentence_zh",
    "prefix_diacritic",
    "suffix_diacritic",
    "full_gold_sentence_diacritic",
    "full_distractor_sentence_diacritic",
    "embedding_distance",
    "gold_frequency_if_available",
    "distractor_frequency_if_available",
    "quality_flags",
    "selection_reason",
]

REVIEW_FIELDS = [
    "id",
    "context_zh_with_blank",
    "gold_zh",
    "distractor_zh",
    "gold_pinyin_diacritic",
    "distractor_pinyin_diacritic",
    "gold_toneless_pinyin",
    "distractor_toneless_pinyin",
    "embedding_distance",
    "source_line",
    "quality_flags",
    "selection_reason",
]


@dataclass(frozen=True)
class Candidate:
    token: str
    frequency: int
    diacritic: str
    toneless: str
    collision_keys: frozenset[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build matched Non-Homophone Control Probe v2.")
    parser.add_argument("--source-jsonl", default=DEFAULT_SOURCE_JSONL)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--collision-csv", default=DEFAULT_COLLISION_CSV)
    parser.add_argument("--frequency-corpus", default=DEFAULT_RAW_TEST)
    parser.add_argument("--seed", type=int, default=20260506)
    parser.add_argument("--target-items", type=int, default=1000)
    parser.add_argument("--min-cjk-chars", type=int, default=2)
    parser.add_argument("--max-cjk-chars", type=int, default=4)
    parser.add_argument("--min-token-frequency", type=int, default=2)
    parser.add_argument("--exclude-chars", default=DEFAULT_EXCLUDE_CHARS)
    parser.add_argument("--exclude-tokens", default=DEFAULT_EXCLUDE_TOKENS)
    parser.add_argument("--exclude-numeral-tokens", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-batch-size", type=int, default=128)
    parser.add_argument("--prefer-local-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-no-embedding", action="store_true")
    parser.add_argument("--primary-min-distance", type=float, default=0.35)
    parser.add_argument("--primary-max-distance", type=float, default=0.65)
    parser.add_argument("--wide-min-distance", type=float, default=0.25)
    parser.add_argument("--wide-max-distance", type=float, default=0.75)
    parser.add_argument("--target-distance", type=float, default=0.475)
    parser.add_argument("--max-items-per-distractor", type=int, default=5)
    parser.add_argument("--print-random-examples", type=int, default=20)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def load_source_items(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                items.append(json.loads(line))
    ids = [item["id"] for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate source homophone item ids")
    return items


def read_collision_resource(path: Path, args: argparse.Namespace) -> tuple[set[str], dict[str, set[str]]]:
    tokens: set[str] = set()
    token_to_keys: dict[str, set[str]] = defaultdict(set)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = row["b_token"]
            for token in json.loads(row["a_tokens_json"]):
                token = token.strip()
                if not token_is_clean_lexical(token, args):
                    continue
                tokens.add(token)
                token_to_keys[token].add(key)
    return tokens, token_to_keys


def count_token_frequency(corpus_path: Path, tokens: set[str], show_progress: bool) -> Counter[str]:
    counts: Counter[str] = Counter()
    trie = TokenTrie(tokens)
    with corpus_path.open("r", encoding="utf-8") as handle:
        iterator = tqdm(handle, desc="count control pool frequency", disable=not show_progress)
        for line in iterator:
            text = line.rstrip("\n")
            seen: set[tuple[str, int]] = set()
            for start, token in trie.find(text):
                key = (token, start)
                if key in seen:
                    continue
                seen.add(key)
                counts[token] += 1
    return counts


def is_valid_nonhomophone(
    gold: str,
    candidate: Candidate,
    source_collision_key: str,
    token_to_keys: dict[str, set[str]],
    gold_diacritic: str | None = None,
    gold_toneless: str | None = None,
) -> bool:
    if candidate.token == gold:
        return False
    gold_diacritic = gold_diacritic if gold_diacritic is not None else to_diacritic(gold)
    gold_toneless = gold_toneless if gold_toneless is not None else to_toneless(gold)
    if candidate.diacritic == gold_diacritic:
        return False
    if candidate.toneless == gold_toneless:
        return False
    if source_collision_key in candidate.collision_keys:
        return False
    if token_to_keys.get(gold, set()) & set(candidate.collision_keys):
        return False
    return True


def index_candidates_by_length(candidates: list[Candidate]) -> dict[int, list[Candidate]]:
    by_length: dict[int, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_length[len(candidate.token)].append(candidate)
    return by_length


def length_matched_candidates(
    candidates_by_length: dict[int, list[Candidate]],
    gold: str,
) -> list[Candidate]:
    gold_len = len(gold)
    return [
        candidate
        for length in (gold_len, gold_len - 1, gold_len + 1)
        for candidate in candidates_by_length.get(length, [])
    ]


def band_score(distance: float, low: float, high: float, target: float) -> float:
    left_span = max(target - low, 1e-6)
    right_span = max(high - target, 1e-6)
    span = left_span if distance <= target else right_span
    return max(0.0, 1.0 - abs(distance - target) / span)


def choose_distractor(
    item: dict[str, Any],
    candidates_by_length: dict[int, list[Candidate]],
    token_to_keys: dict[str, set[str]],
    embeddings: dict[str, Any],
    frequencies: Counter[str],
    per_distractor: Counter[str],
    args: argparse.Namespace,
) -> tuple[dict[str, Any] | None, int]:
    gold = item["gold_zh"]
    source_collision_key = item["collision_key"]
    gold_diacritic = item.get("gold_pinyin_diacritic") or to_diacritic(gold)
    gold_toneless = to_toneless(gold)
    gold_freq = 0
    considered = 0
    base_rows: list[dict[str, Any]] = []
    if gold not in embeddings and not args.allow_no_embedding:
        raise ValueError(f"Missing embedding for gold token: {gold}")

    for candidate in length_matched_candidates(candidates_by_length, gold):
        if per_distractor[candidate.token] >= args.max_items_per_distractor:
            continue
        if not is_valid_nonhomophone(
            gold, candidate, source_collision_key, token_to_keys, gold_diacritic, gold_toneless
        ):
            continue
        considered += 1
        flags = []
        if len(candidate.token) == len(gold):
            flags.append("same_length")
        else:
            flags.append("length_diff_1")
        distance = ""
        if embeddings:
            if candidate.token not in embeddings:
                continue
            distance = cosine_distance(embeddings[gold], embeddings[candidate.token])
        base_rows.append(
            {
                "candidate": candidate,
                "distance": distance,
                "quality_flags": flags,
            }
        )

    if not base_rows:
        return None, considered

    gold_freq = frequencies.get(gold, 0)

    phases = [
        ("primary_exact_length", args.primary_min_distance, args.primary_max_distance, True),
        ("primary_len_plusminus1", args.primary_min_distance, args.primary_max_distance, False),
        ("wide_exact_length", args.wide_min_distance, args.wide_max_distance, True),
        ("wide_len_plusminus1", args.wide_min_distance, args.wide_max_distance, False),
    ]
    for phase, low, high, require_same_length in phases:
        phase_rows = []
        for row in base_rows:
            candidate = row["candidate"]
            if require_same_length and len(candidate.token) != len(gold):
                continue
            distance = row["distance"]
            if distance != "" and not (low <= float(distance) <= high):
                continue
            target_freq = max(gold_freq, 1)
            freq_penalty = abs(math.log(candidate.frequency + 1) - math.log(target_freq + 1)) if gold_freq else 0.0
            distance_value = float(distance) if distance != "" else args.target_distance
            score = (
                band_score(distance_value, low, high, args.target_distance) * 100.0
                - freq_penalty * 3.0
                + math.log(candidate.frequency + 1)
                - per_distractor[candidate.token] * 4.0
                + (5.0 if len(candidate.token) == len(gold) else 0.0)
            )
            quality_flags = list(row["quality_flags"])
            if phase.startswith("wide"):
                quality_flags.append("widened_distance_threshold")
            if not require_same_length:
                quality_flags.append("length_relaxed")
            phase_rows.append(
                {
                    "candidate": candidate,
                    "distance": distance,
                    "quality_flags": quality_flags,
                    "phase": phase,
                    "score": score,
                }
            )
        if phase_rows:
            phase_rows.sort(key=lambda row: (row["score"], row["candidate"].frequency, row["candidate"].token), reverse=True)
            return phase_rows[0], considered
    return None, considered


def build_items(
    source_items: list[dict[str, Any]],
    candidates: list[Candidate],
    token_to_keys: dict[str, set[str]],
    embeddings: dict[str, Any],
    frequencies: Counter[str],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output: list[dict[str, Any]] = []
    per_distractor: Counter[str] = Counter()
    candidates_by_length = index_candidates_by_length(candidates)
    stats: dict[str, Any] = {
        "candidate_pairs_considered": 0,
        "items_skipped": 0,
        "items_requiring_widened_thresholds": 0,
        "items_requiring_length_relaxation": 0,
        "invalid_nonhomophone_after_selection": 0,
    }
    for item in source_items[: args.target_items]:
        selected, considered = choose_distractor(
            item, candidates_by_length, token_to_keys, embeddings, frequencies, per_distractor, args
        )
        stats["candidate_pairs_considered"] += considered
        if selected is None:
            stats["items_skipped"] += 1
            continue

        distractor = selected["candidate"]
        gold = item["gold_zh"]
        gold_diacritic = item["gold_pinyin_diacritic"]
        gold_toneless = to_toneless(gold)
        distractor_diacritic = distractor.diacritic
        distractor_toneless = distractor.toneless
        verified = (
            distractor.token != gold
            and distractor_diacritic != gold_diacritic
            and distractor_toneless != gold_toneless
            and item["collision_key"] not in distractor.collision_keys
        )
        if not verified:
            stats["invalid_nonhomophone_after_selection"] += 1
            continue

        flags = list(selected["quality_flags"])
        if "widened_distance_threshold" in flags:
            stats["items_requiring_widened_thresholds"] += 1
        if "length_relaxed" in flags:
            stats["items_requiring_length_relaxation"] += 1
        per_distractor[distractor.token] += 1

        prefix_diacritic = item["prefix_diacritic"]
        suffix_diacritic = item["suffix_diacritic"]
        control_id = f"nhcv2_{len(output):04d}"
        output.append(
            {
                "id": control_id,
                "source_homophone_item_id": item["id"],
                "source_file": item["source_file"],
                "source_line": item["source_line"],
                "gold_zh": gold,
                "distractor_zh": distractor.token,
                "gold_pinyin_diacritic": gold_diacritic,
                "distractor_pinyin_diacritic": distractor_diacritic,
                "gold_toneless_pinyin": gold_toneless,
                "distractor_toneless_pinyin": distractor_toneless,
                "nonhomophone_verified": verified,
                "prefix_zh": item["prefix_zh"],
                "suffix_zh": item["suffix_zh"],
                "context_zh_with_blank": item["context_zh_with_blank"],
                "full_gold_sentence_zh": item["full_gold_sentence_zh"],
                "full_distractor_sentence_zh": f"{item['prefix_zh']}{distractor.token}{item['suffix_zh']}",
                "prefix_diacritic": prefix_diacritic,
                "suffix_diacritic": suffix_diacritic,
                "full_gold_sentence_diacritic": item["full_gold_sentence_diacritic"],
                "full_distractor_sentence_diacritic": join_diacritic(prefix_diacritic, distractor_diacritic, suffix_diacritic),
                "embedding_distance": selected["distance"],
                "gold_frequency_if_available": frequencies.get(gold, ""),
                "distractor_frequency_if_available": distractor.frequency,
                "quality_flags": flags,
                "selection_reason": (
                    f"matched-context non-homophone control; phase={selected['phase']}; "
                    f"score={selected['score']:.3f}; source_homophone_item={item['id']}"
                ),
            }
        )
    return output, stats


def write_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps({field: item[field] for field in JSONL_FIELDS}, ensure_ascii=False) + "\n")


def write_review_csv(path: Path, items: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        for item in items:
            row = {field: item[field] for field in REVIEW_FIELDS if field != "quality_flags"}
            row["quality_flags"] = ";".join(item["quality_flags"])
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    root = Path.cwd()
    source_path = project_path(root, args.source_jsonl)
    output_dir = project_path(root, args.output_dir)
    collision_csv = project_path(root, args.collision_csv)
    frequency_corpus = project_path(root, args.frequency_corpus)
    output_dir.mkdir(parents=True, exist_ok=True)
    if path_looks_like_train_split(source_path) or path_looks_like_train_split(frequency_corpus):
        raise ValueError(f"Refusing to use train data: {source_path}, {frequency_corpus}")

    source_items = load_source_items(source_path)
    source_items = source_items[: args.target_items]
    print(f"source homophone items: {len(source_items)}")
    print(f"collision resource: {collision_csv}")
    print(f"frequency corpus: {frequency_corpus}")

    pool_tokens, token_to_keys = read_collision_resource(collision_csv, args)
    gold_tokens = {item["gold_zh"] for item in source_items}
    count_tokens = set(pool_tokens) | gold_tokens
    print(f"candidate pool tokens before frequency filter: {len(pool_tokens)}")
    frequencies = count_token_frequency(frequency_corpus, count_tokens, show_progress=not args.no_progress)

    candidate_tokens = sorted(token for token in pool_tokens if frequencies[token] >= args.min_token_frequency)
    print(f"candidate pool tokens after frequency filter: {len(candidate_tokens)}")
    embedding_tokens = sorted(set(candidate_tokens) | gold_tokens)
    embeddings, embedding_meta = load_embeddings(embedding_tokens, args)

    candidates = [
        Candidate(
            token=token,
            frequency=frequencies[token],
            diacritic=to_diacritic(token),
            toneless=to_toneless(token),
            collision_keys=frozenset(token_to_keys.get(token, set())),
        )
        for token in candidate_tokens
    ]

    items, build_stats = build_items(source_items, candidates, token_to_keys, embeddings, frequencies, args)
    distances = [float(item["embedding_distance"]) for item in items if item["embedding_distance"] != ""]
    dist_stats = distance_stats(distances)
    invalid_collisions = [
        item
        for item in items
        if item["gold_pinyin_diacritic"] == item["distractor_pinyin_diacritic"]
        or item["gold_toneless_pinyin"] == item["distractor_toneless_pinyin"]
        or not item["nonhomophone_verified"]
    ]
    if invalid_collisions:
        raise ValueError(f"Invalid non-homophone controls found: {[item['id'] for item in invalid_collisions[:5]]}")

    jsonl_path = output_dir / "nonhomophone_control_v2.jsonl"
    review_path = output_dir / "nonhomophone_control_v2_review.csv"
    write_jsonl(jsonl_path, items)
    write_review_csv(review_path, items)

    meta = {
        "source_jsonl": str(source_path),
        "collision_resource": str(collision_csv),
        "frequency_corpus": str(frequency_corpus),
        "no_train_data_used": not path_looks_like_train_split(source_path) and not path_looks_like_train_split(frequency_corpus),
        "source_items": len(source_items),
        "items_built": len(items),
        "candidate_pool_tokens_before_frequency_filter": len(pool_tokens),
        "candidate_pool_tokens_after_frequency_filter": len(candidate_tokens),
        "embedding": embedding_meta,
        "distance_summary": dist_stats,
        "build_stats": build_stats,
        "invalid_control_collision_count": len(invalid_collisions),
    }
    (output_dir / "nonhomophone_control_v2_build_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"wrote: {jsonl_path}")
    print(f"wrote: {review_path}")
    print(f"dataset size: {len(items)}")
    print(f"candidate pairs considered: {build_stats['candidate_pairs_considered']}")
    print(f"items requiring widened thresholds: {build_stats['items_requiring_widened_thresholds']}")
    print(f"items skipped: {build_stats['items_skipped']}")
    print(
        "embedding distance summary: "
        f"mean={dist_stats['mean']} median={dist_stats['median']} std={dist_stats['std']} "
        f"min={dist_stats['min']} max={dist_stats['max']}"
    )
    print(f"invalid control collisions: {len(invalid_collisions)}")

    sample_count = min(args.print_random_examples, len(items))
    print(f"random examples ({sample_count}):")
    for item in random.Random(args.seed).sample(items, sample_count):
        print(
            f"  {item['id']} <- {item['source_homophone_item_id']} "
            f"{item['context_zh_with_blank']} :: {item['gold_zh']} / {item['distractor_zh']} "
            f"dist={item['embedding_distance']}"
        )


if __name__ == "__main__":
    main()
