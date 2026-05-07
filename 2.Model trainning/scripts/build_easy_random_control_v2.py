#!/usr/bin/env python3
"""Build an easy random non-homophone control probe from Homophone Probe v2 contexts."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from build_nonhomophone_control_v2 import (
    Candidate,
    DEFAULT_COLLISION_CSV,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EXCLUDE_CHARS,
    DEFAULT_EXCLUDE_TOKENS,
    DEFAULT_RAW_TEST,
    DEFAULT_SOURCE_JSONL,
    count_token_frequency,
    index_candidates_by_length,
    is_valid_nonhomophone,
    join_diacritic,
    length_matched_candidates,
    load_embeddings,
    load_source_items,
    path_looks_like_train_split,
    project_path,
    read_collision_resource,
    to_diacritic,
    to_toneless,
)
from probe_build_common_v2 import cosine_distance, summary_stats as stats


DEFAULT_OUTPUT_DIR = "eval_data/easy_random_control_v2"

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
    "embedding_distance_if_available",
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
    "embedding_distance_if_available",
    "source_line",
    "quality_flags",
    "selection_reason",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Easy Random Non-Homophone Control v2.")
    parser.add_argument("--source-jsonl", default=DEFAULT_SOURCE_JSONL)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--collision-csv", default=DEFAULT_COLLISION_CSV)
    parser.add_argument("--frequency-corpus", default=DEFAULT_RAW_TEST)
    parser.add_argument("--seed", type=int, default=42)
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
    parser.add_argument("--min-analysis-distance", type=float, default=0.10)
    parser.add_argument("--max-items-per-distractor", type=int, default=5)
    parser.add_argument("--print-random-examples", type=int, default=20)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def choose_random_distractor(
    item: dict[str, Any],
    candidates_by_length: dict[int, list[Candidate]],
    token_to_keys: dict[str, set[str]],
    embeddings: dict[str, Any],
    per_distractor: Counter[str],
    rng: random.Random,
    args: argparse.Namespace,
) -> tuple[dict[str, Any] | None, int, int]:
    gold = item["gold_zh"]
    source_collision_key = item["collision_key"]
    gold_diacritic = item.get("gold_pinyin_diacritic") or to_diacritic(gold)
    gold_toneless = to_toneless(gold)
    considered = 0
    tiny_distance_filtered = 0
    eligible_same: list[dict[str, Any]] = []
    eligible_relaxed: list[dict[str, Any]] = []

    for candidate in length_matched_candidates(candidates_by_length, gold):
        if per_distractor[candidate.token] >= args.max_items_per_distractor:
            continue
        if not is_valid_nonhomophone(
            gold, candidate, source_collision_key, token_to_keys, gold_diacritic, gold_toneless
        ):
            continue
        considered += 1
        distance: float | str = ""
        if embeddings and gold in embeddings and candidate.token in embeddings:
            distance = cosine_distance(embeddings[gold], embeddings[candidate.token])
            if distance < args.min_analysis_distance:
                tiny_distance_filtered += 1
                continue
        row = {
            "candidate": candidate,
            "distance": distance,
            "quality_flags": ["same_length"] if len(candidate.token) == len(gold) else ["length_diff_1", "length_relaxed"],
        }
        if len(candidate.token) == len(gold):
            eligible_same.append(row)
        else:
            eligible_relaxed.append(row)

    pool = eligible_same if eligible_same else eligible_relaxed
    if not pool:
        return None, considered, tiny_distance_filtered
    return rng.choice(pool), considered, tiny_distance_filtered


def build_items(
    source_items: list[dict[str, Any]],
    candidates: list[Candidate],
    token_to_keys: dict[str, set[str]],
    embeddings: dict[str, Any],
    frequencies: Counter[str],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(args.seed)
    items: list[dict[str, Any]] = []
    per_distractor: Counter[str] = Counter()
    candidates_by_length = index_candidates_by_length(candidates)
    build_stats = {
        "candidate_pairs_considered": 0,
        "items_skipped": 0,
        "tiny_distance_filtered": 0,
        "items_requiring_length_relaxation": 0,
        "invalid_nonhomophone_after_selection": 0,
        "distance_not_used_for_medium_selection": True,
    }

    for source in source_items[: args.target_items]:
        selected, considered, tiny_filtered = choose_random_distractor(
            source, candidates_by_length, token_to_keys, embeddings, per_distractor, rng, args
        )
        build_stats["candidate_pairs_considered"] += considered
        build_stats["tiny_distance_filtered"] += tiny_filtered
        if selected is None:
            build_stats["items_skipped"] += 1
            continue

        distractor = selected["candidate"]
        gold = source["gold_zh"]
        gold_diacritic = source["gold_pinyin_diacritic"]
        distractor_diacritic = distractor.diacritic
        gold_toneless = to_toneless(gold)
        distractor_toneless = distractor.toneless
        verified = (
            distractor.token != gold
            and distractor_diacritic != gold_diacritic
            and distractor_toneless != gold_toneless
            and source["collision_key"] not in distractor.collision_keys
        )
        if not verified:
            build_stats["invalid_nonhomophone_after_selection"] += 1
            continue
        flags = list(selected["quality_flags"])
        if "length_relaxed" in flags:
            build_stats["items_requiring_length_relaxation"] += 1
        per_distractor[distractor.token] += 1
        item_id = f"ercv2_{len(items):04d}"
        items.append(
            {
                "id": item_id,
                "source_homophone_item_id": source["id"],
                "source_file": source["source_file"],
                "source_line": source["source_line"],
                "gold_zh": gold,
                "distractor_zh": distractor.token,
                "gold_pinyin_diacritic": gold_diacritic,
                "distractor_pinyin_diacritic": distractor_diacritic,
                "gold_toneless_pinyin": gold_toneless,
                "distractor_toneless_pinyin": distractor_toneless,
                "nonhomophone_verified": verified,
                "prefix_zh": source["prefix_zh"],
                "suffix_zh": source["suffix_zh"],
                "context_zh_with_blank": source["context_zh_with_blank"],
                "full_gold_sentence_zh": source["full_gold_sentence_zh"],
                "full_distractor_sentence_zh": f"{source['prefix_zh']}{distractor.token}{source['suffix_zh']}",
                "prefix_diacritic": source["prefix_diacritic"],
                "suffix_diacritic": source["suffix_diacritic"],
                "full_gold_sentence_diacritic": source["full_gold_sentence_diacritic"],
                "full_distractor_sentence_diacritic": join_diacritic(
                    source["prefix_diacritic"], distractor_diacritic, source["suffix_diacritic"]
                ),
                "embedding_distance_if_available": selected["distance"],
                "gold_frequency_if_available": frequencies.get(gold, ""),
                "distractor_frequency_if_available": distractor.frequency,
                "quality_flags": flags,
                "selection_reason": (
                    f"easy random non-homophone control; seed={args.seed}; "
                    "distance_not_used_for_medium_selection=true"
                ),
            }
        )
    return items, build_stats


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

    source_items = load_source_items(source_path)[: args.target_items]
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
    invalid = [
        item
        for item in items
        if item["gold_pinyin_diacritic"] == item["distractor_pinyin_diacritic"]
        or item["gold_toneless_pinyin"] == item["distractor_toneless_pinyin"]
        or not item["nonhomophone_verified"]
    ]
    if invalid:
        raise ValueError(f"Invalid easy control collisions: {[item['id'] for item in invalid[:5]]}")

    distances = [
        float(item["embedding_distance_if_available"])
        for item in items
        if item["embedding_distance_if_available"] != ""
    ]
    distractor_freqs = [
        float(item["distractor_frequency_if_available"])
        for item in items
        if item["distractor_frequency_if_available"] != ""
    ]
    distance_summary = stats(distances)
    frequency_summary = stats(distractor_freqs)

    jsonl_path = output_dir / "easy_random_control_v2.jsonl"
    review_path = output_dir / "easy_random_control_v2_review.csv"
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
        "distance_summary": distance_summary,
        "distractor_frequency_summary": frequency_summary,
        "build_stats": build_stats,
        "invalid_control_collision_count": len(invalid),
    }
    (output_dir / "easy_random_control_v2_build_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"wrote: {jsonl_path}")
    print(f"wrote: {review_path}")
    print(f"built items: {len(items)}")
    print(f"skipped items: {build_stats['items_skipped']}")
    print(f"invalid Diacritic collisions: {len(invalid)}")
    print(f"candidate pairs considered: {build_stats['candidate_pairs_considered']}")
    print(f"tiny-distance filters (<{args.min_analysis_distance}): {build_stats['tiny_distance_filtered']}")
    print(
        "embedding distance summary: "
        f"mean={distance_summary['mean']} median={distance_summary['median']} std={distance_summary['std']} "
        f"min={distance_summary['min']} max={distance_summary['max']}"
    )
    print(
        "distractor frequency summary: "
        f"mean={frequency_summary['mean']} median={frequency_summary['median']} std={frequency_summary['std']} "
        f"min={frequency_summary['min']} max={frequency_summary['max']}"
    )
    sample_count = min(args.print_random_examples, len(items))
    print(f"random examples ({sample_count}):")
    for item in random.Random(args.seed).sample(items, sample_count):
        print(
            f"  {item['id']} <- {item['source_homophone_item_id']} "
            f"{item['context_zh_with_blank']} :: {item['gold_zh']} / {item['distractor_zh']} "
            f"dist={item['embedding_distance_if_available']}"
        )


if __name__ == "__main__":
    main()
