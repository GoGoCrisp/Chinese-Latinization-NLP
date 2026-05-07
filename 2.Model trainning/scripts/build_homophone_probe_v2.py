#!/usr/bin/env python3
"""Build Homophone Disambiguation Probe v2."""

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
    BOUNDARY_CHARS,
    DEFAULT_COLLISION_CSV,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EXCLUDE_CHARS,
    DEFAULT_EXCLUDE_TOKENS,
    DEFAULT_RAW_TEST,
    TokenTrie,
    join_diacritic,
    path_looks_like_train_split,
    project_path,
    read_semantic_pair_distances,
    resolve_local_hf_snapshot,
    to_diacritic,
    token_is_clean_lexical,
)

DEFAULT_SEMANTIC_CSV = (
    "../1.Tokenization/decoded_superTokenizers_2048_subset100k/11th_semantic_dispersion/"
    "11th_semantic_dispersion_collision_embeddings_ab/"
    "BAAI_bge-small-zh-v1.5_target_groups.csv"
)
DEFAULT_OUTPUT_DIR = "eval_data/homophone_probe_v2"

JSONL_FIELDS = [
    "id",
    "source_file",
    "source_line",
    "collision_key",
    "gold_zh",
    "distractor_zh",
    "gold_pinyin_diacritic",
    "distractor_pinyin_diacritic",
    "collapsed_diacritic",
    "prefix_zh",
    "suffix_zh",
    "context_zh_with_blank",
    "full_gold_sentence_zh",
    "full_distractor_sentence_zh",
    "prefix_diacritic",
    "suffix_diacritic",
    "full_gold_sentence_diacritic",
    "full_distractor_sentence_diacritic",
    "selection_reason",
    "embedding_distance_if_available",
    "quality_flags",
]

REVIEW_FIELDS = [
    "id",
    "context_zh_with_blank",
    "gold_zh",
    "distractor_zh",
    "gold_pinyin_diacritic",
    "distractor_pinyin_diacritic",
    "collapsed_diacritic",
    "source_line",
    "collision_key",
    "embedding_distance_if_available",
    "quality_flags",
    "selection_reason",
]


@dataclass(frozen=True)
class ContextHit:
    source_file: str
    source_line: int
    text: str
    start: int
    token: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Homophone Probe v2 dataset.")
    parser.add_argument("--collision-csv", default=DEFAULT_COLLISION_CSV)
    parser.add_argument("--semantic-csv", default=DEFAULT_SEMANTIC_CSV)
    parser.add_argument("--context-corpus", action="append", default=[DEFAULT_RAW_TEST])
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-items", type=int, default=1000)
    parser.add_argument("--min-items", type=int, default=800)
    parser.add_argument("--seed", type=int, default=20260504)
    parser.add_argument("--min-cjk-chars", type=int, default=2)
    parser.add_argument("--max-cjk-chars", type=int, default=4)
    parser.add_argument("--exclude-chars", default=DEFAULT_EXCLUDE_CHARS)
    parser.add_argument(
        "--exclude-tokens",
        default=DEFAULT_EXCLUDE_TOKENS,
        help="Comma-separated exact candidate tokens to drop as function-word or list-index junk.",
    )
    parser.add_argument("--exclude-numeral-tokens", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-token-frequency", type=int, default=2)
    parser.add_argument("--min-prefix-chars", type=int, default=4)
    parser.add_argument("--min-suffix-chars", type=int, default=3)
    parser.add_argument("--max-prefix-chars", type=int, default=180)
    parser.add_argument("--max-suffix-chars", type=int, default=180)
    parser.add_argument("--max-items-per-source-line", type=int, default=1)
    parser.add_argument("--max-items-per-collision-key", type=int, default=5)
    parser.add_argument("--max-items-per-gold", type=int, default=3)
    parser.add_argument("--same-length-bonus", type=float, default=5.0)
    parser.add_argument("--collapsed-bonus", type=float, default=1.0)
    parser.add_argument("--no-embedding", action="store_true", help="Disable text embedding distances.")
    parser.add_argument("--require-embedding", action="store_true", help="Fail instead of falling back if embeddings cannot load.")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-batch-size", type=int, default=128)
    parser.add_argument("--prefer-local-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--embedding-min-distance", type=float, default=0.20)
    parser.add_argument("--embedding-max-distance", type=float, default=0.75)
    parser.add_argument("--embedding-target-distance", type=float, default=0.45)
    parser.add_argument("--embedding-weight", type=float, default=20.0)
    parser.add_argument(
        "--allow-embedding-outside-band",
        action="store_true",
        help="Keep non-collapsed pairs outside the preferred semantic-distance band, with a quality flag.",
    )
    parser.add_argument("--print-random-examples", type=int, default=10)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def read_collision_groups(collision_csv: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with collision_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            tokens = sorted(
                set(
                    token.strip()
                    for token in json.loads(row["a_tokens_json"])
                    if token_is_clean_lexical(token.strip(), args)
                )
            )
            if len(tokens) >= 2:
                rows.append({"collision_key": row["b_token"], "tokens": tokens})
    return rows


def compute_embedding_pair_distances(
    groups: list[dict[str, Any]],
    counts: Counter[str],
    args: argparse.Namespace,
) -> tuple[dict[tuple[str, str], float], dict[str, Any]]:
    meta: dict[str, Any] = {
        "enabled": not args.no_embedding,
        "model": args.embedding_model,
        "model_path": None,
        "token_count": 0,
        "pair_distance_count": 0,
        "status": "disabled" if args.no_embedding else "not_started",
        "warning": None,
    }
    if args.no_embedding:
        return {}, meta

    tokens = sorted(
        {
            token
            for group in groups
            for token in group["tokens"]
            if counts[token] >= args.min_token_frequency
        }
    )
    meta["token_count"] = len(tokens)
    if not tokens:
        meta["status"] = "no_tokens"
        return {}, meta

    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        meta["status"] = "import_failed"
        meta["warning"] = str(exc)
        if args.require_embedding:
            raise
        print(f"WARNING: embeddings unavailable; falling back to semantic cache only: {exc}")
        return {}, meta

    model_path = resolve_local_hf_snapshot(args.embedding_model) if args.prefer_local_cache else args.embedding_model
    meta["model_path"] = model_path
    try:
        if model_path != args.embedding_model:
            print(f"embedding model: {args.embedding_model}")
            print(f"using local cache snapshot: {model_path}")
        else:
            print(f"embedding model: {model_path}")
        model = SentenceTransformer(model_path)
        vectors = model.encode(
            tokens,
            normalize_embeddings=True,
            batch_size=args.embedding_batch_size,
            show_progress_bar=not args.no_progress,
        )
    except Exception as exc:
        meta["status"] = "load_or_encode_failed"
        meta["warning"] = str(exc)
        if args.require_embedding:
            raise
        print(f"WARNING: embeddings failed; falling back to semantic cache only: {exc}")
        return {}, meta

    vectors = np.asarray(vectors, dtype=np.float32)
    embeddings = dict(zip(tokens, vectors))
    distances: dict[tuple[str, str], float] = {}
    for group in groups:
        group_tokens = [
            token
            for token in group["tokens"]
            if token in embeddings and counts[token] >= args.min_token_frequency
        ]
        for i, left in enumerate(group_tokens):
            left_vector = embeddings[left]
            for right in group_tokens[i + 1 :]:
                distance = float(1.0 - np.dot(left_vector, embeddings[right]))
                if not math.isfinite(distance):
                    continue
                distances[tuple(sorted((left, right)))] = max(0.0, min(2.0, distance))

    meta["pair_distance_count"] = len(distances)
    meta["status"] = "ok"
    return distances, meta


def load_lines(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                text = line.rstrip("\n")
                if text:
                    records.append({"source_file": str(path), "source_line": index, "text": text})
    return records


def scan_contexts(
    lines: list[dict[str, Any]],
    tokens: set[str],
    show_progress: bool,
) -> tuple[Counter[str], dict[str, list[ContextHit]]]:
    trie = TokenTrie(tokens)
    counts: Counter[str] = Counter()
    contexts: dict[str, list[ContextHit]] = defaultdict(list)
    iterator = tqdm(lines, desc="scan raw contexts", disable=not show_progress)
    for record in iterator:
        seen_in_line: set[tuple[str, int]] = set()
        for start, token in trie.find(record["text"]):
            key = (token, start)
            if key in seen_in_line:
                continue
            seen_in_line.add(key)
            counts[token] += 1
            if len(contexts[token]) < 50:
                contexts[token].append(
                    ContextHit(
                        source_file=record["source_file"],
                        source_line=record["source_line"],
                        text=record["text"],
                        start=start,
                        token=token,
                    )
                )
    return counts, contexts


def sentence_window(text: str, start: int, token: str, args: argparse.Namespace) -> tuple[str, str] | None:
    left = start
    while left > 0 and text[left - 1] not in BOUNDARY_CHARS:
        left -= 1
    right = start + len(token)
    while right < len(text) and text[right] not in BOUNDARY_CHARS:
        right += 1
    prefix = text[left:start].strip()
    suffix = text[start + len(token) : right].strip()
    if len(prefix) < args.min_prefix_chars or len(suffix) < args.min_suffix_chars:
        return None
    return prefix[-args.max_prefix_chars :], suffix[: args.max_suffix_chars]


def pair_quality_flags(gold: str, distractor: str, counts: Counter[str], args: argparse.Namespace) -> list[str]:
    flags: list[str] = []
    if abs(len(gold) - len(distractor)) > 1:
        flags.append("length_diff_gt_1")
    if len(gold) == len(distractor):
        flags.append("same_length")
    if counts[gold] < args.min_token_frequency:
        flags.append("low_gold_frequency")
    if counts[distractor] < args.min_token_frequency:
        flags.append("low_distractor_frequency")
    if to_diacritic(gold) == to_diacritic(distractor):
        flags.append("collapsed_diacritic")
    return flags


def build_pair_candidates(
    groups: list[dict[str, Any]],
    counts: Counter[str],
    semantic_distances: dict[tuple[str, str], float],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for group in groups:
        tokens = [token for token in group["tokens"] if counts[token] >= args.min_token_frequency]
        for i, gold in enumerate(tokens):
            for distractor in tokens:
                if gold == distractor:
                    continue
                flags = pair_quality_flags(gold, distractor, counts, args)
                if "length_diff_gt_1" in flags:
                    continue
                pair_key = tuple(sorted((gold, distractor)))
                distance = semantic_distances.get(pair_key)
                distance_score = 0.0
                embedding_band = "unavailable"
                if distance is not None:
                    flags.append("embedding_distance_available")
                    if args.embedding_min_distance <= distance <= args.embedding_max_distance:
                        flags.append("embedding_distance_medium_band")
                        embedding_band = "medium"
                    else:
                        flags.append("embedding_distance_outside_preferred_band")
                        embedding_band = "outside_preferred"
                        if "collapsed_diacritic" not in flags and not args.allow_embedding_outside_band:
                            continue

                    # Prefer medium-near semantic distances, away from synonym and nonsense extremes.
                    left_span = max(args.embedding_target_distance - args.embedding_min_distance, 1e-6)
                    right_span = max(args.embedding_max_distance - args.embedding_target_distance, 1e-6)
                    span = left_span if distance <= args.embedding_target_distance else right_span
                    distance_score = args.embedding_weight * max(
                        0.0,
                        1.0 - abs(distance - args.embedding_target_distance) / span,
                    )
                score = (
                    min(counts[gold], counts[distractor]) * 3
                    + counts[gold]
                    + counts[distractor]
                    + (args.same_length_bonus if len(gold) == len(distractor) else 0.0)
                    + (args.collapsed_bonus if "collapsed_diacritic" in flags else 0.0)
                    + distance_score
                )
                candidates.append(
                    {
                        "collision_key": group["collision_key"],
                        "gold": gold,
                        "distractor": distractor,
                        "score": score,
                        "quality_flags": flags,
                        "embedding_distance_if_available": distance,
                        "embedding_band": embedding_band,
                    }
                )
    candidates.sort(key=lambda row: row["score"], reverse=True)
    return candidates


def build_items(
    root: Path,
    args: argparse.Namespace,
    pair_candidates: list[dict[str, Any]],
    contexts: dict[str, list[ContextHit]],
) -> tuple[list[dict[str, Any]], dict[str, int], Counter[str]]:
    items: list[dict[str, Any]] = []
    duplicate_counts = {
        "exact_duplicate_contexts_removed": 0,
        "source_line_gold_duplicates_removed": 0,
        "source_line_limit_removed": 0,
        "collision_key_limit_removed": 0,
        "gold_limit_removed": 0,
        "short_prefix_or_suffix_removed": 0,
    }
    seen_contexts: set[str] = set()
    seen_source_gold: set[tuple[str, int, str]] = set()
    per_source_line: Counter[tuple[str, int]] = Counter()
    per_collision: Counter[str] = Counter()
    per_gold: Counter[str] = Counter()
    random.Random(args.seed).shuffle(pair_candidates)
    pair_candidates.sort(key=lambda row: row["score"], reverse=True)

    for pair in pair_candidates:
        if len(items) >= args.target_items:
            break
        gold = pair["gold"]
        for hit in contexts.get(gold, []):
            source_line_key = (hit.source_file, hit.source_line)
            source_gold_key = (hit.source_file, hit.source_line, gold)
            if source_gold_key in seen_source_gold:
                duplicate_counts["source_line_gold_duplicates_removed"] += 1
                continue
            if per_source_line[source_line_key] >= args.max_items_per_source_line:
                duplicate_counts["source_line_limit_removed"] += 1
                continue
            if per_collision[pair["collision_key"]] >= args.max_items_per_collision_key:
                duplicate_counts["collision_key_limit_removed"] += 1
                continue
            if per_gold[gold] >= args.max_items_per_gold:
                duplicate_counts["gold_limit_removed"] += 1
                continue
            window = sentence_window(hit.text, hit.start, gold, args)
            if window is None:
                duplicate_counts["short_prefix_or_suffix_removed"] += 1
                continue
            prefix_zh, suffix_zh = window
            context_zh_with_blank = f"{prefix_zh}____{suffix_zh}"
            if context_zh_with_blank in seen_contexts:
                duplicate_counts["exact_duplicate_contexts_removed"] += 1
                continue

            distractor = pair["distractor"]
            gold_pinyin = to_diacritic(gold)
            distractor_pinyin = to_diacritic(distractor)
            prefix_diacritic = to_diacritic(prefix_zh)
            suffix_diacritic = to_diacritic(suffix_zh)
            quality_flags = list(pair["quality_flags"])
            if pair["embedding_distance_if_available"] is None:
                quality_flags.append("embedding_distance_unavailable")

            item = {
                "id": f"hpv2_{len(items):04d}",
                "source_file": str(Path(hit.source_file).relative_to(root)) if Path(hit.source_file).is_relative_to(root) else hit.source_file,
                "source_line": hit.source_line,
                "collision_key": pair["collision_key"],
                "gold_zh": gold,
                "distractor_zh": distractor,
                "gold_pinyin_diacritic": gold_pinyin,
                "distractor_pinyin_diacritic": distractor_pinyin,
                "collapsed_diacritic": gold_pinyin == distractor_pinyin,
                "prefix_zh": prefix_zh,
                "suffix_zh": suffix_zh,
                "context_zh_with_blank": context_zh_with_blank,
                "full_gold_sentence_zh": f"{prefix_zh}{gold}{suffix_zh}",
                "full_distractor_sentence_zh": f"{prefix_zh}{distractor}{suffix_zh}",
                "prefix_diacritic": prefix_diacritic,
                "suffix_diacritic": suffix_diacritic,
                "full_gold_sentence_diacritic": join_diacritic(prefix_diacritic, gold_pinyin, suffix_diacritic),
                "full_distractor_sentence_diacritic": join_diacritic(prefix_diacritic, distractor_pinyin, suffix_diacritic),
                "selection_reason": (
                    f"same AB toneless collision group; score={pair['score']:.3f}; "
                    f"gold_freq_context={len(contexts.get(gold, []))}; "
                    f"embedding_distance={pair['embedding_distance_if_available']}; "
                    f"embedding_band={pair['embedding_band']}"
                ),
                "embedding_distance_if_available": pair["embedding_distance_if_available"],
                "quality_flags": quality_flags,
            }
            items.append(item)
            seen_contexts.add(context_zh_with_blank)
            seen_source_gold.add(source_gold_key)
            per_source_line[source_line_key] += 1
            per_collision[pair["collision_key"]] += 1
            per_gold[gold] += 1
            break
    return items, duplicate_counts, per_collision


def write_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps({field: item[field] for field in JSONL_FIELDS}, ensure_ascii=False) + "\n")


def write_review_csv(path: Path, items: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        for item in items:
            writer.writerow(
                {
                    **{field: item[field] for field in REVIEW_FIELDS if field != "quality_flags"},
                    "quality_flags": ";".join(item["quality_flags"]),
                }
            )


def main() -> None:
    args = parse_args()
    root = Path.cwd()
    output_dir = project_path(root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    collision_csv = project_path(root, args.collision_csv)
    semantic_csv = project_path(root, args.semantic_csv)
    context_paths = [project_path(root, path) for path in args.context_corpus]
    if any(path_looks_like_train_split(path) for path in context_paths):
        raise ValueError(f"Refusing to build probe from train data: {context_paths}")

    print(f"collision source: {collision_csv}")
    print(f"semantic cache source: {semantic_csv if semantic_csv.exists() else 'unavailable'}")
    print("context corpora:")
    for path in context_paths:
        print(f"  {path}")

    groups = read_collision_groups(collision_csv, args)
    candidate_tokens = {token for group in groups for token in group["tokens"]}
    print(f"filtered collision groups: {len(groups)}")
    print(f"candidate Chinese tokens: {len(candidate_tokens)}")

    lines = load_lines(context_paths)
    counts, contexts = scan_contexts(lines, candidate_tokens, show_progress=not args.no_progress)
    cached_semantic_distances = read_semantic_pair_distances(semantic_csv)
    embedding_distances, embedding_meta = compute_embedding_pair_distances(groups, counts, args)
    semantic_distances = dict(cached_semantic_distances)
    semantic_distances.update(embedding_distances)
    print(f"cached semantic pair distances: {len(cached_semantic_distances)}")
    print(f"computed embedding pair distances: {len(embedding_distances)}")
    pair_candidates = build_pair_candidates(groups, counts, semantic_distances, args)
    items, duplicate_counts, per_collision = build_items(root, args, pair_candidates, contexts)

    if len(items) < args.min_items:
        print(f"WARNING: built only {len(items)} items, below requested minimum {args.min_items}")
        print("Likely causes: held-out corpus coverage, lexical filters, and max-one-item-per-source-line dedup.")

    jsonl_path = output_dir / "probe_v2.jsonl"
    review_path = output_dir / "probe_v2_candidates_review.csv"
    write_jsonl(jsonl_path, items)
    write_review_csv(review_path, items)

    meta = {
        "n_items": len(items),
        "collision_source": str(collision_csv),
        "semantic_cache_source": str(semantic_csv) if semantic_csv.exists() else None,
        "cached_semantic_pair_distances": len(cached_semantic_distances),
        "embedding_distances": embedding_meta,
        "embedding_distance_thresholds": {
            "min": args.embedding_min_distance,
            "target": args.embedding_target_distance,
            "max": args.embedding_max_distance,
            "weight": args.embedding_weight,
            "allow_outside_band": args.allow_embedding_outside_band,
            "non_collapsed_outside_band_filtered": not args.allow_embedding_outside_band,
        },
        "context_corpora": [str(path) for path in context_paths],
        "collapsed_items": sum(1 for item in items if item["collapsed_diacritic"]),
        "items_with_embedding_distance": sum(
            1 for item in items if item["embedding_distance_if_available"] is not None
        ),
        "duplicate_counts": duplicate_counts,
        "items_per_collision_key": dict(per_collision.most_common()),
        "no_train_data_used": not any(path_looks_like_train_split(path) for path in context_paths),
    }
    (output_dir / "probe_v2_build_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"wrote: {jsonl_path}")
    print(f"wrote: {review_path}")
    print(f"dataset size: {len(items)}")
    print(f"collapsed items: {meta['collapsed_items']}")
    print(f"items with embedding distance: {meta['items_with_embedding_distance']}")
    print(f"exact duplicate contexts removed: {duplicate_counts['exact_duplicate_contexts_removed']}")
    print(f"source-line + gold duplicates removed: {duplicate_counts['source_line_gold_duplicates_removed']}")
    print(f"source-line limit removals: {duplicate_counts['source_line_limit_removed']}")
    print("top items per collision key:")
    for key, value in per_collision.most_common(20):
        print(f"  {key}: {value}")

    sample_count = min(args.print_random_examples, len(items))
    print(f"random examples ({sample_count}):")
    for item in random.Random(args.seed).sample(items, sample_count):
        print(
            f"  {item['id']} [{item['collision_key']}] "
            f"{item['context_zh_with_blank']} :: {item['gold_zh']} / {item['distractor_zh']} "
            f"collapsed={item['collapsed_diacritic']}"
        )


if __name__ == "__main__":
    main()
