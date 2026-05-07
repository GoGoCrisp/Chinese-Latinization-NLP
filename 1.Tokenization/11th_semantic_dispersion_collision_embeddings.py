"""
11th semantic dispersion analysis for pinyin collision groups.

2026-04-30 update:
- This script quantifies whether Chinese tokens collapsed into the same pinyin
  token are semantically close or far apart.
- It reads a table2 A-B/A-C/A-D overlap details CSV, embeds the Chinese source
  tokens with an external sentence-embedding model, and reports pairwise cosine
  distances inside each pinyin collision group.
- It filters test tokens to multi-character Chinese tokens by default
  (--min-cjk-chars 2), so one-character collisions do not dominate the report.
- It also computes two baselines, because raw distances such as 0.3 or 0.7 are
  hard to interpret without context:
    1. Random pairs from all Chinese source tokens that appear in collision groups.
    2. Random pairs from the Chinese tokenizer vocabulary classification TSV.
- The readable report includes a stratified sample of collision groups for each
  filtered N, so every available N has some examples.
- It also reports an ALL row: the equal-weight average of per-collision-group
  mean distances across all filtered N>1 collision groups.
- Default models are BAAI/bge-small-zh-v1.5 and
  shibing624/text2vec-base-chinese.

Interpretation:
- cosine_distance = 1 - cosine_similarity.
- Higher distance means the embedding model treats the two Chinese tokens as
  less semantically similar.
- Mean group distance is the average distance over all unordered token pairs
  inside one collision group, not a least-squares or regression metric.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TOKENIZERS_DIR = os.path.join(BASE_DIR, "decoded_superTokenizers_2048_subset100k")
DEFAULT_PAIR = "AD"
TABLE2_DIR = os.path.join(TOKENIZERS_DIR, "table2")
OUTPUT_ROOT = os.path.join(TOKENIZERS_DIR, "11th_semantic_dispersion")


def default_pair_details(pair: str) -> str:
    pair_lower = pair.lower()
    filename = f"table2_{pair_lower}_overlap_superBPE_details.csv"
    grouped_path = os.path.join(
        TABLE2_DIR,
        f"table2_{pair_lower}_overlap_superBPE_outputs",
        filename,
    )
    if os.path.exists(grouped_path):
        return grouped_path
    return os.path.join(
        TOKENIZERS_DIR,
        f"table2_{pair_lower}_overlap_superBPE_outputs",
        filename,
    )


PAIR_DETAILS = {pair: default_pair_details(pair) for pair in ("AB", "AC", "AD")}
DEFAULT_CHINESE_TSV = os.path.join(
    BASE_DIR,
    "decoded_superTokenizers_2048_subset100k",
    "4.4_morphological_coherence",
    "custom_chinese_origin_64k_token_classification.tsv",
)
DEFAULT_OUTPUT_DIR = os.path.join(
    OUTPUT_ROOT,
    f"11th_semantic_dispersion_collision_embeddings_{DEFAULT_PAIR.lower()}",
)
DEFAULT_TARGETS_BY_PAIR = {
    "AB": [
        "ta de",
        "ta",
        "an jian",
        "dian shi",
        "fu shi",
        "lie shi",
        "yu shi",
        "gong shi",
        "xing shi",
        "qi shi",
        "jing li",
        "shi",
        "yi",
    ],
    "AC": [
        "ta1 de",
        "ta1",
        "an4 jian4",
        "dian4 shi4",
        "fu2 shi4",
        "lie4 shi4",
        "yu4 shi4",
        "gong1 shi4",
        "xing2 shi4",
        "qi2 shi4",
        "jing1 li4",
        "shi4",
        "yi4",
    ],
    "AD": [
        "tā de",
        "tā",
        "àn jiàn",
        "diàn shì",
        "fú shì",
        "liè shì",
        "yù shì",
        "gōng shì",
        "xíng shì",
        "qí shì",
        "jīng lì",
        "shì",
        "yì",
    ],
}
DEFAULT_SAMPLE_GROUPS_PER_N = 5
DEFAULT_MIN_CJK_CHARS = 2
DEFAULT_REQUIRED_SAMPLE_GROUPS_BY_PAIR = {
    "AB": ["quan li"],
    "AC": ["quan2 li4"],
    "AD": ["quán lì"],
}
DEFAULT_MODELS = [
    "BAAI/bge-small-zh-v1.5",
    "shibing624/text2vec-base-chinese",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute embedding semantic dispersion for pinyin collision groups."
    )
    parser.add_argument(
        "--pair",
        choices=sorted(PAIR_DETAILS),
        default=DEFAULT_PAIR,
        help="Overlap pair to analyze. Defaults to AD.",
    )
    parser.add_argument(
        "--details",
        default=None,
        help="Details CSV path. Defaults to the selected --pair details CSV.",
    )
    parser.add_argument("--chinese-tsv", default=DEFAULT_CHINESE_TSV)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to a pair-specific 11th output folder.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="SentenceTransformer model names.",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=None,
        help="Specific pinyin collision groups to include in the readable target table.",
    )
    parser.add_argument("--baseline-pairs", type=int, default=10000)
    parser.add_argument(
        "--sample-groups-per-n",
        type=int,
        default=DEFAULT_SAMPLE_GROUPS_PER_N,
        help="Number of filtered collision groups to include per N in the report.",
    )
    parser.add_argument(
        "--required-sample-groups",
        nargs="*",
        default=None,
        help="Pinyin groups that must be included in the sampled-by-N report if eligible.",
    )
    parser.add_argument(
        "--min-cjk-chars",
        type=int,
        default=DEFAULT_MIN_CJK_CHARS,
        help="Minimum number of CJK characters required for a token to be tested.",
    )
    parser.add_argument("--seed", type=int, default=20260430)
    parser.add_argument(
        "--prefer-local-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer an existing local Hugging Face snapshot to avoid network checks.",
    )
    args = parser.parse_args()
    if args.details is None:
        args.details = PAIR_DETAILS[args.pair]
    if args.output_dir is None:
        args.output_dir = os.path.join(
            OUTPUT_ROOT,
            f"11th_semantic_dispersion_collision_embeddings_{args.pair.lower()}",
        )
    if args.targets is None:
        args.targets = DEFAULT_TARGETS_BY_PAIR[args.pair]
    if args.required_sample_groups is None:
        args.required_sample_groups = DEFAULT_REQUIRED_SAMPLE_GROUPS_BY_PAIR[args.pair]
    return args


def has_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def cjk_char_count(text: str) -> int:
    return sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")


def clean_vocab_token(token: str) -> str:
    return re.sub(r"\s+", "", token.replace("##", "").replace("Ġ", " ")).strip()


def keep_test_token(token: str, min_cjk_chars: int) -> bool:
    return cjk_char_count(token) >= min_cjk_chars


def clean_unique_test_tokens(tokens: Sequence[str], min_cjk_chars: int) -> List[str]:
    filtered = []
    seen = set()
    for token in tokens:
        token = clean_vocab_token(token)
        if not token or token in seen or not keep_test_token(token, min_cjk_chars):
            continue
        seen.add(token)
        filtered.append(token)
    return filtered


def infer_collision_token_column(fieldnames: Sequence[str] | None) -> str:
    if not fieldnames:
        raise ValueError("Details CSV has no header.")
    token_columns = [
        name for name in fieldnames
        if name.endswith("_token") and name != "a_token"
    ]
    if len(token_columns) != 1:
        raise ValueError(
            f"Expected exactly one target token column ending in _token; got {token_columns}"
        )
    return token_columns[0]


def load_collision_groups(path: str, min_cjk_chars: int) -> Dict[str, List[str]]:
    groups = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        target_column = infer_collision_token_column(reader.fieldnames)
        for row in reader:
            tokens = clean_unique_test_tokens(
                json.loads(row["a_tokens_json"]),
                min_cjk_chars,
            )
            if len(tokens) >= 2:
                groups[row[target_column]] = tokens
    return groups


def load_chinese_vocab_tokens(path: str, min_cjk_chars: int) -> List[str]:
    tokens = []
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            token = clean_vocab_token(row.get("clean_token", ""))
            if (
                token
                and has_cjk(token)
                and keep_test_token(token, min_cjk_chars)
                and row.get("label") != "EX"
            ):
                tokens.append(token)
    return sorted(set(tokens))


def sample_pairs(
    tokens: Sequence[str],
    pair_count: int,
    rng: random.Random,
) -> List[Tuple[str, str]]:
    tokens = list(dict.fromkeys(tokens))
    if len(tokens) < 2:
        return []

    pairs = []
    seen = set()
    max_unique_pairs = len(tokens) * (len(tokens) - 1) // 2
    target = min(pair_count, max_unique_pairs)
    while len(pairs) < target:
        a, b = rng.sample(tokens, 2)
        if a > b:
            a, b = b, a
        key = (a, b)
        if key in seen:
            continue
        seen.add(key)
        pairs.append(key)
    return pairs


def cosine_distance_matrix(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.clip(norms, 1e-12, None)
    return 1.0 - vectors @ vectors.T


def pair_distances(
    pairs: Iterable[Tuple[str, str]],
    embeddings: Dict[str, np.ndarray],
) -> List[Tuple[float, str, str]]:
    rows = []
    for a, b in pairs:
        va = embeddings[a]
        vb = embeddings[b]
        sim = float(np.dot(va, vb))
        rows.append((1.0 - sim, a, b))
    return rows


def summarize_distances(rows: Sequence[Tuple[float, str, str]]) -> Dict:
    if not rows:
        return {
            "pair_count": 0,
            "mean_dist": "",
            "median_dist": "",
            "p90_dist": "",
            "max_dist": "",
            "max_pair": "",
            "min_dist": "",
            "min_pair": "",
        }

    distances = np.array([row[0] for row in rows], dtype=np.float64)
    max_row = max(rows, key=lambda item: item[0])
    min_row = min(rows, key=lambda item: item[0])
    return {
        "pair_count": len(rows),
        "mean_dist": round(float(np.mean(distances)), 4),
        "median_dist": round(float(np.median(distances)), 4),
        "p90_dist": round(float(np.quantile(distances, 0.9)), 4),
        "max_dist": round(float(max_row[0]), 4),
        "max_pair": f"{max_row[1]}|{max_row[2]}",
        "min_dist": round(float(min_row[0]), 4),
        "min_pair": f"{min_row[1]}|{min_row[2]}",
    }


def group_dispersion(tokens: Sequence[str], embeddings: Dict[str, np.ndarray]) -> Dict:
    pairs = list(itertools.combinations(tokens, 2))
    rows = pair_distances(pairs, embeddings)
    summary = summarize_distances(rows)
    summary["N"] = len(tokens)
    return summary


def summarize_equal_weight_group_means(
    groups: Dict[str, List[str]],
    embeddings: Dict[str, np.ndarray],
) -> Dict:
    group_rows = []
    for key, tokens in groups.items():
        summary = group_dispersion(tokens, embeddings)
        if summary["pair_count"]:
            group_rows.append((float(summary["mean_dist"]), key, len(tokens)))

    if not group_rows:
        return {
            "pinyin_token": "ALL",
            "N": "ALL",
            "pair_count": 0,
            "mean_dist": "",
            "median_dist": "",
            "p90_dist": "",
            "max_dist": "",
            "max_pair": "",
            "min_dist": "",
            "min_pair": "",
            "source_tokens": "equal-weight mean over filtered N>1 collision groups",
        }

    distances = np.array([row[0] for row in group_rows], dtype=np.float64)
    max_row = max(group_rows, key=lambda item: item[0])
    min_row = min(group_rows, key=lambda item: item[0])
    return {
        "pinyin_token": "ALL",
        "N": "ALL",
        "pair_count": len(group_rows),
        "mean_dist": round(float(np.mean(distances)), 4),
        "median_dist": round(float(np.median(distances)), 4),
        "p90_dist": round(float(np.quantile(distances, 0.9)), 4),
        "max_dist": round(float(max_row[0]), 4),
        "max_pair": f"{max_row[1]} (N={max_row[2]})",
        "min_dist": round(float(min_row[0]), 4),
        "min_pair": f"{min_row[1]} (N={min_row[2]})",
        "source_tokens": "equal-weight mean over filtered N>1 collision groups",
    }


def summarize_equal_weight_group_means_by_n(
    groups: Dict[str, List[str]],
    embeddings: Dict[str, np.ndarray],
) -> List[Dict]:
    rows = []
    by_n: Dict[int, Dict[str, List[str]]] = {}
    for key, tokens in groups.items():
        by_n.setdefault(len(tokens), {})[key] = tokens

    for n_value in sorted(by_n):
        row = summarize_equal_weight_group_means(by_n[n_value], embeddings)
        row["pinyin_token"] = f"N={n_value}"
        row["N"] = n_value
        row["source_tokens"] = f"equal-weight mean over filtered N={n_value} collision groups"
        rows.append(row)

    all_row = summarize_equal_weight_group_means(groups, embeddings)
    rows.append(all_row)
    return rows


def sample_collision_groups_by_n(
    groups: Dict[str, List[str]],
    groups_per_n: int,
    required_groups: Sequence[str],
    rng: random.Random,
) -> List[Tuple[str, List[str]]]:
    by_n: Dict[int, List[Tuple[str, List[str]]]] = {}
    for key, tokens in groups.items():
        by_n.setdefault(len(tokens), []).append((key, tokens))

    sampled_groups = []
    required_groups = set(required_groups)
    for n in sorted(by_n):
        candidates = sorted(by_n[n], key=lambda item: item[0])
        if groups_per_n <= 0 or len(candidates) <= groups_per_n:
            selected = candidates
        else:
            required = [item for item in candidates if item[0] in required_groups]
            remaining = [item for item in candidates if item[0] not in required_groups]
            random_slots = max(groups_per_n - len(required), 0)
            selected = sorted(
                required + rng.sample(remaining, random_slots),
                key=lambda item: item[0],
            )
        sampled_groups.extend(selected)
    return sampled_groups


def safe_model_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def short_model_label(name: str) -> str:
    lowered = name.lower()
    if "bge" in lowered:
        return "BGE"
    if "text2vec" in lowered:
        return "text2vec"
    return safe_model_name(name)


def unique_labels(model_names: Sequence[str]) -> List[str]:
    labels = []
    counts = {}
    for model_name in model_names:
        label = short_model_label(model_name)
        counts[label] = counts.get(label, 0) + 1
        if counts[label] > 1:
            label = f"{label}_{counts[label]}"
        labels.append(label)
    return labels


def resolve_local_hf_snapshot(model_name: str) -> str:
    if "/" not in model_name:
        return model_name

    owner, repo = model_name.split("/", 1)
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    model_dir = cache_root / f"models--{owner}--{repo}"
    snapshots_dir = model_dir / "snapshots"
    if not snapshots_dir.exists():
        return model_name

    snapshots = [
        path
        for path in snapshots_dir.iterdir()
        if path.is_dir() and (path / "modules.json").exists()
    ]
    if not snapshots:
        return model_name

    snapshots.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return str(snapshots[0])


def write_csv(path: str, rows: List[Dict], fieldnames: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_markdown_table(rows: List[Dict], columns: List[str]) -> str:
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        values = [
            str(row.get(col, "")).replace("|", "\\|").replace("\n", " ")
            for col in columns
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_two_model_mean_tables(
    output_dir: str,
    model_names: Sequence[str],
    target_rows_by_model: Dict[str, List[Dict]],
    baseline_rows_by_model: Dict[str, List[Dict]],
    target_order: Sequence[str],
) -> None:
    if len(model_names) < 2:
        return

    labels = unique_labels(model_names)
    columns = ["pinyin_token", "N"] + [f"{label}_mean_dist" for label in labels]
    rows = []

    target_by_model = {
        model_name: {row["pinyin_token"]: row for row in rows}
        for model_name, rows in target_rows_by_model.items()
    }
    for pinyin_token in target_order:
        per_model_rows = [
            target_by_model.get(model_name, {}).get(pinyin_token)
            for model_name in model_names
        ]
        present_rows = [row for row in per_model_rows if row]
        if not present_rows:
            continue

        row = {
            "pinyin_token": pinyin_token,
            "N": present_rows[0]["N"],
        }
        for label, model_row in zip(labels, per_model_rows):
            row[f"{label}_mean_dist"] = model_row["mean_dist"] if model_row else ""
        rows.append(row)

    baseline_by_model = {
        model_name: {row["baseline"]: row for row in rows}
        for model_name, rows in baseline_rows_by_model.items()
    }
    for baseline_name in [
        "collision_source_random_pairs",
        "chinese_vocab_random_pairs",
    ]:
        per_model_rows = [
            baseline_by_model.get(model_name, {}).get(baseline_name)
            for model_name in model_names
        ]
        present_rows = [row for row in per_model_rows if row]
        if not present_rows:
            continue

        row = {
            "pinyin_token": f"RANDOM_BASELINE_{baseline_name}",
            "N": f"{present_rows[0]['pair_count']} sampled pairs",
        }
        for label, model_row in zip(labels, per_model_rows):
            row[f"{label}_mean_dist"] = model_row["mean_dist"] if model_row else ""
        rows.append(row)

    csv_path = os.path.join(output_dir, "two_model_collision_mean_dist_table.csv")
    md_path = os.path.join(output_dir, "two_model_collision_mean_dist_table.md")
    write_csv(csv_path, rows, columns)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(format_markdown_table(rows, columns))


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Overlap pair: {args.pair}")
    print(f"Reading details: {args.details}")
    print(f"Writing outputs to: {args.output_dir}")

    groups = load_collision_groups(args.details, args.min_cjk_chars)
    target_groups = {key: groups[key] for key in args.targets if key in groups}
    collision_source_tokens = sorted({token for tokens in groups.values() for token in tokens})
    chinese_vocab_tokens = load_chinese_vocab_tokens(args.chinese_tsv, args.min_cjk_chars)

    rng = random.Random(args.seed)
    sampled_groups_by_n = sample_collision_groups_by_n(
        groups,
        args.sample_groups_per_n,
        args.required_sample_groups,
        rng,
    )
    collision_baseline_pairs = sample_pairs(
        collision_source_tokens,
        args.baseline_pairs,
        rng,
    )
    vocab_baseline_pairs = sample_pairs(
        chinese_vocab_tokens,
        args.baseline_pairs,
        rng,
    )

    all_needed_tokens = set()
    all_needed_tokens.update(collision_source_tokens)
    for tokens in target_groups.values():
        all_needed_tokens.update(tokens)
    for _, tokens in sampled_groups_by_n:
        all_needed_tokens.update(tokens)
    for a, b in collision_baseline_pairs + vocab_baseline_pairs:
        all_needed_tokens.add(a)
        all_needed_tokens.add(b)
    all_needed_tokens = sorted(all_needed_tokens)

    report_sections = []
    target_rows_by_model = {}
    baseline_rows_by_model = {}
    for model_name in args.models:
        print(f"\nLoading model: {model_name}")
        model_path = (
            resolve_local_hf_snapshot(model_name)
            if args.prefer_local_cache
            else model_name
        )
        if model_path != model_name:
            print(f"Using local cache snapshot: {model_path}")
        model = SentenceTransformer(model_path)
        print(f"Encoding {len(all_needed_tokens)} Chinese tokens...")
        vectors = model.encode(
            all_needed_tokens,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        embeddings = dict(zip(all_needed_tokens, vectors))

        baseline_rows = []
        for baseline_name, pairs in [
            ("collision_source_random_pairs", collision_baseline_pairs),
            ("chinese_vocab_random_pairs", vocab_baseline_pairs),
        ]:
            summary = summarize_distances(pair_distances(pairs, embeddings))
            summary["model"] = model_name
            summary["baseline"] = baseline_name
            baseline_rows.append(summary)

        target_rows = []
        all_row = summarize_equal_weight_group_means(groups, embeddings)
        all_row["model"] = model_name
        target_rows.append(all_row)
        for key in args.targets:
            if key not in target_groups:
                continue
            summary = group_dispersion(target_groups[key], embeddings)
            summary["model"] = model_name
            summary["pinyin_token"] = key
            summary["source_tokens"] = " / ".join(target_groups[key])
            target_rows.append(summary)
        target_rows_by_model[model_name] = target_rows

        sampled_n_rows = []
        for key, tokens in sampled_groups_by_n:
            summary = group_dispersion(tokens, embeddings)
            summary["model"] = model_name
            summary["pinyin_token"] = key
            summary["source_tokens"] = " / ".join(tokens)
            sampled_n_rows.append(summary)

        aggregate_rows = summarize_equal_weight_group_means_by_n(groups, embeddings)
        for row in aggregate_rows:
            row["model"] = model_name

        model_slug = safe_model_name(model_name)
        baseline_csv = os.path.join(args.output_dir, f"{model_slug}_baselines.csv")
        targets_csv = os.path.join(args.output_dir, f"{model_slug}_target_groups.csv")
        sampled_n_csv = os.path.join(
            args.output_dir,
            f"{model_slug}_sampled_groups_by_N.csv",
        )
        aggregate_csv = os.path.join(
            args.output_dir,
            f"{model_slug}_aggregate_group_means.csv",
        )
        write_csv(
            baseline_csv,
            baseline_rows,
            [
                "model",
                "baseline",
                "pair_count",
                "mean_dist",
                "median_dist",
                "p90_dist",
                "max_dist",
                "max_pair",
                "min_dist",
                "min_pair",
            ],
        )
        baseline_rows_by_model[model_name] = baseline_rows
        write_csv(
            targets_csv,
            target_rows,
            [
                "model",
                "pinyin_token",
                "N",
                "pair_count",
                "mean_dist",
                "median_dist",
                "p90_dist",
                "max_dist",
                "max_pair",
                "min_dist",
                "min_pair",
                "source_tokens",
            ],
        )
        write_csv(
            sampled_n_csv,
            sampled_n_rows,
            [
                "model",
                "N",
                "pinyin_token",
                "pair_count",
                "mean_dist",
                "median_dist",
                "p90_dist",
                "max_dist",
                "max_pair",
                "min_dist",
                "min_pair",
                "source_tokens",
            ],
        )
        write_csv(
            aggregate_csv,
            aggregate_rows,
            [
                "model",
                "N",
                "pinyin_token",
                "pair_count",
                "mean_dist",
                "median_dist",
                "p90_dist",
                "max_dist",
                "max_pair",
                "min_dist",
                "min_pair",
                "source_tokens",
            ],
        )

        target_columns = [
            "pinyin_token",
            "N",
            "mean_dist",
            "median_dist",
            "max_dist",
            "max_pair",
            "min_dist",
        ]
        baseline_columns = [
            "baseline",
            "pair_count",
            "mean_dist",
            "median_dist",
            "p90_dist",
            "max_dist",
            "min_dist",
        ]
        sampled_n_columns = [
            "N",
            "pinyin_token",
            "pair_count",
            "mean_dist",
            "median_dist",
            "max_dist",
            "max_pair",
            "min_dist",
            "source_tokens",
        ]
        aggregate_columns = [
            "N",
            "pinyin_token",
            "pair_count",
            "mean_dist",
            "median_dist",
            "p90_dist",
            "max_dist",
            "max_pair",
            "min_dist",
            "min_pair",
            "source_tokens",
        ]
        report_sections.append(f"# {model_name}\n")
        report_sections.append(
            f"Token filter: CJK character count >= {args.min_cjk_chars}. "
            f"Collision groups kept only when at least 2 source tokens remain.\n"
        )
        report_sections.append(
            f"Overlap pair: {args.pair}. Details CSV: {args.details}\n"
        )
        report_sections.append("## Baselines\n")
        report_sections.append(format_markdown_table(baseline_rows, baseline_columns))
        report_sections.append("\n## Target collision groups\n")
        report_sections.append(format_markdown_table(target_rows, target_columns))
        report_sections.append("\n## Aggregate collision group means\n")
        report_sections.append(format_markdown_table(aggregate_rows, aggregate_columns))
        report_sections.append("\n## Sampled collision groups by filtered N\n")
        report_sections.append(format_markdown_table(sampled_n_rows, sampled_n_columns))
        report_sections.append("")

        print("\nBaselines:")
        print(format_markdown_table(baseline_rows, baseline_columns))
        print("\nTarget collision groups:")
        print(format_markdown_table(target_rows, target_columns))
        print("\nAggregate collision group means:")
        print(format_markdown_table(aggregate_rows, aggregate_columns))
        print("\nSampled collision groups by filtered N:")
        print(format_markdown_table(sampled_n_rows, sampled_n_columns))

    report_path = os.path.join(args.output_dir, "semantic_dispersion_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_sections))
    print(f"\nSaved report: {report_path}")

    write_two_model_mean_tables(
        args.output_dir,
        args.models,
        target_rows_by_model,
        baseline_rows_by_model,
        ["ALL"] + list(args.targets),
    )


if __name__ == "__main__":
    main()
