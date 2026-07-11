from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import statistics
import subprocess
from collections import Counter, defaultdict
from pathlib import Path


HANJA_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
HANGUL_RE = re.compile(r"[\u1100-\u11ff\u3130-\u318f\ua960-\ua97f\uac00-\ud7a3\ud7b0-\ud7ff]")
LATIN_RE = re.compile(r"[A-Za-z]")
DIGIT_RE = re.compile(r"\d")
WHITESPACE_RE = re.compile(r"\s+", re.UNICODE)
HANJA_PAREN_ANNOTATION_RE = re.compile(r"[\(（][\u3400-\u4dbf\u4e00-\u9fff]+[\)）]")
SPECIAL_TOKENS = {"[UNK]", "[PAD]", "[BOS]", "[EOS]"}
STRICT_HANGUL_ALLOWED = re.compile(r"^[\u1100-\u11ff\u3130-\u318f\ua960-\ua97f\uac00-\ud7a3\ud7b0-\ud7ff·ㆍ\-\u2010\u2011]+$")


DEBUG_EXAMPLES = ["記事", "技師", "騎士", "會社", "社會", "國民", "權利", "權力", "大韓", "日本"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Korean mixed-vocab to Hangulized-vocab N:1 analysis.")
    parser.add_argument(
        "--mixed-tokenizer",
        type=Path,
        default=Path("4.Korean/korean_khdb_magazine_audit/data/tokenizers/korean_mixed_bpe_32k.json"),
    )
    parser.add_argument(
        "--hangulized-tokenizer",
        type=Path,
        default=Path("4.Korean/korean_khdb_magazine_audit/data/tokenizers/korean_hangulized_bpe_32k.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("4.Korean/korean_khdb_magazine_audit/results/korean_n_to_1"),
    )
    parser.add_argument("--converter", choices=["gukhanmun"], default="gukhanmun")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--train-mixed",
        type=Path,
        default=Path(
            "4.Korean/korean_khdb_magazine_audit/data/corpus/final_aligned/splits/seed42_90_10/train.mixed_chunks_nospace.txt"
        ),
    )
    parser.add_argument(
        "--dev-mixed",
        type=Path,
        default=Path(
            "4.Korean/korean_khdb_magazine_audit/data/corpus/final_aligned/splits/seed42_90_10/dev.mixed_chunks_nospace.txt"
        ),
    )
    return parser.parse_args()


def require_tokenizers():
    try:
        import tokenizers
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise SystemExit("Missing dependency. Install with: pip install tokenizers") from exc
    return tokenizers, Tokenizer


def converter_version(command: str) -> str:
    for option in ("--version", "-V"):
        try:
            result = subprocess.run(
                [command, option],
                text=True,
                capture_output=True,
                timeout=5,
                check=False,
            )
        except Exception:
            continue
        output = (result.stdout or result.stderr).strip()
        if output:
            return output.splitlines()[0]
    return "unknown"


def remove_all_whitespace(text: str) -> str:
    return WHITESPACE_RE.sub("", text or "")


def sanitize_hangulized(text: str) -> tuple[str, dict]:
    paren_annotations = HANJA_PAREN_ANNOTATION_RE.findall(text)
    without_paren_annotations = HANJA_PAREN_ANNOTATION_RE.sub("", text)
    leftover_hanja = HANJA_RE.findall(without_paren_annotations)
    cleaned = HANJA_RE.sub("", without_paren_annotations)
    cleaned = remove_all_whitespace(cleaned)
    return cleaned, {
        "removed_hanja_parenthetical_annotations": len(paren_annotations),
        "removed_hanja_parenthetical_chars": sum(len(HANJA_RE.findall(item)) for item in paren_annotations),
        "removed_unconverted_hanja_chars": len(leftover_hanja),
        "removed_unconverted_hanja_examples": leftover_hanja[:20],
    }


def convert_batch_with_gukhanmun(tokens: list[str], timeout: float) -> tuple[list[str], list[dict]]:
    if not tokens:
        return [], []
    result = subprocess.run(
        ["gukhanmun", "--rendering", "hangul-only", "--disambiguation", "off"],
        input="\n".join(tokens),
        text=True,
        capture_output=True,
        timeout=timeout,
        check=True,
    )
    converted_lines = result.stdout.splitlines()
    if len(converted_lines) != len(tokens):
        converted_lines = []
        for token in tokens:
            single = subprocess.run(
                ["gukhanmun", "--rendering", "hangul-only", "--disambiguation", "off"],
                input=token,
                text=True,
                capture_output=True,
                timeout=timeout,
                check=True,
            )
            converted_lines.append(single.stdout.rstrip("\n"))

    sanitized: list[str] = []
    cleanups: list[dict] = []
    for converted in converted_lines:
        cleaned, cleanup = sanitize_hangulized(converted)
        sanitized.append(cleaned)
        cleanups.append(cleanup)
    return sanitized, cleanups


def load_vocab_ordered(tokenizer_path: Path, Tokenizer) -> tuple[dict[str, int], list[tuple[str, int]]]:
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    vocab = tokenizer.get_vocab()
    ordered = sorted(vocab.items(), key=lambda item: item[1])
    return vocab, ordered


def contains_hanja(token: str) -> bool:
    return bool(HANJA_RE.search(token))


def contains_hangul(token: str) -> bool:
    return bool(HANGUL_RE.search(token))


def is_pure_punctuation(token: str) -> bool:
    return bool(token) and all(not ch.isalnum() and not contains_hangul(ch) and not contains_hanja(ch) for ch in token)


def is_pure_digits(token: str) -> bool:
    return bool(token) and all(ch.isdigit() for ch in token)


def is_pure_latin_foreign(token: str) -> bool:
    if not token:
        return False
    if contains_hanja(token) or contains_hangul(token):
        return False
    return bool(LATIN_RE.search(token))


def is_byte_artifact(token: str) -> bool:
    return "�" in token or token in {"Ġ", "Ċ"} or token.startswith("##")


def is_valid_lexical(token: str) -> bool:
    if token in SPECIAL_TOKENS or not token:
        return False
    if any(ch.isspace() for ch in token):
        return False
    if is_byte_artifact(token):
        return False
    if is_pure_punctuation(token) or is_pure_digits(token) or is_pure_latin_foreign(token):
        return False
    return contains_hanja(token) or contains_hangul(token)


def is_pure_hangul_strict(token: str) -> bool:
    if not contains_hangul(token):
        return False
    if contains_hanja(token) or LATIN_RE.search(token) or DIGIT_RE.search(token):
        return False
    return bool(STRICT_HANGUL_ALLOWED.fullmatch(token))


def is_pure_hangul_loose(token: str) -> bool:
    return contains_hangul(token) and not contains_hanja(token)


def token_length_stats(tokens: list[str]) -> dict:
    lengths = [len(token) for token in tokens]
    return {
        "min": min(lengths) if lengths else 0,
        "max": max(lengths) if lengths else 0,
        "mean": statistics.mean(lengths) if lengths else 0.0,
        "median": statistics.median(lengths) if lengths else 0.0,
    }


def conversion_notes(cleanup: dict) -> list[str]:
    return [
        f"{key}={value}"
        for key, value in cleanup.items()
        if value and key != "removed_unconverted_hanja_examples"
    ]


def count_token_frequencies(tokenizer, corpus_path: Path | None) -> Counter[int]:
    counts: Counter[int] = Counter()
    if not corpus_path or not corpus_path.exists():
        return counts
    with corpus_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.rstrip("\n")
            if not text:
                continue
            counts.update(tokenizer.encode(text).ids)
    return counts


def bucket_collision_size(size: int) -> str:
    return ">4:1" if size > 4 else f"{size}:1"


def distribution(groups: list[dict]) -> tuple[dict[str, int], dict[str, int]]:
    group_side: Counter[str] = Counter()
    source_side: Counter[str] = Counter()
    for group in groups:
        bucket = bucket_collision_size(group["group_size"])
        group_side[bucket] += 1
        source_side[bucket] += group["group_size"]
    ordered_keys = ["1:1", "2:1", "3:1", "4:1", ">4:1"]
    return ({key: group_side.get(key, 0) for key in ordered_keys}, {key: source_side.get(key, 0) for key in ordered_keys})


def overlap_stats(rows: list[dict], target_ids: set[int], hanja_only: bool = False) -> dict:
    candidates = [row for row in rows if row["is_valid_lexical"]]
    if hanja_only:
        candidates = [row for row in candidates if row["contains_hanja"]]
    matches = [
        row
        for row in candidates
        if row["converted_hangul"] and row["exact_match_in_hangulized_vocab"] and row["hangulized_token_id"] in target_ids
    ]
    return {
        "candidate_count": len(candidates),
        "exact_overlap_count": len(matches),
        "exact_overlap_rate": len(matches) / len(candidates) if candidates else 0.0,
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_top_size_csv(path: Path, groups: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "converted_hangul",
        "group_size",
        "exists_in_hangulized_vocab",
        "mixed_tokens_joined",
        "train_frequency_sum",
        "dev_frequency_sum",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for group in groups[:200]:
            writer.writerow(
                {
                    "converted_hangul": group["converted_hangul"],
                    "group_size": group["group_size"],
                    "exists_in_hangulized_vocab": group["exists_in_hangulized_vocab"],
                    "mixed_tokens_joined": " / ".join(group["mixed_tokens"]),
                    "train_frequency_sum": group.get("train_frequency_sum", 0),
                    "dev_frequency_sum": group.get("dev_frequency_sum", 0),
                }
            )


def format_distribution_table(group_side: dict[str, int], source_side: dict[str, int]) -> list[str]:
    lines = [
        "| collision size | mixed source tokens | hangulized surfaces |",
        "|---|---:|---:|",
    ]
    for key in ["1:1", "2:1", "3:1", "4:1", ">4:1"]:
        lines.append(f"| {key} | {source_side.get(key, 0)} | {group_side.get(key, 0)} |")
    return lines


def summarize_collision_subset(groups: list[dict], dev_hanja_token_occurrences: int) -> dict:
    group_side, source_side = distribution(groups)
    sizes = [group["group_size"] for group in groups]
    dev_occurrences = sum(group.get("dev_frequency_sum", 0) for group in groups)
    train_occurrences = sum(group.get("train_frequency_sum", 0) for group in groups)
    return {
        "collision_group_count": len(groups),
        "collision_distribution_group_side": group_side,
        "collision_distribution_source_token_side": source_side,
        "max_collision_size": max(sizes) if sizes else 0,
        "mean_collision_size_among_collisions": statistics.mean(sizes) if sizes else 0.0,
        "median_collision_size_among_collisions": statistics.median(sizes) if sizes else 0.0,
        "train_frequency_sum": train_occurrences,
        "dev_frequency_sum": dev_occurrences,
        "percentage_hanja_token_occurrences_in_dev_belonging_to_subset": dev_occurrences
        / dev_hanja_token_occurrences
        * 100
        if dev_hanja_token_occurrences
        else 0.0,
    }


def qualitative_examples(groups: list[dict], seed: int, limit: int = 30) -> list[str]:
    rng = random.Random(seed)
    collision_groups = [group for group in groups if group["group_size"] >= 2]
    high_freq = sorted(collision_groups, key=lambda group: (group.get("dev_frequency_sum", 0), group["group_size"]), reverse=True)
    selected = high_freq[: limit * 2]
    if len(selected) > limit:
        selected = selected[: limit // 2] + rng.sample(selected[limit // 2 :], limit - limit // 2)
    examples = []
    for group in selected[:limit]:
        examples.append(f"{group['converted_hangul']} ← {' / '.join(group['mixed_tokens'][:12])}")
    return examples


def render_report(summary: dict, groups: list[dict], top_by_size: list[dict], top_by_dev: list[dict], debug_rows: list[dict]) -> str:
    lines = [
        "# Korean KHDB Vocabulary N:1 Report",
        "",
        "This is a tokenizer-vocabulary collision diagnostic. Mixed-script BPE",
        "vocabulary tokens are converted to Hangulized surfaces with the same",
        "Gukhanmun settings used in corpus preparation, then grouped by converted",
        "surface form.",
        "",
        "## Inputs",
        "",
        f"- mixed tokenizer: `{summary['inputs']['mixed_tokenizer']}`",
        f"- Hangulized tokenizer: `{summary['inputs']['hangulized_tokenizer']}`",
        f"- train mixed for frequency weighting: `{summary['inputs'].get('train_mixed', '')}`",
        f"- dev mixed for frequency weighting: `{summary['inputs'].get('dev_mixed', '')}`",
        "",
        "## Converter",
        "",
        f"- backend: {summary['converter_backend']}",
        f"- version: {summary['converter_version']}",
        "- command settings: `--rendering hangul-only --disambiguation off`",
        "- cleanup: remove pure-Hanja parenthetical annotations, delete remaining Hanja, remove whitespace",
        "",
        "## Debug Conversion",
        "",
        "| input | output | leftover Hanja after cleanup |",
        "|---|---|---:|",
    ]
    for row in debug_rows:
        lines.append(f"| {row['input']} | {row['converted_hangul']} | {row['leftover_hanja_after_cleanup']} |")
    lines.extend(
        [
            "",
            "## Vocab Filtering",
            "",
            f"- mixed vocab size: `{summary['mixed_vocab_size']}`",
            f"- Hangulized vocab size: `{summary['hangulized_vocab_size']}`",
            f"- mixed valid lexical tokens: `{summary['mixed_valid_lexical_token_count']}`",
            f"- mixed Hanja-containing tokens: `{summary['mixed_hanja_token_count']}`",
            f"- Hangulized valid lexical tokens: `{summary['hangulized_valid_lexical_token_count']}`",
            f"- Hangulized pure Hangul strict tokens: `{summary['hangulized_pure_hangul_token_count_strict']}`",
            f"- Hangulized pure Hangul loose tokens: `{summary['hangulized_pure_hangul_token_count_loose']}`",
            "",
            "## Exact Overlap",
            "",
            f"- all valid exact overlap: `{summary['exact_overlap_count_all_valid']}` / `{summary['converted_valid_mixed_token_count']}` = `{summary['exact_overlap_rate_all_valid']:.6f}`",
            f"- Hanja-token exact overlap: `{summary['exact_overlap_count_hanja_only']}` / `{summary['converted_hanja_mixed_token_count']}` = `{summary['exact_overlap_rate_hanja_only']:.6f}`",
            "",
            "## N:1 Distribution",
            "",
        ]
    )
    lines.extend(format_distribution_table(summary["collision_distribution_group_side"], summary["collision_distribution_source_token_side"]))
    robustness = summary.get("robustness_converted_hangul_len_ge_2", {})
    lines.extend(
        [
            "",
            f"- max collision size: `{summary['max_collision_size']}`",
            f"- mean group size among collision groups: `{summary['mean_collision_size_among_collisions']:.6f}`",
            f"- median group size among collision groups: `{summary['median_collision_size_among_collisions']:.6f}`",
            "",
            "## Length >= 2 Subset Within N:1 Collisions",
            "",
            "This keeps the main N:1 result unchanged and only asks how many collision",
            "groups have `converted_hangul` length at least 2.",
            "",
            f"- length>=2 collision groups: `{robustness.get('collision_group_count', 0)}` / `{summary['collision_group_count']}`",
            f"- length>=2 dev collision occurrences: `{robustness.get('dev_frequency_sum', 0)}`",
            f"- length>=2 dev Hanja-token occurrence share: `{robustness.get('percentage_hanja_token_occurrences_in_dev_belonging_to_subset', 0.0):.2f}%`",
            f"- length>=2 max collision size: `{robustness.get('max_collision_size', 0)}`",
            "",
        ]
    )
    if robustness:
        lines.extend(format_distribution_table(robustness["collision_distribution_group_side"], robustness["collision_distribution_source_token_side"]))
    lines.extend(
        [
            "",
            "## Frequency Weighted Dev Stats",
            "",
            "```json",
            json.dumps(summary.get("frequency_weighted_stats", {}), ensure_ascii=False, indent=2),
            "```",
            "",
            "## Top N:1 Groups by Size",
            "",
            "| converted Hangul | group size | in Hangulized vocab | dev freq | mixed tokens |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for group in top_by_size[:50]:
        lines.append(
            f"| {group['converted_hangul']} | {group['group_size']} | {group['exists_in_hangulized_vocab']} | "
            f"{group.get('dev_frequency_sum', 0)} | {' / '.join(group['mixed_tokens'][:12])} |"
        )
    lines.extend(
        [
            "",
            "## Top N:1 Groups by Dev Frequency",
            "",
            "| converted Hangul | group size | in Hangulized vocab | dev freq | mixed tokens |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for group in top_by_dev[:50]:
        lines.append(
            f"| {group['converted_hangul']} | {group['group_size']} | {group['exists_in_hangulized_vocab']} | "
            f"{group.get('dev_frequency_sum', 0)} | {' / '.join(group['mixed_tokens'][:12])} |"
        )
    lines.extend(["", "## Qualitative Examples", ""])
    for example in summary["qualitative_examples"]:
        lines.append(f"- {example}")
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- Automatic Hanja-to-Hangul conversion may be imperfect.",
            "- This is a tokenizer-vocabulary collision diagnostic, not a gold-standard lexical ambiguity dataset.",
            "- Token-level collisions do not necessarily equal word-level ambiguity.",
            "- BPE tokens can be fragments or multiword pieces.",
            "- Hangulized tokenizer had lower fertility, so this analysis tests the recoverability-cost side of that trade-off.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_robustness_report(summary: dict, top_by_size: list[dict], top_by_dev: list[dict]) -> str:
    robustness = summary["robustness_converted_hangul_len_ge_2"]
    lines = [
        "# Korean KHDB N:1 Robustness Report: converted_hangul length >= 2",
        "",
        "This robustness view keeps only N:1 collision groups whose converted Hangul",
        "surface has length at least 2. It excludes single-syllable surfaces such as",
        "`이`, `기`, `사`, and `구`, which dominate the main collision profile.",
        "",
        "## Summary",
        "",
        f"- collision groups: `{robustness['collision_group_count']}`",
        f"- max collision size: `{robustness['max_collision_size']}`",
        f"- mean collision size: `{robustness['mean_collision_size_among_collisions']:.6f}`",
        f"- median collision size: `{robustness['median_collision_size_among_collisions']:.6f}`",
        f"- dev collision occurrences: `{robustness['dev_frequency_sum']}`",
        f"- dev Hanja-token occurrence share: `{robustness['percentage_hanja_token_occurrences_in_dev_belonging_to_subset']:.2f}%`",
        "",
        "## N:1 Distribution",
        "",
    ]
    lines.extend(format_distribution_table(robustness["collision_distribution_group_side"], robustness["collision_distribution_source_token_side"]))
    lines.extend(
        [
            "",
            "## Top Robust Collision Groups by Size",
            "",
            "| converted Hangul | group size | in Hangulized vocab | dev freq | mixed tokens |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for group in top_by_size[:50]:
        lines.append(
            f"| {group['converted_hangul']} | {group['group_size']} | {group['exists_in_hangulized_vocab']} | "
            f"{group.get('dev_frequency_sum', 0)} | {' / '.join(group['mixed_tokens'][:12])} |"
        )
    lines.extend(
        [
            "",
            "## Top Robust Collision Groups by Dev Frequency",
            "",
            "| converted Hangul | group size | in Hangulized vocab | dev freq | mixed tokens |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for group in top_by_dev[:50]:
        lines.append(
            f"| {group['converted_hangul']} | {group['group_size']} | {group['exists_in_hangulized_vocab']} | "
            f"{group.get('dev_frequency_sum', 0)} | {' / '.join(group['mixed_tokens'][:12])} |"
        )
    lines.extend(
        [
            "",
            "## Caveat",
            "",
            "This is a stricter robustness slice of the same tokenizer-vocabulary diagnostic.",
            "It is not a separate gold ambiguity dataset.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    tokenizers_lib, Tokenizer = require_tokenizers()
    converter_path = shutil.which(args.converter)
    if not converter_path:
        raise RuntimeError("gukhanmun was not found on PATH.")
    converter_ver = converter_version(converter_path)

    mixed_vocab, mixed_ordered = load_vocab_ordered(args.mixed_tokenizer, Tokenizer)
    hangulized_vocab, hangulized_ordered = load_vocab_ordered(args.hangulized_tokenizer, Tokenizer)
    hangulized_id_by_token = {token: token_id for token, token_id in hangulized_vocab.items()}

    mixed_valid = [(token, token_id) for token, token_id in mixed_ordered if is_valid_lexical(token)]
    mixed_hanja_ids = {token_id for token, token_id in mixed_valid if contains_hanja(token)}
    hangulized_valid = [(token, token_id) for token, token_id in hangulized_ordered if is_valid_lexical(token)]
    hangulized_valid_ids = {token_id for token, token_id in hangulized_valid}
    hangulized_strict_ids = {token_id for token, token_id in hangulized_valid if is_pure_hangul_strict(token)}
    hangulized_loose_ids = {token_id for token, token_id in hangulized_valid if is_pure_hangul_loose(token)}

    debug_converted, debug_cleanups = convert_batch_with_gukhanmun(DEBUG_EXAMPLES, args.timeout)
    debug_rows = [
        {
            "input": token,
            "converted_hangul": converted,
            "leftover_hanja_after_cleanup": len(HANJA_RE.findall(converted)),
            "conversion_notes": conversion_notes(cleanup),
        }
        for token, converted, cleanup in zip(DEBUG_EXAMPLES, debug_converted, debug_cleanups)
    ]
    for row in debug_rows:
        print(f"DEBUG {row['input']} -> {row['converted_hangul']}")

    valid_tokens = [token for token, _token_id in mixed_valid]
    converted_tokens, cleanups = convert_batch_with_gukhanmun(valid_tokens, args.timeout)
    train_freq = count_token_frequencies(Tokenizer.from_file(str(args.mixed_tokenizer)), args.train_mixed)
    dev_freq = count_token_frequencies(Tokenizer.from_file(str(args.mixed_tokenizer)), args.dev_mixed)

    converted_rows: list[dict] = []
    remaining_hanja_before_final_cleanup = 0
    for (token, token_id), converted, cleanup in zip(mixed_valid, converted_tokens, cleanups):
        remaining_hanja_before_final_cleanup += int(cleanup.get("removed_unconverted_hanja_chars", 0))
        matched_id = hangulized_id_by_token.get(converted)
        converted_rows.append(
            {
                "mixed_token": token,
                "mixed_token_id": token_id,
                "converted_hangul": converted,
                "contains_hanja": contains_hanja(token),
                "is_valid_lexical": True,
                "exact_match_in_hangulized_vocab": matched_id is not None,
                "hangulized_token_id": matched_id,
                "conversion_success": bool(converted) or not contains_hanja(token),
                "conversion_notes": conversion_notes(cleanup),
                "train_frequency": train_freq.get(token_id, 0),
                "dev_frequency": dev_freq.get(token_id, 0),
            }
        )

    invalid_rows = [
        {
            "mixed_token": token,
            "mixed_token_id": token_id,
            "converted_hangul": "",
            "contains_hanja": contains_hanja(token),
            "is_valid_lexical": False,
            "exact_match_in_hangulized_vocab": False,
            "hangulized_token_id": None,
            "conversion_success": False,
            "conversion_notes": ["invalid_lexical_token_excluded"],
            "train_frequency": train_freq.get(token_id, 0),
            "dev_frequency": dev_freq.get(token_id, 0),
        }
        for token, token_id in mixed_ordered
        if not is_valid_lexical(token)
    ]
    all_conversion_rows = sorted(converted_rows + invalid_rows, key=lambda row: row["mixed_token_id"])

    if any(HANJA_RE.search(row["converted_hangul"]) for row in converted_rows):
        raise RuntimeError("Converted Hangul strings still contain Hanja after cleanup.")
    if any(any(ch.isspace() for ch in row["converted_hangul"]) for row in converted_rows):
        raise RuntimeError("Converted Hangul strings contain whitespace.")

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in converted_rows:
        if row["contains_hanja"] and row["converted_hangul"]:
            grouped[row["converted_hangul"]].append(row)

    groups: list[dict] = []
    for converted_hangul, rows in grouped.items():
        rows_sorted = sorted(rows, key=lambda row: row["mixed_token_id"])
        mixed_tokens = [row["mixed_token"] for row in rows_sorted]
        mixed_token_ids = [row["mixed_token_id"] for row in rows_sorted]
        hangulized_id = hangulized_id_by_token.get(converted_hangul)
        groups.append(
            {
                "converted_hangul": converted_hangul,
                "group_size": len(rows_sorted),
                "mixed_tokens": mixed_tokens,
                "mixed_token_ids": mixed_token_ids,
                "exists_in_hangulized_vocab": hangulized_id is not None,
                "hangulized_token_id": hangulized_id,
                "contains_hanja_token_count": sum(1 for row in rows_sorted if row["contains_hanja"]),
                "only_hanja_hangul_token_count": sum(
                    1
                    for token in mixed_tokens
                    if all(contains_hanja(ch) or contains_hangul(ch) for ch in token)
                ),
                "token_length_stats": token_length_stats(mixed_tokens),
                "train_frequency_sum": sum(train_freq.get(token_id, 0) for token_id in mixed_token_ids),
                "dev_frequency_sum": sum(dev_freq.get(token_id, 0) for token_id in mixed_token_ids),
                "train_frequency_by_token": {
                    token: train_freq.get(token_id, 0) for token, token_id in zip(mixed_tokens, mixed_token_ids)
                },
                "dev_frequency_by_token": {
                    token: dev_freq.get(token_id, 0) for token, token_id in zip(mixed_tokens, mixed_token_ids)
                },
            }
        )

    groups.sort(key=lambda group: (group["group_size"], group["dev_frequency_sum"], group["converted_hangul"]), reverse=True)
    collisions = [group for group in groups if group["group_size"] >= 2]
    if not collisions:
        print("WARNING: no N:1 collision groups found.")

    group_side, source_side = distribution(groups)
    collision_sizes = [group["group_size"] for group in collisions]
    overlap_all = overlap_stats(converted_rows, set(hangulized_vocab.values()), hanja_only=False)
    overlap_hanja = overlap_stats(converted_rows, set(hangulized_vocab.values()), hanja_only=True)
    overlap_valid = overlap_stats(converted_rows, hangulized_valid_ids, hanja_only=True)
    overlap_strict = overlap_stats(converted_rows, hangulized_strict_ids, hanja_only=True)
    overlap_loose = overlap_stats(converted_rows, hangulized_loose_ids, hanja_only=True)

    dev_hanja_token_occurrences = sum(count for token_id, count in dev_freq.items() if token_id in mixed_hanja_ids)
    dev_collision_token_ids = {token_id for group in collisions for token_id in group["mixed_token_ids"]}
    dev_collision_occurrences = sum(dev_freq.get(token_id, 0) for token_id in dev_collision_token_ids)
    frequency_weighted_stats = {
        "train_frequency_file": str(args.train_mixed) if args.train_mixed and args.train_mixed.exists() else "",
        "dev_frequency_file": str(args.dev_mixed) if args.dev_mixed and args.dev_mixed.exists() else "",
        "collision_groups_seen_in_train": sum(1 for group in collisions if group["train_frequency_sum"] > 0),
        "collision_groups_seen_in_dev": sum(1 for group in collisions if group["dev_frequency_sum"] > 0),
        "mixed_token_occurrences_in_dev_collision_groups": dev_collision_occurrences,
        "hanja_token_occurrences_in_dev": dev_hanja_token_occurrences,
        "percentage_hanja_token_occurrences_in_dev_belonging_to_n_to_1_groups": dev_collision_occurrences
        / dev_hanja_token_occurrences
        * 100
        if dev_hanja_token_occurrences
        else 0.0,
    }

    top_by_size = sorted(collisions, key=lambda group: (group["group_size"], group["dev_frequency_sum"], group["converted_hangul"]), reverse=True)
    top_by_dev = sorted(collisions, key=lambda group: (group["dev_frequency_sum"], group["group_size"], group["converted_hangul"]), reverse=True)
    robust_collisions_len_ge_2 = [group for group in collisions if len(group["converted_hangul"]) >= 2]
    robust_top_by_size = sorted(
        robust_collisions_len_ge_2,
        key=lambda group: (group["group_size"], group["dev_frequency_sum"], group["converted_hangul"]),
        reverse=True,
    )
    robust_top_by_dev = sorted(
        robust_collisions_len_ge_2,
        key=lambda group: (group["dev_frequency_sum"], group["group_size"], group["converted_hangul"]),
        reverse=True,
    )
    robustness_len_ge_2 = summarize_collision_subset(robust_collisions_len_ge_2, dev_hanja_token_occurrences)
    qualitative = qualitative_examples(top_by_dev, args.seed, 30)

    output_paths = {
        "summary": args.output_dir / "korean_vocab_n_to_1_summary.json",
        "converted": args.output_dir / "mixed_vocab_converted_to_hangul.jsonl",
        "groups": args.output_dir / "n_to_1_groups.jsonl",
        "collisions": args.output_dir / "n_to_1_groups_collisions_only.jsonl",
        "top_size": args.output_dir / "top_n_to_1_groups_by_size.csv",
        "top_dev": args.output_dir / "top_n_to_1_groups_by_dev_frequency.csv",
        "report": args.output_dir / "korean_n_to_1_report.md",
        "robust_collisions_len_ge_2": args.output_dir / "n_to_1_groups_collisions_converted_len_ge_2.jsonl",
        "robust_top_size_len_ge_2": args.output_dir / "top_n_to_1_groups_len_ge_2_by_size.csv",
        "robust_top_dev_len_ge_2": args.output_dir / "top_n_to_1_groups_len_ge_2_by_dev_frequency.csv",
        "robust_report_len_ge_2": args.output_dir / "korean_n_to_1_robustness_len_ge_2_report.md",
    }
    summary = {
        "mixed_vocab_size": len(mixed_vocab),
        "hangulized_vocab_size": len(hangulized_vocab),
        "mixed_valid_lexical_token_count": len(mixed_valid),
        "mixed_hanja_token_count": len(mixed_hanja_ids),
        "hangulized_valid_lexical_token_count": len(hangulized_valid),
        "hangulized_pure_hangul_token_count_strict": len(hangulized_strict_ids),
        "hangulized_pure_hangul_token_count_loose": len(hangulized_loose_ids),
        "converted_valid_mixed_token_count": len(converted_rows),
        "converted_hanja_mixed_token_count": sum(1 for row in converted_rows if row["contains_hanja"]),
        "exact_overlap_count_all_valid": overlap_all["exact_overlap_count"],
        "exact_overlap_rate_all_valid": overlap_all["exact_overlap_rate"],
        "exact_overlap_count_hanja_only": overlap_hanja["exact_overlap_count"],
        "exact_overlap_rate_hanja_only": overlap_hanja["exact_overlap_rate"],
        "overlap_against_hangulized_valid_lexical_hanja_only": overlap_valid,
        "overlap_against_hangulized_pure_hangul_strict_hanja_only": overlap_strict,
        "overlap_against_hangulized_pure_hangul_loose_hanja_only": overlap_loose,
        "collision_distribution_group_side": group_side,
        "collision_distribution_source_token_side": source_side,
        "collision_group_count": len(collisions),
        "max_collision_size": max(collision_sizes) if collision_sizes else 1,
        "mean_collision_size_among_collisions": statistics.mean(collision_sizes) if collision_sizes else 0.0,
        "median_collision_size_among_collisions": statistics.median(collision_sizes) if collision_sizes else 0.0,
        "remaining_hanja_removed_before_final_cleanup_count": remaining_hanja_before_final_cleanup,
        "converted_strings_with_remaining_hanja_after_cleanup": 0,
        "converted_strings_with_whitespace_after_cleanup": 0,
        "converter_backend": args.converter,
        "converter_executable": args.converter,
        "converter_version": converter_ver,
        "converter_settings": {
            "rendering": "hangul-only",
            "disambiguation": "off",
        },
        "cleanup_rules": [
            "remove pure-Hanja parenthetical annotations",
            "remove remaining Hanja after conversion",
            "remove Unicode whitespace",
        ],
        "frequency_weighted_stats": frequency_weighted_stats,
        "robustness_converted_hangul_len_ge_2": robustness_len_ge_2,
        "qualitative_examples": qualitative,
        "debug_conversion_examples": debug_rows,
        "tokenizers_library": tokenizers_lib.__version__,
        "inputs": {
            "mixed_tokenizer": str(args.mixed_tokenizer),
            "hangulized_tokenizer": str(args.hangulized_tokenizer),
            "train_mixed": str(args.train_mixed),
            "dev_mixed": str(args.dev_mixed),
        },
        "outputs": {key: str(path) for key, path in output_paths.items()},
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_paths["summary"].write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_jsonl(output_paths["converted"], all_conversion_rows)
    write_jsonl(output_paths["groups"], groups)
    write_jsonl(output_paths["collisions"], collisions)
    write_jsonl(output_paths["robust_collisions_len_ge_2"], robust_collisions_len_ge_2)
    write_top_size_csv(output_paths["top_size"], top_by_size)
    write_top_size_csv(output_paths["top_dev"], top_by_dev)
    write_top_size_csv(output_paths["robust_top_size_len_ge_2"], robust_top_by_size)
    write_top_size_csv(output_paths["robust_top_dev_len_ge_2"], robust_top_by_dev)
    output_paths["report"].write_text(render_report(summary, groups, top_by_size, top_by_dev, debug_rows), encoding="utf-8")
    output_paths["robust_report_len_ge_2"].write_text(
        render_robustness_report(summary, robust_top_by_size, robust_top_by_dev),
        encoding="utf-8",
    )

    for path in output_paths.values():
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"Expected non-empty output missing: {path}")

    print(f"Mixed Hanja-containing tokens: {summary['mixed_hanja_token_count']}")
    print(f"Hangulized pure Hangul strict tokens: {summary['hangulized_pure_hangul_token_count_strict']}")
    print(f"Hanja-token exact overlap rate: {summary['exact_overlap_rate_hanja_only']:.6f}")
    print(f"Collision groups: {summary['collision_group_count']}")
    print(f"Max collision size: {summary['max_collision_size']}")
    print(f"Dev collision occurrence pct: {frequency_weighted_stats['percentage_hanja_token_occurrences_in_dev_belonging_to_n_to_1_groups']:.2f}%")
    print(f"Robust len>=2 collision groups: {robustness_len_ge_2['collision_group_count']}")
    print(f"Robust len>=2 dev occurrence pct: {robustness_len_ge_2['percentage_hanja_token_occurrences_in_dev_belonging_to_subset']:.2f}%")
    print(f"Summary: {output_paths['summary']}")
    print(f"Report: {output_paths['report']}")


if __name__ == "__main__":
    main()
