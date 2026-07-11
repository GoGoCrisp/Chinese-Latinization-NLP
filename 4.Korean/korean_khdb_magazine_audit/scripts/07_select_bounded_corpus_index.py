from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from khdb_common import read_jsonl, write_jsonl


SELECTED_FIELDS = [
    "khdb_id",
    "url",
    "magazine_title",
    "issue_title",
    "publication_date",
    "author",
    "article_title",
    "body_length_chars",
    "hangul_count",
    "hanja_count",
    "hanja_ratio",
    "balanced_mixed_pass",
    "hanja_heavy_mixed_pass",
    "loose_pass",
    "strict_pass",
    "raw_html_path",
    "extraction_success",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select the final bounded KHDB diagnostic article index.")
    parser.add_argument(
        "--filtered-jsonl",
        type=Path,
        default=Path("4.Korean/korean_khdb_magazine_audit/data/filtered/all_articles_with_filter_flags_bounded.jsonl"),
    )
    parser.add_argument(
        "--output-selected-index",
        type=Path,
        default=Path("4.Korean/korean_khdb_magazine_audit/data/filtered/selected_diagnostic_article_index.jsonl"),
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("4.Korean/korean_khdb_magazine_audit/data/filtered/selected_diagnostic_corpus_summary.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("4.Korean/korean_khdb_magazine_audit/results/reports/selected_diagnostic_corpus_report.md"),
    )
    parser.add_argument("--target-source-chars", type=int, default=8_000_000)
    parser.add_argument("--minimum-source-chars", type=int, default=5_000_000)
    parser.add_argument("--max-source-chars", type=int, default=12_000_000)
    parser.add_argument("--per-magazine-char-cap", type=int, default=1_600_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-near-classical", action="store_true")
    return parser.parse_args()


def candidate_rows(rows: list[dict], include_near_classical: bool) -> list[dict]:
    candidates: list[dict] = []
    for row in rows:
        if not row.get("loose_pass"):
            continue
        if not row.get("passes_japanese_omission_filter", True):
            continue
        if row.get("japanese_omission_marker_matches"):
            continue
        if row.get("near_classical_or_mostly_hanja") and not include_near_classical:
            continue
        candidates.append(row)
    return candidates


def stratified_order(rows: list[dict], seed: int) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row.get("magazine_title", "")].append(row)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    magazine_titles = sorted(groups)
    rng.shuffle(magazine_titles)
    ordered: list[dict] = []
    while any(groups.values()):
        for magazine_title in magazine_titles:
            if groups[magazine_title]:
                ordered.append(groups[magazine_title].pop())
    return ordered


def select_rows(rows: list[dict], args: argparse.Namespace) -> list[dict]:
    strict_rows = [row for row in rows if row.get("strict_pass")]
    loose_non_strict = [row for row in rows if row.get("loose_pass") and not row.get("strict_pass")]
    ordered = stratified_order(strict_rows, args.seed) + stratified_order(loose_non_strict, args.seed + 1)
    selected: list[dict] = []
    chars_by_magazine: Counter = Counter()
    selected_chars = 0
    seen: set[str] = set()
    for row in ordered:
        key = row.get("khdb_id") or row.get("url")
        source_chars = int(row.get("body_length_chars", 0) or 0)
        magazine_title = row.get("magazine_title", "")
        if not key or key in seen or source_chars <= 0:
            continue
        if selected_chars >= args.target_source_chars:
            break
        if selected_chars + source_chars > args.max_source_chars:
            continue
        if chars_by_magazine[magazine_title] + source_chars > args.per_magazine_char_cap:
            continue
        seen.add(key)
        selected_chars += source_chars
        chars_by_magazine[magazine_title] += source_chars
        reason = "strict_pass_preferred" if row.get("strict_pass") else "non_strict_loose_pass_backfill"
        output = {field: row.get(field, "") for field in SELECTED_FIELDS}
        output["selection_rank"] = len(selected) + 1
        output["selection_reason"] = reason
        selected.append(output)
    return selected


def summarize(selected: list[dict], rows: list[dict], args: argparse.Namespace) -> dict:
    chars_by_magazine: Counter = Counter()
    articles_by_magazine: Counter = Counter()
    for row in selected:
        magazine_title = row.get("magazine_title", "")
        chars_by_magazine[magazine_title] += int(row.get("body_length_chars", 0) or 0)
        articles_by_magazine[magazine_title] += 1
    selected_source_chars = sum(chars_by_magazine.values())
    hanja_ratios = [float(row.get("hanja_ratio", 0.0) or 0.0) for row in selected]
    return {
        "target_source_chars": args.target_source_chars,
        "minimum_source_chars": args.minimum_source_chars,
        "max_source_chars": args.max_source_chars,
        "selected_source_chars": selected_source_chars,
        "selected_article_count": len(selected),
        "selected_strict_count": sum(1 for row in selected if row.get("strict_pass")),
        "selected_loose_count": sum(1 for row in selected if row.get("loose_pass")),
        "selected_balanced_count": sum(1 for row in selected if row.get("balanced_mixed_pass")),
        "selected_hanja_heavy_count": sum(1 for row in selected if row.get("hanja_heavy_mixed_pass")),
        "selected_hanja_chars": sum(int(row.get("hanja_count", 0) or 0) for row in selected),
        "selected_hangul_chars": sum(int(row.get("hangul_count", 0) or 0) for row in selected),
        "mean_hanja_ratio": statistics.mean(hanja_ratios) if hanja_ratios else 0.0,
        "median_hanja_ratio": statistics.median(hanja_ratios) if hanja_ratios else 0.0,
        "chars_by_magazine": dict(chars_by_magazine),
        "articles_by_magazine": dict(articles_by_magazine),
        "whether_target_reached": selected_source_chars >= args.target_source_chars,
        "whether_minimum_reached": selected_source_chars >= args.minimum_source_chars,
        "stop_reason": "target_reached" if selected_source_chars >= args.target_source_chars else "candidate_pool_exhausted",
        "bounded_filtered_article_count": len(rows),
        "bounded_loose_candidate_count": sum(1 for row in rows if row.get("loose_pass")),
    }


def row_preview(row: dict, limit: int = 300) -> str:
    return str(row.get("body_text", "")).replace("\n", " ")[:limit]


def render_report(selected: list[dict], all_rows: list[dict], summary: dict, seed: int) -> str:
    rng = random.Random(seed)
    selected_ids = {row.get("khdb_id") for row in selected}
    selected_full = [row for row in all_rows if row.get("khdb_id") in selected_ids]
    rejected = [row for row in all_rows if row.get("khdb_id") not in selected_ids]
    random_examples = rng.sample(selected_full, min(20, len(selected_full))) if selected_full else []
    high_hanja = sorted(selected_full, key=lambda row: row.get("hanja_ratio", 0.0), reverse=True)[:20]
    rejected_examples = rejected[:20]
    lines = [
        "# KHDB Selected Diagnostic Corpus Report",
        "",
        "This is a small tokenizer diagnostic corpus, not a full historical Korean corpus.",
        "",
        "Later stages should chunk article body text into 300-800 character aligned chunks.",
        "The mixed and hangulized corpora must remain line-aligned.",
        "The tokenizer training target is 32K standard BPE.",
        "The fertility denominator is non-space chars from the original mixed-script source.",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(summary, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Per-Magazine Distribution",
        "",
        "| magazine | articles | source chars |",
        "|---|---:|---:|",
    ]
    for magazine, chars in sorted(summary["chars_by_magazine"].items()):
        lines.append(f"| {magazine} | {summary['articles_by_magazine'].get(magazine, 0)} | {chars} |")
    for heading, rows in [
        ("Random Selected Examples", random_examples),
        ("High-Hanja Selected Examples", high_hanja),
        ("Rejected Examples from Bounded Run", rejected_examples),
    ]:
        lines.extend(["", f"## {heading}", ""])
        for row in rows:
            lines.extend(
                [
                    f"### {row.get('magazine_title', '')} / {row.get('article_title', '')}",
                    "",
                    f"- url: {row.get('url', '')}",
                    f"- chars: {row.get('body_length_chars', 0)}; hanja_ratio: {row.get('hanja_ratio', 0):.4f}",
                    f"- flags: balanced={row.get('balanced_mixed_pass')} hanja_heavy={row.get('hanja_heavy_mixed_pass')} strict={row.get('strict_pass')}",
                    "",
                    "```text",
                    row_preview(row),
                    "```",
                    "",
                ]
            )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    rows = list(read_jsonl(args.filtered_jsonl))
    candidates = candidate_rows(rows, args.include_near_classical)
    selected = select_rows(candidates, args)
    summary = summarize(selected, rows, args)
    write_jsonl(args.output_selected_index, selected)
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_report(selected, rows, summary, args.seed), encoding="utf-8")
    print(f"Selected articles: {len(selected)}")
    print(f"Selected source chars: {summary['selected_source_chars']}")
    print(f"Target reached: {summary['whether_target_reached']}")
    print(f"Selected index: {args.output_selected_index}")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
