from __future__ import annotations

import argparse
import importlib.util
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

from khdb_common import fetch_url, read_jsonl, write_jsonl


SCRIPT_DIR = Path(__file__).resolve().parent


def load_script_module(module_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_DIR / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


extract_module = load_script_module("khdb_extract_article_text", "03_extract_khdb_article_text.py")
filter_module = load_script_module("khdb_filter_mixed_script", "04_filter_mixed_script_articles.py")


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a bounded KHDB article HTML set for tokenizer diagnostics.")
    parser.add_argument(
        "--articles-index",
        type=Path,
        default=Path("4.Korean/korean_khdb_magazine_audit/data/index/articles_index_full19.jsonl"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("4.Korean/korean_khdb_magazine_audit/data/raw_html"),
    )
    parser.add_argument(
        "--output-index",
        type=Path,
        default=Path("4.Korean/korean_khdb_magazine_audit/data/index/articles_downloaded_bounded.jsonl"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("4.Korean/korean_khdb_magazine_audit/data/index/bounded_download_summary.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("4.Korean/korean_khdb_magazine_audit/results/reports/bounded_download_report.md"),
    )
    parser.add_argument("--target-source-chars", type=int, default=8_000_000)
    parser.add_argument("--minimum-source-chars", type=int, default=5_000_000)
    parser.add_argument("--max-source-chars", type=int, default=12_000_000)
    parser.add_argument("--max-articles", type=int, default=8_000)
    parser.add_argument("--per-magazine-char-cap", type=int, default=1_600_000)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefer-strict", type=str_to_bool, default=True)
    parser.add_argument("--include-loose", type=str_to_bool, default=True)
    parser.add_argument("--include-balanced", type=str_to_bool, default=True)
    parser.add_argument("--include-hanja-heavy", type=str_to_bool, default=True)
    parser.add_argument("--exclude-near-classical", type=str_to_bool, default=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def filter_args(seed: int) -> SimpleNamespace:
    return SimpleNamespace(
        max_japanese_omission_markers=1,
        max_japanese_omission_marker_ratio=0.03,
        max_japanese_kana_ratio=0.20,
        min_hanja_context_ratio=0.05,
        max_list_like_line_ratio=0.50,
        max_hanja_parentheses_ratio=0.50,
        seed=seed,
    )


def dedupe_articles(rows: list[dict]) -> list[dict]:
    deduped: list[dict] = []
    seen: set[str] = set()
    for row in rows:
        key = row.get("khdb_id") or row.get("url")
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


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


def should_select(row: dict, args: argparse.Namespace, selected_chars: int, selected_by_magazine: Counter) -> tuple[bool, str]:
    source_chars = int(row.get("body_length_chars", 0) or 0)
    magazine_title = row.get("magazine_title", "")
    if not row.get("extraction_success"):
        return False, "extraction_failed"
    if args.exclude_near_classical and row.get("near_classical_or_mostly_hanja"):
        return False, "near_classical_or_mostly_hanja"
    if not row.get("passes_japanese_omission_filter"):
        return False, "japanese_omission_marker_failed"
    if not row.get("loose_pass") or not args.include_loose:
        return False, "not_loose_pass"
    if row.get("balanced_mixed_pass") and not args.include_balanced:
        return False, "balanced_mixed_excluded"
    if row.get("hanja_heavy_mixed_pass") and not args.include_hanja_heavy:
        return False, "hanja_heavy_mixed_excluded"
    if selected_chars + source_chars > args.max_source_chars:
        return False, "max_source_chars_would_be_exceeded"
    if selected_by_magazine[magazine_title] + source_chars > args.per_magazine_char_cap:
        return False, "per_magazine_char_cap_would_be_exceeded"
    if row.get("strict_pass"):
        return True, "strict_pass_selected"
    return True, "non_strict_loose_pass_selected"


def index_row(source_row: dict, fetched: dict, filtered: dict, selected: bool, reason: str) -> dict:
    fields = [
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
        "near_classical_or_mostly_hanja",
        "loose_pass",
        "strict_pass",
        "passes_japanese_omission_filter",
        "extraction_success",
        "rejection_reasons",
    ]
    output = {field: filtered.get(field, source_row.get(field, "")) for field in fields}
    output.update(
        {
            "url": fetched.get("url") or source_row.get("url", ""),
            "raw_html_path": str(fetched.get("path", "")),
            "download_status": fetched.get("status", ""),
            "downloaded_at": fetched.get("fetched_at", ""),
            "selected_during_download": selected,
            "download_selection_reason": reason,
        }
    )
    return output


def render_report(summary: dict, selected_examples: list[dict], rejected_examples: list[dict]) -> str:
    lines = [
        "# KHDB Bounded Article HTML Download Report",
        "",
        "This bounded run is for a small tokenizer diagnostic corpus, not a full historical Korean corpus.",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(summary, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Selected Chars by Magazine",
        "",
    ]
    for magazine, chars in sorted(summary["selected_chars_by_magazine"].items()):
        lines.append(f"- {magazine}: {chars}")
    lines.extend(["", "## Selected Examples", ""])
    for row in selected_examples[:20]:
        lines.extend(
            [
                f"- {row.get('magazine_title', '')} / {row.get('article_title', '')} / "
                f"{row.get('body_length_chars', 0)} chars / strict={row.get('strict_pass')} / {row.get('url', '')}"
            ]
        )
    lines.extend(["", "## Rejected or Skipped Examples", ""])
    for row in rejected_examples[:20]:
        lines.append(
            f"- {row.get('magazine_title', '')} / {row.get('article_title', '')} / "
            f"{row.get('download_selection_reason', '')} / {row.get('url', '')}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    candidates = dedupe_articles(list(read_jsonl(args.articles_index)))
    ordered = stratified_order(candidates, args.seed)
    local_filter_args = filter_args(args.seed)

    output_rows: list[dict] = []
    selected_rows: list[dict] = []
    rejected_rows: list[dict] = []
    selected_chars_by_magazine: Counter = Counter()
    selected_articles_by_magazine: Counter = Counter()
    failures = 0
    attempted = 0
    downloaded = 0
    extraction_successes = 0
    stop_reason = "candidate_articles_exhausted"

    for candidate in ordered:
        if attempted >= args.max_articles:
            stop_reason = "max_articles_reached"
            break
        if sum(selected_chars_by_magazine.values()) >= args.target_source_chars:
            stop_reason = "target_source_chars_reached"
            break
        if sum(selected_chars_by_magazine.values()) >= args.max_source_chars:
            stop_reason = "max_source_chars_reached"
            break

        magazine_title = candidate.get("magazine_title", "")
        if selected_chars_by_magazine[magazine_title] >= args.per_magazine_char_cap:
            continue

        attempted += 1
        try:
            fetched = fetch_url(
                candidate.get("url", ""),
                cache_dir=args.cache_dir,
                delay=args.delay,
                force=args.force_refresh,
            )
        except Exception as exc:
            failures += 1
            rejected_rows.append(
                {
                    **candidate,
                    "download_selection_reason": f"download_error: {exc}",
                }
            )
            continue

        downloaded += 1
        downloaded_candidate = {**candidate, "raw_html_path": str(fetched["path"])}
        extracted = extract_module.extract_one(downloaded_candidate)
        filtered = filter_module.add_filter_features(extracted, local_filter_args)
        if filtered.get("extraction_success"):
            extraction_successes += 1
        selected, reason = should_select(
            filtered,
            args,
            sum(selected_chars_by_magazine.values()),
            selected_chars_by_magazine,
        )
        row = index_row(downloaded_candidate, fetched, filtered, selected, reason)
        output_rows.append(row)
        if selected:
            source_chars = int(row.get("body_length_chars", 0) or 0)
            selected_chars_by_magazine[row.get("magazine_title", "")] += source_chars
            selected_articles_by_magazine[row.get("magazine_title", "")] += 1
            selected_rows.append(row)
        else:
            rejected_rows.append(row)

        if args.debug and attempted % 50 == 0:
            print(
                f"attempted={attempted} downloaded={downloaded} "
                f"selected_chars={sum(selected_chars_by_magazine.values())}"
            )

    selected_source_chars = sum(selected_chars_by_magazine.values())
    selected_hanja_chars = sum(int(row.get("hanja_count", 0) or 0) for row in selected_rows)
    selected_hangul_chars = sum(int(row.get("hangul_count", 0) or 0) for row in selected_rows)
    hanja_ratios = [float(row.get("hanja_ratio", 0.0) or 0.0) for row in selected_rows]
    summary = {
        "total_candidate_articles_available": len(candidates),
        "total_article_pages_attempted": attempted,
        "total_article_pages_downloaded": downloaded,
        "download_failures": failures,
        "extraction_successes_during_bounded_download": extraction_successes,
        "selected_strict_pass_articles": sum(1 for row in selected_rows if row.get("strict_pass")),
        "selected_loose_pass_articles": sum(1 for row in selected_rows if row.get("loose_pass")),
        "selected_balanced_mixed_articles": sum(1 for row in selected_rows if row.get("balanced_mixed_pass")),
        "selected_hanja_heavy_mixed_articles": sum(1 for row in selected_rows if row.get("hanja_heavy_mixed_pass")),
        "selected_source_chars": selected_source_chars,
        "selected_hanja_chars": selected_hanja_chars,
        "selected_hangul_chars": selected_hangul_chars,
        "mean_hanja_ratio": statistics.mean(hanja_ratios) if hanja_ratios else 0.0,
        "median_hanja_ratio": statistics.median(hanja_ratios) if hanja_ratios else 0.0,
        "selected_chars_by_magazine": dict(selected_chars_by_magazine),
        "selected_articles_by_magazine": dict(selected_articles_by_magazine),
        "target_source_chars": args.target_source_chars,
        "minimum_source_chars": args.minimum_source_chars,
        "max_source_chars": args.max_source_chars,
        "per_magazine_char_cap": args.per_magazine_char_cap,
        "target_source_chars_reached": selected_source_chars >= args.target_source_chars,
        "minimum_source_chars_reached": selected_source_chars >= args.minimum_source_chars,
        "stop_reason": stop_reason,
    }

    write_jsonl(args.output_index, output_rows)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_report(summary, selected_rows, rejected_rows), encoding="utf-8")
    print(f"Candidate articles: {len(candidates)}")
    print(f"Downloaded articles: {downloaded}")
    print(f"Selected source chars: {selected_source_chars}")
    print(f"Target reached: {selected_source_chars >= args.target_source_chars}")
    print(f"Output index: {args.output_index}")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
