from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path

from khdb_common import HANJA_RE, HIRAGANA_RE, KATAKANA_RE, read_jsonl, write_csv, write_jsonl


PARTICLES = [
    "이었다",
    "하였다",
    "되었다",
    "한다",
    "된다",
    "하여",
    "하고",
    "하며",
    "으로",
    "에서",
    "부터",
    "까지",
    "들의",
    "에게",
    "보다",
    "이다",
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "의",
    "에",
    "로",
    "와",
    "과",
    "도",
    "만",
    "중",
    "들",
    "께",
    "하",
]
PARTICLE_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+(?:" + "|".join(map(re.escape, PARTICLES)) + r")")
HANJA_SPAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
NAME_OFFICE_RE = re.compile(r"(氏|君|先生|博士|長|會長|議員|總督|課長|部長|郡|面|府|京城|釜山)")
PAREN_RE = re.compile(r"[\(\[\{（［【].*?[\)\]\}）］】]", re.DOTALL)

JAPANESE_OMISSION_PATTERNS = [
    ("korean_lines_omitted", re.compile(r"이하\s*(?:\d+|[一二三四五六七八九十百千]+|숫자)?\s*줄?\s*일본문", re.I)),
    ("korean_following_japanese_text", re.compile(r"이하.{0,40}일본문", re.I)),
    ("hanja_following_japanese_text", re.compile(r"以下.{0,40}日文", re.I)),
    ("hanja_japanese_text_omitted", re.compile(r"日文\s*省略", re.I)),
    ("korean_japanese_language_omitted", re.compile(r"일본어\s*생략", re.I)),
    ("korean_japanese_text_omitted", re.compile(r"일본문\s*생략", re.I)),
    ("original_japanese_text_omitted", re.compile(r"원문\s*(?:일본문|日文)\s*생략", re.I)),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply mixed-script article quality filters.")
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=Path("4.Korean/korean_khdb_magazine_audit/data/extracted/articles_extracted.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("4.Korean/korean_khdb_magazine_audit/data/filtered"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("4.Korean/korean_khdb_magazine_audit/results/reports/mixed_script_filter_report.md"),
    )
    parser.add_argument("--max-japanese-omission-markers", type=int, default=1)
    parser.add_argument("--max-japanese-omission-marker-ratio", type=float, default=0.03)
    parser.add_argument("--max-japanese-kana-ratio", type=float, default=0.20)
    parser.add_argument("--min-hanja-context-ratio", type=float, default=0.05)
    parser.add_argument("--max-list-like-line-ratio", type=float, default=0.50)
    parser.add_argument("--max-hanja-parentheses-ratio", type=float, default=0.50)
    parser.add_argument(
        "--output-suffix",
        default="",
        help="Optional suffix inserted before output file extensions, e.g. 'bounded'.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def output_name(base_name: str, suffix: str) -> str:
    clean_suffix = suffix.strip()
    if not clean_suffix:
        return base_name
    if not clean_suffix.startswith("_"):
        clean_suffix = "_" + clean_suffix
    path = Path(base_name)
    return f"{path.stem}{clean_suffix}{path.suffix}"


def list_like_line_ratio(text: str) -> float:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return 0.0
    list_like = 0
    separators = re.compile(r"[,、·ㆍ/|;；:：\(\)\[\]（）]")
    sentence_punct = re.compile(r"[.。!?！？다요]$")
    for line in lines:
        chars = len(re.sub(r"\s+", "", line))
        sep_count = len(separators.findall(line))
        hanja_count = len(HANJA_RE.findall(line))
        word_count = len(line.split())
        if chars <= 18 and (sep_count >= 2 or word_count >= 3):
            list_like += 1
        elif sep_count >= 4 and not sentence_punct.search(line):
            list_like += 1
        elif hanja_count >= 6 and chars <= 30 and not sentence_punct.search(line):
            list_like += 1
    return list_like / len(lines)


def hanja_in_parentheses_ratio(text: str) -> float:
    total_hanja = len(HANJA_RE.findall(text))
    if total_hanja == 0:
        return 0.0
    in_parens = sum(len(HANJA_RE.findall(match.group(0))) for match in PAREN_RE.finditer(text))
    return in_parens / total_hanja


def find_japanese_omission_matches(text: str) -> list[dict]:
    matches: list[dict] = []
    seen_spans: set[tuple[int, int]] = set()
    line_starts: list[tuple[int, str]] = []
    offset = 0
    for line in text.splitlines():
        line_starts.append((offset, line))
        offset += len(line) + 1
    for line_start, line in line_starts:
        for pattern_name, pattern in JAPANESE_OMISSION_PATTERNS:
            for match in pattern.finditer(line):
                span = (line_start + match.start(), line_start + match.end())
                if span in seen_spans:
                    continue
                seen_spans.add(span)
                matches.append(
                    {
                        "pattern_name": pattern_name,
                        "matched_text": match.group(0),
                        "line": line,
                        "char_start": span[0],
                        "char_end": span[1],
                    }
                )
    return matches


def add_filter_features(row: dict, args: argparse.Namespace) -> dict:
    text = row.get("body_text", "")
    spans = HANJA_SPAN_RE.findall(text)
    particle_context_count = len(PARTICLE_RE.findall(text))
    name_office_marker_count = len(NAME_OFFICE_RE.findall(text))
    japanese_matches = find_japanese_omission_matches(text)
    line_count = max(1, row.get("line_count", 0) or len([line for line in text.splitlines() if line.strip()]))
    kana_count = len(HIRAGANA_RE.findall(text)) + len(KATAKANA_RE.findall(text))
    average_span_length = statistics.mean([len(span) for span in spans]) if spans else 0.0
    output = {
        **row,
        "list_like_line_ratio": list_like_line_ratio(text),
        "hanja_particle_context_count": particle_context_count,
        "number_of_hanja_spans": len(spans),
        "hanja_particle_context_ratio": particle_context_count / max(1, len(spans)),
        "name_office_marker_count": name_office_marker_count,
        "name_office_marker_ratio": name_office_marker_count / max(1, len(spans)),
        "hanja_in_parentheses_ratio": hanja_in_parentheses_ratio(text),
        "average_hanja_span_length": average_span_length,
        "japanese_omission_marker_matches": japanese_matches,
        "japanese_omission_marker_count": len(japanese_matches),
        "japanese_omission_marker_lines": len({match["line"] for match in japanese_matches}),
        "japanese_omission_marker_ratio": len({match["line"] for match in japanese_matches}) / line_count,
        "japanese_kana_count": kana_count,
    }
    output["passes_japanese_omission_filter"] = (
        output["japanese_omission_marker_count"] <= args.max_japanese_omission_markers
        and output["japanese_omission_marker_ratio"] <= args.max_japanese_omission_marker_ratio
    )
    output["passes_japanese_kana_diagnostic"] = (
        kana_count < 20 or output.get("japanese_kana_ratio", 0.0) <= args.max_japanese_kana_ratio
    )
    output["basic_pass"] = (
        bool(output.get("extraction_success"))
        and output.get("body_length_chars", 0) >= 200
        and output.get("hanja_count", 0) >= 20
        and output.get("hangul_count", 0) >= 20
    )
    output["near_classical_or_mostly_hanja"] = (
        output.get("hanja_ratio", 0.0) > 0.90 or output.get("hangul_count", 0) < 100
    )
    output["balanced_mixed_pass"] = (
        output["basic_pass"]
        and 0.05 <= output.get("hanja_ratio", 0.0) <= 0.70
        and output["passes_japanese_omission_filter"]
    )
    output["hanja_heavy_mixed_pass"] = (
        bool(output.get("extraction_success"))
        and output.get("body_length_chars", 0) >= 200
        and output.get("hanja_count", 0) >= 20
        and output.get("hangul_count", 0) >= 100
        and 0.70 < output.get("hanja_ratio", 0.0) <= 0.90
        and output["passes_japanese_omission_filter"]
        and output["hanja_particle_context_ratio"] >= args.min_hanja_context_ratio
    )
    output["passes_content_mixed_filter"] = (
        output["list_like_line_ratio"] <= args.max_list_like_line_ratio
        and output["hanja_particle_context_ratio"] >= args.min_hanja_context_ratio
        and output["hanja_in_parentheses_ratio"] <= args.max_hanja_parentheses_ratio
    )
    output["loose_pass"] = output["balanced_mixed_pass"] or output["hanja_heavy_mixed_pass"]
    output["strict_pass"] = output["loose_pass"] and output["passes_content_mixed_filter"]
    output["passes_basic"] = output["basic_pass"]
    output["passes_japanese_marker_filter"] = output["passes_japanese_omission_filter"]

    reasons = []
    if not output.get("extraction_success") or output.get("body_length_chars", 0) < 200:
        reasons.append("too_short")
    if output.get("hanja_count", 0) < 20:
        reasons.append("too_few_hanja")
    if output.get("hangul_count", 0) < 20:
        reasons.append("too_few_hangul")
    if output.get("hanja_ratio", 0.0) < 0.05:
        reasons.append("hanja_ratio_too_low")
    if output["near_classical_or_mostly_hanja"]:
        reasons.append("near_classical_or_mostly_hanja")
    if not output["passes_japanese_omission_filter"]:
        reasons.append("japanese_omission_marker_failed")
    if not output["passes_content_mixed_filter"]:
        reasons.append("content_mixed_filter_failed")
    if output["list_like_line_ratio"] > args.max_list_like_line_ratio:
        reasons.append("list_like_failed")
    if output["hanja_particle_context_ratio"] < args.min_hanja_context_ratio:
        reasons.append("hanja_context_too_low")
    if output["hanja_in_parentheses_ratio"] > args.max_hanja_parentheses_ratio:
        reasons.append("hanja_parentheses_too_high")
    if not output["loose_pass"] and not any(reason in reasons for reason in ["near_classical_or_mostly_hanja", "japanese_omission_marker_failed", "hanja_ratio_too_low"]):
        if output.get("hanja_ratio", 0.0) > 0.90:
            reasons.append("near_classical_or_mostly_hanja")
    output["rejection_reasons"] = reasons
    return output


INDEX_FIELDS = [
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
    "basic_pass",
    "balanced_mixed_pass",
    "hanja_heavy_mixed_pass",
    "near_classical_or_mostly_hanja",
    "loose_pass",
    "strict_pass",
    "list_like_line_ratio",
    "hanja_particle_context_ratio",
    "hanja_in_parentheses_ratio",
    "name_office_marker_count",
    "name_office_marker_ratio",
    "japanese_omission_marker_count",
    "raw_html_path",
]


def index_row(row: dict, include_reasons: bool = False) -> dict:
    output = {field: row.get(field, "") for field in INDEX_FIELDS}
    if include_reasons:
        output["rejection_reasons"] = row.get("rejection_reasons", [])
        output["japanese_omission_marker_matches"] = row.get("japanese_omission_marker_matches", [])
    return output


def summarize_by_magazine(rows: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row.get("magazine_title", "")].append(row)
    summary_rows: list[dict] = []
    for magazine_title, group in groups.items():
        loose = [row for row in group if row.get("loose_pass")]
        hanja_ratios = [row.get("hanja_ratio", 0.0) for row in loose]
        summary_rows.append(
            {
                "magazine_title": magazine_title,
                "total_extracted_articles": len(group),
                "extraction_success_count": sum(1 for row in group if row.get("extraction_success")),
                "basic_pass_count": sum(1 for row in group if row.get("basic_pass")),
                "balanced_mixed_count": sum(1 for row in group if row.get("balanced_mixed_pass")),
                "hanja_heavy_mixed_count": sum(1 for row in group if row.get("hanja_heavy_mixed_pass")),
                "loose_pass_count": len(loose),
                "strict_pass_count": sum(1 for row in group if row.get("strict_pass")),
                "near_classical_or_mostly_hanja_count": sum(1 for row in group if row.get("near_classical_or_mostly_hanja")),
                "total_body_chars_loose": sum(row.get("body_length_chars", 0) for row in loose),
                "total_hangul_chars_loose": sum(row.get("hangul_count", 0) for row in loose),
                "total_hanja_chars_loose": sum(row.get("hanja_count", 0) for row in loose),
                "mean_hanja_ratio_loose": statistics.mean(hanja_ratios) if hanja_ratios else 0.0,
                "median_hanja_ratio_loose": statistics.median(hanja_ratios) if hanja_ratios else 0.0,
            }
        )
    return sorted(summary_rows, key=lambda row: row["magazine_title"])


def overall_summary(rows: list[dict], by_magazine: list[dict]) -> dict:
    loose = [row for row in rows if row.get("loose_pass")]
    return {
        "total_candidate_pages": len(rows),
        "total_extracted_articles": len(rows),
        "extraction_success_count": sum(1 for row in rows if row.get("extraction_success")),
        "basic_pass_count": sum(1 for row in rows if row.get("basic_pass")),
        "balanced_mixed_count": sum(1 for row in rows if row.get("balanced_mixed_pass")),
        "hanja_heavy_mixed_count": sum(1 for row in rows if row.get("hanja_heavy_mixed_pass")),
        "near_classical_or_mostly_hanja_count": sum(1 for row in rows if row.get("near_classical_or_mostly_hanja")),
        "japanese_omission_filter_pass_count": sum(1 for row in rows if row.get("passes_japanese_omission_filter")),
        "loose_pass_count": len(loose),
        "strict_pass_count": sum(1 for row in rows if row.get("strict_pass")),
        "per_magazine_counts": by_magazine,
        "total_body_chars_in_loose_pass": sum(row.get("body_length_chars", 0) for row in loose),
        "total_hanja_chars_in_loose_pass": sum(row.get("hanja_count", 0) for row in loose),
        "total_hangul_chars_in_loose_pass": sum(row.get("hangul_count", 0) for row in loose),
    }


def example_block(row: dict, include_body: bool = True) -> list[str]:
    lines = [
        f"### {row.get('magazine_title', '')} {row.get('issue_title', '')} / {row.get('article_title', '')}".strip(),
        "",
        f"- khdb_id: `{row.get('khdb_id', '')}`",
        f"- url: {row.get('url', '')}",
        f"- hanja: {row.get('hanja_count', 0)}; hangul: {row.get('hangul_count', 0)}; hanja_ratio: {row.get('hanja_ratio', 0):.4f}",
        f"- flags: basic={row.get('basic_pass')} balanced={row.get('balanced_mixed_pass')} hanja_heavy={row.get('hanja_heavy_mixed_pass')} loose={row.get('loose_pass')} strict={row.get('strict_pass')}",
        f"- rejection reasons: {row.get('rejection_reasons', [])}",
    ]
    if row.get("japanese_omission_marker_matches"):
        lines.append(f"- Japanese omission matches: {json.dumps(row.get('japanese_omission_marker_matches'), ensure_ascii=False)}")
    if include_body:
        lines.extend(["", "```text", row.get("body_text", "")[:300], "```"])
    lines.append("")
    return lines


def render_report(rows: list[dict], summary: dict, seed: int) -> str:
    strict = [row for row in rows if row.get("strict_pass")]
    balanced = [row for row in rows if row.get("balanced_mixed_pass")]
    hanja_heavy = [row for row in rows if row.get("hanja_heavy_mixed_pass")]
    rejected = [row for row in rows if not row.get("strict_pass")]
    marker_rows = [row for row in rows if row.get("japanese_omission_marker_matches")]
    rng = random.Random(seed)
    sections = [
        ("Top Strict-Pass Articles by Hanja Count", sorted(strict, key=lambda row: row.get("hanja_count", 0), reverse=True)[:20]),
        ("Top Hanja-Heavy Articles by Hanja Count", sorted(hanja_heavy, key=lambda row: row.get("hanja_count", 0), reverse=True)[:20]),
        ("Random Balanced Examples", rng.sample(balanced, min(20, len(balanced))) if balanced else []),
        ("Random Hanja-Heavy Examples", rng.sample(hanja_heavy, min(20, len(hanja_heavy))) if hanja_heavy else []),
        ("Rejected Examples", rejected[:20]),
        ("Japanese Omission Marker Matches", marker_rows[:20]),
    ]
    lines = [
        "# KHDB Mixed-Script Filter Report",
        "",
        "The name/office/list filter is heuristic and should be manually audited before downstream use.",
        "",
        "## Overall Counts",
        "",
        "```json",
        json.dumps({k: v for k, v in summary.items() if k != "per_magazine_counts"}, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Per-Magazine Counts",
        "",
        "| magazine | extracted | basic | balanced | hanja-heavy | loose | strict | near-classical |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["per_magazine_counts"]:
        lines.append(
            f"| {row['magazine_title']} | {row['total_extracted_articles']} | {row['basic_pass_count']} | {row['balanced_mixed_count']} | {row['hanja_heavy_mixed_count']} | {row['loose_pass_count']} | {row['strict_pass_count']} | {row['near_classical_or_mostly_hanja_count']} |"
        )
    for heading, section_rows in sections:
        lines.extend(["", f"## {heading}", ""])
        for row in section_rows:
            lines.extend(example_block(row))
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    rows = [add_filter_features(row, args) for row in read_jsonl(args.input_jsonl)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / output_name("all_articles_with_filter_flags.jsonl", args.output_suffix), rows)
    write_jsonl(
        args.output_dir / output_name("balanced_mixed_article_index.jsonl", args.output_suffix),
        [index_row(row) for row in rows if row.get("balanced_mixed_pass")],
    )
    write_jsonl(
        args.output_dir / output_name("hanja_heavy_mixed_article_index.jsonl", args.output_suffix),
        [index_row(row) for row in rows if row.get("hanja_heavy_mixed_pass")],
    )
    write_jsonl(
        args.output_dir / output_name("strict_pass_article_index.jsonl", args.output_suffix),
        [index_row(row) for row in rows if row.get("strict_pass")],
    )
    write_jsonl(
        args.output_dir / output_name("loose_pass_article_index.jsonl", args.output_suffix),
        [index_row(row) for row in rows if row.get("loose_pass")],
    )
    write_jsonl(
        args.output_dir / output_name("near_classical_or_mostly_hanja_index.jsonl", args.output_suffix),
        [index_row(row) for row in rows if row.get("near_classical_or_mostly_hanja")],
    )
    write_jsonl(
        args.output_dir / output_name("rejected_article_index.jsonl", args.output_suffix),
        [index_row(row, include_reasons=True) for row in rows if not row.get("strict_pass")],
    )
    by_magazine = summarize_by_magazine(rows)
    write_csv(args.output_dir / output_name("filter_summary_by_magazine.csv", args.output_suffix), by_magazine)
    summary = overall_summary(rows, by_magazine)
    (args.output_dir / output_name("filter_summary_overall.json", args.output_suffix)).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_report(rows, summary, args.seed), encoding="utf-8")
    print(f"Articles: {len(rows)}")
    print(f"Balanced mixed: {summary['balanced_mixed_count']}")
    print(f"Hanja-heavy mixed: {summary['hanja_heavy_mixed_count']}")
    print(f"Loose pass: {summary['loose_pass_count']}")
    print(f"Strict pass: {summary['strict_pass_count']}")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
