from __future__ import annotations

import argparse
import importlib.metadata
import json
import random
import re
import shutil
import subprocess
import sys
import unicodedata
from pathlib import Path

from khdb_common import HANJA_RE, HANGUL_RE, count_chars, read_jsonl, write_jsonl


PAGE_MARKER_RE = re.compile(r"<\s*\d+\s*(?:-\s*\d+\s*)?>")
WHITESPACE_RE = re.compile(r"\s+", re.UNICODE)
PUNCT_BOUNDARY_RE = re.compile(r"[。.!?！？\.\)]")
HANJA_PAREN_ANNOTATION_RE = re.compile(r"[\(（][\u3400-\u4dbf\u4e00-\u9fff]+[\)）]")

ARTICLE_FIELDS = [
    "khdb_id",
    "url",
    "magazine_title",
    "issue_title",
    "publication_date",
    "author",
    "article_title",
    "balanced_mixed_pass",
    "hanja_heavy_mixed_pass",
    "loose_pass",
    "strict_pass",
    "selection_rank",
    "selection_reason",
    "raw_html_path",
    "extraction_success",
]


SANITY_EXAMPLES = [
    ("大韓民國은民主共和國이다.", "대한민국은민주공화국이다."),
    ("漢字北京標識", "한자베이징표지"),
    ("國民의自由와權利를保障한다.", "국민의자유와권리를보장한다."),
    ("會社定款의作成과其外一定順序經야成立되故로", "historical_hangul_preserved"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare no-space mixed and Hangulized KHDB diagnostic corpora.")
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=Path(
            "4.Korean/korean_khdb_magazine_audit/data/corpus/article_level/selected_diagnostic_mixed_articles.jsonl"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("4.Korean/korean_khdb_magazine_audit/data/corpus"),
    )
    parser.add_argument("--converter", choices=["gukhanmun"], default="gukhanmun")
    parser.add_argument("--fallback-converter", choices=["none", "hanja"], default="hanja")
    parser.add_argument("--chunk-size", type=int, default=600)
    parser.add_argument("--min-chunk-size", type=int, default=300)
    parser.add_argument("--max-chunk-size", type=int, default=800)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-articles", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def clean_to_nospace(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = PAGE_MARKER_RE.sub("", text)
    return WHITESPACE_RE.sub("", text)


def remove_all_whitespace(text: str) -> str:
    return WHITESPACE_RE.sub("", text or "")


def script_counts(text: str) -> dict:
    counts = count_chars(text)
    return {
        "source_chars": counts["body_length_chars"],
        "hanja_count": counts["hanja_count"],
        "hangul_count": counts["hangul_count"],
    }


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


def hanja_version() -> str:
    try:
        return importlib.metadata.version("hanja")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def converter_status(args: argparse.Namespace) -> dict:
    gukhanmun_path = shutil.which("gukhanmun")
    hanja_available = False
    if args.fallback_converter == "hanja":
        try:
            import hanja  # noqa: F401

            hanja_available = True
        except ImportError:
            hanja_available = False
    return {
        "gukhanmun_path": gukhanmun_path,
        "gukhanmun_version": converter_version(gukhanmun_path) if gukhanmun_path else "",
        "hanja_available": hanja_available,
        "hanja_version": hanja_version() if hanja_available else "",
    }


def convert_with_gukhanmun(text: str, timeout: float) -> str:
    result = subprocess.run(
        ["gukhanmun", "--rendering", "hangul-only", "--disambiguation", "off"],
        input=text,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=True,
    )
    return remove_all_whitespace(result.stdout.rstrip("\n"))


def convert_with_hanja(text: str) -> str:
    import hanja

    return remove_all_whitespace(hanja.translate(text, "substitution"))


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


def convert_text(text: str, args: argparse.Namespace, status: dict) -> tuple[str, str, bool, list[str]]:
    notes: list[str] = []
    if args.converter == "gukhanmun" and status["gukhanmun_path"]:
        try:
            converted, cleanup = sanitize_hangulized(convert_with_gukhanmun(text, args.timeout))
            cleanup_notes = [f"{key}={value}" for key, value in cleanup.items() if value and key != "removed_unconverted_hanja_examples"]
            return converted, "gukhanmun", True, notes + cleanup_notes
        except subprocess.TimeoutExpired:
            notes.append("gukhanmun_timeout")
        except subprocess.CalledProcessError as exc:
            notes.append(f"gukhanmun_failed_returncode_{exc.returncode}")
    elif args.converter == "gukhanmun":
        notes.append("gukhanmun_not_found")

    if args.fallback_converter == "hanja" and status["hanja_available"]:
        try:
            converted, cleanup = sanitize_hangulized(convert_with_hanja(text))
            cleanup_notes = [f"{key}={value}" for key, value in cleanup.items() if value and key != "removed_unconverted_hanja_examples"]
            return converted, "hanja_fallback", True, notes + ["fallback_used"] + cleanup_notes
        except Exception as exc:
            notes.append(f"hanja_fallback_failed: {exc}")

    install_message = (
        "No usable converter found. Install Gukhanmun with `cargo install gukhanmun-cli gukhanmun-mkdict` "
        "or download a prebuilt `gukhanmun` binary from https://github.com/dahlia/gukhanmun/releases. "
        "Optional fallback: `.venv/bin/pip install hanja`."
    )
    raise RuntimeError(install_message + " Notes: " + "; ".join(notes))


def split_chunk_at_boundary(text: str, start: int, args: argparse.Namespace) -> int:
    remaining = len(text) - start
    if remaining <= args.max_chunk_size:
        return len(text)
    target = start + args.chunk_size
    min_pos = start + args.min_chunk_size
    max_pos = min(start + args.max_chunk_size, len(text))
    window = text[min_pos:max_pos]
    candidates = [min_pos + match.end() for match in PUNCT_BOUNDARY_RE.finditer(window)]
    if candidates:
        return min(candidates, key=lambda pos: abs(pos - target))
    return min(target, max_pos)


def chunk_text(text: str, args: argparse.Namespace) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = split_chunk_at_boundary(text, start, args)
        chunks.append(text[start:end])
        start = end
    return [chunk for chunk in chunks if chunk]


def article_output_row(row: dict, mixed_nospace: str, hangulized: str, backend: str, success: bool, notes: list[str]) -> dict:
    mixed_counts = script_counts(mixed_nospace)
    hangulized_counts = script_counts(hangulized)
    output = {field: row.get(field, "") for field in ARTICLE_FIELDS}
    output.update(
        {
            "body_text_original": row.get("body_text", ""),
            "body_text_nospace": mixed_nospace,
            "body_text_hangulized_nospace": hangulized,
            "source_chars_nospace": mixed_counts["source_chars"],
            "mixed_hanja_count": mixed_counts["hanja_count"],
            "mixed_hangul_count": mixed_counts["hangul_count"],
            "hangulized_hanja_count": hangulized_counts["hanja_count"],
            "hangulized_hangul_count": hangulized_counts["hangul_count"],
            "converter_backend": backend,
            "conversion_success": success,
            "conversion_notes": notes,
        }
    )
    return output


def render_example(row: dict) -> list[str]:
    return [
        f"### {row.get('magazine_title', '')} / {row.get('article_title', '')}",
        "",
        f"- khdb_id: `{row.get('khdb_id', '')}`",
        f"- backend: {row.get('converter_backend', '')}; success: {row.get('conversion_success')}",
        f"- mixed chars: {row.get('source_chars_nospace', 0)}; leftover hanja: {row.get('hangulized_hanja_count', 0)}",
        "",
        "```text",
        row.get("body_text_original", "")[:300].replace("\n", " "),
        "```",
        "",
        "```text",
        row.get("body_text_nospace", "")[:300],
        "```",
        "",
        "```text",
        row.get("body_text_hangulized_nospace", "")[:300],
        "```",
        "",
    ]


def render_report(summary: dict, sanity_rows: list[dict], article_rows: list[dict], seed: int) -> str:
    rng = random.Random(seed)
    random_examples = rng.sample(article_rows, min(20, len(article_rows))) if article_rows else []
    high_hanja = sorted(article_rows, key=lambda row: row.get("mixed_hanja_count", 0), reverse=True)[:20]
    leftovers = [row for row in article_rows if row.get("hangulized_hanja_count", 0) > 0][:20]
    lines = [
        "# KHDB Step 2 No-Space and Hangulized Corpus Report",
        "",
        "This is an automatic preprocessing step for a tokenizer diagnostic corpus, not gold annotation.",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(summary, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Converter",
        "",
        f"- backend: {summary.get('converter_backend', '')}",
        f"- version: {summary.get('converter_version', '')}",
        "- Gukhanmun is an external GPL-3.0 tool; its source is not vendored in this repository.",
        "- Optional fallback is Python `hanja` when explicitly enabled and installed.",
        "",
        "## Cleaning Rules",
        "",
        "- Unicode NFKC normalization.",
        "- Remove KHDB page markers such as `<44>`, `<24-42>`, and `<256-257>`.",
        "- Remove all Unicode whitespace.",
        "- Preserve Hanja, Hangul, punctuation, digits, Latin text, and symbols except page markers.",
        "",
        "## Sanity Examples",
        "",
    ]
    for row in sanity_rows:
        lines.extend(
            [
                f"### {row['input']}",
                "",
                f"- expected: {row['expected']}",
                f"- output: {row['output']}",
                f"- leftover_hanja: {row['leftover_hanja_count']}",
                f"- backend: {row['converter_backend']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Validation",
            "",
            f"- article line alignment: {summary.get('mixed_article_lines')} == {summary.get('hangulized_article_lines')}",
            f"- chunk line alignment: {summary.get('mixed_chunk_lines')} == {summary.get('hangulized_chunk_lines')}",
            f"- no-space check passed: {summary.get('whether_no_space_check_passed')}",
            f"- line alignment passed: {summary.get('whether_line_alignment_passed')}",
            f"- Hanja leftovers after conversion: {summary.get('total_hangulized_hanja_leftover_chars')}",
            "",
            "## Caveats",
            "",
            "- Hangulized corpus is generated by automatic Hanja-to-Hangul conversion.",
            "- Conversion may be imperfect for historical spellings, names, rare Hanja, and Classical Chinese-heavy passages.",
            "- Do not treat this as a gold-standard Korean transliteration corpus.",
        ]
    )
    for heading, rows in [
        ("Random Conversion Examples", random_examples),
        ("High-Hanja Examples", high_hanja),
        ("Examples with Hanja Leftovers", leftovers),
    ]:
        lines.extend(["", f"## {heading}", ""])
        for row in rows:
            lines.extend(render_example(row))
    return "\n".join(lines)


def validate_no_space(lines: list[str]) -> bool:
    return all(not any(ch.isspace() for ch in line) for line in lines)


def main() -> None:
    args = parse_args()
    if args.min_chunk_size > args.chunk_size or args.chunk_size > args.max_chunk_size:
        raise ValueError("--min-chunk-size <= --chunk-size <= --max-chunk-size is required")
    status = converter_status(args)
    rows = list(read_jsonl(args.input_jsonl))
    if args.max_articles is not None:
        rows = rows[: args.max_articles]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    article_dir = args.output_dir / "article_level"
    final_aligned_dir = args.output_dir / "final_aligned"
    summary_dir = args.output_dir / "summaries"
    report_path = Path("4.Korean/korean_khdb_magazine_audit/results/reports/step2_no_space_hangulized_report.md")
    if args.max_articles is not None:
        debug_dir = args.output_dir / "debug" / f"debug{args.max_articles}"
        article_jsonl_path = debug_dir / "selected_diagnostic_mixed_articles_nospace.jsonl"
        mixed_article_text_path = debug_dir / "selected_diagnostic_mixed_nospace.txt"
        hangulized_article_text_path = debug_dir / "selected_diagnostic_hangulized_nospace.txt"
        mixed_chunk_text_path = debug_dir / "selected_diagnostic_mixed_chunks_nospace.txt"
        hangulized_chunk_text_path = debug_dir / "selected_diagnostic_hangulized_chunks_nospace.txt"
        chunk_index_path = debug_dir / "selected_diagnostic_chunk_index.jsonl"
        summary_path = debug_dir / "step2_no_space_hangulized_summary.json"
        report_path = Path(
            f"4.Korean/korean_khdb_magazine_audit/results/reports/step2_no_space_hangulized_report_debug{args.max_articles}.md"
        )
    else:
        article_jsonl_path = article_dir / "selected_diagnostic_mixed_articles_nospace.jsonl"
        mixed_article_text_path = article_dir / "selected_diagnostic_mixed_nospace.txt"
        hangulized_article_text_path = article_dir / "selected_diagnostic_hangulized_nospace.txt"
        mixed_chunk_text_path = final_aligned_dir / "selected_diagnostic_mixed_chunks_nospace.txt"
        hangulized_chunk_text_path = final_aligned_dir / "selected_diagnostic_hangulized_chunks_nospace.txt"
        chunk_index_path = final_aligned_dir / "selected_diagnostic_chunk_index.jsonl"
        summary_path = summary_dir / "step2_no_space_hangulized_summary.json"

    sanity_rows: list[dict] = []
    fallback_used_count = 0
    conversion_failure_count = 0
    for text, expected in SANITY_EXAMPLES:
        cleaned = clean_to_nospace(text)
        converted, backend, success, notes = convert_text(cleaned, args, status)
        fallback_used_count += int(backend == "hanja_fallback")
        conversion_failure_count += int(not success)
        sanity_rows.append(
            {
                "input": text,
                "expected": expected,
                "cleaned": cleaned,
                "output": converted,
                "leftover_hanja_count": len(HANJA_RE.findall(converted)),
                "converter_backend": backend,
                "conversion_success": success,
                "conversion_notes": notes,
            }
        )

    article_rows: list[dict] = []
    mixed_article_lines: list[str] = []
    hangulized_article_lines: list[str] = []
    mixed_chunk_lines: list[str] = []
    hangulized_chunk_lines: list[str] = []
    chunk_index_rows: list[dict] = []
    examples_with_hanja_leftover: list[dict] = []
    previous_selected_source_chars = sum(int(row.get("body_length_chars", 0) or 0) for row in rows)

    for row in rows:
        mixed_nospace = clean_to_nospace(row.get("body_text", "") or row.get("body_text_original", ""))
        hangulized, backend, success, notes = convert_text(mixed_nospace, args, status)
        fallback_used_count += int(backend == "hanja_fallback")
        conversion_failure_count += int(not success)
        article_row = article_output_row(row, mixed_nospace, hangulized, backend, success, notes)
        article_rows.append(article_row)
        mixed_article_lines.append(mixed_nospace)
        hangulized_article_lines.append(hangulized)
        if article_row["hangulized_hanja_count"] > 0 and len(examples_with_hanja_leftover) < 50:
            examples_with_hanja_leftover.append(
                {
                    "khdb_id": row.get("khdb_id", ""),
                    "article_title": row.get("article_title", ""),
                    "leftover_hanja_count": article_row["hangulized_hanja_count"],
                    "mixed_preview": mixed_nospace[:120],
                    "hangulized_preview": hangulized[:120],
                }
            )

        for chunk_number, mixed_chunk in enumerate(chunk_text(mixed_nospace, args), start=1):
            hangulized_chunk, chunk_backend, chunk_success, chunk_notes = convert_text(mixed_chunk, args, status)
            fallback_used_count += int(chunk_backend == "hanja_fallback")
            conversion_failure_count += int(not chunk_success)
            mixed_counts = script_counts(mixed_chunk)
            hangulized_counts = script_counts(hangulized_chunk)
            chunk_id = f"{row.get('khdb_id', '')}_{chunk_number:04d}"
            chunk_index_rows.append(
                {
                    "chunk_id": chunk_id,
                    "khdb_id": row.get("khdb_id", ""),
                    "article_title": row.get("article_title", ""),
                    "magazine_title": row.get("magazine_title", ""),
                    "chunk_index_within_article": chunk_number,
                    "mixed_source_chars": mixed_counts["source_chars"],
                    "mixed_hanja_count": mixed_counts["hanja_count"],
                    "mixed_hangul_count": mixed_counts["hangul_count"],
                    "hangulized_chars": hangulized_counts["source_chars"],
                    "hangulized_hanja_count": hangulized_counts["hanja_count"],
                    "converter_backend": chunk_backend,
                    "conversion_success": chunk_success,
                    "conversion_notes": chunk_notes,
                    "raw_article_url": row.get("url", ""),
                }
            )
            mixed_chunk_lines.append(mixed_chunk)
            hangulized_chunk_lines.append(hangulized_chunk)

    write_jsonl(article_jsonl_path, article_rows)
    for path in [
        mixed_article_text_path,
        hangulized_article_text_path,
        mixed_chunk_text_path,
        hangulized_chunk_text_path,
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
    mixed_article_text_path.write_text("\n".join(mixed_article_lines) + "\n", encoding="utf-8")
    hangulized_article_text_path.write_text("\n".join(hangulized_article_lines) + "\n", encoding="utf-8")
    mixed_chunk_text_path.write_text("\n".join(mixed_chunk_lines) + "\n", encoding="utf-8")
    hangulized_chunk_text_path.write_text("\n".join(hangulized_chunk_lines) + "\n", encoding="utf-8")
    write_jsonl(chunk_index_path, chunk_index_rows)

    total_mixed_text = "\n".join(mixed_article_lines)
    total_hangulized_text = "\n".join(hangulized_article_lines)
    total_mixed_counts = count_chars(total_mixed_text)
    total_hangulized_counts = count_chars(total_hangulized_text)
    line_alignment_passed = (
        len(mixed_article_lines) == len(hangulized_article_lines)
        and len(mixed_chunk_lines) == len(hangulized_chunk_lines)
        and len(chunk_index_rows) == len(mixed_chunk_lines)
    )
    no_space_check_passed = validate_no_space(mixed_article_lines + hangulized_article_lines + mixed_chunk_lines + hangulized_chunk_lines)
    backends = sorted({row["converter_backend"] for row in article_rows})
    backend_label = "+".join(backends) if backends else args.converter
    if backend_label == "gukhanmun":
        version = status["gukhanmun_version"]
    elif backend_label == "hanja_fallback":
        version = status["hanja_version"]
    else:
        version = json.dumps(
            {
                "gukhanmun": status["gukhanmun_version"],
                "hanja": status["hanja_version"],
            },
            ensure_ascii=False,
        )
    summary = {
        "input_article_count": len(rows),
        "output_article_count": len(article_rows),
        "mixed_article_lines": len(mixed_article_lines),
        "hangulized_article_lines": len(hangulized_article_lines),
        "chunk_count": len(chunk_index_rows),
        "mixed_chunk_lines": len(mixed_chunk_lines),
        "hangulized_chunk_lines": len(hangulized_chunk_lines),
        "previous_selected_source_chars": previous_selected_source_chars,
        "current_total_mixed_source_chars_nospace": total_mixed_counts["body_length_chars"],
        "source_char_difference_after_marker_removal": total_mixed_counts["body_length_chars"] - previous_selected_source_chars,
        "total_mixed_source_chars_nospace": total_mixed_counts["body_length_chars"],
        "total_mixed_hanja_chars": total_mixed_counts["hanja_count"],
        "total_mixed_hangul_chars": total_mixed_counts["hangul_count"],
        "total_hangulized_chars": total_hangulized_counts["body_length_chars"],
        "total_hangulized_hanja_leftover_chars": total_hangulized_counts["hanja_count"],
        "articles_with_hanja_leftover_after_conversion": sum(1 for row in article_rows if row["hangulized_hanja_count"] > 0),
        "chunks_with_hanja_leftover_after_conversion": sum(1 for row in chunk_index_rows if row["hangulized_hanja_count"] > 0),
        "converter_backend": backend_label,
        "converter_version": version,
        "fallback_used_count": fallback_used_count,
        "conversion_failure_count": conversion_failure_count,
        "examples_with_hanja_leftover": examples_with_hanja_leftover,
        "sanity_examples": sanity_rows,
        "whether_line_alignment_passed": line_alignment_passed,
        "whether_no_space_check_passed": no_space_check_passed,
        "outputs": {
            "article_jsonl": str(article_jsonl_path),
            "mixed_article_text": str(mixed_article_text_path),
            "hangulized_article_text": str(hangulized_article_text_path),
            "mixed_chunk_text": str(mixed_chunk_text_path),
            "hangulized_chunk_text": str(hangulized_chunk_text_path),
            "chunk_index": str(chunk_index_path),
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(summary, sanity_rows, article_rows, args.seed), encoding="utf-8")

    print(f"Articles: {len(article_rows)}")
    print(f"Chunks: {len(chunk_index_rows)}")
    print(f"Converter backend: {backend_label}")
    print(f"Converter version: {version}")
    print(f"No-space check passed: {no_space_check_passed}")
    print(f"Line alignment passed: {line_alignment_passed}")
    print(f"Hanja leftovers: {total_hangulized_counts['hanja_count']}")
    print(f"Summary: {summary_path}")
    print(f"Report: {report_path}")
    if not line_alignment_passed or not no_space_check_passed:
        sys.exit(2)


if __name__ == "__main__":
    main()
