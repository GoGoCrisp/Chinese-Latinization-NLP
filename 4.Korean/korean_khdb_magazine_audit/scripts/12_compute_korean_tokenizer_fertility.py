from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute tokenizer fertility on the held-out Korean KHDB dev split."
    )
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
        "--dev-mixed",
        type=Path,
        default=Path(
            "4.Korean/korean_khdb_magazine_audit/data/corpus/final_aligned/splits/seed42_90_10/dev.mixed_chunks_nospace.txt"
        ),
    )
    parser.add_argument(
        "--dev-hangulized",
        type=Path,
        default=Path(
            "4.Korean/korean_khdb_magazine_audit/data/corpus/final_aligned/splits/seed42_90_10/dev.hangulized_chunks_nospace.txt"
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("4.Korean/korean_khdb_magazine_audit/data/tokenizers/fertility_dev_summary.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("4.Korean/korean_khdb_magazine_audit/data/tokenizers/fertility_dev_results.csv"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("4.Korean/korean_khdb_magazine_audit/results/reports/tokenizer_dev_fertility_report.md"),
    )
    return parser.parse_args()


def require_tokenizers():
    try:
        import tokenizers
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise SystemExit("Missing dependency. Install with: pip install tokenizers") from exc
    return tokenizers, Tokenizer


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def nonspace_len(text: str) -> int:
    return sum(1 for ch in text if not ch.isspace())


def validate_inputs(mixed_lines: list[str], hangulized_lines: list[str]) -> None:
    if len(mixed_lines) != len(hangulized_lines):
        raise ValueError(
            f"Dev split line alignment failed: mixed={len(mixed_lines)} hangulized={len(hangulized_lines)}"
        )
    for label, lines in [("mixed", mixed_lines), ("hangulized", hangulized_lines)]:
        bad = [idx for idx, line in enumerate(lines, start=1) if any(ch.isspace() for ch in line)]
        if bad:
            raise ValueError(f"{label} dev split contains whitespace on line {bad[0]}.")


def tokenizer_fertility(tokenizer, lines: list[str], original_source_chars: int) -> dict:
    unk_id = tokenizer.token_to_id("[UNK]")
    token_counts: list[int] = []
    total_tokens = 0
    total_unk = 0
    lines_with_unk = 0
    total_surface_chars = 0
    total_surface_bytes = 0

    for line in lines:
        encoded = tokenizer.encode(line)
        token_count = len(encoded.tokens)
        unk_count = sum(1 for token_id in encoded.ids if token_id == unk_id)
        token_counts.append(token_count)
        total_tokens += token_count
        total_unk += unk_count
        lines_with_unk += int(unk_count > 0)
        total_surface_chars += nonspace_len(line)
        total_surface_bytes += len(line.encode("utf-8"))

    line_count = len(lines)
    return {
        "line_count": line_count,
        "total_tokens": total_tokens,
        "total_surface_chars": total_surface_chars,
        "total_original_source_chars": original_source_chars,
        "total_surface_bytes": total_surface_bytes,
        "tokens_per_sample": total_tokens / line_count if line_count else 0.0,
        "tokens_per_surface_char": total_tokens / total_surface_chars if total_surface_chars else 0.0,
        "tokens_per_original_source_char": total_tokens / original_source_chars if original_source_chars else 0.0,
        "chars_per_token": total_surface_chars / total_tokens if total_tokens else 0.0,
        "bytes_per_token": total_surface_bytes / total_tokens if total_tokens else 0.0,
        "mean_tokens_per_line": statistics.mean(token_counts) if token_counts else 0.0,
        "median_tokens_per_line": statistics.median(token_counts) if token_counts else 0.0,
        "max_tokens_per_line": max(token_counts) if token_counts else 0,
        "total_unk_tokens": total_unk,
        "unk_tokens_per_10k_original_source_chars": total_unk / original_source_chars * 10_000
        if original_source_chars
        else 0.0,
        "lines_with_unk": lines_with_unk,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "corpus",
        "tokenizer",
        "vocab_size",
        "line_count",
        "tokens_per_sample",
        "tokens_per_surface_char",
        "tokens_per_original_source_char",
        "total_tokens",
        "total_surface_chars",
        "total_original_source_chars",
        "chars_per_token",
        "bytes_per_token",
        "total_unk_tokens",
        "unk_tokens_per_10k_original_source_chars",
        "lines_with_unk",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def render_report(summary: dict, rows: list[dict]) -> str:
    comparison = summary["comparison"]
    lines = [
        "# Korean KHDB Dev Fertility Report",
        "",
        "Fertility is computed on the held-out 10% dev split only.",
        "The denominator for cross-representation comparison is the non-space",
        "character count of the original mixed-script dev source.",
        "",
        "## Inputs",
        "",
        f"- mixed tokenizer: `{summary['inputs']['mixed_tokenizer']}`",
        f"- Hangulized tokenizer: `{summary['inputs']['hangulized_tokenizer']}`",
        f"- mixed dev: `{summary['inputs']['dev_mixed']}`",
        f"- Hangulized dev: `{summary['inputs']['dev_hangulized']}`",
        "",
        "## Results",
        "",
        "| corpus | vocab | lines | tokens/sample | tokens/surface char | tokens/original source char | total tokens | original source chars | UNK/10k original chars |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {corpus} | {vocab_size} | {line_count} | {tokens_per_sample:.6f} | "
            "{tokens_per_surface_char:.6f} | {tokens_per_original_source_char:.6f} | "
            "{total_tokens} | {total_original_source_chars} | "
            "{unk_tokens_per_10k_original_source_chars:.6f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Comparison",
            "",
            f"- absolute fertility reduction: `{comparison['absolute_fertility_reduction']:.6f}` tokens/original source char",
            f"- relative fertility reduction: `{comparison['relative_fertility_reduction_percent']:.2f}%`",
            f"- total dev token reduction: `{comparison['total_token_reduction']}` tokens",
            "",
            "## Notes",
            "",
            "- `tokens_per_surface_char` uses each representation's own non-space character count.",
            "- `tokens_per_original_source_char` uses the mixed-script dev source denominator for both tokenizers.",
            "- No special tokens are added during encoding.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    tokenizers_lib, Tokenizer = require_tokenizers()
    mixed_lines = read_lines(args.dev_mixed)
    hangulized_lines = read_lines(args.dev_hangulized)
    validate_inputs(mixed_lines, hangulized_lines)

    mixed_tokenizer = Tokenizer.from_file(str(args.mixed_tokenizer))
    hangulized_tokenizer = Tokenizer.from_file(str(args.hangulized_tokenizer))
    original_source_chars = sum(nonspace_len(line) for line in mixed_lines)

    rows = []
    for corpus, tokenizer, tokenizer_path, lines in [
        ("mixed", mixed_tokenizer, args.mixed_tokenizer, mixed_lines),
        ("hangulized", hangulized_tokenizer, args.hangulized_tokenizer, hangulized_lines),
    ]:
        stats = tokenizer_fertility(tokenizer, lines, original_source_chars)
        rows.append(
            {
                "corpus": corpus,
                "tokenizer": str(tokenizer_path),
                "vocab_size": tokenizer.get_vocab_size(),
                **stats,
            }
        )

    mixed_result = next(row for row in rows if row["corpus"] == "mixed")
    hangulized_result = next(row for row in rows if row["corpus"] == "hangulized")
    fertility_delta = (
        mixed_result["tokens_per_original_source_char"] - hangulized_result["tokens_per_original_source_char"]
    )
    comparison = {
        "absolute_fertility_reduction": fertility_delta,
        "relative_fertility_reduction_percent": fertility_delta
        / mixed_result["tokens_per_original_source_char"]
        * 100
        if mixed_result["tokens_per_original_source_char"]
        else 0.0,
        "total_token_reduction": mixed_result["total_tokens"] - hangulized_result["total_tokens"],
    }

    summary = {
        "metric": "tokenizer_fertility_on_dev",
        "definition": {
            "tokens_per_sample": "total_tokens / dev_line_count",
            "tokens_per_surface_char": "total_tokens / non-space chars in the evaluated representation",
            "tokens_per_original_source_char": "total_tokens / non-space chars in mixed-script dev source",
        },
        "tokenizers_library": tokenizers_lib.__version__,
        "inputs": {
            "mixed_tokenizer": str(args.mixed_tokenizer),
            "hangulized_tokenizer": str(args.hangulized_tokenizer),
            "dev_mixed": str(args.dev_mixed),
            "dev_hangulized": str(args.dev_hangulized),
        },
        "line_alignment_check_passed": True,
        "no_space_check_passed": True,
        "results": rows,
        "comparison": comparison,
        "outputs": {
            "summary": str(args.output_json),
            "csv": str(args.output_csv),
            "report": str(args.report),
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(args.output_csv, rows)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_report(summary, rows), encoding="utf-8")

    for row in rows:
        print(
            "{corpus}: tokens/original_source_char={tokens_per_original_source_char:.6f}, "
            "tokens/surface_char={tokens_per_surface_char:.6f}, tokens/sample={tokens_per_sample:.6f}".format(
                **row
            )
        )
    print(f"Summary: {args.output_json}")
    print(f"CSV: {args.output_csv}")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
