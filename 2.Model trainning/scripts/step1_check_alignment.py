#!/usr/bin/env python3
# Step 1 program: paired-corpus alignment check.
"""Check line-level alignment and basic text statistics for paired corpora."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check that two text corpora are aligned line by line."
    )
    parser.add_argument("--source", required=True, help="Path to source text file.")
    parser.add_argument("--target", required=True, help="Path to target text file.")
    parser.add_argument(
        "--show_examples",
        type=int,
        default=0,
        help="Number of aligned examples to print from the beginning of the files.",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Text encoding to use when reading files. Default: utf-8.",
    )
    return parser.parse_args()


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"{label} path is not a file: {path}")


def summarize(path: Path, encoding: str) -> dict[str, object]:
    line_count = 0
    total_chars = 0
    empty_count = 0
    empty_examples: list[int] = []

    with path.open("r", encoding=encoding) as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.rstrip("\n\r")
            line_count += 1
            total_chars += len(text)
            if not text.strip():
                empty_count += 1
                if len(empty_examples) < 10:
                    empty_examples.append(line_no)

    avg_chars = total_chars / line_count if line_count else 0.0
    return {
        "lines": line_count,
        "total_chars": total_chars,
        "avg_chars_per_line": avg_chars,
        "empty_lines": empty_count,
        "empty_line_examples": empty_examples,
    }


def print_summary(label: str, stats: dict[str, object]) -> None:
    print(f"{label}:")
    print(f"  lines: {stats['lines']}")
    print(f"  total_chars: {stats['total_chars']}")
    print(f"  avg_chars_per_line: {stats['avg_chars_per_line']:.2f}")
    print(f"  empty_lines: {stats['empty_lines']}")
    examples = stats["empty_line_examples"]
    if examples:
        print(f"  empty_line_examples: {examples}")


def print_examples(source: Path, target: Path, n: int, encoding: str) -> None:
    if n <= 0:
        return

    print("\nAligned examples:")
    with source.open("r", encoding=encoding) as src_handle, target.open(
        "r", encoding=encoding
    ) as tgt_handle:
        for idx, (src_line, tgt_line) in enumerate(zip(src_handle, tgt_handle), start=1):
            if idx > n:
                break
            print(f"\n[{idx}] SOURCE")
            print(src_line.rstrip("\n\r"))
            print(f"[{idx}] TARGET")
            print(tgt_line.rstrip("\n\r"))


def main() -> int:
    args = parse_args()
    source = Path(args.source)
    target = Path(args.target)

    try:
        require_file(source, "source")
        require_file(target, "target")
        source_stats = summarize(source, args.encoding)
        target_stats = summarize(target, args.encoding)
    except (OSError, UnicodeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print("Alignment check")
    print(f"source: {source}")
    print(f"target: {target}\n")
    print_summary("Source stats", source_stats)
    print()
    print_summary("Target stats", target_stats)

    line_match = source_stats["lines"] == target_stats["lines"]
    print("\nChecks:")
    print(f"  same_number_of_lines: {'YES' if line_match else 'NO'}")

    source_empty = int(source_stats["empty_lines"])
    target_empty = int(target_stats["empty_lines"])
    if source_empty or target_empty:
        print(f"  empty_lines_found: YES source={source_empty} target={target_empty}")
    else:
        print("  empty_lines_found: NO")

    if line_match:
        print("  status: PASS")
        print_examples(source, target, args.show_examples, args.encoding)
        return 0

    print("  status: FAIL")
    print(
        "ERROR: source and target have different line counts; "
        "fix alignment before tokenization.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
