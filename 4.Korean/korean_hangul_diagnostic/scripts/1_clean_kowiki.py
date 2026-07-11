from __future__ import annotations

import argparse
import json
import re
import statistics
import unicodedata
from pathlib import Path
from typing import Optional


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_JSONL = PROJECT_DIR / "1_cleaned_wiki.jsonl"
DEFAULT_OUTPUT_NORMAL = PROJECT_DIR / "corpora" / "1_korean_normal.txt"
DEFAULT_DIAGNOSTICS = PROJECT_DIR / "results" / "1_cleaning_diagnostics.json"

HANGUL_RE = re.compile(r"[\u1100-\u11ff\u3130-\u318f\uac00-\ud7af]")
HANJA_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
KOREAN_SCRIPT_RE = re.compile(
    r"[\u1100-\u11ff\u3130-\u318f\uac00-\ud7af\u3400-\u4dbf\u4e00-\u9fff]"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean Korean WikiExtractor JSON into a normal Korean corpus."
    )
    parser.add_argument("--input-extracted-dir", type=Path, default=None)
    parser.add_argument("--input-jsonl", type=Path, default=None)
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument("--output-normal-txt", type=Path, default=DEFAULT_OUTPUT_NORMAL)
    parser.add_argument("--diagnostics", type=Path, default=DEFAULT_DIAGNOSTICS)
    parser.add_argument("--max-lines", type=int, default=None)
    parser.add_argument("--min-chars", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--korean-ratio-threshold", type=float, default=0.5)
    return parser.parse_args()


def iter_input_files(args: argparse.Namespace) -> list[Path]:
    if args.input_jsonl:
        return [args.input_jsonl]
    if not args.input_extracted_dir:
        raise ValueError("Pass either --input-jsonl or --input-extracted-dir.")
    files = [
        path
        for path in args.input_extracted_dir.rglob("*")
        if path.is_file() and path.name.startswith("wiki_")
    ]
    return sorted(files)


def parse_json_line(line: str) -> Optional[dict]:
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = re.sub(r"\[(?:\d+|주\s*\d+|출처\s*필요|깨진\s*링크)\]", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\{\{[^{}]*\}\}", "", text)
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"^\s*[*#;:]+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def korean_script_ratio(text: str) -> float:
    chars = [ch for ch in text if not ch.isspace()]
    if not chars:
        return 0.0
    korean_chars = sum(1 for ch in chars if KOREAN_SCRIPT_RE.match(ch))
    return korean_chars / len(chars)


def is_valid(text: str, min_chars: int, ratio_threshold: float) -> tuple[bool, Optional[str]]:
    if not text or len(text) < min_chars:
        return False, "dropped_short"
    if korean_script_ratio(text) < ratio_threshold:
        return False, "dropped_low_korean_ratio"
    return True, None


def main() -> None:
    args = parse_args()
    input_files = iter_input_files(args)
    if not input_files:
        raise FileNotFoundError("No WikiExtractor JSON files found.")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_normal_txt.parent.mkdir(parents=True, exist_ok=True)
    args.diagnostics.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "input_files": [str(path) for path in input_files],
        "seed": args.seed,
        "min_chars": args.min_chars,
        "korean_ratio_threshold": args.korean_ratio_threshold,
        "max_lines": args.max_lines,
        "total_input_records": 0,
        "total_output_lines": 0,
        "dropped_parse_error": 0,
        "dropped_short": 0,
        "dropped_low_korean_ratio": 0,
        "lines_with_hangul": 0,
        "lines_with_hanja": 0,
    }
    lengths: list[int] = []

    with args.output_jsonl.open("w", encoding="utf-8") as json_out, args.output_normal_txt.open(
        "w", encoding="utf-8"
    ) as text_out:
        for input_file in input_files:
            with input_file.open("r", encoding="utf-8") as handle:
                for raw_line in handle:
                    obj = parse_json_line(raw_line)
                    if obj is None:
                        stats["dropped_parse_error"] += 1
                        continue

                    stats["total_input_records"] += 1
                    text = clean_text(str(obj.get("text", "")))
                    valid, drop_reason = is_valid(
                        text, args.min_chars, args.korean_ratio_threshold
                    )
                    if not valid:
                        stats[str(drop_reason)] += 1
                        continue

                    out_obj = {
                        "id": obj.get("id", ""),
                        "title": obj.get("title", ""),
                        "text": text,
                    }
                    json_out.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                    text_out.write(text + "\n")

                    stats["total_output_lines"] += 1
                    lengths.append(len(text))
                    if HANGUL_RE.search(text):
                        stats["lines_with_hangul"] += 1
                    if HANJA_RE.search(text):
                        stats["lines_with_hanja"] += 1

                    if args.max_lines and stats["total_output_lines"] >= args.max_lines:
                        break
            if args.max_lines and stats["total_output_lines"] >= args.max_lines:
                break

    total = stats["total_output_lines"]
    stats["average_line_length"] = statistics.mean(lengths) if lengths else 0.0
    stats["hangul_line_percentage"] = (
        stats["lines_with_hangul"] / total * 100 if total else 0.0
    )
    stats["hanja_line_percentage"] = (
        stats["lines_with_hanja"] / total * 100 if total else 0.0
    )

    args.diagnostics.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Cleaned lines: {stats['total_output_lines']}")
    print(f"Normal corpus: {args.output_normal_txt}")
    print(f"Cleaned jsonl: {args.output_jsonl}")
    print(f"Diagnostics: {args.diagnostics}")


if __name__ == "__main__":
    main()
