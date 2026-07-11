from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from khdb_common import read_jsonl, write_jsonl


HANJA_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split aligned mixed/Hangulized KHDB chunk corpora.")
    parser.add_argument(
        "--mixed",
        type=Path,
        default=Path(
            "4.Korean/korean_khdb_magazine_audit/data/corpus/final_aligned/selected_diagnostic_mixed_chunks_nospace.txt"
        ),
    )
    parser.add_argument(
        "--hangulized",
        type=Path,
        default=Path(
            "4.Korean/korean_khdb_magazine_audit/data/corpus/final_aligned/selected_diagnostic_hangulized_chunks_nospace.txt"
        ),
    )
    parser.add_argument(
        "--chunk-index",
        type=Path,
        default=Path("4.Korean/korean_khdb_magazine_audit/data/corpus/final_aligned/selected_diagnostic_chunk_index.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("4.Korean/korean_khdb_magazine_audit/data/corpus/final_aligned/splits/seed42_90_10"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    return parser.parse_args()


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def no_space(lines: list[str]) -> bool:
    return all(not any(ch.isspace() for ch in line) for line in lines)


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def subset_rows(rows: list[dict], indices: list[int], split_name: str) -> list[dict]:
    output: list[dict] = []
    for split_rank, index in enumerate(indices, start=1):
        row = dict(rows[index])
        row["split"] = split_name
        row["split_rank"] = split_rank
        row["original_line_index"] = index + 1
        output.append(row)
    return output


def main() -> None:
    args = parse_args()
    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError("--train-ratio must be between 0 and 1")

    mixed_lines = read_lines(args.mixed)
    hangulized_lines = read_lines(args.hangulized)
    index_rows = list(read_jsonl(args.chunk_index))
    if not (len(mixed_lines) == len(hangulized_lines) == len(index_rows)):
        raise ValueError(
            "Input line counts are not aligned: "
            f"mixed={len(mixed_lines)} hangulized={len(hangulized_lines)} index={len(index_rows)}"
        )

    indices = list(range(len(mixed_lines)))
    rng = random.Random(args.seed)
    rng.shuffle(indices)
    train_count = round(len(indices) * args.train_ratio)
    train_indices = indices[:train_count]
    dev_indices = indices[train_count:]

    outputs = {
        "train_mixed": args.output_dir / "train.mixed_chunks_nospace.txt",
        "train_hangulized": args.output_dir / "train.hangulized_chunks_nospace.txt",
        "dev_mixed": args.output_dir / "dev.mixed_chunks_nospace.txt",
        "dev_hangulized": args.output_dir / "dev.hangulized_chunks_nospace.txt",
        "train_index": args.output_dir / "train.chunk_index.jsonl",
        "dev_index": args.output_dir / "dev.chunk_index.jsonl",
        "summary": args.output_dir / "split_summary.json",
    }
    write_lines(outputs["train_mixed"], [mixed_lines[i] for i in train_indices])
    write_lines(outputs["train_hangulized"], [hangulized_lines[i] for i in train_indices])
    write_lines(outputs["dev_mixed"], [mixed_lines[i] for i in dev_indices])
    write_lines(outputs["dev_hangulized"], [hangulized_lines[i] for i in dev_indices])
    write_jsonl(outputs["train_index"], subset_rows(index_rows, train_indices, "train"))
    write_jsonl(outputs["dev_index"], subset_rows(index_rows, dev_indices, "dev"))

    summary = {
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "input_line_count": len(indices),
        "train_line_count": len(train_indices),
        "dev_line_count": len(dev_indices),
        "train_fraction": len(train_indices) / max(1, len(indices)),
        "dev_fraction": len(dev_indices) / max(1, len(indices)),
        "line_alignment_passed": (
            len(train_indices) == len(read_lines(outputs["train_mixed"])) == len(read_lines(outputs["train_hangulized"]))
            and len(dev_indices) == len(read_lines(outputs["dev_mixed"])) == len(read_lines(outputs["dev_hangulized"]))
        ),
        "no_space_check_passed": no_space(mixed_lines + hangulized_lines),
        "hangulized_hanja_leftover_chars": len(HANJA_RE.findall("\n".join(hangulized_lines))),
        "outputs": {key: str(value) for key, value in outputs.items()},
    }
    outputs["summary"].write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Input lines: {len(indices)}")
    print(f"Train lines: {len(train_indices)}")
    print(f"Dev lines: {len(dev_indices)}")
    print(f"Line alignment passed: {summary['line_alignment_passed']}")
    print(f"No-space check passed: {summary['no_space_check_passed']}")
    print(f"Hangulized Hanja leftovers: {summary['hangulized_hanja_leftover_chars']}")
    print(f"Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
