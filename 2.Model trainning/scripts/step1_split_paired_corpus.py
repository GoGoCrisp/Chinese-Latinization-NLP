#!/usr/bin/env python3
# Step 1 helper program: reproducible paired-corpus train/valid/test split.
"""Create reproducible train/valid/test splits for aligned paired corpora."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split aligned source/target text files into train/valid/test files."
    )
    parser.add_argument("--source", required=True, help="Path to source corpus.")
    parser.add_argument("--target", required=True, help="Path to target corpus.")
    parser.add_argument("--output_dir", required=True, help="Directory for split files.")
    parser.add_argument("--source_name", default="zh", help="Source filename tag.")
    parser.add_argument("--target_name", default="diacritic", help="Target filename tag.")
    parser.add_argument(
        "--train_ratio", type=float, default=0.98, help="Training split ratio."
    )
    parser.add_argument(
        "--valid_ratio", type=float, default=0.01, help="Validation split ratio."
    )
    parser.add_argument("--test_ratio", type=float, default=0.01, help="Test split ratio.")
    parser.add_argument("--seed", type=int, default=20260502, help="Random seed.")
    parser.add_argument("--encoding", default="utf-8", help="Text encoding.")
    return parser.parse_args()


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"{label} path is not a file: {path}")


def count_lines(path: Path, encoding: str) -> int:
    with path.open("r", encoding=encoding) as handle:
        return sum(1 for _ in handle)


def validate_ratios(train_ratio: float, valid_ratio: float, test_ratio: float) -> None:
    total = train_ratio + valid_ratio + test_ratio
    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            "train_ratio + valid_ratio + test_ratio must equal 1.0; "
            f"got {total:.12f}"
        )
    for name, ratio in (
        ("train_ratio", train_ratio),
        ("valid_ratio", valid_ratio),
        ("test_ratio", test_ratio),
    ):
        if ratio < 0:
            raise ValueError(f"{name} must be non-negative; got {ratio}")


def build_split_assignments(
    n_lines: int, train_ratio: float, valid_ratio: float, seed: int
) -> tuple[set[int], set[int], set[int]]:
    valid_count = int(n_lines * valid_ratio)
    test_count = int(n_lines * (1.0 - train_ratio - valid_ratio))

    indices = list(range(n_lines))
    random.Random(seed).shuffle(indices)

    valid_indices = set(indices[:valid_count])
    test_indices = set(indices[valid_count : valid_count + test_count])
    train_indices = set(indices[valid_count + test_count :])
    return train_indices, valid_indices, test_indices


def open_split_files(output_dir: Path, source_name: str, target_name: str, encoding: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    handles = {}
    for split in ("train", "valid", "test"):
        handles[(split, "source")] = (
            output_dir / f"{split}.{source_name}.txt"
        ).open("w", encoding=encoding)
        handles[(split, "target")] = (
            output_dir / f"{split}.{target_name}.txt"
        ).open("w", encoding=encoding)
    return handles


def close_handles(handles: dict[tuple[str, str], object]) -> None:
    for handle in handles.values():
        handle.close()


def write_splits(
    source: Path,
    target: Path,
    output_dir: Path,
    source_name: str,
    target_name: str,
    train_indices: set[int],
    valid_indices: set[int],
    test_indices: set[int],
    encoding: str,
) -> dict[str, int]:
    counts = {"train": 0, "valid": 0, "test": 0}
    handles = open_split_files(output_dir, source_name, target_name, encoding)

    try:
        with source.open("r", encoding=encoding) as src_handle, target.open(
            "r", encoding=encoding
        ) as tgt_handle:
            for idx, (src_line, tgt_line) in enumerate(zip(src_handle, tgt_handle)):
                if idx in valid_indices:
                    split = "valid"
                elif idx in test_indices:
                    split = "test"
                elif idx in train_indices:
                    split = "train"
                else:
                    raise RuntimeError(f"Line {idx + 1} was not assigned to a split.")

                handles[(split, "source")].write(src_line)
                handles[(split, "target")].write(tgt_line)
                counts[split] += 1
    finally:
        close_handles(handles)

    return counts


def main() -> int:
    args = parse_args()
    source = Path(args.source)
    target = Path(args.target)
    output_dir = Path(args.output_dir)

    try:
        validate_ratios(args.train_ratio, args.valid_ratio, args.test_ratio)
        require_file(source, "source")
        require_file(target, "target")

        source_lines = count_lines(source, args.encoding)
        target_lines = count_lines(target, args.encoding)
        if source_lines != target_lines:
            raise ValueError(
                "source and target have different line counts: "
                f"{source_lines} vs {target_lines}"
            )

        train_indices, valid_indices, test_indices = build_split_assignments(
            source_lines, args.train_ratio, args.valid_ratio, args.seed
        )
        counts = write_splits(
            source,
            target,
            output_dir,
            args.source_name,
            args.target_name,
            train_indices,
            valid_indices,
            test_indices,
            args.encoding,
        )

        metadata = {
            "source": str(source),
            "target": str(target),
            "output_dir": str(output_dir),
            "source_name": args.source_name,
            "target_name": args.target_name,
            "seed": args.seed,
            "ratios": {
                "train": args.train_ratio,
                "valid": args.valid_ratio,
                "test": args.test_ratio,
            },
            "counts": counts,
            "total_lines": source_lines,
            "files": {
                "train_source": f"train.{args.source_name}.txt",
                "train_target": f"train.{args.target_name}.txt",
                "valid_source": f"valid.{args.source_name}.txt",
                "valid_target": f"valid.{args.target_name}.txt",
                "test_source": f"test.{args.source_name}.txt",
                "test_target": f"test.{args.target_name}.txt",
            },
        }
        metadata_path = output_dir / "split_metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding=args.encoding,
        )

        print("Split complete")
        print(f"  total_lines: {source_lines}")
        print(f"  train: {counts['train']}")
        print(f"  valid: {counts['valid']}")
        print(f"  test: {counts['test']}")
        print(f"  metadata: {metadata_path}")
        return 0
    except (OSError, UnicodeError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
