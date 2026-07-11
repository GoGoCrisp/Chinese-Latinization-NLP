from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_NORMAL = PROJECT_DIR / "corpora" / "1_korean_normal.txt"
DEFAULT_HANGUL = PROJECT_DIR / "corpora" / "2_korean_hangul_only.txt"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "corpora" / "splits"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create shared train/test splits for normal and Hangul-only Korean corpora."
    )
    parser.add_argument("--normal", type=Path, default=DEFAULT_NORMAL)
    parser.add_argument("--hangul", type=Path, default=DEFAULT_HANGUL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_lines(path: Path) -> list[str]:
    return [line.rstrip("\n") for line in path.open("r", encoding="utf-8")]


def write_lines(path: Path, lines: list[str], indices: list[int]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for idx in indices:
            handle.write(lines[idx] + "\n")


def main() -> None:
    args = parse_args()
    normal_lines = read_lines(args.normal)
    hangul_lines = read_lines(args.hangul)

    if len(normal_lines) != len(hangul_lines):
        raise ValueError(
            f"Line count mismatch: normal={len(normal_lines)}, hangul={len(hangul_lines)}"
        )
    if not normal_lines:
        raise ValueError("Input corpora are empty.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    indices = list(range(len(normal_lines)))
    rng = random.Random(args.seed)
    rng.shuffle(indices)

    split_idx = int(len(indices) * args.train_ratio)
    train_indices = sorted(indices[:split_idx])
    test_indices = sorted(indices[split_idx:])

    outputs = {
        "normal_train": args.output_dir / "3_korean_normal_train90.txt",
        "normal_test": args.output_dir / "3_korean_normal_test10.txt",
        "hangul_train": args.output_dir / "3_korean_hangul_only_train90.txt",
        "hangul_test": args.output_dir / "3_korean_hangul_only_test10.txt",
        "indices": args.output_dir / "3_split_indices.json",
    }

    write_lines(outputs["normal_train"], normal_lines, train_indices)
    write_lines(outputs["normal_test"], normal_lines, test_indices)
    write_lines(outputs["hangul_train"], hangul_lines, train_indices)
    write_lines(outputs["hangul_test"], hangul_lines, test_indices)

    metadata = {
        "normal": str(args.normal),
        "hangul": str(args.hangul),
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "total_lines": len(indices),
        "train_lines": len(train_indices),
        "test_lines": len(test_indices),
        "train_indices": train_indices,
        "test_indices": test_indices,
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    outputs["indices"].write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Total lines: {len(indices)}")
    print(f"Train lines: {len(train_indices)}")
    print(f"Test lines: {len(test_indices)}")
    print(f"Seed: {args.seed}")
    print(f"Train ratio: {args.train_ratio}")
    print(f"Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
