from __future__ import annotations

import argparse
import importlib.metadata
import json
import random
import sys
from pathlib import Path
from typing import Optional

from tokenizers import Regex, Tokenizer, normalizers, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel, Split
from tokenizers.trainers import BpeTrainer


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_FILES = [
    PROJECT_DIR / "corpora" / "splits" / "3_korean_normal_train90.txt",
    PROJECT_DIR / "corpora" / "splits" / "3_korean_hangul_only_train90.txt",
]
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "tokenizers_bpe"
DEFAULT_SUBSET_DIR = PROJECT_DIR / "corpora" / "subsets"

REPRESENTATION_BY_STEM = {
    "3_korean_normal_train90": "korean_normal",
    "3_korean_hangul_only_train90": "korean_hangul_only",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train standard 32K BPE tokenizers for normal and Hangul-only Korean corpora."
    )
    parser.add_argument(
        "--train-files",
        type=Path,
        nargs="+",
        default=DEFAULT_TRAIN_FILES,
        help="Aligned train split files.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--subset-dir", type=Path, default=DEFAULT_SUBSET_DIR)
    parser.add_argument("--subset-size", type=int, default=100000)
    parser.add_argument("--subset-seed", type=int, default=42)
    parser.add_argument("--vocab-sizes", type=int, nargs="+", default=[32000])
    parser.add_argument("--special-tokens", nargs="*", default=[])
    parser.add_argument(
        "--punctuation-regex",
        default=r"[^\p{L}\p{N}\s]+|[\r\n]+",
        help="Isolated punctuation/newline regex. No whitespace pre-tokenizer is used.",
    )
    return parser.parse_args()


def package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def representation_name(train_file: Path) -> str:
    return REPRESENTATION_BY_STEM.get(train_file.stem, train_file.stem)


def read_lines(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_subset(train_file: Path, subset_path: Path, subset_size: int, seed: int) -> dict:
    lines = read_lines(train_file)
    rng = random.Random(seed)
    selected_indices = list(range(len(lines)))
    rng.shuffle(selected_indices)
    selected_indices = sorted(selected_indices[: min(subset_size, len(lines))])
    subset_path.parent.mkdir(parents=True, exist_ok=True)
    with subset_path.open("w", encoding="utf-8") as handle:
        for idx in selected_indices:
            handle.write(lines[idx] + "\n")
    return {
        "train_file": str(train_file),
        "subset_file": str(subset_path),
        "subset_size_requested": subset_size,
        "subset_seed": seed,
        "actual_subset_line_count": len(selected_indices),
        "total_train_lines": len(lines),
        "subset_indices": selected_indices,
    }


def train_bpe(
    subset_path: Path,
    output_path: Path,
    vocab_size: int,
    punctuation_regex: str,
    special_tokens: list[str],
) -> None:
    tokenizer = Tokenizer(BPE(unk_token=special_tokens[0] if special_tokens else None))
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            Split(
                pattern=Regex(punctuation_regex),
                behavior="isolated",
                invert=False,
            ),
            ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=False),
        ]
    )
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        show_progress=True,
        special_tokens=special_tokens,
    )
    tokenizer.train([str(subset_path)], trainer)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))


def vocab_label(vocab_size: int) -> str:
    if vocab_size % 1000 == 0:
        return f"{vocab_size // 1000}k"
    return str(vocab_size)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.subset_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "tokenizer_type": "standard BPE",
        "normalizer": "NFKC",
        "pre_tokenizer": {
            "sequence": [
                {
                    "type": "Split",
                    "regex": args.punctuation_regex,
                    "behavior": "isolated",
                },
                {
                    "type": "ByteLevel",
                    "add_prefix_space": False,
                    "trim_offsets": True,
                    "use_regex": False,
                },
            ],
            "notes": "No Whitespace pre-tokenizer and no artificial spaces are inserted.",
        },
        "special_tokens": args.special_tokens,
        "subset_size": args.subset_size,
        "subset_seed": args.subset_seed,
        "vocab_sizes": args.vocab_sizes,
        "command_line": " ".join(sys.argv),
        "package_versions": {
            "tokenizers": package_version("tokenizers"),
        },
        "runs": [],
    }

    for train_file in args.train_files:
        if not train_file.exists():
            raise FileNotFoundError(train_file)
        representation = representation_name(train_file)
        subset_path = args.subset_dir / f"4_{representation}_subset100k.txt"
        subset_meta = build_subset(
            train_file=train_file,
            subset_path=subset_path,
            subset_size=args.subset_size,
            seed=args.subset_seed,
        )

        for vocab_size in args.vocab_sizes:
            label = vocab_label(vocab_size)
            output_path = args.output_dir / f"4_{representation}_{label}.json"
            print(f"Training {representation} vocab={vocab_size} -> {output_path}")
            train_bpe(
                subset_path=subset_path,
                output_path=output_path,
                vocab_size=vocab_size,
                punctuation_regex=args.punctuation_regex,
                special_tokens=args.special_tokens,
            )
            metadata["runs"].append(
                {
                    "representation": representation,
                    "vocab_size": vocab_size,
                    "tokenizer_file": str(output_path),
                    **subset_meta,
                }
            )

    metadata_path = args.output_dir / "4_training_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Training metadata: {metadata_path}")


if __name__ == "__main__":
    main()
