#!/usr/bin/env python3
# Step 2 program: tokenize raw text into fixed-length LM blocks.
"""Tokenize a text file into fixed-length causal-LM blocks."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tokenize a line-based corpus and save fixed-length LM blocks."
    )
    parser.add_argument("--tokenizer", required=True, help="Local tokenizer directory.")
    parser.add_argument("--input", required=True, help="Input text file.")
    parser.add_argument("--output", required=True, help="Output dataset directory.")
    parser.add_argument(
        "--block_size", type=int, default=512, help="Fixed token block size."
    )
    parser.add_argument(
        "--max_lines",
        type=int,
        default=None,
        help="Only process the first N lines. Default: process all lines.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Decode and print the first saved block.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output directory if it already exists.",
    )
    parser.add_argument(
        "--require_vocab_size",
        type=int,
        default=None,
        help="Fail unless len(tokenizer) equals this value.",
    )
    parser.add_argument(
        "--require_eos_id",
        type=int,
        default=None,
        help="Fail unless tokenizer.eos_token_id equals this value.",
    )
    parser.add_argument(
        "--require_pad_id",
        type=int,
        default=None,
        help="Fail unless tokenizer.pad_token_id equals this value.",
    )
    parser.add_argument("--encoding", default="utf-8", help="Input text encoding.")
    return parser.parse_args()


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"{label} is not a file: {path}")


def require_dir(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"{label} is not a directory: {path}")


def load_tokenizer(tokenizer_dir: Path):
    try:
        from transformers import AutoTokenizer, PreTrainedTokenizerFast
    except ImportError as exc:
        raise ImportError(
            "transformers is required. Install it with: pip install transformers"
        ) from exc

    tokenizer_json = tokenizer_dir / "tokenizer.json"
    expected_vocab_size = None
    if tokenizer_json.exists():
        with tokenizer_json.open("r", encoding="utf-8") as handle:
            tokenizer_payload = json.load(handle)
        model_vocab = tokenizer_payload.get("model", {}).get("vocab", {})
        added_tokens = tokenizer_payload.get("added_tokens", [])
        expected_vocab_size = len(model_vocab) + len(added_tokens)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir,
            use_fast=True,
            local_files_only=True,
        )
        if expected_vocab_size is None or len(tokenizer) == expected_vocab_size:
            return tokenizer
        print(
            "  warning: AutoTokenizer vocab size does not match tokenizer.json; "
            f"auto={len(tokenizer)} expected={expected_vocab_size}. "
            "Falling back to PreTrainedTokenizerFast."
        )
    except Exception:
        tokenizer = None

    if tokenizer_json.exists():
        try:
            return PreTrainedTokenizerFast.from_pretrained(
                tokenizer_dir,
                local_files_only=True,
            )
        except Exception:
            return PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_json))

    raise RuntimeError(f"No tokenizer.json found in {tokenizer_dir}")


def strip_newline(line: str) -> str:
    if line.endswith("\n"):
        line = line[:-1]
    if line.endswith("\r"):
        line = line[:-1]
    return line


def count_lines(path: Path, encoding: str) -> int:
    with path.open("r", encoding=encoding) as handle:
        return sum(1 for _ in handle)


def tokenize_to_blocks(
    tokenizer,
    input_path: Path,
    block_size: int,
    max_lines: int | None,
    encoding: str,
    cache_dir: Path,
) -> tuple[object, dict[str, int], list[int] | None]:
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise ImportError(
            "datasets is required for save_to_disk. Install it with: pip install datasets"
        ) from exc

    stream: list[int] = []
    stats = {
        "num_lines_read": 0,
        "empty_lines": 0,
        "num_tokens": 0,
        "num_blocks": 0,
        "dropped_tokens": 0,
    }
    first_block: list[int] | None = None
    eos_token_id = tokenizer.eos_token_id
    total_lines = max_lines if max_lines is not None else count_lines(input_path, encoding)

    def block_generator():
        nonlocal first_block
        with input_path.open("r", encoding=encoding) as handle:
            progress = tqdm(
                handle,
                total=total_lines,
                desc=f"tokenizing {input_path.name}",
                unit="line",
            )
            for raw_line in progress:
                if max_lines is not None and stats["num_lines_read"] >= max_lines:
                    break

                line = strip_newline(raw_line)
                stats["num_lines_read"] += 1
                if line == "":
                    stats["empty_lines"] += 1

                token_ids = tokenizer.encode(line, add_special_tokens=False)
                if eos_token_id is not None:
                    token_ids.append(eos_token_id)

                stats["num_tokens"] += len(token_ids)
                stream.extend(token_ids)

                while len(stream) >= block_size:
                    block = stream[:block_size]
                    del stream[:block_size]
                    stats["num_blocks"] += 1
                    if first_block is None:
                        first_block = list(block)
                    yield {"input_ids": block}

        stats["dropped_tokens"] = len(stream)

    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset = Dataset.from_generator(block_generator, cache_dir=str(cache_dir))
    return dataset, stats, first_block


def validate_tokenizer_requirements(tokenizer, args: argparse.Namespace) -> None:
    if args.require_vocab_size is not None and len(tokenizer) != args.require_vocab_size:
        raise ValueError(
            f"Tokenizer vocab size must be {args.require_vocab_size}; got {len(tokenizer)}"
        )
    if args.require_eos_id is not None and tokenizer.eos_token_id != args.require_eos_id:
        raise ValueError(
            f"Tokenizer eos_token_id must be {args.require_eos_id}; "
            f"got {tokenizer.eos_token_id}"
        )
    if args.require_pad_id is not None and tokenizer.pad_token_id != args.require_pad_id:
        raise ValueError(
            f"Tokenizer pad_token_id must be {args.require_pad_id}; "
            f"got {tokenizer.pad_token_id}"
        )


def main() -> int:
    args = parse_args()
    tokenizer_dir = Path(args.tokenizer)
    input_path = Path(args.input)
    output_dir = Path(args.output)

    try:
        if args.block_size <= 0:
            raise ValueError(f"--block_size must be positive; got {args.block_size}")
        if args.max_lines is not None and args.max_lines <= 0:
            raise ValueError(f"--max_lines must be positive when set; got {args.max_lines}")

        require_dir(tokenizer_dir, "tokenizer directory")
        require_file(input_path, "input file")
        if output_dir.exists():
            if not args.overwrite:
                raise ValueError(
                    f"Output directory already exists: {output_dir}. "
                    "Use --overwrite to replace it."
                )
            shutil.rmtree(output_dir)
        generator_cache_dir = output_dir.parent / ".generator_cache" / output_dir.name
        if generator_cache_dir.exists():
            shutil.rmtree(generator_cache_dir)

        tokenizer = load_tokenizer(tokenizer_dir)
        validate_tokenizer_requirements(tokenizer, args)
        vocab_size = len(tokenizer)
        eos_token_id = tokenizer.eos_token_id

        print("Tokenizer")
        print(f"  class: {tokenizer.__class__.__name__}")
        print(f"  vocab_size: {vocab_size}")
        print(f"  eos_token: {tokenizer.eos_token}")
        print(f"  eos_token_id: {eos_token_id}")
        print(f"  pad_token: {tokenizer.pad_token}")
        print(f"  pad_token_id: {tokenizer.pad_token_id}")
        if eos_token_id is None:
            print("  warning: tokenizer has no eos_token_id; no EOS will be appended")

        dataset, stats, first_block = tokenize_to_blocks(
            tokenizer=tokenizer,
            input_path=input_path,
            block_size=args.block_size,
            max_lines=args.max_lines,
            encoding=args.encoding,
            cache_dir=generator_cache_dir,
        )

        if stats["num_blocks"] == 0:
            raise ValueError(
                "Tokenization produced zero complete blocks. "
                "Use more input lines or a smaller --block_size."
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(output_dir))

        metadata = {
            "input_path": str(input_path),
            "tokenizer_path": str(tokenizer_dir),
            "max_lines": args.max_lines,
            "num_lines_read": stats["num_lines_read"],
            "empty_lines": stats["empty_lines"],
            "num_tokens": stats["num_tokens"],
            "block_size": args.block_size,
            "num_blocks": stats["num_blocks"],
            "dropped_tokens": stats["dropped_tokens"],
            "dropped_remainder_tokens": stats["dropped_tokens"],
            "eos_token_id": eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "vocab_size": vocab_size,
        }
        metadata_path = output_dir / "metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        print("\nTokenization complete")
        print(f"  input: {input_path}")
        print(f"  output: {output_dir}")
        print(f"  num_lines_read: {stats['num_lines_read']}")
        print(f"  empty_lines: {stats['empty_lines']}")
        print(f"  num_tokens: {stats['num_tokens']}")
        print(f"  block_size: {args.block_size}")
        print(f"  num_blocks: {stats['num_blocks']}")
        print(f"  dropped_tokens: {stats['dropped_tokens']}")
        print(f"  metadata: {metadata_path}")

        if args.preview:
            print("\nPreview first block:")
            if first_block is None:
                print("<no complete block>")
            else:
                print(tokenizer.decode(first_block))

        return 0
    except (OSError, UnicodeError, ValueError, RuntimeError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
