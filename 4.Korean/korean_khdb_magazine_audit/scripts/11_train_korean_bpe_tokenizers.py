from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from pathlib import Path


HANJA_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
SPECIAL_TOKENS = ["[UNK]", "[PAD]", "[BOS]", "[EOS]"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 32K BPE tokenizers for Korean KHDB diagnostic corpora.")
    parser.add_argument(
        "--train-mixed",
        type=Path,
        default=Path(
            "4.Korean/korean_khdb_magazine_audit/data/corpus/final_aligned/splits/seed42_90_10/train.mixed_chunks_nospace.txt"
        ),
    )
    parser.add_argument(
        "--train-hangulized",
        type=Path,
        default=Path(
            "4.Korean/korean_khdb_magazine_audit/data/corpus/final_aligned/splits/seed42_90_10/train.hangulized_chunks_nospace.txt"
        ),
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
        "--output-dir",
        type=Path,
        default=Path("4.Korean/korean_khdb_magazine_audit/data/tokenizers"),
    )
    parser.add_argument("--vocab-size", type=int, default=32_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--byte-fallback", action="store_true")
    parser.add_argument("--show-progress", action="store_true")
    return parser.parse_args()


def require_tokenizers():
    try:
        import tokenizers
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.normalizers import NFKC
        from tokenizers.trainers import BpeTrainer
    except ImportError as exc:
        raise SystemExit("Missing dependency. Install with: pip install tokenizers tqdm") from exc
    return tokenizers, Tokenizer, BPE, NFKC, BpeTrainer


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def nonspace_len(text: str) -> int:
    return sum(1 for ch in text if not ch.isspace())


def no_space(lines: list[str]) -> bool:
    return all(not any(ch.isspace() for ch in line) for line in lines)


def no_hanja(lines: list[str]) -> bool:
    return not HANJA_RE.search("\n".join(lines))


def validate_inputs(train_mixed: list[str], train_hangulized: list[str], dev_mixed: list[str], dev_hangulized: list[str]) -> dict:
    train_aligned = len(train_mixed) == len(train_hangulized)
    dev_aligned = len(dev_mixed) == len(dev_hangulized)
    no_space_passed = no_space(train_mixed + train_hangulized + dev_mixed + dev_hangulized)
    hangulized_no_hanja = no_hanja(train_hangulized + dev_hangulized)
    if not train_aligned or not dev_aligned:
        raise ValueError(
            f"Line alignment failed: train mixed={len(train_mixed)} train hangulized={len(train_hangulized)} "
            f"dev mixed={len(dev_mixed)} dev hangulized={len(dev_hangulized)}"
        )
    if not no_space_passed:
        raise ValueError("No-space check failed: at least one corpus line contains Unicode whitespace.")
    if not hangulized_no_hanja:
        raise ValueError("Hangulized corpus still contains Hanja.")
    return {
        "train_aligned": train_aligned,
        "dev_aligned": dev_aligned,
        "no_space_check_passed": no_space_passed,
        "hangulized_no_hanja_check_passed": hangulized_no_hanja,
    }


def make_tokenizer(BPE, NFKC, byte_fallback: bool):
    tokenizer = __import__("tokenizers").Tokenizer(BPE(unk_token="[UNK]", byte_fallback=byte_fallback))
    tokenizer.normalizer = NFKC()
    return tokenizer


def train_bpe(name: str, lines: list[str], output_path: Path, vocab_path: Path, args: argparse.Namespace, BPE, NFKC, BpeTrainer):
    tokenizer = make_tokenizer(BPE, NFKC, args.byte_fallback)
    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=SPECIAL_TOKENS,
        show_progress=args.show_progress,
    )
    tokenizer.train_from_iterator(lines, trainer=trainer, length=len(lines))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))
    write_vocab(tokenizer, vocab_path)
    print(f"Trained {name}: vocab={tokenizer.get_vocab_size()} output={output_path}")
    return tokenizer


def write_vocab(tokenizer, path: Path) -> None:
    vocab = tokenizer.get_vocab()
    ordered = sorted(vocab.items(), key=lambda item: item[1])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for token, idx in ordered:
            handle.write(f"{idx}\t{token}\n")


def tokenizer_stats(tokenizer, lines: list[str]) -> dict:
    unk_id = tokenizer.token_to_id("[UNK]")
    total_chars = 0
    total_tokens = 0
    total_unk = 0
    lines_with_unk = 0
    token_counts: list[int] = []
    for line in lines:
        encoded = tokenizer.encode(line)
        ids = encoded.ids
        token_count = len(ids)
        unk_count = sum(1 for idx in ids if idx == unk_id)
        chars = nonspace_len(line)
        total_chars += chars
        total_tokens += token_count
        total_unk += unk_count
        lines_with_unk += int(unk_count > 0)
        token_counts.append(token_count)
    return {
        "total_lines": len(lines),
        "total_non_space_chars": total_chars,
        "total_tokens": total_tokens,
        "total_unk_tokens": total_unk,
        "unk_tokens_per_10k_chars": total_unk / max(1, total_chars) * 10_000,
        "lines_with_unk": lines_with_unk,
        "max_tokens_per_line": max(token_counts) if token_counts else 0,
        "mean_tokens_per_line": statistics.mean(token_counts) if token_counts else 0.0,
        "median_tokens_per_line": statistics.median(token_counts) if token_counts else 0.0,
        "tokens_per_source_char_preliminary": total_tokens / max(1, total_chars),
    }


def warning_flag(stats: dict) -> str:
    rate = stats["unk_tokens_per_10k_chars"]
    if rate > 50:
        return "serious_warning"
    if rate > 5:
        return "warning"
    return "ok"


def vocab_preview(tokenizer, n: int = 30) -> list[str]:
    ordered = sorted(tokenizer.get_vocab().items(), key=lambda item: item[1])
    return [token for token, _ in ordered[:n]]


def encoding_examples(tokenizer, lines: list[str], seed: int, n: int = 20) -> list[dict]:
    rng = random.Random(seed)
    selected = rng.sample(lines, min(n, len(lines))) if lines else []
    examples: list[dict] = []
    for line in selected:
        encoded = tokenizer.encode(line)
        examples.append(
            {
                "line_preview": line[:160],
                "token_count": len(encoded.tokens),
                "first_30_tokens": encoded.tokens[:30],
            }
        )
    return examples


def validate_loaded_tokenizer(path: Path, lines: list[str], seed: int, Tokenizer) -> dict:
    tokenizer = Tokenizer.from_file(str(path))
    rng = random.Random(seed)
    sample = rng.sample(lines, min(20, len(lines))) if lines else []
    failures = 0
    for line in sample:
        if line and len(tokenizer.encode(line).tokens) == 0:
            failures += 1
    return {
        "path": str(path),
        "loaded": True,
        "sample_size": len(sample),
        "empty_encoding_failures": failures,
    }


def render_report(summary: dict, mixed_tokenizer, hangulized_tokenizer, mixed_dev_examples: list[dict], hangulized_dev_examples: list[dict]) -> str:
    lines = [
        "# Korean KHDB 32K BPE Tokenizer Training Report",
        "",
        "This trains two standard BPE tokenizers for a small tokenizer diagnostic experiment.",
        "Only the 90% training split is used for tokenizer training; dev is used for sanity checks only.",
        "",
        "## Inputs",
        "",
    ]
    for key, value in summary["training_files"].items():
        lines.append(f"- {key}: `{value}`")
    for key, value in summary["dev_files"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Configuration",
            "",
            f"- tokenizer type: BPE",
            f"- vocab size requested: {summary['vocab_size_requested']}",
            f"- special tokens: {summary['special_tokens']}",
            f"- normalizer: NFKC",
            f"- whitespace pre-tokenizer: none",
            f"- byte fallback used: {summary['byte_fallback_used']}",
            f"- tokenizers version: {summary['tokenizer_library_version']}",
            "",
            "## Outputs",
            "",
        ]
    )
    for key, value in summary["output_paths"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Line Counts",
            "",
            f"- train lines: {summary['train_line_count']}",
            f"- dev lines: {summary['dev_line_count']}",
            f"- line alignment check: {summary['line_alignment_check_passed']}",
            f"- no-space check: {summary['no_space_check_passed']}",
            f"- hangulized no-Hanja check: {summary['hangulized_no_hanja_check_passed']}",
            "",
            "## UNK and Token Sanity Stats",
            "",
            "```json",
            json.dumps(
                {
                    "mixed_train_stats": summary["mixed_train_stats"],
                    "mixed_dev_stats": summary["mixed_dev_stats"],
                    "hangulized_train_stats": summary["hangulized_train_stats"],
                    "hangulized_dev_stats": summary["hangulized_dev_stats"],
                    "unk_warning_flags": summary["unk_warning_flags"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            "```",
            "",
            "## Mixed Vocab Preview",
            "",
            ", ".join(vocab_preview(mixed_tokenizer, 30)),
            "",
            "## Hangulized Vocab Preview",
            "",
            ", ".join(vocab_preview(hangulized_tokenizer, 30)),
            "",
            "## Mixed Dev Encoding Examples",
            "",
        ]
    )
    for example in mixed_dev_examples:
        lines.extend(
            [
                f"- chars: `{example['line_preview']}`",
                f"  token_count: {example['token_count']}",
                f"  first_30_tokens: {example['first_30_tokens']}",
            ]
        )
    lines.extend(["", "## Hangulized Dev Encoding Examples", ""])
    for example in hangulized_dev_examples:
        lines.extend(
            [
                f"- chars: `{example['line_preview']}`",
                f"  token_count: {example['token_count']}",
                f"  first_30_tokens: {example['first_30_tokens']}",
            ]
        )
    lines.extend(
        [
            "",
            "## Caveat",
            "",
            "These are tokenizer sanity checks only. Final fertility and N:1 pair analysis will be performed in the next step on the dev split.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    tokenizers_lib, Tokenizer, BPE, NFKC, BpeTrainer = require_tokenizers()
    train_mixed = read_lines(args.train_mixed)
    train_hangulized = read_lines(args.train_hangulized)
    dev_mixed = read_lines(args.dev_mixed)
    dev_hangulized = read_lines(args.dev_hangulized)
    validation = validate_inputs(train_mixed, train_hangulized, dev_mixed, dev_hangulized)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mixed_tokenizer_path = args.output_dir / "korean_mixed_bpe_32k.json"
    hangulized_tokenizer_path = args.output_dir / "korean_hangulized_bpe_32k.json"
    mixed_vocab_path = args.output_dir / "korean_mixed_bpe_32k_vocab.txt"
    hangulized_vocab_path = args.output_dir / "korean_hangulized_bpe_32k_vocab.txt"
    summary_path = args.output_dir / "tokenizer_training_summary.json"
    report_path = args.output_dir / "tokenizer_training_report.md"

    mixed_tokenizer = train_bpe(
        "mixed",
        train_mixed,
        mixed_tokenizer_path,
        mixed_vocab_path,
        args,
        BPE,
        NFKC,
        BpeTrainer,
    )
    hangulized_tokenizer = train_bpe(
        "hangulized",
        train_hangulized,
        hangulized_tokenizer_path,
        hangulized_vocab_path,
        args,
        BPE,
        NFKC,
        BpeTrainer,
    )

    mixed_train_stats = tokenizer_stats(mixed_tokenizer, train_mixed)
    mixed_dev_stats = tokenizer_stats(mixed_tokenizer, dev_mixed)
    hangulized_train_stats = tokenizer_stats(hangulized_tokenizer, train_hangulized)
    hangulized_dev_stats = tokenizer_stats(hangulized_tokenizer, dev_hangulized)
    loaded_validation = {
        "mixed_train": validate_loaded_tokenizer(mixed_tokenizer_path, train_mixed, args.seed, Tokenizer),
        "mixed_dev": validate_loaded_tokenizer(mixed_tokenizer_path, dev_mixed, args.seed + 1, Tokenizer),
        "hangulized_train": validate_loaded_tokenizer(hangulized_tokenizer_path, train_hangulized, args.seed + 2, Tokenizer),
        "hangulized_dev": validate_loaded_tokenizer(hangulized_tokenizer_path, dev_hangulized, args.seed + 3, Tokenizer),
    }
    output_paths = {
        "mixed_tokenizer": str(mixed_tokenizer_path),
        "hangulized_tokenizer": str(hangulized_tokenizer_path),
        "mixed_vocab": str(mixed_vocab_path),
        "hangulized_vocab": str(hangulized_vocab_path),
        "summary": str(summary_path),
        "report": str(report_path),
    }
    summary = {
        "vocab_size_requested": args.vocab_size,
        "actual_vocab_size_mixed": mixed_tokenizer.get_vocab_size(),
        "actual_vocab_size_hangulized": hangulized_tokenizer.get_vocab_size(),
        "special_tokens": SPECIAL_TOKENS,
        "training_files": {
            "train_mixed": str(args.train_mixed),
            "train_hangulized": str(args.train_hangulized),
        },
        "dev_files": {
            "dev_mixed": str(args.dev_mixed),
            "dev_hangulized": str(args.dev_hangulized),
        },
        "train_line_count": len(train_mixed),
        "dev_line_count": len(dev_mixed),
        "no_space_check_passed": validation["no_space_check_passed"],
        "line_alignment_check_passed": validation["train_aligned"] and validation["dev_aligned"],
        "hangulized_no_hanja_check_passed": validation["hangulized_no_hanja_check_passed"],
        "tokenizer_library": "huggingface_tokenizers",
        "tokenizer_library_version": tokenizers_lib.__version__,
        "byte_fallback_used": args.byte_fallback,
        "mixed_train_stats": mixed_train_stats,
        "mixed_dev_stats": mixed_dev_stats,
        "hangulized_train_stats": hangulized_train_stats,
        "hangulized_dev_stats": hangulized_dev_stats,
        "unk_warning_flags": {
            "mixed_train": warning_flag(mixed_train_stats),
            "mixed_dev": warning_flag(mixed_dev_stats),
            "hangulized_train": warning_flag(hangulized_train_stats),
            "hangulized_dev": warning_flag(hangulized_dev_stats),
        },
        "loaded_tokenizer_validation": loaded_validation,
        "output_paths": output_paths,
    }
    mixed_dev_examples = encoding_examples(mixed_tokenizer, dev_mixed, args.seed, 20)
    hangulized_dev_examples = encoding_examples(hangulized_tokenizer, dev_hangulized, args.seed, 20)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(render_report(summary, mixed_tokenizer, hangulized_tokenizer, mixed_dev_examples, hangulized_dev_examples), encoding="utf-8")

    for path in [mixed_tokenizer_path, hangulized_tokenizer_path, mixed_vocab_path, hangulized_vocab_path, summary_path, report_path]:
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"Expected non-empty output missing: {path}")

    print(f"Mixed vocab size: {summary['actual_vocab_size_mixed']}")
    print(f"Hangulized vocab size: {summary['actual_vocab_size_hangulized']}")
    print(f"Mixed dev UNK/10k chars: {mixed_dev_stats['unk_tokens_per_10k_chars']:.4f}")
    print(f"Hangulized dev UNK/10k chars: {hangulized_dev_stats['unk_tokens_per_10k_chars']:.4f}")
    print(f"Loaded tokenizer validation: {json.dumps(loaded_validation, ensure_ascii=False)}")
    print(f"Summary: {summary_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
