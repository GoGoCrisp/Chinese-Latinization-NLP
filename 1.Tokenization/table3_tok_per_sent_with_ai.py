from __future__ import annotations

"""
Build Table 3: tok/sent for four 64K custom tokenizers and three public models.

Rules:
- Chinese and public AI models use `chinese_origin_中国_test10.txt`.
- Pinyin tokenizers use the matching spaced `*_test10.txt` files, following
  the same mapping logic as `5th_Analyzation for 16 tokenization.py`.

Outputs:
- table3_tok_per_sent.csv
- table3_tok_per_sent.md

Quick accessibility check:
- `python3 table3_tok_per_sent_with_ai.py --check-ai-only --max-lines 3`
"""

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
CORPORA_DIR = BASE_DIR / "corpora"
SUPER_TOKENIZER_DIR = BASE_DIR / "superTokenizers_BPE"

OUTPUT_CSV = BASE_DIR / "table3_tok_per_sent.csv"
OUTPUT_MD = BASE_DIR / "table3_tok_per_sent.md"

HF_TOKEN = os.environ.get("HF_TOKEN")
LOCAL_CL100K_CANDIDATES = (
    Path.home() / ".cache" / "tiktoken" / "cl100k_base.tiktoken",
    Path.home() / ".vscode" / "extensions" / "github.copilot-chat-0.43.0" / "dist" / "cl100k_base.tiktoken",
)

try:
    from tokenizers import Tokenizer as LocalTokenizer

    HAS_TOKENIZERS = True
    TOKENIZERS_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - depends on local env
    HAS_TOKENIZERS = False
    TOKENIZERS_IMPORT_ERROR = str(exc)
    LocalTokenizer = None  # type: ignore[assignment]

try:
    import tiktoken

    HAS_TIKTOKEN = True
    TIKTOKEN_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - depends on local env
    HAS_TIKTOKEN = False
    TIKTOKEN_IMPORT_ERROR = str(exc)
    tiktoken = None  # type: ignore[assignment]

try:
    from transformers import AutoTokenizer

    HAS_TRANSFORMERS = True
    TRANSFORMERS_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - depends on local env
    HAS_TRANSFORMERS = False
    TRANSFORMERS_IMPORT_ERROR = str(exc)
    AutoTokenizer = None  # type: ignore[assignment]

try:
    from tqdm.auto import tqdm

    HAS_TQDM = True
    TQDM_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - depends on local env
    HAS_TQDM = False
    TQDM_IMPORT_ERROR = str(exc)
    tqdm = None  # type: ignore[assignment]


@dataclass(frozen=True)
class TokenizerSpec:
    label: str
    source_type: str
    corpus_type: str
    local_path: Path | None = None
    encoding_name: str | None = None
    model_candidates: tuple[str, ...] = ()


def find_test_file(corpus_type: str) -> Path:
    for file in os.listdir(CORPORA_DIR):
        fname = file.lower()

        if corpus_type == "origin" and fname == "chinese_origin_中国_test10.txt":
            return CORPORA_DIR / file

        if corpus_type == "diacritic" and fname == "pinyin_diacritic_spaced_test10.txt":
            return CORPORA_DIR / file

        if corpus_type == "toneless" and fname == "pinyin_toneless_spaced_test10.txt":
            return CORPORA_DIR / file

        if corpus_type == "toned" and fname == "pinyin_toned_spaced_test10.txt":
            return CORPORA_DIR / file

    raise FileNotFoundError(f"No test file found for corpus type: {corpus_type}")


def load_texts(path: Path, max_lines: int | None = None, progress_label: str = "") -> list[str]:
    texts: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        progress = None
        if HAS_TQDM:
            progress = tqdm(
                desc=progress_label or f"Read {path.name}",
                total=max_lines,
                unit="line",
                dynamic_ncols=True,
                leave=True,
            )

        try:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                texts.append(line)
                if progress is not None:
                    progress.update(1)
                if max_lines is not None and len(texts) >= max_lines:
                    break
        finally:
            if progress is not None:
                progress.close()
    return texts


def iter_with_tqdm(iterable: Any, desc: str, total: int | None = None, unit: str = "it") -> Any:
    if not HAS_TQDM:
        return iterable
    return tqdm(
        iterable,
        desc=desc,
        total=total,
        unit=unit,
        dynamic_ncols=True,
        leave=True,
    )


def print_progress_bar(label: str, current: int, total: int, detail: str = "") -> None:
    width = 28
    ratio = current / total if total else 1.0
    filled = min(width, int(width * ratio))
    bar = "#" * filled + "-" * (width - filled)
    suffix = f"  {detail}" if detail else ""
    print(f"{label}: [{bar}] {current}/{total}{suffix}", flush=True)


def short_error(exc: Exception, limit: int = 160) -> str:
    message = " ".join(str(exc).split())
    if len(message) <= limit:
        return message
    return message[: limit - 3] + "..."


def is_ai_spec(spec: TokenizerSpec) -> bool:
    return spec.source_type in {"tiktoken", "transformers"}


def build_specs() -> list[TokenizerSpec]:
    return [
        TokenizerSpec(
            label="Char-BPE (64K)",
            source_type="custom",
            corpus_type="origin",
            local_path=SUPER_TOKENIZER_DIR
            / "chinese_origin_subset100k_superbpe_64000"
            / "tokenizer.json",
        ),
        TokenizerSpec(
            label="Pinyin-Toned-BPE (64K)",
            source_type="custom",
            corpus_type="toned",
            local_path=SUPER_TOKENIZER_DIR
            / "pinyin_toned_subset100k_superbpe_64000"
            / "tokenizer.json",
        ),
        TokenizerSpec(
            label="Pinyin-Toneless-BPE (64K)",
            source_type="custom",
            corpus_type="toneless",
            local_path=SUPER_TOKENIZER_DIR
            / "pinyin_toneless_subset100k_superbpe_64000"
            / "tokenizer.json",
        ),
        TokenizerSpec(
            label="Pinyin-Diacritic-BPE (64K)",
            source_type="custom",
            corpus_type="diacritic",
            local_path=SUPER_TOKENIZER_DIR
            / "pinyin_diacritic_subset100k_superbpe_64000"
            / "tokenizer.json",
        ),
        TokenizerSpec(
            label="GPT-4 (cl100k_base)",
            source_type="tiktoken",
            corpus_type="origin",
            encoding_name="cl100k_base",
        ),
        TokenizerSpec(
            label="Llama-3",
            source_type="transformers",
            corpus_type="origin",
            model_candidates=(
                "NousResearch/Meta-Llama-3-8B",
            ),
        ),
        TokenizerSpec(
            label="Qwen",
            source_type="transformers",
            corpus_type="origin",
            model_candidates=(
                "Qwen/Qwen-7B",
            ),
        ),
    ]


def find_local_cl100k_file() -> Path | None:
    env_path = os.environ.get("CL100K_BASE_FILE")
    if env_path:
        candidate = Path(env_path).expanduser()
        if candidate.exists():
            return candidate

    for candidate in LOCAL_CL100K_CANDIDATES:
        if candidate.exists():
            return candidate

    vscode_extensions = Path.home() / ".vscode" / "extensions"
    if vscode_extensions.exists():
        matches = sorted(vscode_extensions.glob("github.copilot-chat-*/dist/cl100k_base.tiktoken"))
        if matches:
            return matches[-1]

    return None


def load_local_cl100k_encoding() -> tuple[Any | None, str]:
    if not HAS_TIKTOKEN:
        return None, f"missing dependency: tiktoken ({TIKTOKEN_IMPORT_ERROR})"

    local_file = find_local_cl100k_file()
    if local_file is None:
        return None, "local cl100k_base.tiktoken file not found"

    try:
        from tiktoken import Encoding
        from tiktoken.load import load_tiktoken_bpe

        mergeable_ranks = load_tiktoken_bpe(str(local_file))
        special_tokens = {
            "<|endoftext|>": 100257,
            "<|fim_prefix|>": 100258,
            "<|fim_middle|>": 100259,
            "<|fim_suffix|>": 100260,
            "<|endofprompt|>": 100276,
        }
        pat_str = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s"""
        encoding = Encoding(
            name="cl100k_base_local",
            pat_str=pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens,
        )
        return encoding, str(local_file)
    except Exception as exc:  # pragma: no cover - depends on local env
        return None, f"{local_file}: {exc}"


def load_runtime_tokenizer(spec: TokenizerSpec) -> tuple[Any | None, str]:
    try:
        if spec.source_type == "custom":
            if not HAS_TOKENIZERS:
                return None, f"missing dependency: tokenizers ({TOKENIZERS_IMPORT_ERROR})"
            if spec.local_path is None or not spec.local_path.exists():
                return None, f"tokenizer file not found: {spec.local_path}"
            tokenizer = LocalTokenizer.from_file(str(spec.local_path))
            return tokenizer, str(spec.local_path)

        if spec.source_type == "tiktoken":
            print(f"  Loading {spec.label} tokenizer...", flush=True)
            if not HAS_TIKTOKEN:
                return None, f"missing dependency: tiktoken ({TIKTOKEN_IMPORT_ERROR})"
            if spec.encoding_name is None:
                return None, "encoding_name is not configured"
            tokenizer, backend = load_local_cl100k_encoding()
            if tokenizer is not None:
                print(f"  Loaded {spec.label} from local file.", flush=True)
                return tokenizer, backend
            print(f"  Local cl100k file not found; loading {spec.encoding_name} via tiktoken.", flush=True)
            tokenizer = tiktoken.get_encoding(spec.encoding_name)
            print(f"  Loaded {spec.label}.", flush=True)
            return tokenizer, f"{spec.encoding_name} (downloaded)"

        if spec.source_type == "transformers":
            print(f"  Loading {spec.label} tokenizer...", flush=True)
            if not HAS_TRANSFORMERS:
                return None, f"missing dependency: transformers ({TRANSFORMERS_IMPORT_ERROR})"

            errors: list[str] = []
            for model_name in spec.model_candidates:
                try:
                    print(f"  Attempting remote load: {model_name}", flush=True)
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_name,
                        trust_remote_code=True,
                        token=HF_TOKEN,
                    )
                    print(f"  Loaded {model_name} from Hugging Face.", flush=True)
                    return tokenizer, model_name
                except Exception as exc:  # pragma: no cover - depends on local env/network
                    print(f"  Remote load failed: {short_error(exc)}", flush=True)
                    errors.append(f"{model_name} remote: {exc}")
                try:
                    print(f"  Attempting local cache load: {model_name}", flush=True)
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_name,
                        trust_remote_code=True,
                        token=HF_TOKEN,
                        local_files_only=True,
                    )
                    print(f"  Loaded {model_name} from local cache.", flush=True)
                    return tokenizer, f"{model_name} (local cache)"
                except Exception as exc:  # pragma: no cover - depends on local env/network
                    print(f"  Local cache load failed: {short_error(exc)}", flush=True)
                    errors.append(f"{model_name} local: {exc}")

            return None, " | ".join(errors)

        return None, f"unsupported source_type: {spec.source_type}"
    except Exception as exc:  # pragma: no cover - depends on local env/network
        return None, str(exc)


def count_tokens(text: str, runtime_tokenizer: Any, spec: TokenizerSpec) -> int:
    if spec.source_type == "custom":
        return len(runtime_tokenizer.encode(text).ids)

    if spec.source_type == "tiktoken":
        return len(runtime_tokenizer.encode(text))

    if spec.source_type == "transformers":
        return len(runtime_tokenizer.encode(text, add_special_tokens=False))

    raise ValueError(f"Unsupported source_type: {spec.source_type}")


def get_vocab_size(runtime_tokenizer: Any, spec: TokenizerSpec) -> int | str:
    try:
        if spec.source_type == "custom":
            return runtime_tokenizer.get_vocab_size(with_added_tokens=True)

        if spec.source_type == "tiktoken":
            return runtime_tokenizer.n_vocab

        if spec.source_type == "transformers":
            return len(runtime_tokenizer)
    except Exception as exc:  # pragma: no cover - depends on tokenizer implementation
        return f"unknown: {exc}"

    return "unknown"


def evaluate_spec(
    spec: TokenizerSpec,
    max_lines: int | None = None,
    check_ai_only: bool = False,
) -> dict[str, Any]:
    if check_ai_only and spec.source_type == "custom":
        return {}

    test_file = find_test_file(spec.corpus_type)
    texts = load_texts(
        test_file,
        max_lines=max_lines,
        progress_label=f"Read test text ({spec.label})",
    )
    runtime_tokenizer, backend = load_runtime_tokenizer(spec)

    result: dict[str, Any] = {
        "Tokenizer": spec.label,
        "Tok/Sent": "—",
        "status": "failed",
        "backend": backend,
        "test_file": str(test_file),
        "sentences": len(texts),
        "total_tokens": "",
        "vocab_size": "",
    }

    if runtime_tokenizer is None:
        return result

    result["vocab_size"] = get_vocab_size(runtime_tokenizer, spec)

    try:
        total_tokens = 0
        for text in iter_with_tqdm(
            texts,
            desc=f"Tokenize test text ({spec.label})",
            total=len(texts),
            unit="sent",
        ):
            total_tokens += count_tokens(text, runtime_tokenizer, spec)
    except Exception as exc:  # pragma: no cover - depends on tokenizer/runtime
        result["backend"] = f"{backend} | tokenization failed: {exc}"
        return result

    tok_per_sent = total_tokens / len(texts) if texts else 0.0
    result["Tok/Sent"] = f"{tok_per_sent:.4f}"
    result["status"] = "ok"
    result["total_tokens"] = total_tokens
    return result


def render_markdown_table(results: list[dict[str, Any]]) -> str:
    lines = [
        "# Table 3. Tokenizer Tok/Sent",
        "",
        "| Tokenizer | Vocab Size | Tok/Sent |",
        "|---|---:|---:|",
    ]
    for row in results:
        vocab_size = row["vocab_size"]
        if isinstance(vocab_size, int):
            vocab_size = f"{vocab_size:,}"
        lines.append(f"| {row['Tokenizer']} | {vocab_size} | {row['Tok/Sent']} |")
    return "\n".join(lines) + "\n"


def write_outputs(results: list[dict[str, Any]]) -> None:
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "Tokenizer",
                "vocab_size",
                "Tok/Sent",
                "status",
                "backend",
                "test_file",
                "sentences",
                "total_tokens",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    OUTPUT_MD.write_text(render_markdown_table(results), encoding="utf-8")


def print_accessibility_summary(results: list[dict[str, Any]]) -> None:
    print("\nAccessibility summary")
    print("=" * 80)
    for row in results:
        print(f"{row['Tokenizer']}: {row['status']}")
        print(f"  Vocab size: {row['vocab_size']}")
        print(f"  Tok/Sent: {row['Tok/Sent']}")
        print(f"  Backend: {row['backend']}")
        print(f"  Test file: {row['test_file']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build table3 tok/sent results.")
    parser.add_argument(
        "--max-lines",
        type=int,
        default=None,
        help="Only evaluate the first N non-empty lines for quick testing.",
    )
    parser.add_argument(
        "--check-ai-only",
        action="store_true",
        help="Only test GPT-4 / Llama-3 / Qwen accessibility and tok/sent.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = []
    specs = build_specs()
    if args.check_ai_only:
        specs = [spec for spec in specs if is_ai_spec(spec)]

    total_specs = len(specs)
    total_ai_specs = sum(1 for spec in specs if is_ai_spec(spec))
    tested_specs = 0
    loaded_ai_specs = 0

    print()
    print_progress_bar("Tokenizer tests", tested_specs, total_specs, "starting")
    if total_ai_specs:
        print_progress_bar("AI tokenizer loading", loaded_ai_specs, total_ai_specs, "waiting")

    for spec in specs:
        print(f"\nTesting {spec.label}...", flush=True)
        if is_ai_spec(spec):
            print_progress_bar("AI tokenizer loading", loaded_ai_specs, total_ai_specs, f"loading {spec.label}")

        result = evaluate_spec(
            spec,
            max_lines=args.max_lines,
        )
        if result:
            results.append(result)
            if is_ai_spec(spec):
                loaded_ai_specs += 1
                print_progress_bar(
                    "AI tokenizer loading",
                    loaded_ai_specs,
                    total_ai_specs,
                    f"{spec.label}: {result['status']}",
                )

        tested_specs += 1
        status = result["status"] if result else "skipped"
        print_progress_bar("Tokenizer tests", tested_specs, total_specs, f"{spec.label}: {status}")

    write_outputs(results)
    print_accessibility_summary(results)
    print("\nSaved:")
    print(f"  {OUTPUT_MD}")


if __name__ == "__main__":
    main()
