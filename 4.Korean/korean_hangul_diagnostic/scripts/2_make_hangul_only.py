from __future__ import annotations

import argparse
import importlib.metadata
import json
import random
import re
from pathlib import Path
from typing import Optional


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_NORMAL = PROJECT_DIR / "corpora" / "1_korean_normal.txt"
DEFAULT_OUTPUT_HANGUL = PROJECT_DIR / "corpora" / "2_korean_hangul_only.txt"
DEFAULT_DIAGNOSTICS = PROJECT_DIR / "results" / "2_hangul_conversion_diagnostics.json"
DEFAULT_EXAMPLES = PROJECT_DIR / "results" / "2_hangul_conversion_examples.md"

HANJA_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
HANGUL_RE = re.compile(r"[\u1100-\u11ff\u3130-\u318f\uac00-\ud7af]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Korean normal lines to aligned Hangul-only lines where possible."
    )
    parser.add_argument("--input-normal", type=Path, default=DEFAULT_INPUT_NORMAL)
    parser.add_argument("--output-hangul", type=Path, default=DEFAULT_OUTPUT_HANGUL)
    parser.add_argument("--diagnostics", type=Path, default=DEFAULT_DIAGNOSTICS)
    parser.add_argument("--examples", type=Path, default=DEFAULT_EXAMPLES)
    parser.add_argument("--max-lines", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


class HanjaConverter:
    def __init__(self) -> None:
        self.name = "none"
        self.version = None
        self._hanja = None
        try:
            import hanja

            self._hanja = hanja
            self.name = "hanja"
            self.version = package_version("hanja")
        except Exception:
            self._hanja = None

    @property
    def available(self) -> bool:
        return self._hanja is not None

    def convert(self, text: str) -> str:
        if self._hanja is None:
            return text
        try:
            return self._hanja.translate(text, "substitution")
        except Exception:
            return text


def count_hanja(text: str) -> int:
    return len(HANJA_RE.findall(text))


def count_hangul(text: str) -> int:
    return len(HANGUL_RE.findall(text))


def write_examples(path: Path, examples: list[dict]) -> None:
    lines = ["# Hangul Conversion Examples", ""]
    for idx, item in enumerate(examples, start=1):
        lines.append(f"## Example {idx}")
        lines.append("")
        lines.append("Normal:")
        lines.append("")
        lines.append(item["normal"])
        lines.append("")
        lines.append("Hangul-only:")
        lines.append("")
        lines.append(item["hangul"])
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    converter = HanjaConverter()
    rng = random.Random(args.seed)

    args.output_hangul.parent.mkdir(parents=True, exist_ok=True)
    args.diagnostics.parent.mkdir(parents=True, exist_ok=True)
    args.examples.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        line.rstrip("\n")
        for line in args.input_normal.open("r", encoding="utf-8")
        if line.strip()
    ]
    if args.max_lines:
        lines = lines[: args.max_lines]

    diagnostics = {
        "input_normal": str(args.input_normal),
        "output_hangul": str(args.output_hangul),
        "seed": args.seed,
        "total_lines": len(lines),
        "converted_lines": 0,
        "lines_with_hanja_before": 0,
        "lines_with_hanja_after": 0,
        "hanja_chars_before": 0,
        "hanja_chars_after": 0,
        "hangul_chars_before": 0,
        "hangul_chars_after": 0,
        "converter_name": converter.name,
        "converter_version": converter.version,
        "converter_available": converter.available,
        "notes": (
            "If converter_available is false, Hanja is preserved. Korean Wikipedia "
            "is already mostly Hangul, so this diagnostic may be close to the normal corpus."
        ),
    }
    examples: list[dict] = []

    with args.output_hangul.open("w", encoding="utf-8") as out:
        for line_no, line in enumerate(lines, start=1):
            before_hanja = count_hanja(line)
            before_hangul = count_hangul(line)
            converted = converter.convert(line)
            after_hanja = count_hanja(converted)
            after_hangul = count_hangul(converted)

            diagnostics["converted_lines"] += 1
            diagnostics["hanja_chars_before"] += before_hanja
            diagnostics["hanja_chars_after"] += after_hanja
            diagnostics["hangul_chars_before"] += before_hangul
            diagnostics["hangul_chars_after"] += after_hangul
            if before_hanja:
                diagnostics["lines_with_hanja_before"] += 1
            if after_hanja:
                diagnostics["lines_with_hanja_after"] += 1

            out.write(converted + "\n")

            should_keep = before_hanja > 0 or rng.random() < 0.01 or len(lines) <= 50
            if should_keep and len(examples) < 50:
                examples.append(
                    {"line_no": line_no, "normal": line, "hangul": converted}
                )

    if len(examples) < 50:
        output_lines = args.output_hangul.read_text(encoding="utf-8").splitlines()
        candidate_indices = list(range(len(lines)))
        rng.shuffle(candidate_indices)
        existing = {item["line_no"] for item in examples}
        for idx in candidate_indices:
            line_no = idx + 1
            if line_no in existing:
                continue
            examples.append(
                {"line_no": line_no, "normal": lines[idx], "hangul": output_lines[idx]}
            )
            if len(examples) >= 50:
                break

    before = diagnostics["hanja_chars_before"]
    after = diagnostics["hanja_chars_after"]
    diagnostics["hanja_conversion_rate"] = (before - after) / before if before else 0.0
    diagnostics["residual_hanja_rate"] = after / before if before else 0.0

    args.diagnostics.write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_examples(args.examples, examples)

    normal_count = len(lines)
    hangul_count = sum(1 for _ in args.output_hangul.open("r", encoding="utf-8"))
    assert normal_count == hangul_count, "Normal and Hangul-only files must be line-aligned."

    print(f"Converted lines: {diagnostics['converted_lines']}")
    print(f"Hangul-only corpus: {args.output_hangul}")
    print(f"Diagnostics: {args.diagnostics}")
    print(f"Examples: {args.examples}")


if __name__ == "__main__":
    main()
