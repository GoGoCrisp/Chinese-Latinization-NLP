from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
SUMMARY_SCRIPT = BASE_DIR / "APPENDIX1_PARA_summarize_64k_transition_robustness.py"
OUTPUT_MD = BASE_DIR / "APPENDIX1_PARA_64k_transition_robustness_detailed_tables.md"


def load_summary_module():
    spec = importlib.util.spec_from_file_location("transition_summary", SUMMARY_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def fmt_int(value) -> str:
    return f"{int(float(value)):,}"


def fmt_float(value, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def build_rows(summary) -> tuple[list[dict], dict[str, dict]]:
    fertility = summary.load_fertility_rows()
    rows = []
    table2_by_param = {}

    for parameter in summary.PARAMETERS:
        key = parameter["key"]
        label = parameter["label"]
        decoded_dir = parameter["decoded_dir"]
        table2_by_param[label] = summary.analyze_table2(decoded_dir)

        for token_type, tokenizer_label, filename in summary.TOKENIZERS:
            fert = fertility[(key, token_type)]
            unique_vocab, raw_vocab = summary.vocab_total(decoded_dir / filename)
            rows.append(
                {
                    "parameter": label,
                    "tokenizer": tokenizer_label,
                    "tokens_per_sample": fert["tokens_per_sample"],
                    "tokens_per_surface_char": fert["tokens_per_surface_char"],
                    "tokens_per_original_char": fert["tokens_per_original_char"],
                    "total_tokens": fert["total_tokens"],
                    "total_chars": fert["total_chars"],
                    "total_original_chars": fert["total_original_chars"],
                    "morph_score": fert["morph_score"],
                    "valid_tokens": fert["valid_tokens"],
                    "checked_tokens": fert["checked_tokens"],
                    "invalid_tokens": fert["invalid_tokens"],
                    "skipped_punctuation_tokens": fert["skipped_punctuation_tokens"],
                    "overlap": fert["overlap"],
                    "chars_per_token": fert["chars_per_token"],
                    "bytes_per_token": fert["bytes_per_token"],
                    "chars_per_byte": fert["chars_per_byte"],
                    "bytes_per_original_char": fert["bytes_per_original_char"],
                    "vocab_unique": unique_vocab,
                    "vocab_raw": raw_vocab,
                }
            )

    return rows, table2_by_param


def append_fertility_table(lines: list[str], rows: list[dict]) -> None:
    lines.append("## Table 1. Fertility")
    lines.append("")
    lines.append("| Parameter | Tokenizer | Tokens/sample | Tokens/surface char | Tokens/original char | Total tokens | Surface chars | Original chars |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| {parameter} | {tokenizer} | {sample} | {surface} | {original} | {total_tokens} | {total_chars} | {total_original_chars} |".format(
                parameter=row["parameter"],
                tokenizer=row["tokenizer"],
                sample=fmt_float(row["tokens_per_sample"]),
                surface=fmt_float(row["tokens_per_surface_char"]),
                original=fmt_float(row["tokens_per_original_char"]),
                total_tokens=fmt_int(row["total_tokens"]),
                total_chars=fmt_int(row["total_chars"]),
                total_original_chars=fmt_int(row["total_original_chars"]),
            )
        )
    lines.append("")


def append_morph_table(lines: list[str], rows: list[dict]) -> None:
    lines.append("## Table 2. Morph / Validity Check")
    lines.append("")
    lines.append("| Parameter | Tokenizer | Morph score | Valid tokens | Checked tokens | Invalid tokens | Skipped punctuation | Overlap |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| {parameter} | {tokenizer} | {morph} | {valid} | {checked} | {invalid} | {skipped} | {overlap} |".format(
                parameter=row["parameter"],
                tokenizer=row["tokenizer"],
                morph=fmt_float(row["morph_score"]),
                valid=fmt_int(row["valid_tokens"]),
                checked=fmt_int(row["checked_tokens"]),
                invalid=fmt_int(row["invalid_tokens"]),
                skipped=fmt_int(row["skipped_punctuation_tokens"]),
                overlap=fmt_float(row["overlap"]),
            )
        )
    lines.append("")


def append_length_table(lines: list[str], rows: list[dict]) -> None:
    lines.append("## Table 3. Token Length / Byte Metrics")
    lines.append("")
    lines.append("| Parameter | Tokenizer | Chars/token | Bytes/token | Chars/byte | Bytes/original char |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| {parameter} | {tokenizer} | {chars_token} | {bytes_token} | {chars_byte} | {bytes_original} |".format(
                parameter=row["parameter"],
                tokenizer=row["tokenizer"],
                chars_token=fmt_float(row["chars_per_token"]),
                bytes_token=fmt_float(row["bytes_per_token"]),
                chars_byte=fmt_float(row["chars_per_byte"]),
                bytes_original=fmt_float(row["bytes_per_original_char"]),
            )
        )
    lines.append("")


def append_vocab_table(lines: list[str], rows: list[dict]) -> None:
    lines.append("## Table 4. Vocabulary Total")
    lines.append("")
    lines.append("| Parameter | Tokenizer | Unique whitespace-insensitive entries | Raw vocabulary entries | Display total |")
    lines.append("|---|---|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| {parameter} | {tokenizer} | {unique} | {raw} | {display} |".format(
                parameter=row["parameter"],
                tokenizer=row["tokenizer"],
                unique=fmt_int(row["vocab_unique"]),
                raw=fmt_int(row["vocab_raw"]),
                display=f"{fmt_int(row['vocab_unique'])} ({fmt_int(row['vocab_raw'])})",
            )
        )
    lines.append("")


def append_mapping_table(lines: list[str], table2_by_param: dict[str, dict]) -> None:
    lines.append("## Table 5. Chinese-Origin to Pinyin-Toned Mapping")
    lines.append("")
    lines.append("| Parameter | Mapping type | Chinese source tokens | Pinyin target tokens |")
    lines.append("|---|---|---:|---:|")
    for label, t2 in table2_by_param.items():
        mapping_rows = [
            ("Shared 1:1", t2["shared_1to1_source"], t2["shared_1to1_target"]),
            ("Shared 2:1", t2["shared_2to1_source"], t2["shared_2to1_target"]),
            ("Shared 3:1", t2["shared_3to1_source"], t2["shared_3to1_target"]),
            ("Shared 4:1", t2["shared_4to1_source"], t2["shared_4to1_target"]),
            ("Shared >4:1", t2["shared_gt4to1_source"], t2["shared_gt4to1_target"]),
            ("Unique", t2["unique_source"], t2["unique_target"]),
        ]
        for mapping_type, source, target in mapping_rows:
            lines.append(f"| {label} | {mapping_type} | {source:,} | {target:,} |")
    lines.append("")


def build_markdown(rows: list[dict], table2_by_param: dict[str, dict]) -> str:
    lines = [
        "# Appendix: 64k SuperBPE Transition Robustness Detailed Tables",
        "",
        "This file lists every measured column used in the 64k transition-point comparison. The settings are K=0, K=1000, 5% (K=3200 for 64k), K=2048, and 10% (K=6400 for 64k).",
        "",
    ]
    append_fertility_table(lines, rows)
    append_morph_table(lines, rows)
    append_length_table(lines, rows)
    append_vocab_table(lines, rows)
    append_mapping_table(lines, table2_by_param)
    return "\n".join(lines)


def main() -> None:
    summary = load_summary_module()
    rows, table2_by_param = build_rows(summary)
    OUTPUT_MD.write_text(build_markdown(rows, table2_by_param) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_MD}")


if __name__ == "__main__":
    main()
