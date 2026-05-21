#!/usr/bin/env python3
"""Generate manual-review samples for Chinese-to-Pinyin conversion quality.

This script deliberately reuses the repository's existing conversion settings:

* corpus pipeline: jieba.cut(..., cut_all=False), then pypinyin.pinyin(...,
  style=Style.NORMAL/TONE3/TONE, strict=False) per jieba token.
* Eval2/Eval4 stored fields: existing diacritic fields are preserved when
  present; missing values are filled with the same evaluation helper convention
  used by the builders, lazy_pinyin(..., style=Style.TONE,
  neutral_tone_with_five=False, errors=lambda chunk: list(chunk)).

Automatic polyphone flags are review aids only. They are not error labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import jieba
from pypinyin import Style, lazy_pinyin, pinyin


DEFAULT_SAMPLE_SIZE = 500
DEFAULT_SEED = 42
SCRIPT_PATH = Path(__file__).resolve()
MODEL_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]

GENERAL_FIELDS = [
    "sample_id",
    "source_line",
    "source_n_chars",
    "jieba_tokens",
    "pinyin_toneless_pipeline",
    "pinyin_diacritic_pipeline",
    "pinyin_toned_pipeline",
    "n_pinyin_syllables",
    "potential_polyphone_chars",
    "potential_polyphone_count",
    "notes_auto",
    "manual_polyphone_error_count",
    "manual_segmentation_error_count",
    "manual_affects_meaning_or_eval",
    "manual_corrected_pinyin_optional",
    "manual_comments",
]

EVAL2_FIELDS = [
    "sample_id",
    "eval2_subset",
    "item_id",
    "source_line",
    "chinese_context",
    "gold_candidate_zh",
    "distractor_candidate_zh",
    "gold_candidate_pinyin",
    "distractor_candidate_pinyin",
    "pinyin_forms_collapse",
    "full_gold_sentence_zh",
    "full_distractor_sentence_zh",
    "full_gold_sentence_pinyin",
    "full_distractor_sentence_pinyin",
    "quality_flags",
    "manual_conversion_error_good",
    "manual_conversion_error_bad",
    "manual_error_changes_contrast",
    "manual_comments",
]

EVAL4_FIELDS = [
    "sample_id",
    "item_id",
    "phenomenon",
    "type",
    "subtype",
    "good_sentence_zh",
    "bad_sentence_zh",
    "good_sentence_pinyin",
    "bad_sentence_pinyin",
    "pinyin_pair_collapses_exactly",
    "data_source",
    "quality_flags",
    "manual_conversion_error_good",
    "manual_conversion_error_bad",
    "manual_error_changes_contrast",
    "manual_comments",
]

COLLISION_FIELDS = [
    "rank",
    "pinyin_form",
    "chinese_forms",
    "group_size",
    "count_or_frequency_if_available",
    "source_file",
    "manual_status",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create conversion-quality audit samples and summarize manual labels."
    )
    parser.add_argument("--zhwiki_test", default=None, help="Held-out zh-wiki Chinese test split.")
    parser.add_argument("--eval2_path", default=None, help="Optional Eval2 jsonl file or directory.")
    parser.add_argument("--eval4_path", default=None, help="Optional Eval4 jsonl file or directory.")
    parser.add_argument("--out_dir", default="conversion_audit")
    parser.add_argument("--sample_size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--manual_labels",
        action="append",
        default=[],
        help="Labeled audit CSV. May be passed more than once.",
    )
    parser.add_argument("--eval2_manual_labels", default=None)
    parser.add_argument("--eval4_manual_labels", default=None)
    parser.add_argument(
        "--collision_path",
        default=None,
        help="Optional Experiment 1 collision details CSV. Auto-discovered if omitted.",
    )
    parser.add_argument("--collision_top_n", type=int, default=50)
    return parser.parse_args()


def resolve_path(value: str | None, base: Path = MODEL_ROOT) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    cwd_path = (Path.cwd() / path).resolve()
    if cwd_path.exists():
        return cwd_path
    return (base / path).resolve()


def normalize_text(text: str) -> str:
    table = {
        "，": ",",
        "。": ".",
        "！": "!",
        "？": "?",
        "；": ";",
        "：": ":",
        "（": "(",
        "）": ")",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
    }
    for old, new in table.items():
        text = text.replace(old, new)
    return text


def jieba_tokens(text: str) -> list[str]:
    return list(jieba.cut(normalize_text(text), cut_all=False))


def pipeline_pinyin_for_tokens(tokens: list[str], style: Style) -> str:
    words: list[str] = []
    for word in tokens:
        py = pinyin(word, style=style, strict=False)
        word_py = " ".join(item[0] for item in py if item)
        if word_py:
            words.append(word_py)
    return " ".join(words)


def pipeline_convert(text: str) -> dict[str, Any]:
    tokens = jieba_tokens(text)
    toneless = pipeline_pinyin_for_tokens(tokens, Style.NORMAL)
    toned = pipeline_pinyin_for_tokens(tokens, Style.TONE3)
    diacritic = pipeline_pinyin_for_tokens(tokens, Style.TONE)
    return {
        "tokens": tokens,
        "toneless": toneless,
        "toned": toned,
        "diacritic": diacritic,
        "n_syllables": len([part for part in diacritic.split() if part]),
    }


def eval_diacritic(text: str) -> str:
    parts = lazy_pinyin(
        str(text),
        style=Style.TONE,
        neutral_tone_with_five=False,
        errors=lambda chunk: list(chunk),
    )
    return re.sub(r"\s+", " ", " ".join(part.strip() for part in parts if part.strip())).strip()


def eval_toneless(text: str) -> str:
    parts = lazy_pinyin(str(text), style=Style.NORMAL, errors=lambda chunk: list(chunk))
    return re.sub(r"\s+", " ", " ".join(part.strip() for part in parts if part.strip())).strip()


def is_cjk(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff"


def potential_polyphone_chars(text: str) -> tuple[str, int]:
    flagged: list[str] = []
    for ch in text:
        if not is_cjk(ch):
            continue
        heteronyms = pinyin(
            ch,
            style=Style.TONE,
            heteronym=True,
            strict=False,
            errors=lambda chunk: list(chunk),
        )
        variants = sorted({item for group in heteronyms for item in group if item})
        if len(variants) > 1:
            flagged.append(ch)
    return "|".join(flagged), len(flagged)


def read_nonempty_lines(path: Path) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.rstrip("\n")
            if text.strip():
                rows.append((line_no, text))
    return rows


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def sample_rows(rows: list[Any], sample_size: int, seed: int) -> list[Any]:
    rng = random.Random(seed)
    if len(rows) <= sample_size:
        out = list(rows)
        rng.shuffle(out)
        return out
    return rng.sample(rows, sample_size)


def sample_stratified(
    rows: list[dict[str, Any]], strata_key: str, max_items: int, seed: int
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(strata_key) or "unknown")].append(row)
    for group_rows in groups.values():
        rng.shuffle(group_rows)
    if len(rows) <= max_items:
        out = list(rows)
        rng.shuffle(out)
        return out
    strata = sorted(groups)
    base = max_items // max(len(strata), 1)
    remainder = max_items % max(len(strata), 1)
    selected: list[dict[str, Any]] = []
    for idx, key in enumerate(strata):
        take = base + (1 if idx < remainder else 0)
        selected.extend(groups[key][:take])
    if len(selected) < max_items:
        selected_ids = {id(row) for row in selected}
        leftovers = [row for key in strata for row in groups[key] if id(row) not in selected_ids]
        rng.shuffle(leftovers)
        selected.extend(leftovers[: max_items - len(selected)])
    rng.shuffle(selected)
    return selected[:max_items]


def generate_zhwiki_sample(zhwiki_test: Path, out_dir: Path, sample_size: int, seed: int) -> dict[str, Any]:
    rows = read_nonempty_lines(zhwiki_test)
    sampled = sample_rows(rows, sample_size, seed)
    out_rows: list[dict[str, Any]] = []
    suspicious = {"polyphone_flagged_lines": 0, "non_cjk_or_ascii_present": 0}
    for sample_idx, (line_no, text) in enumerate(sampled, start=1):
        conv = pipeline_convert(text)
        poly_chars, poly_count = potential_polyphone_chars(text)
        notes: list[str] = []
        if poly_count:
            notes.append("potential_polyphone_review_flag")
            suspicious["polyphone_flagged_lines"] += 1
        if any(not is_cjk(ch) and not ch.isspace() for ch in text):
            notes.append("contains_non_cjk_or_punctuation_or_latin")
            suspicious["non_cjk_or_ascii_present"] += 1
        out_rows.append(
            {
                "sample_id": f"zhwiki_{sample_idx:04d}",
                "source_line": text,
                "source_n_chars": len(text),
                "jieba_tokens": " | ".join(conv["tokens"]),
                "pinyin_toneless_pipeline": conv["toneless"],
                "pinyin_diacritic_pipeline": conv["diacritic"],
                "pinyin_toned_pipeline": conv["toned"],
                "n_pinyin_syllables": conv["n_syllables"],
                "potential_polyphone_chars": poly_chars,
                "potential_polyphone_count": poly_count,
                "notes_auto": ";".join(notes),
            }
        )
    output_path = out_dir / "zhwiki_conversion_audit_sample.csv"
    write_csv(output_path, GENERAL_FIELDS, out_rows)
    return {
        "path": str(output_path),
        "available_lines": len(rows),
        "sampled_items": len(out_rows),
        "suspicious": suspicious,
    }


def infer_eval2_subset(path: Path, row: dict[str, Any]) -> str:
    text = "/".join(part.lower() for part in path.parts)
    row_id = str(row.get("id", "")).lower()
    if "easy_random_control" in text or row_id.startswith("erc"):
        return "easy_control"
    if "nonhomophone_control" in text or row_id.startswith("nhc"):
        return "hard_control"
    return "homophone"


def discover_eval2_files(path: Path | None) -> list[Path]:
    if path is not None:
        if path.is_file():
            return [path]
        if path.is_dir():
            return sorted(path.rglob("*.jsonl"))
        return []
    candidates = [
        MODEL_ROOT / "eval_data/homophone_probe_v2/probe_v2.jsonl",
        MODEL_ROOT / "eval_data/nonhomophone_control_v2/nonhomophone_control_v2.jsonl",
        MODEL_ROOT / "eval_data/easy_random_control_v2/easy_random_control_v2.jsonl",
    ]
    return [path for path in candidates if path.exists()]


def generate_eval2_sample(eval2_path: Path | None, out_dir: Path, seed: int) -> dict[str, Any]:
    files = discover_eval2_files(eval2_path)
    rows: list[dict[str, Any]] = []
    for path in files:
        if path.name.endswith(".jsonl"):
            for row in read_jsonl(path):
                row = dict(row)
                row["_eval2_subset"] = infer_eval2_subset(path, row)
                rows.append(row)
    if not rows:
        return {"path": None, "sampled_items": 0, "available_items": 0, "files": []}
    sampled = sample_stratified(rows, "_eval2_subset", 100, seed)
    out_rows: list[dict[str, Any]] = []
    collapse_count = 0
    for idx, row in enumerate(sampled, start=1):
        gold_py = row.get("gold_pinyin_diacritic") or eval_diacritic(row.get("gold_zh", ""))
        distractor_py = row.get("distractor_pinyin_diacritic") or eval_diacritic(row.get("distractor_zh", ""))
        collapsed = str(gold_py).strip() == str(distractor_py).strip()
        collapse_count += int(collapsed)
        out_rows.append(
            {
                "sample_id": f"eval2_{idx:04d}",
                "eval2_subset": row.get("_eval2_subset", ""),
                "item_id": row.get("id", ""),
                "source_line": row.get("source_line", ""),
                "chinese_context": row.get("context_zh_with_blank", ""),
                "gold_candidate_zh": row.get("gold_zh", ""),
                "distractor_candidate_zh": row.get("distractor_zh", ""),
                "gold_candidate_pinyin": gold_py,
                "distractor_candidate_pinyin": distractor_py,
                "pinyin_forms_collapse": collapsed,
                "full_gold_sentence_zh": row.get("full_gold_sentence_zh", ""),
                "full_distractor_sentence_zh": row.get("full_distractor_sentence_zh", ""),
                "full_gold_sentence_pinyin": row.get("full_gold_sentence_diacritic")
                or eval_diacritic(row.get("full_gold_sentence_zh", "")),
                "full_distractor_sentence_pinyin": row.get("full_distractor_sentence_diacritic")
                or eval_diacritic(row.get("full_distractor_sentence_zh", "")),
                "quality_flags": json.dumps(row.get("quality_flags", []), ensure_ascii=False),
            }
        )
    output_path = out_dir / "eval2_conversion_audit_sample.csv"
    write_csv(output_path, EVAL2_FIELDS, out_rows)
    return {
        "path": str(output_path),
        "available_items": len(rows),
        "sampled_items": len(out_rows),
        "files": [str(path) for path in files],
        "automatic_collapse_count": collapse_count,
    }


def discover_eval4_file(path: Path | None) -> Path | None:
    if path is not None:
        if path.is_file():
            return path
        if path.is_dir():
            hits = sorted(path.rglob("*eval4*.jsonl"))
            if hits:
                return hits[0]
            hits = sorted(path.rglob("*.jsonl"))
            return hits[0] if hits else None
        return None
    candidate = MODEL_ROOT / "eval_data/eval4_chinese_blimp_style/eval4_chinese_blimp_style.jsonl"
    return candidate if candidate.exists() else None


def generate_eval4_sample(eval4_path: Path | None, out_dir: Path, seed: int) -> dict[str, Any]:
    path = discover_eval4_file(eval4_path)
    if path is None:
        return {"path": None, "sampled_items": 0, "available_items": 0, "file": None}
    rows = read_jsonl(path)
    sampled = sample_stratified(rows, "phenomenon", 150, seed)
    out_rows: list[dict[str, Any]] = []
    collapse_count = 0
    for idx, row in enumerate(sampled, start=1):
        good_py = row.get("good_sentence_diacritic") or row.get("good_sentence_pinyin") or eval_diacritic(
            row.get("good_sentence_zh", "")
        )
        bad_py = row.get("bad_sentence_diacritic") or row.get("bad_sentence_pinyin") or eval_diacritic(
            row.get("bad_sentence_zh", "")
        )
        collapsed = str(good_py).strip() == str(bad_py).strip()
        collapse_count += int(collapsed)
        out_rows.append(
            {
                "sample_id": f"eval4_{idx:04d}",
                "item_id": row.get("id", ""),
                "phenomenon": row.get("phenomenon", ""),
                "type": row.get("type", row.get("phenomenon", "")),
                "subtype": row.get("subtype_if_any", row.get("subtype", "")),
                "good_sentence_zh": row.get("good_sentence_zh", ""),
                "bad_sentence_zh": row.get("bad_sentence_zh", ""),
                "good_sentence_pinyin": good_py,
                "bad_sentence_pinyin": bad_py,
                "pinyin_pair_collapses_exactly": collapsed,
                "data_source": row.get("data_source", ""),
                "quality_flags": json.dumps(row.get("quality_flags", []), ensure_ascii=False),
            }
        )
    output_path = out_dir / "eval4_conversion_audit_sample.csv"
    write_csv(output_path, EVAL4_FIELDS, out_rows)
    return {
        "path": str(output_path),
        "available_items": len(rows),
        "sampled_items": len(out_rows),
        "file": str(path),
        "automatic_collapse_count": collapse_count,
    }


def find_collision_path(explicit: Path | None) -> Path | None:
    if explicit and explicit.exists():
        return explicit
    candidates = [
        PROJECT_ROOT
        / "1.Tokenization/decoded_superTokenizers_2048_subset100k/table2/table2_ad_overlap_superBPE_outputs/table2_ad_overlap_superBPE_details.csv",
        PROJECT_ROOT
        / "1.Tokenization/decoded_superTokenizers/table2_ac_overlap_superBPE_outputs/table2_ac_overlap_superBPE_details.csv",
    ]
    return next((path for path in candidates if path.exists()), None)


def generate_collision_sanity(
    collision_path: Path | None, out_dir: Path, top_n: int
) -> dict[str, Any]:
    path = find_collision_path(collision_path)
    if path is None:
        return {"path": None, "items": 0, "source": None}
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            pinyin_form = row.get("d_token") or row.get("c_token") or row.get("b_token") or row.get("pinyin_form")
            forms_raw = row.get("a_tokens_json") or row.get("chinese_forms_json") or "[]"
            try:
                forms = json.loads(forms_raw)
            except json.JSONDecodeError:
                forms = [forms_raw]
            if not pinyin_form or not isinstance(forms, list) or len(forms) < 2:
                continue
            group_size = int(row.get("N") or row.get("source_token_count_for_pair") or len(forms))
            rows.append(
                {
                    "pinyin_form": pinyin_form,
                    "chinese_forms": " | ".join(str(item) for item in forms),
                    "group_size": group_size,
                    "count_or_frequency_if_available": row.get("source_token_count_for_pair", ""),
                }
            )
    rows.sort(key=lambda item: (-int(item["group_size"]), str(item["pinyin_form"])))
    out_rows = []
    for rank, row in enumerate(rows[:top_n], start=1):
        out_rows.append(
            {
                "rank": rank,
                **row,
                "source_file": str(path),
                "manual_status": "needs_manual_genuine_homophone_check",
            }
        )
    output_path = out_dir / "diacritic_collision_sanity_check.csv"
    write_csv(output_path, COLLISION_FIELDS, out_rows)
    return {"path": str(output_path), "items": len(out_rows), "source": str(path)}


def write_manual_readme(out_dir: Path) -> Path:
    path = out_dir / "README_manual_audit.md"
    text = """# Conversion Quality Manual Audit

This directory contains CSV samples for checking whether Chinese-to-Pinyin conversion quality could confound the reported model comparisons. The automatic columns are review aids only; do not treat them as gold error labels.

## How to label zhwiki_conversion_audit_sample.csv

- `manual_polyphone_error_count`: enter the number of characters or words whose Pinyin pronunciation is wrong in context. A polyphone error means the Pinyin pronunciation is wrong in context.
- `manual_segmentation_error_count`: enter the number of jieba segmentation boundary errors that create an unnatural or misleading word boundary affecting Pinyin grouping or a downstream contrast.
- `manual_affects_meaning_or_eval`: use `1`/`true` when the conversion error changes meaning or would plausibly affect an evaluation contrast; otherwise use `0`/`false`.
- `manual_corrected_pinyin_optional`: optionally enter a corrected Pinyin string.
- `manual_comments`: add short evidence or uncertainty notes.

The `potential_polyphone_chars` column is only an automatic review flag produced with pypinyin heteronym mode. Do not mark it as an error unless the pipeline pronunciation is wrong in context.

## How to label Eval2 and Eval4 samples

- `manual_conversion_error_good`: use `1`/`true` if the good/gold side has a conversion error.
- `manual_conversion_error_bad`: use `1`/`true` if the bad/distractor side has a conversion error.
- `manual_error_changes_contrast`: use `1`/`true` if the conversion error changes the intended good/bad contrast, or creates/removes a Pinyin collapse not licensed by true pronunciation.
- `manual_comments`: add the corrected pronunciation or the reason for the label.

Do not mark genuine homophones as conversion errors. Genuine same-pronunciation pairs are expected in homophone probes and collision groups.

## After labeling

Save a labeled copy, for example:

```bash
python scripts/audit_conversion_quality.py \\
  --manual_labels conversion_audit/zhwiki_conversion_audit_sample_labeled.csv \\
  --manual_labels conversion_audit/eval2_conversion_audit_sample_labeled.csv \\
  --manual_labels conversion_audit/eval4_conversion_audit_sample_labeled.csv
```
"""
    path.write_text(text, encoding="utf-8")
    return path


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "是"}


def numeric(value: Any) -> float:
    text = str(value).strip()
    if not text:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def summarize_zhwiki_labels(rows: list[dict[str, str]]) -> dict[str, Any]:
    reviewed = [
        row
        for row in rows
        if any(
            str(row.get(col, "")).strip()
            for col in (
                "manual_polyphone_error_count",
                "manual_segmentation_error_count",
                "manual_affects_meaning_or_eval",
                "manual_comments",
            )
        )
    ]
    chars = sum(int(numeric(row.get("source_n_chars", 0))) for row in reviewed)
    poly_errors = sum(numeric(row.get("manual_polyphone_error_count", 0)) for row in reviewed)
    seg_errors = sum(numeric(row.get("manual_segmentation_error_count", 0)) for row in reviewed)
    return {
        "type": "zhwiki",
        "reviewed_items": len(reviewed),
        "total_chinese_chars": chars,
        "sentence_level_polyphone_error_rate": (
            sum(1 for row in reviewed if numeric(row.get("manual_polyphone_error_count", 0)) > 0) / len(reviewed)
            if reviewed
            else None
        ),
        "polyphone_errors_per_1000_chars": (poly_errors / chars * 1000 if chars else None),
        "sentence_level_segmentation_boundary_error_rate": (
            sum(1 for row in reviewed if numeric(row.get("manual_segmentation_error_count", 0)) > 0) / len(reviewed)
            if reviewed
            else None
        ),
        "segmentation_errors_per_1000_chars": (seg_errors / chars * 1000 if chars else None),
        "eval_affecting_error_rate": (
            sum(1 for row in reviewed if truthy(row.get("manual_affects_meaning_or_eval", ""))) / len(reviewed)
            if reviewed
            else None
        ),
        "top_examples": top_manual_examples(reviewed),
    }


def summarize_eval_labels(rows: list[dict[str, str]], label_type: str) -> dict[str, Any]:
    reviewed = [
        row
        for row in rows
        if any(
            str(row.get(col, "")).strip()
            for col in (
                "manual_conversion_error_good",
                "manual_conversion_error_bad",
                "manual_error_changes_contrast",
                "manual_comments",
            )
        )
    ]
    return {
        "type": label_type,
        "reviewed_items": len(reviewed),
        "conversion_error_good_rate": (
            sum(1 for row in reviewed if truthy(row.get("manual_conversion_error_good", ""))) / len(reviewed)
            if reviewed
            else None
        ),
        "conversion_error_bad_rate": (
            sum(1 for row in reviewed if truthy(row.get("manual_conversion_error_bad", ""))) / len(reviewed)
            if reviewed
            else None
        ),
        "eval_affecting_conversion_error_rate": (
            sum(1 for row in reviewed if truthy(row.get("manual_error_changes_contrast", ""))) / len(reviewed)
            if reviewed
            else None
        ),
        "top_examples": top_manual_examples(reviewed),
    }


def top_manual_examples(rows: list[dict[str, str]], limit: int = 5) -> list[dict[str, str]]:
    examples: list[dict[str, str]] = []
    for row in rows:
        has_error = (
            numeric(row.get("manual_polyphone_error_count", 0)) > 0
            or numeric(row.get("manual_segmentation_error_count", 0)) > 0
            or truthy(row.get("manual_affects_meaning_or_eval", ""))
            or truthy(row.get("manual_conversion_error_good", ""))
            or truthy(row.get("manual_conversion_error_bad", ""))
            or truthy(row.get("manual_error_changes_contrast", ""))
        )
        if has_error:
            examples.append(
                {
                    "sample_id": row.get("sample_id", ""),
                    "source_or_good": row.get("source_line")
                    or row.get("good_sentence_zh")
                    or row.get("chinese_context", ""),
                    "manual_comments": row.get("manual_comments", ""),
                }
            )
        if len(examples) >= limit:
            break
    return examples


def infer_label_type(path: Path, rows: list[dict[str, str]]) -> str:
    fields = set(rows[0].keys()) if rows else set()
    name = path.name.lower()
    if "eval2" in name or "gold_candidate_zh" in fields:
        return "eval2"
    if "eval4" in name or "good_sentence_zh" in fields:
        return "eval4"
    return "zhwiki"


def write_summary(out_dir: Path, generation: dict[str, Any], label_paths: list[Path]) -> dict[str, Any]:
    labels: list[dict[str, Any]] = []
    for path in label_paths:
        if not path.exists():
            labels.append({"path": str(path), "status": "missing"})
            continue
        rows = read_csv(path)
        label_type = infer_label_type(path, rows)
        if label_type == "zhwiki":
            summary = summarize_zhwiki_labels(rows)
        else:
            summary = summarize_eval_labels(rows, label_type)
        summary["path"] = str(path)
        labels.append(summary)

    manual_pending = not labels or all(item.get("reviewed_items", 0) == 0 for item in labels if item.get("status") != "missing")
    summary = {
        "generation": generation,
        "manual_label_summaries": labels,
        "manual_review_pending": manual_pending,
    }
    json_path = out_dir / "conversion_audit_summary.json"
    md_path = out_dir / "conversion_audit_summary.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown_summary(summary), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path), "manual_review_pending": manual_pending}


def render_markdown_summary(summary: dict[str, Any]) -> str:
    gen = summary.get("generation", {})
    lines = [
        "# Conversion Audit Summary",
        "",
        "## Pipeline Settings Used",
        "",
        "- Jieba segmentation: `jieba.cut(text, cut_all=False)` after the existing light punctuation normalization.",
        "- Pinyin-Toneless: `pypinyin.pinyin(word, style=Style.NORMAL, strict=False)` per jieba token.",
        "- Pinyin-Toned: `pypinyin.pinyin(word, style=Style.TONE3, strict=False)` per jieba token.",
        "- Pinyin-Diacritic: `pypinyin.pinyin(word, style=Style.TONE, strict=False)` per jieba token.",
        "- Main split files: `data/raw/train.zh.txt`, `valid.zh.txt`, `test.zh.txt`, aligned to `train.diacritic.txt`, `valid.diacritic.txt`, `test.diacritic.txt`.",
        "- Eval2/Eval4 stored diacritic fields are preserved when present; missing fields use the existing eval builder convention with `lazy_pinyin(..., style=Style.TONE, neutral_tone_with_five=False, errors=lambda chunk: list(chunk))`.",
        "",
        "## Generated Samples",
        "",
    ]
    for key in ("zhwiki", "eval2", "eval4", "collision_sanity"):
        value = gen.get(key)
        if not value:
            continue
        lines.append(f"- {key}: `{value.get('path')}`; sampled/items = {value.get('sampled_items', value.get('items', 0))}")
    lines.extend(["", "## Manual Labels", ""])
    labels = summary.get("manual_label_summaries", [])
    if summary.get("manual_review_pending"):
        lines.append("Manual review is pending. Fill the manual columns in copied labeled CSVs, then rerun with `--manual_labels`.")
    for item in labels:
        if item.get("status") == "missing":
            lines.append(f"- Missing label file: `{item.get('path')}`")
            continue
        lines.append(f"- {item.get('type')}: reviewed_items={item.get('reviewed_items')}")
        for metric in (
            "sentence_level_polyphone_error_rate",
            "polyphone_errors_per_1000_chars",
            "sentence_level_segmentation_boundary_error_rate",
            "segmentation_errors_per_1000_chars",
            "eval_affecting_error_rate",
            "eval_affecting_conversion_error_rate",
        ):
            if metric in item:
                lines.append(f"  - {metric}: {item.get(metric)}")
        examples = item.get("top_examples") or []
        if examples:
            lines.append("  - Top examples:")
            for example in examples:
                source = str(example.get("source_or_good", "")).replace("\n", " ")[:160]
                lines.append(f"    - {example.get('sample_id')}: {source}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    out_dir = resolve_path(args.out_dir, MODEL_ROOT) or (MODEL_ROOT / "conversion_audit")
    out_dir.mkdir(parents=True, exist_ok=True)

    zhwiki_test = resolve_path(args.zhwiki_test, MODEL_ROOT)
    if zhwiki_test is None:
        default_zhwiki = MODEL_ROOT / "data/raw/test.zh.txt"
        zhwiki_test = default_zhwiki if default_zhwiki.exists() else None
    eval2_path = resolve_path(args.eval2_path, MODEL_ROOT)
    eval4_path = resolve_path(args.eval4_path, MODEL_ROOT)
    collision_path = resolve_path(args.collision_path, MODEL_ROOT)

    generation: dict[str, Any] = {
        "seed": args.seed,
        "sample_size": args.sample_size,
        "paths": {
            "zhwiki_test": str(zhwiki_test) if zhwiki_test else None,
            "eval2_path": str(eval2_path) if eval2_path else None,
            "eval4_path": str(eval4_path) if eval4_path else None,
        },
    }

    if zhwiki_test and zhwiki_test.exists():
        generation["zhwiki"] = generate_zhwiki_sample(zhwiki_test, out_dir, args.sample_size, args.seed)
    else:
        generation["zhwiki"] = {"path": None, "sampled_items": 0, "status": "missing_zhwiki_test"}

    generation["eval2"] = generate_eval2_sample(eval2_path, out_dir, args.seed)
    generation["eval4"] = generate_eval4_sample(eval4_path, out_dir, args.seed)
    generation["collision_sanity"] = generate_collision_sanity(collision_path, out_dir, args.collision_top_n)

    readme_path = write_manual_readme(out_dir)
    generation["manual_readme"] = str(readme_path)

    label_paths = [resolve_path(path, MODEL_ROOT) for path in args.manual_labels]
    if args.eval2_manual_labels:
        label_paths.append(resolve_path(args.eval2_manual_labels, MODEL_ROOT))
    if args.eval4_manual_labels:
        label_paths.append(resolve_path(args.eval4_manual_labels, MODEL_ROOT))
    label_paths = [path for path in label_paths if path is not None]

    summary_paths = write_summary(out_dir, generation, label_paths)
    print(json.dumps({"generation": generation, "summary": summary_paths}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
