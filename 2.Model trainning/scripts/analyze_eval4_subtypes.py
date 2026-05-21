#!/usr/bin/env python3
"""Fine-grained subtype analysis for Eval 4 / ZhoBLiMP.

This script only reads the existing standardized Eval 4 dataset and model score
CSV. It does not rerun scoring and does not modify original Eval 4 result files.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any


DEFAULT_DATASET_CANDIDATES = [
    "eval_data/eval4_chinese_blimp_style/eval4_chinese_blimp_style_standardized.jsonl",
    "eval_data/eval4_chinese_blimp_style/eval4_chinese_blimp_style.jsonl",
]
DEFAULT_SCORES = "eval_results/eval4_chinese_blimp_style/item_scores.csv"
DEFAULT_OUTPUT_DIR = "eval_results/eval4_chinese_blimp_style/subtype_analysis"
DEFAULT_ZHOBLIMP_REPO_URL = "https://github.com/sjtu-compling/ZhoBLiMP.git"
EXPECTED_ITEMS = 35_400
EXPECTED_MODELS = {"chinese_4epoch", "diacritic_matched_token_4epoch"}
BASELINE = 0.5
MAIN_MIN_ITEMS = 10
EXAMPLE_ROWS_PER_SUBTYPE = 3

FUNCTION_WORDS = [
    "没有",
    "任何",
    "无论",
    "所有",
    "自己",
    "的",
    "地",
    "得",
    "了",
    "着",
    "过",
    "吗",
    "呢",
    "吧",
    "被",
    "把",
    "不",
    "没",
    "们",
    "都",
    "也",
    "每",
]

SUBTYPE_SUMMARY_FIELDS = [
    "phenomenon",
    "subtype_key",
    "subtype_label",
    "phenomenon_subtype",
    "n_items",
    "edit_type",
    "good_span",
    "bad_span",
    "good_span_len_chars",
    "bad_span_len_chars",
    "diff_len_delta",
    "mean_diff_position_ratio",
    "span_contains_other_count",
    "span_contains_other_rate",
    "function_words_involved",
    "diacritic_identical_count",
    "diacritic_identical_rate",
    "chinese_accuracy",
    "diacritic_accuracy",
    "gap_chinese_minus_diacritic",
    "chinese_mean_margin",
    "diacritic_mean_margin",
    "margin_gap",
    "chinese_median_margin",
    "diacritic_median_margin",
    "chinese_n_ties",
    "diacritic_n_ties",
    "n_collapsed_diacritic",
    "chinese_minus_baseline",
    "diacritic_minus_baseline",
    "interpretation_bucket",
    "uids",
    "example_item_ids",
]

PHENOMENON_FIELDS = [
    "phenomenon",
    "n_items",
    "chinese_accuracy",
    "diacritic_accuracy",
    "gap",
    "chinese_minus_baseline",
    "diacritic_minus_baseline",
    "chinese_mean_margin",
    "diacritic_mean_margin",
    "diacritic_exact_identical_count",
    "diacritic_exact_identical_rate",
    "collapsed_count",
    "collapsed_rate",
    "tie_count",
    "tie_rate",
    "interpretation_bucket",
    "top_subtypes_by_n",
    "top_subtypes_by_gap",
]

ANAPHOR_FIELDS = [
    "phenomenon",
    "subtype_key",
    "subtype_label",
    "n_items",
    "collapsed_count",
    "collapsed_rate",
    "chinese_accuracy_all",
    "diacritic_accuracy_all",
    "gap_all",
    "noncollapsed_n_items",
    "chinese_accuracy_noncollapsed",
    "diacritic_accuracy_noncollapsed",
    "noncollapsed_gap",
    "examples",
]

EXAMPLE_FIELDS = [
    "item_id",
    "phenomenon",
    "subtype_key",
    "subtype_label",
    "good_sentence_zh",
    "bad_sentence_zh",
    "good_sentence_diacritic",
    "bad_sentence_diacritic",
    "model_correct_chinese",
    "model_correct_diacritic",
    "chinese_margin",
    "diacritic_margin",
    "common_prefix",
    "good_diff_span",
    "bad_diff_span",
    "common_suffix",
]

PARADIGM_INVENTORY_FIELDS = [
    "phenomenon",
    "uid",
    "template_filename",
    "strict_MP",
    "good_rule_text",
    "bad_rule_text",
    "rule_diff_summary",
    "n_items_in_dataset",
    "chinese_accuracy",
    "diacritic_accuracy",
    "gap",
    "chinese_minus_baseline",
    "diacritic_minus_baseline",
    "diacritic_identical_count",
    "diacritic_identical_rate",
    "tie_count",
    "tie_rate",
]

PARADIGM_SUMMARY_FIELDS = [
    "phenomenon",
    "uid",
    "template_filename",
    "strict_MP",
    "good_rule_text",
    "bad_rule_text",
    "rule_diff_summary",
    "n_items",
    "chinese_accuracy",
    "diacritic_accuracy",
    "gap",
    "chinese_minus_baseline",
    "diacritic_minus_baseline",
    "chinese_mean_margin",
    "diacritic_mean_margin",
    "diacritic_exact_identical_count",
    "diacritic_exact_identical_rate",
    "tie_count",
    "tie_rate",
]

PARADIGM_SURFACE_DIFF_FIELDS = [
    "phenomenon",
    "uid",
    "template_filename",
    "strict_MP",
    "observed_surface_diff_key",
    "observed_surface_diff_label",
    "observed_edit_type",
    "good_diff_span",
    "bad_diff_span",
    "observed_diacritic_diff_key",
    "n_items",
    "chinese_accuracy",
    "diacritic_accuracy",
    "gap",
    "chinese_minus_baseline",
    "diacritic_minus_baseline",
    "chinese_mean_margin",
    "diacritic_mean_margin",
    "diacritic_exact_identical_count",
    "diacritic_exact_identical_rate",
    "tie_count",
    "tie_rate",
]

PARADIGM_EXAMPLE_FIELDS = [
    "example_group",
    "item_id",
    "phenomenon",
    "uid",
    "template_filename",
    "strict_MP",
    "good_rule_text",
    "bad_rule_text",
    "rule_diff_summary",
    "observed_surface_diff_label",
    "observed_diacritic_diff_key",
    "good_sentence_zh",
    "bad_sentence_zh",
    "good_sentence_diacritic",
    "bad_sentence_diacritic",
    "model_correct_chinese",
    "model_correct_diacritic",
    "chinese_margin",
    "diacritic_margin",
    "common_prefix",
    "good_diff_span",
    "bad_diff_span",
    "common_suffix",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze fine-grained Eval 4 / ZhoBLiMP subtypes.")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--scores", default=DEFAULT_SCORES)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--zhoblimp-template-dir",
        default=None,
        help="Directory containing official ZhoBLiMP template JSONs, usually projects/ZhoBLiMP.",
    )
    parser.add_argument(
        "--zhoblimp-repo-dir",
        default=None,
        help="Local ZhoBLiMP repo checkout. Defaults to <output-dir>/ZhoBLiMP.",
    )
    parser.add_argument("--zhoblimp-repo-url", default=DEFAULT_ZHOBLIMP_REPO_URL)
    parser.add_argument("--expected-items", type=int, default=EXPECTED_ITEMS)
    parser.add_argument("--main-min-items", type=int, default=MAIN_MIN_ITEMS)
    parser.add_argument("--examples-per-subtype", type=int, default=EXAMPLE_ROWS_PER_SUBTYPE)
    return parser.parse_args()


def project_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def resolve_dataset_path(root: Path, dataset_arg: str | None) -> Path:
    if dataset_arg:
        path = project_path(root, dataset_arg)
        if not path.exists():
            raise FileNotFoundError(f"Dataset does not exist: {path}")
        return path
    for candidate in DEFAULT_DATASET_CANDIDATES:
        path = project_path(root, candidate)
        if path.exists():
            return path
    raise FileNotFoundError(f"No Eval 4 dataset found among: {DEFAULT_DATASET_CANDIDATES}")


def resolve_template_dir(
    root: Path,
    output_dir: Path,
    template_dir_arg: str | None,
    repo_dir_arg: str | None,
    repo_url: str,
) -> Path:
    if template_dir_arg:
        template_dir = project_path(root, template_dir_arg)
        if not template_dir.exists():
            raise FileNotFoundError(f"ZhoBLiMP template dir does not exist: {template_dir}")
        return template_dir

    candidates = []
    if repo_dir_arg:
        repo_dir = project_path(root, repo_dir_arg)
        candidates.append(repo_dir / "projects" / "ZhoBLiMP")
    candidates.extend(
        [
            output_dir / "ZhoBLiMP" / "projects" / "ZhoBLiMP",
            root / "ZhoBLiMP" / "projects" / "ZhoBLiMP",
            root.parent / "ZhoBLiMP" / "projects" / "ZhoBLiMP",
        ]
    )
    for candidate in candidates:
        if candidate.exists() and any(candidate.glob("*.json")):
            return candidate

    repo_dir = project_path(root, repo_dir_arg) if repo_dir_arg else output_dir / "ZhoBLiMP"
    if not repo_dir.exists():
        print(f"official ZhoBLiMP templates not found; cloning {repo_url} into {repo_dir}")
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(repo_dir)],
            check=True,
        )
    template_dir = repo_dir / "projects" / "ZhoBLiMP"
    if not template_dir.exists() or not any(template_dir.glob("*.json")):
        raise FileNotFoundError(f"No official ZhoBLiMP template JSONs found in {template_dir}")
    return template_dir


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def mean(values: list[float]) -> float | str:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else ""


def median_or_blank(values: list[float]) -> float | str:
    finite = [value for value in values if math.isfinite(value)]
    return median(finite) if finite else ""


def accuracy(rows: list[dict[str, Any]]) -> float | str:
    finite = [row for row in rows if int(row.get("non_finite", 0)) == 0]
    if not finite:
        return ""
    return sum(int(row["correct"]) for row in finite) / len(finite)


def diff_analysis(good: str, bad: str, good_py: str, bad_py: str) -> dict[str, Any]:
    matcher = difflib.SequenceMatcher(a=good, b=bad, autojunk=False)
    opcodes = matcher.get_opcodes()
    diff_ops = [op for op in opcodes if op[0] != "equal"]
    if not diff_ops:
        edit_type = "identical_zh"
    elif len(diff_ops) > 1:
        edit_type = "multiple_edits"
    else:
        tag, *_ = diff_ops[0]
        if tag == "replace":
            edit_type = "replace"
        elif tag == "insert":
            edit_type = "insert_in_bad"
        elif tag == "delete":
            edit_type = "delete_in_bad"
        else:
            edit_type = tag

    if diff_ops:
        first = diff_ops[0]
        last = diff_ops[-1]
        common_prefix = good[: first[1]]
        common_suffix = good[last[2] :]
        start_index = first[1]
    else:
        common_prefix = good
        common_suffix = ""
        start_index = 0

    changed_pairs: list[tuple[str, str]] = []
    for _tag, i1, i2, j1, j2 in diff_ops:
        changed_pairs.append((good[i1:i2], bad[j1:j2]))

    good_diff_span = "|".join(pair[0] for pair in changed_pairs)
    bad_diff_span = "|".join(pair[1] for pair in changed_pairs)

    if edit_type == "replace":
        good_span, bad_span = changed_pairs[0]
        subtype_key = f"{good_span}->{bad_span}"
        subtype_label = f"{good_span} → {bad_span}" if len(good_span) <= 8 and len(bad_span) <= 8 else subtype_key
    elif edit_type == "insert_in_bad":
        good_span, bad_span = changed_pairs[0]
        subtype_key = f"INSERT_BAD:{bad_span}"
        subtype_label = f"bad inserts {bad_span}"
    elif edit_type == "delete_in_bad":
        good_span, bad_span = changed_pairs[0]
        subtype_key = f"DELETE_BAD:{good_span}"
        subtype_label = f"bad deletes {good_span}"
    elif edit_type == "multiple_edits":
        compact_parts = [f"{left}->{right}" for left, right in changed_pairs]
        subtype_key = "MULTI:" + "|".join(compact_parts)
        subtype_label = "multiple edits: " + "|".join(compact_parts[:4])
        if len(compact_parts) > 4:
            subtype_label += "|..."
    else:
        subtype_key = "IDENTICAL_ZH"
        subtype_label = "identical Chinese strings"

    contains_other = False
    if good_diff_span and bad_diff_span:
        contains_other = good_diff_span in bad_diff_span or bad_diff_span in good_diff_span

    involved = sorted({word for word in FUNCTION_WORDS if word in good_diff_span or word in bad_diff_span}, key=FUNCTION_WORDS.index)

    py_diff = diff_analysis_simple(good_py, bad_py)

    return {
        "common_prefix": common_prefix,
        "good_diff_span": good_diff_span,
        "bad_diff_span": bad_diff_span,
        "common_suffix": common_suffix,
        "diff_opcodes": json.dumps(opcodes, ensure_ascii=False),
        "n_diff_ops": len(diff_ops),
        "edit_type": edit_type,
        "subtype_key": subtype_key,
        "subtype_label": subtype_label,
        "good_span_len_chars": len(good_diff_span),
        "bad_span_len_chars": len(bad_diff_span),
        "diff_len_delta": len(bad_diff_span) - len(good_diff_span),
        "diff_position_ratio": start_index / max(len(good), 1),
        "span_contains_other": int(contains_other),
        "function_words_involved": ";".join(involved),
        "pinyin_edit_type": py_diff["edit_type"],
        "pinyin_good_diff_span": py_diff["good_diff_span"],
        "pinyin_bad_diff_span": py_diff["bad_diff_span"],
        "diacritic_identical": int(good_py == bad_py),
    }


def diff_analysis_simple(good: str, bad: str) -> dict[str, str]:
    matcher = difflib.SequenceMatcher(a=good, b=bad, autojunk=False)
    diff_ops = [op for op in matcher.get_opcodes() if op[0] != "equal"]
    if not diff_ops:
        edit_type = "identical"
    elif len(diff_ops) > 1:
        edit_type = "multiple_edits"
    else:
        tag = diff_ops[0][0]
        edit_type = {
            "replace": "replace",
            "insert": "insert_in_bad",
            "delete": "delete_in_bad",
        }.get(tag, tag)
    changed_pairs = [(good[i1:i2], bad[j1:j2]) for _tag, i1, i2, j1, j2 in diff_ops]
    return {
        "edit_type": edit_type,
        "good_diff_span": "|".join(pair[0] for pair in changed_pairs),
        "bad_diff_span": "|".join(pair[1] for pair in changed_pairs),
    }


def validate_inputs(items: list[dict[str, Any]], score_rows: list[dict[str, str]], expected_items: int) -> dict[str, Any]:
    required_item_fields = {
        "id",
        "phenomenon",
        "good_sentence_zh",
        "bad_sentence_zh",
        "good_sentence_diacritic",
        "bad_sentence_diacritic",
    }
    missing_item_fields = sorted(required_item_fields - set(items[0]))
    if missing_item_fields:
        raise ValueError(f"Dataset is missing required fields: {missing_item_fields}")
    if len(items) != expected_items:
        raise ValueError(f"Expected {expected_items} items; loaded {len(items)}")
    item_ids = {item["id"] for item in items}
    if len(item_ids) != len(items):
        raise ValueError("Dataset item ids are not unique")

    required_score_fields = {"id", "model_run", "correct", "margin"}
    missing_score_fields = sorted(required_score_fields - set(score_rows[0]))
    if missing_score_fields:
        raise ValueError(f"Scores CSV is missing required fields: {missing_score_fields}")
    model_names = {row["model_run"] for row in score_rows}
    if model_names != EXPECTED_MODELS:
        raise ValueError(f"Expected models {sorted(EXPECTED_MODELS)}; got {sorted(model_names)}")
    score_ids = {row["id"] for row in score_rows}
    missing_scores = sorted(item_ids - score_ids)
    extra_scores = sorted(score_ids - item_ids)
    if missing_scores or extra_scores:
        raise ValueError(
            f"Dataset-score id mismatch: missing_scores={len(missing_scores)}, extra_scores={len(extra_scores)}"
        )

    expected_rows = len(items) * len(EXPECTED_MODELS)
    if len(score_rows) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} scored rows for one scoring mode; got {len(score_rows)}"
        )

    counts = Counter((row["id"], row["model_run"]) for row in score_rows)
    duplicates = [key for key, count in counts.items() if count != 1]
    if duplicates:
        raise ValueError(f"Expected exactly one score row per item/model; bad pairs={duplicates[:5]}")

    return {
        "n_items": len(items),
        "n_score_rows": len(score_rows),
        "models": sorted(model_names),
        "n_phenomena": len({item["phenomenon"] for item in items}),
    }


def enrich_items(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    enriched = {}
    for item in items:
        diff = diff_analysis(
            item["good_sentence_zh"],
            item["bad_sentence_zh"],
            item["good_sentence_diacritic"],
            item["bad_sentence_diacritic"],
        )
        row = {**item, **diff}
        row["phenomenon_subtype"] = f"{row['phenomenon']}::{row['subtype_key']}"
        enriched[row["id"]] = row
    return enriched


def normalize_score_rows(score_rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, Any]]:
    output = {}
    for row in score_rows:
        normalized = dict(row)
        normalized["correct"] = int(row.get("correct") or 0)
        normalized["tie"] = int(row.get("tie") or 0)
        normalized["non_finite"] = int(row.get("non_finite") or 0)
        normalized["margin"] = safe_float(row.get("margin"))
        normalized["good_mean_logprob"] = safe_float(
            row.get("good_mean_logprob", row.get("score_good"))
        )
        normalized["bad_mean_logprob"] = safe_float(
            row.get("bad_mean_logprob", row.get("score_bad"))
        )
        output[(row["id"], row["model_run"])] = normalized
    return output


def rule_to_text(rule: Any) -> str:
    if isinstance(rule, list):
        return " | ".join(str(part) for part in rule)
    return str(rule)


def rule_diff_summary(good_rule: Any, bad_rule: Any) -> str:
    good_parts = [str(part) for part in good_rule] if isinstance(good_rule, list) else [str(good_rule)]
    bad_parts = [str(part) for part in bad_rule] if isinstance(bad_rule, list) else [str(bad_rule)]
    matcher = difflib.SequenceMatcher(a=good_parts, b=bad_parts, autojunk=False)
    chunks = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        left = " + ".join(good_parts[i1:i2])
        right = " + ".join(bad_parts[j1:j2])
        if tag == "replace":
            chunks.append(f"{left} -> {right}")
        elif tag == "insert":
            chunks.append(f"INSERT_BAD:{right}")
        elif tag == "delete":
            chunks.append(f"DELETE_BAD:{left}")
        else:
            chunks.append(f"{tag}:{left}->{right}")
    return " | ".join(chunks) if chunks else "IDENTICAL_RULES"


def load_official_paradigms(template_dir: Path) -> dict[str, dict[str, Any]]:
    paradigms: dict[str, dict[str, Any]] = {}
    duplicate_uids: defaultdict[str, list[str]] = defaultdict(list)
    for path in sorted(template_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        uid = str(data.get("uid", "")).strip()
        if not uid:
            continue
        good_rule = data.get("good_rule", [])
        bad_rule = data.get("bad_rule", [])
        record = {
            "uid": uid,
            "phenomenon": str(data.get("phenomenon", "")).strip(),
            "strict_MP": data.get("strict_MP", ""),
            "template_filename": path.name,
            "good_rule_text": rule_to_text(good_rule),
            "bad_rule_text": rule_to_text(bad_rule),
            "rule_diff_summary": rule_diff_summary(good_rule, bad_rule),
        }
        if uid in paradigms:
            duplicate_uids[uid].append(path.name)
            if path.name < paradigms[uid]["template_filename"]:
                paradigms[uid] = record
        else:
            paradigms[uid] = record
    if duplicate_uids:
        print(f"warning: duplicate template UIDs found: {len(duplicate_uids)}; using lexicographically first filename")
    return paradigms


def load_official_template_records(template_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(template_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        uid = str(data.get("uid", "")).strip()
        if not uid:
            continue
        good_rule = data.get("good_rule", [])
        bad_rule = data.get("bad_rule", [])
        rows.append(
            {
                "uid": uid,
                "phenomenon": str(data.get("phenomenon", "")).strip(),
                "strict_MP": data.get("strict_MP", ""),
                "template_filename": path.name,
                "good_rule_text": rule_to_text(good_rule),
                "bad_rule_text": rule_to_text(bad_rule),
                "rule_diff_summary": rule_diff_summary(good_rule, bad_rule),
            }
        )
    return rows


def attach_paradigms(
    enriched_items: dict[str, dict[str, Any]],
    paradigms: dict[str, dict[str, Any]],
) -> None:
    for item in enriched_items.values():
        uid = str(item.get("subtype_if_any", "")).strip()
        paradigm = paradigms.get(uid, {})
        item["official_uid"] = uid
        item["template_filename"] = paradigm.get("template_filename", "")
        item["official_strict_MP"] = paradigm.get("strict_MP", "")
        item["official_good_rule_text"] = paradigm.get("good_rule_text", "")
        item["official_bad_rule_text"] = paradigm.get("bad_rule_text", "")
        item["official_rule_diff_summary"] = paradigm.get("rule_diff_summary", "")
        item["observed_surface_diff_key"] = item["subtype_key"]
        item["observed_surface_diff_label"] = item["subtype_label"]
        item["observed_diacritic_diff_key"] = make_diacritic_diff_key(item)


def make_diacritic_diff_key(item: dict[str, Any]) -> str:
    if item["diacritic_identical"]:
        return "IDENTICAL_DIACRITIC"
    edit_type = item.get("pinyin_edit_type", "")
    good = item.get("pinyin_good_diff_span", "")
    bad = item.get("pinyin_bad_diff_span", "")
    if edit_type == "replace":
        return f"{good}->{bad}"
    if edit_type == "insert_in_bad":
        return f"INSERT_BAD:{bad}"
    if edit_type == "delete_in_bad":
        return f"DELETE_BAD:{good}"
    if edit_type == "multiple_edits":
        return f"MULTI:{good}->{bad}"
    return edit_type or ""


def aggregate_item_group(
    group: list[dict[str, Any]],
    scores: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    ids = [item["id"] for item in group]
    ch_rows = [scores[(item_id, "chinese_4epoch")] for item_id in ids]
    di_rows = [scores[(item_id, "diacritic_matched_token_4epoch")] for item_id in ids]
    ch_acc = accuracy(ch_rows)
    di_acc = accuracy(di_rows)
    gap = float(ch_acc) - float(di_acc) if ch_acc != "" and di_acc != "" else ""
    ch_margins = [row["margin"] for row in ch_rows]
    di_margins = [row["margin"] for row in di_rows]
    collapsed = sum(int(item["diacritic_identical"]) for item in group)
    ties = sum(row["tie"] for row in di_rows)
    n = len(group)
    return {
        "n_items": n,
        "chinese_accuracy": ch_acc,
        "diacritic_accuracy": di_acc,
        "gap": gap,
        "chinese_minus_baseline": float(ch_acc) - BASELINE if ch_acc != "" else "",
        "diacritic_minus_baseline": float(di_acc) - BASELINE if di_acc != "" else "",
        "chinese_mean_margin": mean(ch_margins),
        "diacritic_mean_margin": mean(di_margins),
        "diacritic_exact_identical_count": collapsed,
        "diacritic_exact_identical_rate": collapsed / n if n else "",
        "tie_count": ties,
        "tie_rate": ties / n if n else "",
    }


def official_paradigm_inventory(
    enriched_items: dict[str, dict[str, Any]],
    scores: dict[tuple[str, str], dict[str, Any]],
    paradigms: dict[str, dict[str, Any]],
    template_records: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in enriched_items.values():
        groups[item["official_uid"]].append(item)

    rows: list[dict[str, Any]] = []
    emitted_uids = set()
    records = template_records if template_records is not None else [paradigms[uid] for uid in sorted(paradigms)]
    for template in records:
        uid = template["uid"]
        emitted_uids.add(uid)
        group = groups.get(uid, [])
        metrics = aggregate_item_group(group, scores) if group else {}
        rows.append(
            {
                "phenomenon": template.get("phenomenon") or (group[0]["phenomenon"] if group else ""),
                "uid": uid,
                "template_filename": template.get("template_filename", ""),
                "strict_MP": template.get("strict_MP", ""),
                "good_rule_text": template.get("good_rule_text", ""),
                "bad_rule_text": template.get("bad_rule_text", ""),
                "rule_diff_summary": template.get("rule_diff_summary", ""),
                "n_items_in_dataset": len(group),
                "chinese_accuracy": metrics.get("chinese_accuracy", ""),
                "diacritic_accuracy": metrics.get("diacritic_accuracy", ""),
                "gap": metrics.get("gap", ""),
                "chinese_minus_baseline": metrics.get("chinese_minus_baseline", ""),
                "diacritic_minus_baseline": metrics.get("diacritic_minus_baseline", ""),
                "diacritic_identical_count": metrics.get("diacritic_exact_identical_count", ""),
                "diacritic_identical_rate": metrics.get("diacritic_exact_identical_rate", ""),
                "tie_count": metrics.get("tie_count", ""),
                "tie_rate": metrics.get("tie_rate", ""),
            }
        )
    for uid in sorted(set(groups) - emitted_uids):
        group = groups[uid]
        metrics = aggregate_item_group(group, scores)
        rows.append(
            {
                "phenomenon": group[0]["phenomenon"],
                "uid": uid,
                "template_filename": "",
                "strict_MP": "",
                "good_rule_text": "",
                "bad_rule_text": "",
                "rule_diff_summary": "",
                "n_items_in_dataset": len(group),
                "chinese_accuracy": metrics.get("chinese_accuracy", ""),
                "diacritic_accuracy": metrics.get("diacritic_accuracy", ""),
                "gap": metrics.get("gap", ""),
                "chinese_minus_baseline": metrics.get("chinese_minus_baseline", ""),
                "diacritic_minus_baseline": metrics.get("diacritic_minus_baseline", ""),
                "diacritic_identical_count": metrics.get("diacritic_exact_identical_count", ""),
                "diacritic_identical_rate": metrics.get("diacritic_exact_identical_rate", ""),
                "tie_count": metrics.get("tie_count", ""),
                "tie_rate": metrics.get("tie_rate", ""),
            }
        )
    return rows


def aggregate_official_paradigms(
    enriched_items: dict[str, dict[str, Any]],
    scores: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in enriched_items.values():
        groups[item["official_uid"]].append(item)
    rows = []
    for uid, group in sorted(groups.items()):
        first = group[0]
        metrics = aggregate_item_group(group, scores)
        rows.append(
            {
                "phenomenon": first["phenomenon"],
                "uid": uid,
                "template_filename": first.get("template_filename", ""),
                "strict_MP": first.get("official_strict_MP", ""),
                "good_rule_text": first.get("official_good_rule_text", ""),
                "bad_rule_text": first.get("official_bad_rule_text", ""),
                "rule_diff_summary": first.get("official_rule_diff_summary", ""),
                **metrics,
            }
        )
    return rows


def aggregate_paradigm_surface_diffs(
    enriched_items: dict[str, dict[str, Any]],
    scores: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in enriched_items.values():
        groups[(item["official_uid"], item["observed_surface_diff_key"])].append(item)
    rows = []
    for (_uid, _surface), group in sorted(groups.items()):
        first = group[0]
        metrics = aggregate_item_group(group, scores)
        rows.append(
            {
                "phenomenon": first["phenomenon"],
                "uid": first["official_uid"],
                "template_filename": first.get("template_filename", ""),
                "strict_MP": first.get("official_strict_MP", ""),
                "observed_surface_diff_key": first["observed_surface_diff_key"],
                "observed_surface_diff_label": first["observed_surface_diff_label"],
                "observed_edit_type": first["edit_type"],
                "good_diff_span": first["good_diff_span"],
                "bad_diff_span": first["bad_diff_span"],
                "observed_diacritic_diff_key": first["observed_diacritic_diff_key"],
                **metrics,
            }
        )
    return rows


def bucket_labels(ch_acc: float | str, di_acc: float | str, gap: float | str, collapse_rate: float) -> str:
    if ch_acc == "" or di_acc == "" or gap == "":
        return ""
    labels = []
    ch = float(ch_acc)
    di = float(di_acc)
    gp = float(gap)
    if ch >= 0.60 and di >= 0.50 and gp >= 0.10:
        labels.append("strong_chinese_above_baseline")
    if ch >= 0.60 and abs(di - 0.50) <= 0.05 and gp >= 0.10:
        labels.append("chinese_above_diacritic_near_chance")
    if di - ch >= 0.05:
        labels.append("diacritic_better")
    if ch >= 0.60 and di >= 0.60 and abs(gp) < 0.05:
        labels.append("both_above_chance_close")
    if abs(ch - 0.50) <= 0.05 and abs(di - 0.50) <= 0.05:
        labels.append("both_near_chance")
    if ch < 0.50 and di < 0.50:
        labels.append("both_below_chance_or_unstable")
    if collapse_rate >= 0.10:
        labels.append("collapse_affected")
    return ";".join(labels)


def aggregate_subtypes(
    enriched_items: dict[str, dict[str, Any]],
    scores: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in enriched_items.values():
        groups[item["phenomenon_subtype"]].append(item)

    rows = []
    for phenomenon_subtype, group in sorted(groups.items()):
        first = group[0]
        ids = [item["id"] for item in group]
        ch_rows = [scores[(item_id, "chinese_4epoch")] for item_id in ids]
        di_rows = [scores[(item_id, "diacritic_matched_token_4epoch")] for item_id in ids]
        ch_acc = accuracy(ch_rows)
        di_acc = accuracy(di_rows)
        gap = float(ch_acc) - float(di_acc) if ch_acc != "" and di_acc != "" else ""
        ch_margins = [row["margin"] for row in ch_rows]
        di_margins = [row["margin"] for row in di_rows]
        ch_mean = mean(ch_margins)
        di_mean = mean(di_margins)
        collapse_count = sum(int(item["diacritic_identical"]) for item in group)
        collapse_rate = collapse_count / len(group)
        function_words = sorted(
            {
                word
                for item in group
                for word in str(item.get("function_words_involved", "")).split(";")
                if word
            },
            key=lambda value: FUNCTION_WORDS.index(value) if value in FUNCTION_WORDS else 999,
        )
        row = {
            "phenomenon": first["phenomenon"],
            "subtype_key": first["subtype_key"],
            "subtype_label": first["subtype_label"],
            "phenomenon_subtype": phenomenon_subtype,
            "n_items": len(group),
            "edit_type": first["edit_type"],
            "good_span": first["good_diff_span"],
            "bad_span": first["bad_diff_span"],
            "good_span_len_chars": first["good_span_len_chars"],
            "bad_span_len_chars": first["bad_span_len_chars"],
            "diff_len_delta": first["diff_len_delta"],
            "mean_diff_position_ratio": mean([item["diff_position_ratio"] for item in group]),
            "span_contains_other_count": sum(int(item["span_contains_other"]) for item in group),
            "span_contains_other_rate": sum(int(item["span_contains_other"]) for item in group) / len(group),
            "function_words_involved": ";".join(function_words),
            "diacritic_identical_count": collapse_count,
            "diacritic_identical_rate": collapse_rate,
            "chinese_accuracy": ch_acc,
            "diacritic_accuracy": di_acc,
            "gap_chinese_minus_diacritic": gap,
            "chinese_mean_margin": ch_mean,
            "diacritic_mean_margin": di_mean,
            "margin_gap": float(ch_mean) - float(di_mean) if ch_mean != "" and di_mean != "" else "",
            "chinese_median_margin": median_or_blank(ch_margins),
            "diacritic_median_margin": median_or_blank(di_margins),
            "chinese_n_ties": sum(row["tie"] for row in ch_rows),
            "diacritic_n_ties": sum(row["tie"] for row in di_rows),
            "n_collapsed_diacritic": collapse_count,
            "chinese_minus_baseline": float(ch_acc) - BASELINE if ch_acc != "" else "",
            "diacritic_minus_baseline": float(di_acc) - BASELINE if di_acc != "" else "",
            "uids": json.dumps(sorted({item.get("subtype_if_any", "") for item in group}), ensure_ascii=False),
            "example_item_ids": json.dumps(ids[:5], ensure_ascii=False),
        }
        row["interpretation_bucket"] = bucket_labels(ch_acc, di_acc, gap, collapse_rate)
        rows.append(row)
    return rows


def aggregate_phenomena(
    enriched_items: dict[str, dict[str, Any]],
    scores: dict[tuple[str, str], dict[str, Any]],
    subtype_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    subtype_by_ph: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in subtype_rows:
        subtype_by_ph[row["phenomenon"]].append(row)
    item_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in enriched_items.values():
        item_groups[item["phenomenon"]].append(item)

    rows = []
    for phenomenon, group in sorted(item_groups.items()):
        metrics = aggregate_item_group(group, scores)
        ch_acc = metrics["chinese_accuracy"]
        di_acc = metrics["diacritic_accuracy"]
        gap = metrics["gap"]
        collapsed_count = metrics["diacritic_exact_identical_count"]
        tie_count = metrics["tie_count"]
        top_by_n = sorted(subtype_by_ph[phenomenon], key=lambda row: int(row["n_items"]), reverse=True)[:5]
        eligible = [row for row in subtype_by_ph[phenomenon] if int(row["n_items"]) >= 10]
        top_by_gap = sorted(
            eligible,
            key=lambda row: abs(float(row["gap_chinese_minus_diacritic"])),
            reverse=True,
        )[:5]
        collapse_rate = metrics["diacritic_exact_identical_rate"]
        rows.append(
            {
                "phenomenon": phenomenon,
                "n_items": metrics["n_items"],
                "chinese_accuracy": ch_acc,
                "diacritic_accuracy": di_acc,
                "gap": gap,
                "chinese_minus_baseline": metrics["chinese_minus_baseline"],
                "diacritic_minus_baseline": metrics["diacritic_minus_baseline"],
                "chinese_mean_margin": metrics["chinese_mean_margin"],
                "diacritic_mean_margin": metrics["diacritic_mean_margin"],
                "diacritic_exact_identical_count": collapsed_count,
                "diacritic_exact_identical_rate": collapse_rate,
                "collapsed_count": collapsed_count,
                "collapsed_rate": collapse_rate,
                "tie_count": tie_count,
                "tie_rate": metrics["tie_rate"],
                "interpretation_bucket": bucket_labels(ch_acc, di_acc, gap, collapse_rate),
                "top_subtypes_by_n": json.dumps(
                    [(row["subtype_label"], row["n_items"]) for row in top_by_n],
                    ensure_ascii=False,
                ),
                "top_subtypes_by_gap": json.dumps(
                    [
                        (
                            row["subtype_label"],
                            row["n_items"],
                            row["gap_chinese_minus_diacritic"],
                        )
                        for row in top_by_gap
                    ],
                    ensure_ascii=False,
                ),
            }
        )
    return rows


def anaphor_analysis(
    enriched_items: dict[str, dict[str, Any]],
    scores: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in enriched_items.values():
        if item["phenomenon"] == "anaphor":
            groups[item["phenomenon_subtype"]].append(item)

    rows = []
    for _key, group in sorted(groups.items(), key=lambda pair: (-len(pair[1]), pair[0])):
        first = group[0]
        ids = [item["id"] for item in group]
        noncollapsed = [item for item in group if not item["diacritic_identical"]]
        ch_all = [scores[(item_id, "chinese_4epoch")] for item_id in ids]
        di_all = [scores[(item_id, "diacritic_matched_token_4epoch")] for item_id in ids]
        ch_nc = [scores[(item["id"], "chinese_4epoch")] for item in noncollapsed]
        di_nc = [scores[(item["id"], "diacritic_matched_token_4epoch")] for item in noncollapsed]
        ch_all_acc = accuracy(ch_all)
        di_all_acc = accuracy(di_all)
        ch_nc_acc = accuracy(ch_nc)
        di_nc_acc = accuracy(di_nc)
        collapsed_count = len(group) - len(noncollapsed)
        examples = [
            {
                "id": item["id"],
                "good": item["good_sentence_zh"],
                "bad": item["bad_sentence_zh"],
                "collapsed": bool(item["diacritic_identical"]),
            }
            for item in group[:5]
        ]
        rows.append(
            {
                "phenomenon": "anaphor",
                "subtype_key": first["subtype_key"],
                "subtype_label": first["subtype_label"],
                "n_items": len(group),
                "collapsed_count": collapsed_count,
                "collapsed_rate": collapsed_count / len(group),
                "chinese_accuracy_all": ch_all_acc,
                "diacritic_accuracy_all": di_all_acc,
                "gap_all": float(ch_all_acc) - float(di_all_acc) if ch_all_acc != "" and di_all_acc != "" else "",
                "noncollapsed_n_items": len(noncollapsed),
                "chinese_accuracy_noncollapsed": ch_nc_acc,
                "diacritic_accuracy_noncollapsed": di_nc_acc,
                "noncollapsed_gap": float(ch_nc_acc) - float(di_nc_acc) if ch_nc_acc != "" and di_nc_acc != "" else "",
                "examples": json.dumps(examples, ensure_ascii=False),
            }
        )
    return rows


def example_rows_for_subtypes(
    subtype_rows: list[dict[str, Any]],
    enriched_items: dict[str, dict[str, Any]],
    scores: dict[tuple[str, str], dict[str, Any]],
    selected_subtypes: list[str],
    examples_per_subtype: int,
) -> list[dict[str, Any]]:
    subtype_lookup = {row["phenomenon_subtype"]: row for row in subtype_rows}
    items_by_subtype: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in enriched_items.values():
        items_by_subtype[item["phenomenon_subtype"]].append(item)

    rows = []
    for phenomenon_subtype in selected_subtypes:
        subtype = subtype_lookup[phenomenon_subtype]
        for item in items_by_subtype[phenomenon_subtype][:examples_per_subtype]:
            ch = scores[(item["id"], "chinese_4epoch")]
            di = scores[(item["id"], "diacritic_matched_token_4epoch")]
            rows.append(
                {
                    "item_id": item["id"],
                    "phenomenon": item["phenomenon"],
                    "subtype_key": subtype["subtype_key"],
                    "subtype_label": subtype["subtype_label"],
                    "good_sentence_zh": item["good_sentence_zh"],
                    "bad_sentence_zh": item["bad_sentence_zh"],
                    "good_sentence_diacritic": item["good_sentence_diacritic"],
                    "bad_sentence_diacritic": item["bad_sentence_diacritic"],
                    "model_correct_chinese": ch["correct"],
                    "model_correct_diacritic": di["correct"],
                    "chinese_margin": ch["margin"],
                    "diacritic_margin": di["margin"],
                    "common_prefix": item["common_prefix"],
                    "good_diff_span": item["good_diff_span"],
                    "bad_diff_span": item["bad_diff_span"],
                    "common_suffix": item["common_suffix"],
                }
            )
    return rows


def paradigm_example_row(
    item: dict[str, Any],
    scores: dict[tuple[str, str], dict[str, Any]],
    example_group: str,
) -> dict[str, Any]:
    ch = scores[(item["id"], "chinese_4epoch")]
    di = scores[(item["id"], "diacritic_matched_token_4epoch")]
    return {
        "example_group": example_group,
        "item_id": item["id"],
        "phenomenon": item["phenomenon"],
        "uid": item["official_uid"],
        "template_filename": item.get("template_filename", ""),
        "strict_MP": item.get("official_strict_MP", ""),
        "good_rule_text": item.get("official_good_rule_text", ""),
        "bad_rule_text": item.get("official_bad_rule_text", ""),
        "rule_diff_summary": item.get("official_rule_diff_summary", ""),
        "observed_surface_diff_label": item["observed_surface_diff_label"],
        "observed_diacritic_diff_key": item["observed_diacritic_diff_key"],
        "good_sentence_zh": item["good_sentence_zh"],
        "bad_sentence_zh": item["bad_sentence_zh"],
        "good_sentence_diacritic": item["good_sentence_diacritic"],
        "bad_sentence_diacritic": item["bad_sentence_diacritic"],
        "model_correct_chinese": ch["correct"],
        "model_correct_diacritic": di["correct"],
        "chinese_margin": ch["margin"],
        "diacritic_margin": di["margin"],
        "common_prefix": item["common_prefix"],
        "good_diff_span": item["good_diff_span"],
        "bad_diff_span": item["bad_diff_span"],
        "common_suffix": item["common_suffix"],
    }


def example_rows_for_paradigms(
    enriched_items: dict[str, dict[str, Any]],
    scores: dict[tuple[str, str], dict[str, Any]],
    selected_uids: list[str],
    examples_per_uid: int,
    example_group: str,
) -> list[dict[str, Any]]:
    items_by_uid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in enriched_items.values():
        items_by_uid[item["official_uid"]].append(item)

    rows = []
    for uid in selected_uids:
        for item in items_by_uid.get(uid, [])[:examples_per_uid]:
            rows.append(paradigm_example_row(item, scores, example_group))
    return rows


def example_rows_for_surface_diffs(
    enriched_items: dict[str, dict[str, Any]],
    scores: dict[tuple[str, str], dict[str, Any]],
    selected_keys: list[tuple[str, str]],
    examples_per_key: int,
    example_group: str,
) -> list[dict[str, Any]]:
    items_by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in enriched_items.values():
        items_by_key[(item["official_uid"], item["observed_surface_diff_key"])].append(item)

    rows = []
    for key in selected_keys:
        for item in items_by_key.get(key, [])[:examples_per_key]:
            rows.append(paradigm_example_row(item, scores, example_group))
    return rows


def special_pattern_examples(
    enriched_items: dict[str, dict[str, Any]],
    scores: dict[tuple[str, str], dict[str, Any]],
    limit_per_pattern: int = 30,
) -> list[dict[str, Any]]:
    pattern_items: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in sorted(enriched_items.values(), key=lambda row: row["id"]):
        text = "".join(
            [
                item["phenomenon"],
                item["official_uid"],
                item.get("official_rule_diff_summary", ""),
                item["observed_surface_diff_label"],
                item["good_sentence_zh"],
                item["bad_sentence_zh"],
            ]
        )
        if item["phenomenon"] == "anaphor" and item["diacritic_identical"]:
            pattern_items["anaphor_collapse"].append(item)
        if "question" in item["phenomenon"].lower() or "吗" in text or "呢" in text:
            pattern_items["question_particles"].append(item)
        if item["phenomenon"] in {"BA", "passive"} or "把" in text or "被" in text:
            pattern_items["ba_bei_alternations"].append(item)
        if any(marker in text for marker in ["的", "地", "得"]):
            pattern_items["de_di_de_particles"].append(item)

    rows = []
    for pattern in [
        "anaphor_collapse",
        "question_particles",
        "ba_bei_alternations",
        "de_di_de_particles",
    ]:
        for item in pattern_items.get(pattern, [])[:limit_per_pattern]:
            rows.append(paradigm_example_row(item, scores, pattern))
    return rows


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]], limit: int) -> list[str]:
    output = []
    header = "| " + " | ".join(label for label, _field in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    output.extend([header, sep])
    for row in rows[:limit]:
        cells = []
        for _label, field in columns:
            value = row.get(field, "")
            if isinstance(value, float):
                cells.append(f"{value:.4f}")
            else:
                cells.append(str(value).replace("|", "\\|").replace("\n", " "))
        output.append("| " + " | ".join(cells) + " |")
    return output


def write_report(
    path: Path,
    validation: dict[str, Any],
    subtype_all: list[dict[str, Any]],
    subtype_main: list[dict[str, Any]],
    phenomenon_rows: list[dict[str, Any]],
    anaphor_rows: list[dict[str, Any]],
) -> None:
    top_by_n = sorted(subtype_all, key=lambda row: int(row["n_items"]), reverse=True)
    gap_eligible = [row for row in subtype_main if int(row["n_items"]) >= 20]
    top_chinese = sorted(gap_eligible, key=lambda row: float(row["gap_chinese_minus_diacritic"]), reverse=True)
    top_diacritic = sorted(gap_eligible, key=lambda row: float(row["gap_chinese_minus_diacritic"]))
    collapse = sorted(
        [row for row in subtype_all if float(row["diacritic_identical_rate"]) > 0],
        key=lambda row: (float(row["diacritic_identical_rate"]), int(row["n_items"])),
        reverse=True,
    )
    chinese_near = [
        row
        for row in subtype_main
        if "chinese_above_diacritic_near_chance" in row["interpretation_bucket"]
    ]
    both_solve = [row for row in subtype_main if "both_above_chance_close" in row["interpretation_bucket"]]
    both_fail = [row for row in subtype_main if "both_below_chance_or_unstable" in row["interpretation_bucket"]]
    di_better = [row for row in subtype_main if "diacritic_better" in row["interpretation_bucket"]]
    anaphor_total = next((row for row in phenomenon_rows if row["phenomenon"] == "anaphor"), {})

    columns = [
        ("phenomenon", "phenomenon"),
        ("subtype", "subtype_label"),
        ("n", "n_items"),
        ("ch_acc", "chinese_accuracy"),
        ("di_acc", "diacritic_accuracy"),
        ("gap", "gap_chinese_minus_diacritic"),
        ("collapse", "diacritic_identical_rate"),
    ]
    lines = [
        "# Eval 4 ZhoBLiMP Subtype Analysis",
        "",
        "## Loaded Data",
        "",
        f"- items: {validation['n_items']}",
        f"- score rows: {validation['n_score_rows']}",
        f"- models: {', '.join(validation['models'])}",
        f"- unique phenomena: {validation['n_phenomena']}",
        f"- unique subtypes: {len(subtype_all)}",
        f"- main-table subtypes with n >= {MAIN_MIN_ITEMS}: {len(subtype_main)}",
        "",
        "## Top 20 Subtypes By n_items",
        "",
    ]
    lines.extend(markdown_table(top_by_n, columns, 20))
    lines.extend(["", "## Top 20 Chinese-Advantage Subtypes", ""])
    lines.extend(markdown_table(top_chinese, columns, 20))
    lines.extend(["", "## Top 20 Diacritic-Advantage Subtypes", ""])
    lines.extend(markdown_table(top_diacritic, columns, 20))
    lines.extend(["", "## Highest Diacritic Collapse Rate", ""])
    lines.extend(markdown_table(collapse, columns, 20))
    lines.extend(["", "## Anaphor Collapse Summary", ""])
    if anaphor_total:
        lines.extend(
            [
                f"- n_items: {anaphor_total['n_items']}",
                f"- collapsed_count: {anaphor_total['collapsed_count']}",
                f"- collapsed_rate: {float(anaphor_total['collapsed_rate']):.4f}",
                f"- tie_count: {anaphor_total['tie_count']}",
                f"- tie_rate: {float(anaphor_total['tie_rate']):.4f}",
            ]
        )
    lines.extend(["", "Top anaphor subtypes:"])
    lines.extend(
        markdown_table(
            anaphor_rows,
            [
                ("subtype", "subtype_label"),
                ("n", "n_items"),
                ("collapse_rate", "collapsed_rate"),
                ("ch_all", "chinese_accuracy_all"),
                ("di_all", "diacritic_accuracy_all"),
                ("ch_noncollapsed", "chinese_accuracy_noncollapsed"),
                ("di_noncollapsed", "diacritic_accuracy_noncollapsed"),
            ],
            20,
        )
    )
    lines.extend(["", "## Baseline-Aware Interpretation", ""])
    lines.append(f"- Chinese above baseline and Diacritic near chance: {len(chinese_near)} subtypes")
    lines.append(f"- Both models solve / close: {len(both_solve)} subtypes")
    lines.append(f"- Both models below chance or unstable: {len(both_fail)} subtypes")
    lines.append(f"- Diacritic-favoring: {len(di_better)} subtypes")
    lines.extend(["", "### Chinese Above, Diacritic Near Chance", ""])
    lines.extend(markdown_table(chinese_near, columns, 20))
    lines.extend(["", "### Both Models Solve", ""])
    lines.extend(markdown_table(both_solve, columns, 20))
    lines.extend(["", "### Both Models Fail", ""])
    lines.extend(markdown_table(both_fail, columns, 20))
    lines.extend(["", "### Diacritic-Favoring", ""])
    lines.extend(markdown_table(di_better, columns, 20))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_official_paradigm_report(
    path: Path,
    validation: dict[str, Any],
    paradigms: dict[str, dict[str, Any]],
    inventory_rows: list[dict[str, Any]],
    paradigm_rows: list[dict[str, Any]],
    surface_rows: list[dict[str, Any]],
    phenomenon_rows: list[dict[str, Any]],
) -> None:
    scored_inventory = [row for row in inventory_rows if int(row.get("n_items_in_dataset") or 0) > 0]
    top_paradigms_by_n = sorted(paradigm_rows, key=lambda row: int(row["n_items"]), reverse=True)
    eligible = [row for row in paradigm_rows if int(row["n_items"]) >= 20]
    top_chinese = sorted(eligible, key=lambda row: float(row["gap"]), reverse=True)
    top_diacritic = sorted(eligible, key=lambda row: float(row["gap"]))
    top_collapse = sorted(
        [row for row in paradigm_rows if float(row["diacritic_exact_identical_rate"]) > 0],
        key=lambda row: (float(row["diacritic_exact_identical_rate"]), int(row["n_items"])),
        reverse=True,
    )
    top_surface_by_n = sorted(surface_rows, key=lambda row: int(row["n_items"]), reverse=True)
    anaphor_collapse = [
        row
        for row in paradigm_rows
        if row["phenomenon"] == "anaphor" and float(row["diacritic_exact_identical_rate"]) > 0
    ]
    question = [
        row
        for row in paradigm_rows
        if "question" in row["phenomenon"].lower()
        or "吗" in row["rule_diff_summary"]
        or "呢" in row["rule_diff_summary"]
    ]
    ba_bei = [
        row
        for row in paradigm_rows
        if row["phenomenon"] in {"BA", "passive"}
        or "把" in row["rule_diff_summary"]
        or "被" in row["rule_diff_summary"]
    ]
    de_particles = [
        row
        for row in paradigm_rows
        if any(marker in row["rule_diff_summary"] for marker in ["的", "地", "得"])
    ]

    paradigm_cols = [
        ("phenomenon", "phenomenon"),
        ("uid", "uid"),
        ("n", "n_items"),
        ("rule_diff", "rule_diff_summary"),
        ("ch_acc", "chinese_accuracy"),
        ("di_acc", "diacritic_accuracy"),
        ("gap", "gap"),
        ("ch_margin", "chinese_mean_margin"),
        ("di_margin", "diacritic_mean_margin"),
        ("collapse", "diacritic_exact_identical_rate"),
        ("tie", "tie_rate"),
    ]
    surface_cols = [
        ("phenomenon", "phenomenon"),
        ("uid", "uid"),
        ("surface_diff", "observed_surface_diff_label"),
        ("n", "n_items"),
        ("ch_acc", "chinese_accuracy"),
        ("di_acc", "diacritic_accuracy"),
        ("gap", "gap"),
        ("ch_margin", "chinese_mean_margin"),
        ("di_margin", "diacritic_mean_margin"),
        ("collapse", "diacritic_exact_identical_rate"),
    ]
    phenomenon_cols = [
        ("phenomenon", "phenomenon"),
        ("n", "n_items"),
        ("ch_acc", "chinese_accuracy"),
        ("di_acc", "diacritic_accuracy"),
        ("gap", "gap"),
        ("ch_margin", "chinese_mean_margin"),
        ("di_margin", "diacritic_mean_margin"),
        ("collapse", "diacritic_exact_identical_rate"),
        ("tie", "tie_rate"),
    ]

    lines = [
        "# Eval 4 Official ZhoBLiMP Paradigm Analysis",
        "",
        "## Loaded Data",
        "",
        f"- dataset: {validation['dataset_path']}",
        f"- scores: {validation['scores_path']}",
        f"- items: {validation['n_items']}",
        f"- score rows: {validation['n_score_rows']}",
        f"- official template JSON files loaded: {validation.get('n_official_template_json_files', '')}",
        f"- official unique template UIDs loaded: {len(paradigms)}",
        f"- official paradigms represented in dataset: {len(scored_inventory)}",
        f"- unique phenomenon labels: {len({row['phenomenon'] for row in phenomenon_rows})}",
        f"- unique paradigm + observed surface diff rows: {len(surface_rows)}",
        "",
        "## Phenomenon Level",
        "",
    ]
    lines.extend(markdown_table(phenomenon_rows, phenomenon_cols, 50))
    lines.extend(["", "## All Official Paradigms", ""])
    lines.append(
        "This is the compact subtype report: each row is the official ZhoBLiMP UID, "
        "the intended template rule replacement, and both models' accuracy and mean margin."
    )
    lines.extend([""])
    lines.extend(markdown_table(sorted(paradigm_rows, key=lambda row: (row["phenomenon"], row["uid"])), paradigm_cols, 200))
    lines.extend(["", "## Top 20 Official Paradigms By n_items", ""])
    lines.extend(markdown_table(top_paradigms_by_n, paradigm_cols, 20))
    lines.extend(["", "## Top 20 Chinese-Advantage Official Paradigms", ""])
    lines.extend(markdown_table(top_chinese, paradigm_cols, 20))
    lines.extend(["", "## Top 20 Diacritic-Advantage Official Paradigms", ""])
    lines.extend(markdown_table(top_diacritic, paradigm_cols, 20))
    lines.extend(["", "## Top 20 Collapse-Affected Official Paradigms", ""])
    lines.extend(markdown_table(top_collapse, paradigm_cols, 20))
    lines.extend(["", "## Top 30 Paradigm + Observed Surface Diff Rows By n_items", ""])
    lines.extend(markdown_table(top_surface_by_n, surface_cols, 30))
    lines.extend(["", "## Anaphor Collapse Paradigms", ""])
    lines.extend(markdown_table(sorted(anaphor_collapse, key=lambda row: float(row["diacritic_exact_identical_rate"]), reverse=True), paradigm_cols, 30))
    lines.extend(["", "## Question Particle Paradigms", ""])
    lines.extend(markdown_table(sorted(question, key=lambda row: abs(float(row["gap"])), reverse=True), paradigm_cols, 30))
    lines.extend(["", "## BA/BEI Paradigms", ""])
    lines.extend(markdown_table(sorted(ba_bei, key=lambda row: abs(float(row["gap"])), reverse=True), paradigm_cols, 30))
    lines.extend(["", "## 的/地/得 Paradigms", ""])
    lines.extend(markdown_table(sorted(de_particles, key=lambda row: abs(float(row["gap"])), reverse=True), paradigm_cols, 30))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_console_summary(
    phenomenon_rows: list[dict[str, Any]],
    subtype_main: list[dict[str, Any]],
    paradigm_rows: list[dict[str, Any]] | None = None,
    surface_rows: list[dict[str, Any]] | None = None,
) -> None:
    print("\nBroad phenomenon table")
    for row in sorted(phenomenon_rows, key=lambda row: row["phenomenon"]):
        print(
            f"{row['phenomenon']}: n={row['n_items']} "
            f"ch={float(row['chinese_accuracy']):.4f} "
            f"di={float(row['diacritic_accuracy']):.4f} "
            f"gap={float(row['gap']):+.4f} "
            f"ch_base={float(row['chinese_minus_baseline']):+.4f} "
            f"di_base={float(row['diacritic_minus_baseline']):+.4f} "
            f"collapse={float(row['collapsed_rate']):.4f} "
            f"tie={float(row['tie_rate']):.4f} "
            f"bucket={row['interpretation_bucket']}"
        )

    print("\nTop 10 subtypes by n_items")
    for row in sorted(subtype_main, key=lambda row: int(row["n_items"]), reverse=True)[:10]:
        print(f"{row['phenomenon']}::{row['subtype_label']} n={row['n_items']}")

    print("\nTop 10 Chinese-favoring subtypes by gap")
    for row in sorted(subtype_main, key=lambda row: float(row["gap_chinese_minus_diacritic"]), reverse=True)[:10]:
        print(
            f"{row['phenomenon']}::{row['subtype_label']} "
            f"n={row['n_items']} gap={float(row['gap_chinese_minus_diacritic']):+.4f} "
            f"ch={float(row['chinese_accuracy']):.4f} di={float(row['diacritic_accuracy']):.4f}"
        )

    print("\nTop 10 Diacritic-favoring subtypes by gap")
    for row in sorted(subtype_main, key=lambda row: float(row["gap_chinese_minus_diacritic"]))[:10]:
        print(
            f"{row['phenomenon']}::{row['subtype_label']} "
            f"n={row['n_items']} gap={float(row['gap_chinese_minus_diacritic']):+.4f} "
            f"ch={float(row['chinese_accuracy']):.4f} di={float(row['diacritic_accuracy']):.4f}"
        )

    print("\nTop 10 collapse-affected subtypes")
    collapse = [row for row in subtype_main if float(row["diacritic_identical_rate"]) > 0]
    for row in sorted(collapse, key=lambda row: (float(row["diacritic_identical_rate"]), int(row["n_items"])), reverse=True)[:10]:
        print(
            f"{row['phenomenon']}::{row['subtype_label']} "
            f"n={row['n_items']} collapse_rate={float(row['diacritic_identical_rate']):.4f} "
            f"ch={float(row['chinese_accuracy']):.4f} di={float(row['diacritic_accuracy']):.4f}"
        )

    if paradigm_rows is None or surface_rows is None:
        return

    print("\nTop 10 official paradigms by n_items")
    for row in sorted(paradigm_rows, key=lambda row: int(row["n_items"]), reverse=True)[:10]:
        print(f"{row['phenomenon']}::{row['uid']} n={row['n_items']} diff={row['rule_diff_summary']}")

    print("\nTop 10 Chinese-favoring official paradigms")
    for row in sorted(paradigm_rows, key=lambda row: float(row["gap"]), reverse=True)[:10]:
        print(
            f"{row['phenomenon']}::{row['uid']} n={row['n_items']} "
            f"gap={float(row['gap']):+.4f} ch={float(row['chinese_accuracy']):.4f} "
            f"di={float(row['diacritic_accuracy']):.4f} diff={row['rule_diff_summary']}"
        )

    print("\nTop 10 Diacritic-favoring official paradigms")
    for row in sorted(paradigm_rows, key=lambda row: float(row["gap"]))[:10]:
        print(
            f"{row['phenomenon']}::{row['uid']} n={row['n_items']} "
            f"gap={float(row['gap']):+.4f} ch={float(row['chinese_accuracy']):.4f} "
            f"di={float(row['diacritic_accuracy']):.4f} diff={row['rule_diff_summary']}"
        )

    print("\nTop 10 collapse-affected official paradigms")
    collapse_paradigms = [row for row in paradigm_rows if float(row["diacritic_exact_identical_rate"]) > 0]
    for row in sorted(collapse_paradigms, key=lambda row: (float(row["diacritic_exact_identical_rate"]), int(row["n_items"])), reverse=True)[:10]:
        print(
            f"{row['phenomenon']}::{row['uid']} n={row['n_items']} "
            f"collapse_rate={float(row['diacritic_exact_identical_rate']):.4f} "
            f"tie_rate={float(row['tie_rate']):.4f}"
        )

    print("\nTop 10 paradigm + observed surface diffs by n_items")
    for row in sorted(surface_rows, key=lambda row: int(row["n_items"]), reverse=True)[:10]:
        print(
            f"{row['phenomenon']}::{row['uid']}::{row['observed_surface_diff_label']} "
            f"n={row['n_items']} ch={float(row['chinese_accuracy']):.4f} "
            f"di={float(row['diacritic_accuracy']):.4f}"
        )


def main() -> None:
    args = parse_args()
    root = Path.cwd()
    dataset_path = resolve_dataset_path(root, args.dataset)
    scores_path = project_path(root, args.scores)
    output_dir = project_path(root, args.output_dir)
    if not scores_path.exists():
        raise FileNotFoundError(f"Scores CSV does not exist: {scores_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    template_dir = resolve_template_dir(
        root,
        output_dir,
        args.zhoblimp_template_dir,
        args.zhoblimp_repo_dir,
        args.zhoblimp_repo_url,
    )

    items = read_jsonl(dataset_path)
    score_rows = read_csv(scores_path)
    validation = validate_inputs(items, score_rows, args.expected_items)
    validation["dataset_path"] = str(dataset_path)
    validation["scores_path"] = str(scores_path)
    validation["zhoblimp_template_dir"] = str(template_dir)

    enriched_items = enrich_items(items)
    scores = normalize_score_rows(score_rows)
    paradigms = load_official_paradigms(template_dir)
    template_records = load_official_template_records(template_dir)
    validation["n_official_template_json_files"] = len(template_records)
    validation["n_official_template_unique_uids"] = len(paradigms)
    attach_paradigms(enriched_items, paradigms)
    subtype_all = aggregate_subtypes(enriched_items, scores)
    subtype_main = [row for row in subtype_all if int(row["n_items"]) >= args.main_min_items]
    phenomenon_rows = aggregate_phenomena(enriched_items, scores, subtype_all)
    inventory_rows = official_paradigm_inventory(enriched_items, scores, paradigms, template_records)
    paradigm_rows = aggregate_official_paradigms(enriched_items, scores)
    surface_rows = aggregate_paradigm_surface_diffs(enriched_items, scores)
    anaphor_rows = anaphor_analysis(enriched_items, scores)

    write_csv(output_dir / "eval4_subtype_summary_all.csv", subtype_all, SUBTYPE_SUMMARY_FIELDS)
    write_csv(output_dir / "eval4_subtype_summary_main.csv", subtype_main, SUBTYPE_SUMMARY_FIELDS)
    write_csv(output_dir / "eval4_phenomenon_baseline_summary.csv", phenomenon_rows, PHENOMENON_FIELDS)
    write_csv(output_dir / "zhoblimp_paradigm_inventory.csv", inventory_rows, PARADIGM_INVENTORY_FIELDS)
    write_csv(output_dir / "eval4_paradigm_summary.csv", paradigm_rows, PARADIGM_SUMMARY_FIELDS)
    write_csv(
        output_dir / "eval4_paradigm_surface_diff_summary.csv",
        surface_rows,
        PARADIGM_SURFACE_DIFF_FIELDS,
    )
    write_csv(output_dir / "anaphor_subtype_analysis.csv", anaphor_rows, ANAPHOR_FIELDS)

    gap_eligible = [row for row in subtype_main if int(row["n_items"]) >= 20]
    top_chinese = sorted(gap_eligible, key=lambda row: float(row["gap_chinese_minus_diacritic"]), reverse=True)[:30]
    top_diacritic = [
        row
        for row in sorted(gap_eligible, key=lambda row: float(row["gap_chinese_minus_diacritic"]))
        if float(row["diacritic_accuracy"]) - float(row["chinese_accuracy"]) >= 0.05
    ][:30]
    collapse_subtypes = [
        row
        for row in sorted(
            subtype_all,
            key=lambda row: (float(row["diacritic_identical_rate"]), int(row["n_items"])),
            reverse=True,
        )
        if int(row["diacritic_identical_count"]) > 0
    ][:30]

    paradigm_eligible = [row for row in paradigm_rows if int(row["n_items"]) >= 20]
    top_chinese_paradigms = sorted(paradigm_eligible, key=lambda row: float(row["gap"]), reverse=True)[:30]
    top_diacritic_paradigms = sorted(paradigm_eligible, key=lambda row: float(row["gap"]))[:30]
    top_collapse_paradigms = [
        row
        for row in sorted(
            paradigm_rows,
            key=lambda row: (float(row["diacritic_exact_identical_rate"]), int(row["n_items"])),
            reverse=True,
        )
        if int(row["diacritic_exact_identical_count"]) > 0
    ][:30]
    top_surface_diffs = sorted(surface_rows, key=lambda row: int(row["n_items"]), reverse=True)[:30]

    write_csv(
        output_dir / "top_chinese_advantage_subtypes_examples.csv",
        example_rows_for_subtypes(
            subtype_all,
            enriched_items,
            scores,
            [row["phenomenon_subtype"] for row in top_chinese],
            args.examples_per_subtype,
        ),
        EXAMPLE_FIELDS,
    )
    write_csv(
        output_dir / "top_diacritic_advantage_subtypes_examples.csv",
        example_rows_for_subtypes(
            subtype_all,
            enriched_items,
            scores,
            [row["phenomenon_subtype"] for row in top_diacritic],
            args.examples_per_subtype,
        ),
        EXAMPLE_FIELDS,
    )
    write_csv(
        output_dir / "collapse_subtype_examples.csv",
        example_rows_for_subtypes(
            subtype_all,
            enriched_items,
            scores,
            [row["phenomenon_subtype"] for row in collapse_subtypes],
            args.examples_per_subtype,
        ),
        EXAMPLE_FIELDS,
    )
    write_csv(
        output_dir / "top_chinese_advantage_paradigms_examples.csv",
        example_rows_for_paradigms(
            enriched_items,
            scores,
            [row["uid"] for row in top_chinese_paradigms],
            args.examples_per_subtype,
            "top_chinese_advantage_paradigms",
        ),
        PARADIGM_EXAMPLE_FIELDS,
    )
    write_csv(
        output_dir / "top_diacritic_advantage_paradigms_examples.csv",
        example_rows_for_paradigms(
            enriched_items,
            scores,
            [row["uid"] for row in top_diacritic_paradigms],
            args.examples_per_subtype,
            "top_diacritic_advantage_paradigms",
        ),
        PARADIGM_EXAMPLE_FIELDS,
    )
    write_csv(
        output_dir / "collapse_affected_paradigms_examples.csv",
        example_rows_for_paradigms(
            enriched_items,
            scores,
            [row["uid"] for row in top_collapse_paradigms],
            args.examples_per_subtype,
            "collapse_affected_paradigms",
        ),
        PARADIGM_EXAMPLE_FIELDS,
    )
    write_csv(
        output_dir / "top_surface_diffs_by_n_examples.csv",
        example_rows_for_surface_diffs(
            enriched_items,
            scores,
            [(row["uid"], row["observed_surface_diff_key"]) for row in top_surface_diffs],
            args.examples_per_subtype,
            "top_surface_diffs_by_n",
        ),
        PARADIGM_EXAMPLE_FIELDS,
    )
    write_csv(
        output_dir / "special_pattern_examples.csv",
        special_pattern_examples(enriched_items, scores),
        PARADIGM_EXAMPLE_FIELDS,
    )
    write_report(
        output_dir / "eval4_subtype_analysis_report.md",
        validation,
        subtype_all,
        subtype_main,
        phenomenon_rows,
        anaphor_rows,
    )
    write_official_paradigm_report(
        output_dir / "eval4_official_paradigm_analysis_report.md",
        validation,
        paradigms,
        inventory_rows,
        paradigm_rows,
        surface_rows,
        phenomenon_rows,
    )
    (output_dir / "eval4_subtype_analysis_meta.json").write_text(
        json.dumps(
            {
                **validation,
                "output_dir": str(output_dir),
                "main_min_items": args.main_min_items,
                "examples_per_subtype": args.examples_per_subtype,
                "n_unique_subtypes_all": len(subtype_all),
                "n_unique_subtypes_main": len(subtype_main),
                "n_official_template_json_files": len(template_records),
                "n_official_template_unique_uids": len(paradigms),
                "n_official_paradigms_in_dataset": len(paradigm_rows),
                "n_paradigm_surface_diff_rows": len(surface_rows),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"loaded items: {validation['n_items']}")
    print(f"loaded score rows: {validation['n_score_rows']}")
    print(f"unique phenomena: {validation['n_phenomena']}")
    print(f"unique subtypes: {len(subtype_all)}")
    print(f"official template JSON files loaded: {len(template_records)}")
    print(f"official unique template UIDs loaded: {len(paradigms)}")
    print(f"official paradigms in dataset: {len(paradigm_rows)}")
    print(f"paradigm + surface diff rows: {len(surface_rows)}")
    print(f"wrote output dir: {output_dir}")
    print_console_summary(phenomenon_rows, subtype_main, paradigm_rows, surface_rows)


if __name__ == "__main__":
    main()
