#!/usr/bin/env python3
"""Human-readable Eval 4 subtype tables by phenomenon and observed surface diff.

This script only reads the standardized Eval 4 dataset and the existing
item-level score CSV. It does not rerun model scoring, download data, or modify
existing score files.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_DATASET = "eval_data/eval4_chinese_blimp_style/eval4_chinese_blimp_style.jsonl"
DEFAULT_SCORES = "eval_results/eval4_chinese_blimp_style/item_scores.csv"
DEFAULT_OUTPUT_DIR = "eval_results/eval4_chinese_blimp_style/subtype_analysis"
EXPECTED_ITEMS = 35_400
EXPECTED_MODELS = {"chinese_4epoch", "diacritic_matched_token_4epoch"}
BASELINE = 0.5

OUTPUT_CSV_FIELDS = [
    "phenomenon",
    "uid",
    "surface_diff_label",
    "n_items",
    "chinese_accuracy",
    "diacritic_accuracy",
    "gap",
    "chinese_minus_baseline",
    "diacritic_minus_baseline",
    "chinese_mean_margin",
    "diacritic_mean_margin",
    "collapse_count",
    "collapse_rate",
    "tie_count",
    "tie_rate",
    "example_good_zh",
    "example_bad_zh",
    "example_good_diacritic",
    "example_bad_diacritic",
    "example_chinese_correct",
    "example_diacritic_correct",
    "example_chinese_margin",
    "example_diacritic_margin",
    "common_prefix",
    "good_diff_span",
    "bad_diff_span",
    "common_suffix",
    "diacritic_surface_diff_label",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create readable Eval 4 tables grouped by phenomenon and observed Chinese surface diff."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--scores", default=DEFAULT_SCORES)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--expected-items", type=int, default=EXPECTED_ITEMS)
    return parser.parse_args()


def project_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


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


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        raise ValueError("Cannot compute mean over zero finite values")
    return sum(finite) / len(finite)


def accuracy(rows: list[dict[str, Any]]) -> float:
    finite = [row for row in rows if row["non_finite"] == 0]
    if not finite:
        raise ValueError("Cannot compute accuracy over zero finite score rows")
    return sum(row["correct"] for row in finite) / len(finite)


def clean_span(span: str) -> str:
    return span if span else ""


def edit_part_label(good_span: str, bad_span: str, arrow: str = " → ") -> str:
    if good_span and bad_span:
        return f"{good_span}{arrow}{bad_span}"
    if bad_span:
        return f"bad inserts {bad_span}"
    if good_span:
        return f"bad deletes {good_span}"
    return "empty edit"


def surface_diff(good: str, bad: str) -> dict[str, Any]:
    matcher = difflib.SequenceMatcher(a=good, b=bad, autojunk=False)
    opcodes = matcher.get_opcodes()
    diff_ops = [op for op in opcodes if op[0] != "equal"]

    if not diff_ops:
        return {
            "common_prefix": good,
            "good_diff_span": "",
            "bad_diff_span": "",
            "common_suffix": "",
            "edit_type": "identical_zh",
            "surface_diff_label": "identical Chinese strings",
            "surface_diff_key": "IDENTICAL",
        }

    first = diff_ops[0]
    last = diff_ops[-1]
    common_prefix = good[: first[1]]
    common_suffix = good[last[2] :]
    changed_pairs = [(good[i1:i2], bad[j1:j2]) for _tag, i1, i2, j1, j2 in diff_ops]

    if len(diff_ops) == 1:
        tag, _i1, _i2, _j1, _j2 = diff_ops[0]
        good_span, bad_span = changed_pairs[0]
        if tag == "replace":
            edit_type = "replace"
            label = f"{good_span} → {bad_span}"
            key = f"REPLACE:{good_span}->{bad_span}"
        elif tag == "insert":
            edit_type = "insert_in_bad"
            label = f"bad inserts {bad_span}"
            key = f"INSERT_BAD:{bad_span}"
        elif tag == "delete":
            edit_type = "delete_in_bad"
            label = f"bad deletes {good_span}"
            key = f"DELETE_BAD:{good_span}"
        else:
            edit_type = tag
            label = edit_part_label(good_span, bad_span)
            key = f"{tag}:{good_span}->{bad_span}"
    else:
        edit_type = "multiple_edits"
        parts = [edit_part_label(left, right, arrow=" -> ") for left, right in changed_pairs]
        label = "multiple edits: " + "; ".join(parts)
        key = "MULTI:" + ";".join(f"{left}->{right}" for left, right in changed_pairs)

    return {
        "common_prefix": common_prefix,
        "good_diff_span": "|".join(clean_span(pair[0]) for pair in changed_pairs),
        "bad_diff_span": "|".join(clean_span(pair[1]) for pair in changed_pairs),
        "common_suffix": common_suffix,
        "edit_type": edit_type,
        "surface_diff_label": label,
        "surface_diff_key": key,
    }


def validate_inputs(
    items: list[dict[str, Any]],
    score_rows: list[dict[str, str]],
    expected_items: int,
) -> dict[str, Any]:
    if not items:
        raise ValueError("Dataset is empty")
    if not score_rows:
        raise ValueError("Score CSV is empty")

    required_item_fields = {
        "id",
        "phenomenon",
        "subtype_if_any",
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

    required_score_fields = {"id", "model_run", "correct", "tie", "margin", "non_finite"}
    missing_score_fields = sorted(required_score_fields - set(score_rows[0]))
    if missing_score_fields:
        raise ValueError(f"Scores CSV is missing required fields: {missing_score_fields}")
    model_names = {row["model_run"] for row in score_rows}
    if model_names != EXPECTED_MODELS:
        raise ValueError(f"Expected models {sorted(EXPECTED_MODELS)}; got {sorted(model_names)}")
    if len(score_rows) != len(items) * len(EXPECTED_MODELS):
        raise ValueError(f"Expected {len(items) * len(EXPECTED_MODELS)} score rows; got {len(score_rows)}")
    n_phenomena = len({item["phenomenon"] for item in items})
    if n_phenomena != 15:
        raise ValueError(f"Expected 15 phenomenon labels; got {n_phenomena}")
    score_ids = {row["id"] for row in score_rows}
    if score_ids != item_ids:
        raise ValueError(
            f"Dataset-score id mismatch: missing={len(item_ids - score_ids)}, extra={len(score_ids - item_ids)}"
        )
    counts = Counter((row["id"], row["model_run"]) for row in score_rows)
    bad_pairs = [key for key, count in counts.items() if count != 1]
    if bad_pairs:
        raise ValueError(f"Expected exactly one score row per item/model; bad pairs={bad_pairs[:5]}")

    non_finite_rows = [row for row in score_rows if int(row.get("non_finite") or 0) != 0]
    score_fields = [
        "margin",
        "good_mean_logprob",
        "bad_mean_logprob",
        "good_total_logprob",
        "bad_total_logprob",
    ]
    non_finite_score_values = [
        (row["id"], row["model_run"], field)
        for row in score_rows
        for field in score_fields
        if field in row and not math.isfinite(safe_float(row.get(field)))
    ]
    if non_finite_rows or non_finite_score_values:
        raise ValueError(
            f"Found non-finite scores: non_finite_rows={len(non_finite_rows)}, "
            f"non_finite_score_values={len(non_finite_score_values)}"
        )

    return {
        "n_items": len(items),
        "n_score_rows": len(score_rows),
        "n_phenomena": n_phenomena,
        "models": sorted(model_names),
    }


def normalize_scores(score_rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, Any]]:
    scores = {}
    for row in score_rows:
        normalized = dict(row)
        normalized["correct"] = int(row["correct"])
        normalized["tie"] = int(row["tie"])
        normalized["non_finite"] = int(row["non_finite"])
        normalized["margin"] = safe_float(row["margin"])
        scores[(row["id"], row["model_run"])] = normalized
    return scores


def enrich_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched = []
    for item in items:
        zh_diff = surface_diff(item["good_sentence_zh"], item["bad_sentence_zh"])
        di_diff = surface_diff(item["good_sentence_diacritic"], item["bad_sentence_diacritic"])
        row = dict(item)
        row.update(
            {
                "uid": str(item.get("subtype_if_any", "")).strip(),
                "common_prefix": zh_diff["common_prefix"],
                "good_diff_span": zh_diff["good_diff_span"],
                "bad_diff_span": zh_diff["bad_diff_span"],
                "common_suffix": zh_diff["common_suffix"],
                "edit_type": zh_diff["edit_type"],
                "surface_diff_label": zh_diff["surface_diff_label"],
                "surface_diff_key": zh_diff["surface_diff_key"],
                "diacritic_identical": item["good_sentence_diacritic"] == item["bad_sentence_diacritic"],
                "diacritic_surface_diff_label": (
                    "identical diacritic strings"
                    if item["good_sentence_diacritic"] == item["bad_sentence_diacritic"]
                    else di_diff["surface_diff_label"]
                ),
                "diacritic_surface_diff_key": di_diff["surface_diff_key"],
            }
        )
        enriched.append(row)
    return enriched


def subtype_key(item: dict[str, Any]) -> tuple[str, str, str]:
    return (item["phenomenon"], item["uid"], item["surface_diff_key"])


def select_example(
    group: list[dict[str, Any]],
    scores: dict[tuple[str, str], dict[str, Any]],
    gap: float,
    ch_mean_margin: float,
    di_mean_margin: float,
) -> dict[str, Any]:
    sorted_group = sorted(group, key=lambda item: item["id"])

    if gap >= 0.10:
        preferred = [
            item
            for item in sorted_group
            if scores[(item["id"], "chinese_4epoch")]["correct"] == 1
            and scores[(item["id"], "diacritic_matched_token_4epoch")]["correct"] == 0
        ]
    elif gap <= -0.10:
        preferred = [
            item
            for item in sorted_group
            if scores[(item["id"], "chinese_4epoch")]["correct"] == 0
            and scores[(item["id"], "diacritic_matched_token_4epoch")]["correct"] == 1
        ]
    else:
        preferred = [
            item
            for item in sorted_group
            if scores[(item["id"], "chinese_4epoch")]["correct"] == 1
            and scores[(item["id"], "diacritic_matched_token_4epoch")]["correct"] == 1
        ]

    candidates = preferred if preferred else sorted_group

    def distance(item: dict[str, Any]) -> tuple[float, str]:
        ch_margin = scores[(item["id"], "chinese_4epoch")]["margin"]
        di_margin = scores[(item["id"], "diacritic_matched_token_4epoch")]["margin"]
        return (abs(ch_margin - ch_mean_margin) + abs(di_margin - di_mean_margin), item["id"])

    return min(candidates, key=distance)


def aggregate_subtypes(
    items: list[dict[str, Any]],
    scores: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        groups[subtype_key(item)].append(item)

    rows = []
    for key, group in sorted(groups.items()):
        first = group[0]
        item_ids = [item["id"] for item in group]
        ch_rows = [scores[(item_id, "chinese_4epoch")] for item_id in item_ids]
        di_rows = [scores[(item_id, "diacritic_matched_token_4epoch")] for item_id in item_ids]
        ch_acc = accuracy(ch_rows)
        di_acc = accuracy(di_rows)
        gap = ch_acc - di_acc
        ch_mean_margin = mean([row["margin"] for row in ch_rows])
        di_mean_margin = mean([row["margin"] for row in di_rows])
        collapse_count = sum(1 for item in group if item["diacritic_identical"])
        tie_count = sum(row["tie"] for row in di_rows)
        example = select_example(group, scores, gap, ch_mean_margin, di_mean_margin)
        ch_example = scores[(example["id"], "chinese_4epoch")]
        di_example = scores[(example["id"], "diacritic_matched_token_4epoch")]
        n_items = len(group)

        rows.append(
            {
                "phenomenon": key[0],
                "uid": key[1],
                "surface_diff_label": first["surface_diff_label"],
                "surface_diff_key": key[2],
                "n_items": n_items,
                "chinese_accuracy": ch_acc,
                "diacritic_accuracy": di_acc,
                "gap": gap,
                "chinese_minus_baseline": ch_acc - BASELINE,
                "diacritic_minus_baseline": di_acc - BASELINE,
                "chinese_mean_margin": ch_mean_margin,
                "diacritic_mean_margin": di_mean_margin,
                "collapse_count": collapse_count,
                "collapse_rate": collapse_count / n_items,
                "tie_count": tie_count,
                "tie_rate": tie_count / n_items,
                "example_good_zh": example["good_sentence_zh"],
                "example_bad_zh": example["bad_sentence_zh"],
                "example_good_diacritic": example["good_sentence_diacritic"],
                "example_bad_diacritic": example["bad_sentence_diacritic"],
                "example_chinese_correct": ch_example["correct"],
                "example_diacritic_correct": di_example["correct"],
                "example_chinese_margin": ch_example["margin"],
                "example_diacritic_margin": di_example["margin"],
                "common_prefix": first["common_prefix"],
                "good_diff_span": first["good_diff_span"],
                "bad_diff_span": first["bad_diff_span"],
                "common_suffix": first["common_suffix"],
                "diacritic_surface_diff_label": first["diacritic_surface_diff_label"],
            }
        )
    return rows


def fmt_float(value: Any, signed: bool = False) -> str:
    number = float(value)
    return f"{number:+.4f}" if signed else f"{number:.4f}"


def md_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def example_cell(row: dict[str, Any]) -> str:
    return f"Good: {row['example_good_zh']}<br>Bad: {row['example_bad_zh']}"


def sorted_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (-abs(float(row["gap"])), -int(row["n_items"]), row["surface_diff_label"], row["uid"]),
    )


def write_main_markdown(path: Path, rows: list[dict[str, Any]], validation: dict[str, Any]) -> None:
    rows_by_phenomenon: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_phenomenon[row["phenomenon"]].append(row)

    lines = [
        "# Eval 4 By Type And Observed Surface Difference",
        "",
        "Rows are grouped by broad phenomenon/type. The main subtype label is the observed "
        "good/bad Chinese surface difference, while UID is kept as a secondary identifier "
        "to avoid merging distinct official ZhoBLiMP paradigms.",
        "",
        "## Loaded Data",
        "",
        f"- items: {validation['n_items']}",
        f"- score rows: {validation['n_score_rows']}",
        f"- phenomena: {validation['n_phenomena']}",
        f"- models: {', '.join(validation['models'])}",
        f"- subtype rows: {len(rows)}",
        "",
    ]

    for phenomenon in sorted(rows_by_phenomenon):
        lines.extend(
            [
                f"## {phenomenon}",
                "",
                "| Subtype / good-bad difference | UID | n | Chinese | Diacritic | Gap | Collapse | Example |",
                "|---|---|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in sorted_rows(rows_by_phenomenon[phenomenon]):
            lines.append(
                "| "
                + " | ".join(
                    [
                        md_escape(row["surface_diff_label"]),
                        md_escape(row["uid"]),
                        str(row["n_items"]),
                        fmt_float(row["chinese_accuracy"]),
                        fmt_float(row["diacritic_accuracy"]),
                        fmt_float(row["gap"], signed=True),
                        fmt_float(row["collapse_rate"]),
                        md_escape(example_cell(row)),
                    ]
                )
                + " |"
            )
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compact_table(rows: list[dict[str, Any]], limit: int | None = None) -> list[str]:
    selected = sorted_rows(rows)
    if limit is not None:
        selected = selected[:limit]
    lines = [
        "| phenomenon | surface diff | uid | n | Chinese | Diacritic | gap | example |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in selected:
        lines.append(
            "| "
            + " | ".join(
                [
                    md_escape(row["phenomenon"]),
                    md_escape(row["surface_diff_label"]),
                    md_escape(row["uid"]),
                    str(row["n_items"]),
                    fmt_float(row["chinese_accuracy"]),
                    fmt_float(row["diacritic_accuracy"]),
                    fmt_float(row["gap"], signed=True),
                    md_escape(example_cell(row)),
                ]
            )
            + " |"
        )
    return lines


def write_discussion_markdown(path: Path, rows: list[dict[str, Any]]) -> dict[str, int]:
    sections = [
        (
            "Strong Chinese advantage, both meaningful",
            [
                row
                for row in rows
                if float(row["chinese_accuracy"]) >= 0.60
                and float(row["gap"]) >= 0.15
                and float(row["collapse_rate"]) < 0.10
            ],
        ),
        (
            "Chinese advantage but Diacritic near/below chance",
            [
                row
                for row in rows
                if float(row["chinese_accuracy"]) >= 0.60
                and float(row["diacritic_accuracy"]) <= 0.55
                and float(row["gap"]) >= 0.15
            ],
        ),
        (
            "Diacritic advantage",
            [row for row in rows if float(row["diacritic_accuracy"]) - float(row["chinese_accuracy"]) >= 0.10],
        ),
        (
            "Collapse-affected",
            [row for row in rows if float(row["collapse_rate"]) >= 0.10],
        ),
        (
            "Both models fail",
            [
                row
                for row in rows
                if float(row["chinese_accuracy"]) < 0.50 and float(row["diacritic_accuracy"]) < 0.50
            ],
        ),
    ]

    lines = [
        "# Eval 4 Key Surface-Diff Subtypes For Discussion",
        "",
        "These compact tables use observed Chinese good/bad differences as the main labels. "
        "Examples are selected from actual Eval 4 items.",
        "",
    ]
    counts = {}
    for title, section_rows in sections:
        counts[title] = len(section_rows)
        lines.extend([f"## {title}", "", f"Rows: {len(section_rows)}", ""])
        lines.extend(compact_table(section_rows))
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return counts


def print_console_summary(rows: list[dict[str, Any]], validation: dict[str, Any], paths: dict[str, Path]) -> None:
    print(f"subtype rows generated: {len(rows)}")
    print(f"phenomena: {validation['n_phenomena']}")

    print("\nTop 10 Chinese-advantage subtypes")
    for row in sorted(rows, key=lambda item: float(item["gap"]), reverse=True)[:10]:
        print(
            f"{row['phenomenon']} | {row['surface_diff_label']} | {row['uid']} | "
            f"n={row['n_items']} ch={float(row['chinese_accuracy']):.4f} "
            f"di={float(row['diacritic_accuracy']):.4f} gap={float(row['gap']):+.4f}"
        )
        print(f"  Good: {row['example_good_zh']}")
        print(f"  Bad:  {row['example_bad_zh']}")

    print("\nTop 10 Diacritic-advantage subtypes")
    for row in sorted(rows, key=lambda item: float(item["gap"]))[:10]:
        print(
            f"{row['phenomenon']} | {row['surface_diff_label']} | {row['uid']} | "
            f"n={row['n_items']} ch={float(row['chinese_accuracy']):.4f} "
            f"di={float(row['diacritic_accuracy']):.4f} gap={float(row['gap']):+.4f}"
        )
        print(f"  Good: {row['example_good_zh']}")
        print(f"  Bad:  {row['example_bad_zh']}")

    print("\nCollapse-affected subtypes")
    collapse_rows = [row for row in rows if float(row["collapse_rate"]) > 0]
    for row in sorted(collapse_rows, key=lambda item: (-float(item["collapse_rate"]), -int(item["n_items"]))):
        print(
            f"{row['phenomenon']} | {row['surface_diff_label']} | {row['uid']} | "
            f"n={row['n_items']} collapse={float(row['collapse_rate']):.4f} "
            f"ch={float(row['chinese_accuracy']):.4f} di={float(row['diacritic_accuracy']):.4f}"
        )
        print(f"  Good: {row['example_good_zh']}")
        print(f"  Bad:  {row['example_bad_zh']}")

    print("\nOutputs")
    for label, path in paths.items():
        print(f"{label}: {path}")


def main() -> None:
    args = parse_args()
    root = Path.cwd()
    dataset_path = project_path(root, args.dataset)
    scores_path = project_path(root, args.scores)
    output_dir = project_path(root, args.output_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset does not exist: {dataset_path}")
    if not scores_path.exists():
        raise FileNotFoundError(f"Scores CSV does not exist: {scores_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    items = read_jsonl(dataset_path)
    score_rows = read_csv(scores_path)
    validation = validate_inputs(items, score_rows, args.expected_items)
    scores = normalize_scores(score_rows)
    enriched = enrich_items(items)
    rows = aggregate_subtypes(enriched, scores)

    csv_path = output_dir / "eval4_by_type_surface_diff_table.csv"
    md_path = output_dir / "eval4_by_type_surface_diff_table.md"
    discussion_path = output_dir / "eval4_key_subtypes_for_discussion.md"

    write_csv(csv_path, rows, OUTPUT_CSV_FIELDS)
    write_main_markdown(md_path, rows, validation)
    write_discussion_markdown(discussion_path, rows)

    print_console_summary(
        rows,
        validation,
        {
            "markdown": md_path,
            "csv": csv_path,
            "discussion_markdown": discussion_path,
        },
    )


if __name__ == "__main__":
    main()
