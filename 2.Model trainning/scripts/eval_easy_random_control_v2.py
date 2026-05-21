#!/usr/bin/env python3
"""Evaluate Easy Random Non-Homophone Control v2 and compare three probe gaps."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from eval2_common import load_model_runs_json, read_csv
from eval_nonhomophone_control_v2 import (
    ITEM_SCORE_FIELDS,
    SCORING_MODES,
    SUMMARY_FIELDS,
    choose_device_and_dtype,
    evaluate_run,
    item_has_invalid_control_collision,
    load_items,
    project_path,
    write_csv,
)


THREE_PROBE_FIELDS = [
    "scoring_mode",
    "homophone_chinese_noncollapsed_accuracy",
    "homophone_diacritic_noncollapsed_accuracy",
    "homophone_gap",
    "hard_control_chinese_accuracy",
    "hard_control_diacritic_accuracy",
    "hard_control_gap",
    "easy_control_chinese_accuracy",
    "easy_control_diacritic_accuracy",
    "easy_control_gap",
    "homophone_minus_hard_gap",
    "homophone_minus_easy_gap",
    "hard_minus_easy_gap",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Easy Random Non-Homophone Control v2.")
    parser.add_argument("--control-jsonl", default="eval_data/easy_random_control_v2/easy_random_control_v2.jsonl")
    parser.add_argument("--output-dir", default="eval_results/eval2/easy_random_control_v2")
    parser.add_argument(
        "--model-runs-json",
        default=None,
        help="Optional JSON manifest with model runs to evaluate. Defaults to the built-in seed42 pair.",
    )
    parser.add_argument(
        "--homophone-matched-summary",
        default="eval_results/eval2/homophone_probe_v2/summary_matched_subsets.csv",
    )
    parser.add_argument(
        "--hard-control-summary",
        default="eval_results/eval2/nonhomophone_control_v2/summary_by_model_and_scoring.csv",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def summary_lookup(rows: list[dict[str, str]], model_field: str, metric_field: str) -> dict[tuple[str, str], float]:
    output: dict[tuple[str, str], float] = {}
    for row in rows:
        output[(row["scoring_mode"], row[model_field])] = float(row[metric_field])
    return output


def build_three_probe_comparison(
    homophone_summary_path: Path,
    hard_summary_path: Path,
    easy_summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not homophone_summary_path.exists():
        print(f"WARNING: missing homophone matched summary: {homophone_summary_path}")
        return []
    if not hard_summary_path.exists():
        print(f"WARNING: missing hard control summary: {hard_summary_path}")
        return []

    homophone_rows = read_csv(homophone_summary_path)
    hard_rows = read_csv(hard_summary_path)
    homophone = summary_lookup(homophone_rows, "model", "noncollapsed_subset_accuracy")
    hard = summary_lookup(hard_rows, "model_run", "accuracy")
    easy = {
        (row["scoring_mode"], row["model_run"]): float(row["accuracy"])
        for row in easy_summaries
    }

    rows: list[dict[str, Any]] = []
    for mode in SCORING_MODES:
        required = [
            (homophone, (mode, "chinese_4epoch")),
            (homophone, (mode, "diacritic_matched_token_4epoch")),
            (hard, (mode, "chinese_4epoch")),
            (hard, (mode, "diacritic_matched_token_4epoch")),
            (easy, (mode, "chinese_4epoch")),
            (easy, (mode, "diacritic_matched_token_4epoch")),
        ]
        if not all(key in table for table, key in required):
            print(f"WARNING: skipping default three-probe comparison for {mode}; default seed42 pair is absent")
            continue
        h_ch = homophone[(mode, "chinese_4epoch")]
        h_di = homophone[(mode, "diacritic_matched_token_4epoch")]
        hard_ch = hard[(mode, "chinese_4epoch")]
        hard_di = hard[(mode, "diacritic_matched_token_4epoch")]
        easy_ch = easy[(mode, "chinese_4epoch")]
        easy_di = easy[(mode, "diacritic_matched_token_4epoch")]
        h_gap = h_ch - h_di
        hard_gap = hard_ch - hard_di
        easy_gap = easy_ch - easy_di
        rows.append(
            {
                "scoring_mode": mode,
                "homophone_chinese_noncollapsed_accuracy": h_ch,
                "homophone_diacritic_noncollapsed_accuracy": h_di,
                "homophone_gap": h_gap,
                "hard_control_chinese_accuracy": hard_ch,
                "hard_control_diacritic_accuracy": hard_di,
                "hard_control_gap": hard_gap,
                "easy_control_chinese_accuracy": easy_ch,
                "easy_control_diacritic_accuracy": easy_di,
                "easy_control_gap": easy_gap,
                "homophone_minus_hard_gap": h_gap - hard_gap,
                "homophone_minus_easy_gap": h_gap - easy_gap,
                "hard_minus_easy_gap": hard_gap - easy_gap,
            }
        )
    return rows


def main() -> None:
    import os

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    root = Path.cwd()
    control_path = project_path(root, args.control_jsonl)
    output_dir = project_path(root, args.output_dir)
    homophone_summary_path = project_path(root, args.homophone_matched_summary)
    hard_summary_path = project_path(root, args.hard_control_summary)
    output_dir.mkdir(parents=True, exist_ok=True)

    items = load_items(control_path)
    invalid_count = sum(1 for item in items if item_has_invalid_control_collision(item))
    print(f"easy control items: {len(items)}")
    print(f"invalid control collisions: {invalid_count}")
    device, dtype, dtype_name = choose_device_and_dtype()
    print(f"device: {device.type}, dtype: {dtype_name}")
    model_runs = load_model_runs_json(args.model_runs_json, root)
    print(f"model runs: {', '.join(run.run_name for run in model_runs)}")

    all_rows: list[dict[str, Any]] = []
    all_summaries: list[dict[str, Any]] = []
    for run in model_runs:
        rows, summaries = evaluate_run(root, run, items, output_dir, device, dtype, dtype_name, args)
        all_rows.extend(rows)
        all_summaries.extend(summaries)

    write_csv(output_dir / "item_scores.csv", all_rows, ITEM_SCORE_FIELDS)
    write_csv(output_dir / "summary_by_model_and_scoring.csv", all_summaries, SUMMARY_FIELDS)
    comparison_rows = build_three_probe_comparison(homophone_summary_path, hard_summary_path, all_summaries)
    write_csv(output_dir / "three_probe_gap_comparison.csv", comparison_rows, THREE_PROBE_FIELDS)

    print(f"wrote: {output_dir / 'item_scores.csv'}")
    print(f"wrote: {output_dir / 'summary_by_model_and_scoring.csv'}")
    print(f"wrote: {output_dir / 'three_probe_gap_comparison.csv'}")
    print("\nSummary")
    for row in all_summaries:
        print(
            f"{row['model_run']} {row['scoring_mode']}: "
            f"n_valid={row['n_valid']} invalid={row['invalid_control_collision_count']} "
            f"accuracy={row['accuracy']} margin={row['mean_margin']}"
        )
    if comparison_rows:
        print("\nThree-probe gap comparison")
        for row in comparison_rows:
            print(
                f"{row['scoring_mode']}: homophone={row['homophone_gap']} "
                f"hard={row['hard_control_gap']} easy={row['easy_control_gap']} "
                f"homophone_minus_easy={row['homophone_minus_easy_gap']}"
            )


if __name__ == "__main__":
    main()
