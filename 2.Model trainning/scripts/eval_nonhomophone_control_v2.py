#!/usr/bin/env python3
"""Evaluate the matched Non-Homophone Control Probe v2."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from statistics import median
from typing import Any

import torch
from tqdm.auto import tqdm

from eval2_common import (
    MODEL_RUNS,
    SCORING_MODES,
    ModelRun,
    bootstrap_ci,
    choose_device_and_dtype,
    completion_parts,
    load_model,
    project_path,
    read_csv,
    score_completion,
    validate_checkpoint_and_tokenizer,
    write_csv,
)
from probe_build_common_v2 import path_looks_like_train_split

ITEM_SCORE_FIELDS = [
    "id",
    "source_homophone_item_id",
    "model_run",
    "script",
    "scoring_mode",
    "status",
    "invalid_control_collision",
    "correct",
    "prediction",
    "gold_score",
    "distractor_score",
    "margin",
    "gold_token_count",
    "distractor_token_count",
    "gold_zh",
    "distractor_zh",
    "source_line",
]

SUMMARY_FIELDS = [
    "model_run",
    "script",
    "scoring_mode",
    "primary",
    "n_items",
    "n_valid",
    "invalid_control_collision_count",
    "accuracy",
    "mean_margin",
    "median_margin",
    "mean_gold_score",
    "mean_distractor_score",
    "accuracy_ci_low",
    "accuracy_ci_high",
    "mean_margin_ci_low",
    "mean_margin_ci_high",
    "device",
    "dtype",
    "notes",
]

COMPARISON_FIELDS = [
    "scoring_mode",
    "homophone_chinese_noncollapsed_accuracy",
    "homophone_diacritic_noncollapsed_accuracy",
    "homophone_gap",
    "nonhomophone_chinese_accuracy",
    "nonhomophone_diacritic_accuracy",
    "nonhomophone_gap",
    "gap_difference",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Non-Homophone Control Probe v2.")
    parser.add_argument("--control-jsonl", default="eval_data/nonhomophone_control_v2/nonhomophone_control_v2.jsonl")
    parser.add_argument("--output-dir", default="eval_results/nonhomophone_control_v2")
    parser.add_argument(
        "--homophone-matched-summary",
        default="eval_results/homophone_probe_v2/summary_matched_subsets.csv",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260506)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def load_items(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                items.append(json.loads(line))
    ids = [item["id"] for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate control item ids found")
    train_sources = [item["source_file"] for item in items if path_looks_like_train_split(item["source_file"])]
    if train_sources:
        raise ValueError(f"Control items reference train data: {train_sources[:5]}")
    return items


def item_has_invalid_control_collision(item: dict[str, Any]) -> bool:
    return (
        not item.get("nonhomophone_verified", False)
        or item["gold_pinyin_diacritic"] == item["distractor_pinyin_diacritic"]
        or item["gold_toneless_pinyin"] == item["distractor_toneless_pinyin"]
    )


def summarize(rows: list[dict[str, Any]], run: ModelRun, mode: str, device: str, dtype: str, args: argparse.Namespace) -> dict[str, Any]:
    n_items = len(rows)
    invalid = [row for row in rows if row["status"] == "invalid_control_collision"]
    scored = [row for row in rows if row["status"] == "scored"]
    correct_values = [float(row["correct"]) for row in scored]
    margins = [float(row["margin"]) for row in scored]
    gold_scores = [float(row["gold_score"]) for row in scored]
    distractor_scores = [float(row["distractor_score"]) for row in scored]
    accuracy = sum(correct_values) / len(correct_values) if correct_values else ""
    acc_low, acc_high = bootstrap_ci(
        correct_values, lambda sample: sum(sample) / len(sample), args.bootstrap_samples, args.seed + 17
    )
    margin_low, margin_high = bootstrap_ci(
        margins, lambda sample: sum(sample) / len(sample), args.bootstrap_samples, args.seed + 31
    )
    return {
        "model_run": run.run_name,
        "script": run.script,
        "scoring_mode": mode,
        "primary": mode == "candidate_plus_suffix",
        "n_items": n_items,
        "n_valid": len(scored),
        "invalid_control_collision_count": len(invalid),
        "accuracy": accuracy,
        "mean_margin": sum(margins) / len(margins) if margins else "",
        "median_margin": median(margins) if margins else "",
        "mean_gold_score": sum(gold_scores) / len(gold_scores) if gold_scores else "",
        "mean_distractor_score": sum(distractor_scores) / len(distractor_scores) if distractor_scores else "",
        "accuracy_ci_low": acc_low,
        "accuracy_ci_high": acc_high,
        "mean_margin_ci_low": margin_low,
        "mean_margin_ci_high": margin_high,
        "device": device,
        "dtype": dtype,
        "notes": "Non-homophone control; candidate_plus_suffix is primary.",
    }


def invalid_row(item: dict[str, Any], run: ModelRun, mode: str) -> dict[str, Any]:
    return {
        "id": item["id"],
        "source_homophone_item_id": item["source_homophone_item_id"],
        "model_run": run.run_name,
        "script": run.script,
        "scoring_mode": mode,
        "status": "invalid_control_collision",
        "invalid_control_collision": True,
        "correct": "",
        "prediction": "",
        "gold_score": "",
        "distractor_score": "",
        "margin": "",
        "gold_token_count": "",
        "distractor_token_count": "",
        "gold_zh": item["gold_zh"],
        "distractor_zh": item["distractor_zh"],
        "source_line": item["source_line"],
    }


def evaluate_run(
    root: Path,
    run: ModelRun,
    items: list[dict[str, Any]],
    output_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    dtype_name: str,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    checkpoint, tokenizer = validate_checkpoint_and_tokenizer(run, root)
    print(f"\n== {run.run_name} ==")
    print(f"checkpoint: {checkpoint}")
    print(f"tokenizer vocab/eos/pad validated: {len(tokenizer)}/{tokenizer.eos_token_id}/{tokenizer.pad_token_id}")
    model = load_model(checkpoint, device, dtype)
    rows: list[dict[str, Any]] = []
    iterator = tqdm(items, desc=run.run_name, disable=args.no_progress)
    for index, item in enumerate(iterator, start=1):
        if args.progress_every and index % args.progress_every == 0:
            print(f"{run.run_name}: scored {index}/{len(items)} items")
        for mode in SCORING_MODES:
            if item_has_invalid_control_collision(item):
                rows.append(invalid_row(item, run, mode))
                continue
            prefix, gold_completion, distractor_completion = completion_parts(item, run, mode)
            gold_score, gold_count = score_completion(model, tokenizer, device, prefix, gold_completion)
            distractor_score, distractor_count = score_completion(model, tokenizer, device, prefix, distractor_completion)
            margin = gold_score - distractor_score
            if not math.isfinite(margin):
                raise ValueError(f"Non-finite margin for {run.run_name}/{mode}/{item['id']}")
            rows.append(
                {
                    "id": item["id"],
                    "source_homophone_item_id": item["source_homophone_item_id"],
                    "model_run": run.run_name,
                    "script": run.script,
                    "scoring_mode": mode,
                    "status": "scored",
                    "invalid_control_collision": False,
                    "correct": int(margin > 0),
                    "prediction": item["gold_zh"] if margin > 0 else item["distractor_zh"],
                    "gold_score": gold_score,
                    "distractor_score": distractor_score,
                    "margin": margin,
                    "gold_token_count": gold_count,
                    "distractor_token_count": distractor_count,
                    "gold_zh": item["gold_zh"],
                    "distractor_zh": item["distractor_zh"],
                    "source_line": item["source_line"],
                }
            )
    summaries = [
        summarize([row for row in rows if row["scoring_mode"] == mode], run, mode, device.type, dtype_name, args)
        for mode in SCORING_MODES
    ]
    (output_dir / f"{run.run_name}.json").write_text(
        json.dumps({"run": run.__dict__, "summary": summaries}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows, summaries


def build_gap_comparison(homophone_summary_path: Path, control_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not homophone_summary_path.exists():
        print(f"WARNING: missing homophone matched summary: {homophone_summary_path}")
        return []
    homophone_rows = read_csv(homophone_summary_path)
    homophone_lookup = {
        (row["scoring_mode"], row["model"]): row
        for row in homophone_rows
    }
    control_lookup = {
        (row["scoring_mode"], row["model_run"]): row
        for row in control_summaries
    }
    rows: list[dict[str, Any]] = []
    for mode in SCORING_MODES:
        h_ch = homophone_lookup.get((mode, "chinese_4epoch"), {})
        h_di = homophone_lookup.get((mode, "diacritic_matched_token_4epoch"), {})
        c_ch = control_lookup.get((mode, "chinese_4epoch"), {})
        c_di = control_lookup.get((mode, "diacritic_matched_token_4epoch"), {})
        if not (h_ch and h_di and c_ch and c_di):
            print(f"WARNING: incomplete gap comparison inputs for {mode}")
            continue
        h_ch_acc = float(h_ch["noncollapsed_subset_accuracy"])
        h_di_acc = float(h_di["noncollapsed_subset_accuracy"])
        c_ch_acc = float(c_ch["accuracy"])
        c_di_acc = float(c_di["accuracy"])
        h_gap = h_ch_acc - h_di_acc
        c_gap = c_ch_acc - c_di_acc
        rows.append(
            {
                "scoring_mode": mode,
                "homophone_chinese_noncollapsed_accuracy": h_ch_acc,
                "homophone_diacritic_noncollapsed_accuracy": h_di_acc,
                "homophone_gap": h_gap,
                "nonhomophone_chinese_accuracy": c_ch_acc,
                "nonhomophone_diacritic_accuracy": c_di_acc,
                "nonhomophone_gap": c_gap,
                "gap_difference": h_gap - c_gap,
            }
        )
    return rows


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    root = Path.cwd()
    control_path = project_path(root, args.control_jsonl)
    output_dir = project_path(root, args.output_dir)
    homophone_summary_path = project_path(root, args.homophone_matched_summary)
    output_dir.mkdir(parents=True, exist_ok=True)

    items = load_items(control_path)
    invalid_count = sum(1 for item in items if item_has_invalid_control_collision(item))
    print(f"control items: {len(items)}")
    print(f"invalid control collisions: {invalid_count}")
    device, dtype, dtype_name = choose_device_and_dtype()
    print(f"device: {device.type}, dtype: {dtype_name}")

    all_rows: list[dict[str, Any]] = []
    all_summaries: list[dict[str, Any]] = []
    for run in MODEL_RUNS:
        rows, summaries = evaluate_run(root, run, items, output_dir, device, dtype, dtype_name, args)
        all_rows.extend(rows)
        all_summaries.extend(summaries)

    write_csv(output_dir / "item_scores.csv", all_rows, ITEM_SCORE_FIELDS)
    write_csv(output_dir / "summary_by_model_and_scoring.csv", all_summaries, SUMMARY_FIELDS)
    comparison_rows = build_gap_comparison(homophone_summary_path, all_summaries)
    write_csv(output_dir / "homophone_vs_control_gap.csv", comparison_rows, COMPARISON_FIELDS)

    print(f"wrote: {output_dir / 'item_scores.csv'}")
    print(f"wrote: {output_dir / 'summary_by_model_and_scoring.csv'}")
    print(f"wrote: {output_dir / 'homophone_vs_control_gap.csv'}")
    print("\nSummary")
    for row in all_summaries:
        print(
            f"{row['model_run']} {row['scoring_mode']}: "
            f"n_valid={row['n_valid']} invalid={row['invalid_control_collision_count']} "
            f"accuracy={row['accuracy']} margin={row['mean_margin']}"
        )
    if comparison_rows:
        print("\nHomophone vs control gaps")
        for row in comparison_rows:
            print(
                f"{row['scoring_mode']}: homophone_gap={row['homophone_gap']} "
                f"control_gap={row['nonhomophone_gap']} gap_difference={row['gap_difference']}"
            )


if __name__ == "__main__":
    main()
