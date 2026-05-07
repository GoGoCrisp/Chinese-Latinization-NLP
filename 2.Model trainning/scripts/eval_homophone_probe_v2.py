#!/usr/bin/env python3
"""Evaluate Homophone Disambiguation Probe v2 with causal LM likelihood scoring."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
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
    score_completion,
    validate_checkpoint_and_tokenizer,
    write_csv,
)


ITEM_SCORE_FIELDS = [
    "id",
    "model_run",
    "script",
    "scoring_mode",
    "status",
    "collapsed_diacritic",
    "correct",
    "prediction",
    "gold_score",
    "distractor_score",
    "margin",
    "gold_token_count",
    "distractor_token_count",
    "collision_key",
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
    "n_collapsed",
    "collapse_rate",
    "n_resolvable",
    "resolvable_accuracy",
    "chance_adjusted_all_accuracy",
    "mean_margin",
    "median_margin",
    "mean_gold_score",
    "mean_distractor_score",
    "resolvable_accuracy_ci_low",
    "resolvable_accuracy_ci_high",
    "mean_margin_ci_low",
    "mean_margin_ci_high",
    "device",
    "dtype",
    "notes",
]

MATCHED_SUBSET_FIELDS = [
    "scoring_mode",
    "model",
    "n_all",
    "n_collapsed",
    "n_noncollapsed",
    "all_item_accuracy",
    "chance_adjusted_all_accuracy",
    "noncollapsed_subset_accuracy",
    "collapsed_subset_accuracy",
    "mean_margin_all_scored",
    "mean_margin_noncollapsed",
    "mean_margin_collapsed_for_chinese",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Homophone Probe v2.")
    parser.add_argument("--probe-jsonl", default="eval_data/homophone_probe_v2/probe_v2.jsonl")
    parser.add_argument("--output-dir", default="eval_results/homophone_probe_v2")
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Only aggregate existing item_scores.csv; do not load models or score items.",
    )
    parser.add_argument("--item-scores-csv", default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260504)
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
        raise ValueError("Duplicate item ids found in probe JSONL")
    return items


def summarize(rows: list[dict[str, Any]], run: ModelRun, mode: str, device: str, dtype: str, bootstrap_samples: int, seed: int) -> dict[str, Any]:
    n_items = len(rows)
    collapsed = [row for row in rows if row["status"] == "collapsed"]
    scored = [row for row in rows if row["status"] == "scored"]
    correct_values = [float(row["correct"]) for row in scored]
    margins = [float(row["margin"]) for row in scored]
    gold_scores = [float(row["gold_score"]) for row in scored]
    distractor_scores = [float(row["distractor_score"]) for row in scored]
    correct_count = sum(correct_values)

    acc = correct_count / len(scored) if scored else ""
    chance_adjusted = (correct_count + 0.5 * len(collapsed)) / n_items if n_items else ""
    acc_ci_low, acc_ci_high = bootstrap_ci(
        correct_values, lambda sample: sum(sample) / len(sample), bootstrap_samples, seed + 17
    )
    margin_ci_low, margin_ci_high = bootstrap_ci(
        margins, lambda sample: sum(sample) / len(sample), bootstrap_samples, seed + 31
    )

    return {
        "model_run": run.run_name,
        "script": run.script,
        "scoring_mode": mode,
        "primary": mode == "candidate_plus_suffix",
        "n_items": n_items,
        "n_collapsed": len(collapsed),
        "collapse_rate": len(collapsed) / n_items if n_items else "",
        "n_resolvable": len(scored),
        "resolvable_accuracy": acc,
        "chance_adjusted_all_accuracy": chance_adjusted,
        "mean_margin": sum(margins) / len(margins) if margins else "",
        "median_margin": median(margins) if margins else "",
        "mean_gold_score": sum(gold_scores) / len(gold_scores) if gold_scores else "",
        "mean_distractor_score": sum(distractor_scores) / len(distractor_scores) if distractor_scores else "",
        "resolvable_accuracy_ci_low": acc_ci_low,
        "resolvable_accuracy_ci_high": acc_ci_high,
        "mean_margin_ci_low": margin_ci_low,
        "mean_margin_ci_high": margin_ci_high,
        "device": device,
        "dtype": dtype,
        "notes": (
            "candidate_plus_suffix is primary. Diacritic collapsed items are representational collapses "
            "and excluded from resolvable accuracy."
        ),
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
            collapsed = run.script == "pinyin_diacritic" and item["collapsed_diacritic"]
            if collapsed:
                rows.append(
                    {
                        "id": item["id"],
                        "model_run": run.run_name,
                        "script": run.script,
                        "scoring_mode": mode,
                        "status": "collapsed",
                        "collapsed_diacritic": True,
                        "correct": "",
                        "prediction": "",
                        "gold_score": "",
                        "distractor_score": "",
                        "margin": "",
                        "gold_token_count": "",
                        "distractor_token_count": "",
                        "collision_key": item["collision_key"],
                        "gold_zh": item["gold_zh"],
                        "distractor_zh": item["distractor_zh"],
                        "source_line": item["source_line"],
                    }
                )
                continue
            prefix, gold_completion, distractor_completion = completion_parts(item, run, mode)
            gold_score, gold_count = score_completion(model, tokenizer, device, prefix, gold_completion)
            distractor_score, distractor_count = score_completion(
                model, tokenizer, device, prefix, distractor_completion
            )
            margin = gold_score - distractor_score
            if not math.isfinite(margin):
                raise ValueError(f"Non-finite margin for {run.run_name}/{mode}/{item['id']}")
            rows.append(
                {
                    "id": item["id"],
                    "model_run": run.run_name,
                    "script": run.script,
                    "scoring_mode": mode,
                    "status": "scored",
                    "collapsed_diacritic": False,
                    "correct": int(margin > 0),
                    "prediction": item["gold_zh"] if margin > 0 else item["distractor_zh"],
                    "gold_score": gold_score,
                    "distractor_score": distractor_score,
                    "margin": margin,
                    "gold_token_count": gold_count,
                    "distractor_token_count": distractor_count,
                    "collision_key": item["collision_key"],
                    "gold_zh": item["gold_zh"],
                    "distractor_zh": item["distractor_zh"],
                    "source_line": item["source_line"],
                }
            )
    summaries = [
        summarize(
            [row for row in rows if row["scoring_mode"] == mode],
            run,
            mode,
            device.type,
            dtype_name,
            args.bootstrap_samples,
            args.seed,
        )
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


def read_item_scores(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def mean_or_blank(values: list[float]) -> float | str:
    return sum(values) / len(values) if values else ""


def accuracy_or_blank(rows: list[dict[str, Any]]) -> float | str:
    scored = [row for row in rows if row["status"] == "scored" and row["correct"] != ""]
    if not scored:
        return ""
    return sum(float(row["correct"]) for row in scored) / len(scored)


def margin_mean(rows: list[dict[str, Any]]) -> float | str:
    margins = [
        float(row["margin"])
        for row in rows
        if row["status"] == "scored" and row.get("margin") not in ("", None)
    ]
    return mean_or_blank(margins)


def build_matched_subset_summary(items: list[dict[str, Any]], score_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    collapsed_ids = {item["id"] for item in items if item["collapsed_diacritic"]}
    noncollapsed_ids = {item["id"] for item in items if not item["collapsed_diacritic"]}
    all_ids = {item["id"] for item in items}
    n_all = len(all_ids)
    n_collapsed = len(collapsed_ids)
    n_noncollapsed = len(noncollapsed_ids)

    rows_by_mode_model: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in score_rows:
        rows_by_mode_model[(row["scoring_mode"], row["model_run"])].append(row)

    output: list[dict[str, Any]] = []
    for mode in SCORING_MODES:
        models = sorted(model for scoring_mode, model in rows_by_mode_model if scoring_mode == mode)
        for model in models:
            model_rows = rows_by_mode_model[(mode, model)]
            by_id = {row["id"]: row for row in model_rows}
            missing = sorted(all_ids - set(by_id))
            if missing:
                raise ValueError(f"Missing item scores for {model}/{mode}: {missing[:5]}")

            script = model_rows[0]["script"] if model_rows else ""
            all_rows = [by_id[item_id] for item_id in all_ids]
            noncollapsed_rows = [by_id[item_id] for item_id in noncollapsed_ids]
            collapsed_rows = [by_id[item_id] for item_id in collapsed_ids]
            scored_all = [row for row in all_rows if row["status"] == "scored"]
            correct_all_scored = sum(float(row["correct"]) for row in scored_all if row["correct"] != "")

            if script == "pinyin_diacritic":
                all_item_accuracy = (correct_all_scored + 0.5 * n_collapsed) / n_all if n_all else ""
                chance_adjusted = all_item_accuracy
                collapsed_subset_accuracy: float | str = 0.5 if n_collapsed else ""
                collapsed_margin_for_chinese: float | str = ""
                notes = (
                    "Diacritic collapsed items are unresolvable; all-item accuracy is chance-adjusted "
                    "with 0.5 credit per collapsed two-choice item."
                )
            else:
                all_item_accuracy = accuracy_or_blank(all_rows)
                chance_adjusted = all_item_accuracy
                collapsed_subset_accuracy = accuracy_or_blank(collapsed_rows)
                collapsed_margin_for_chinese = margin_mean(collapsed_rows)
                notes = (
                    "Chinese is scored on all items; collapsed/noncollapsed subsets are defined by "
                    "the Diacritic surface collision flag for matched comparison."
                )

            output.append(
                {
                    "scoring_mode": mode,
                    "model": model,
                    "n_all": n_all,
                    "n_collapsed": n_collapsed,
                    "n_noncollapsed": n_noncollapsed,
                    "all_item_accuracy": all_item_accuracy,
                    "chance_adjusted_all_accuracy": chance_adjusted,
                    "noncollapsed_subset_accuracy": accuracy_or_blank(noncollapsed_rows),
                    "collapsed_subset_accuracy": collapsed_subset_accuracy,
                    "mean_margin_all_scored": margin_mean(all_rows),
                    "mean_margin_noncollapsed": margin_mean(noncollapsed_rows),
                    "mean_margin_collapsed_for_chinese": collapsed_margin_for_chinese,
                    "notes": notes,
                }
            )
    return output


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    root = Path.cwd()
    probe_path = project_path(root, args.probe_jsonl)
    output_dir = project_path(root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    items = load_items(probe_path)
    print(f"probe items: {len(items)}")
    print(f"collapsed items in dataset: {sum(1 for item in items if item['collapsed_diacritic'])}")
    item_scores_path = project_path(root, args.item_scores_csv) if args.item_scores_csv else output_dir / "item_scores.csv"
    if args.aggregate_only:
        score_rows = read_item_scores(item_scores_path)
        matched_rows = build_matched_subset_summary(items, score_rows)
        matched_path = output_dir / "summary_matched_subsets.csv"
        write_csv(matched_path, matched_rows, MATCHED_SUBSET_FIELDS)
        print(f"read: {item_scores_path}")
        print(f"wrote: {matched_path}")
        return

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
    matched_rows = build_matched_subset_summary(items, all_rows)
    write_csv(output_dir / "summary_matched_subsets.csv", matched_rows, MATCHED_SUBSET_FIELDS)
    print(f"wrote: {output_dir / 'item_scores.csv'}")
    print(f"wrote: {output_dir / 'summary_by_model_and_scoring.csv'}")
    print(f"wrote: {output_dir / 'summary_matched_subsets.csv'}")

    print("\nSummary")
    for row in all_summaries:
        print(
            f"{row['model_run']} {row['scoring_mode']}: "
            f"collapse={row['collapse_rate']} resolvable_acc={row['resolvable_accuracy']} "
            f"chance_adj={row['chance_adjusted_all_accuracy']} margin={row['mean_margin']}"
        )

    primary = {row["model_run"]: row for row in all_summaries if row["scoring_mode"] == "candidate_plus_suffix"}
    mt = primary.get("diacritic_matched_token_4epoch", {})
    ch = primary.get("chinese_4epoch", {})
    if ch and mt:
        print(
            "Chinese vs Diacritic primary resolvable accuracy: "
            f"{ch.get('resolvable_accuracy')} vs {mt.get('resolvable_accuracy')}"
        )


if __name__ == "__main__":
    main()
