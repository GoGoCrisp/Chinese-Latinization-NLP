#!/usr/bin/env python3
"""Aggregate robustness metrics across matched-token seeds and fixed-data control."""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "eval_results" / "robustness_seed_aggregate"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppl-summary", default="eval_results/eval1/normalized_ppl_4epoch_linelevel/summary.csv")
    parser.add_argument("--homophone", default="eval_results/eval2/homophone_probe_v2/item_scores.csv")
    parser.add_argument("--zhoblimp", default="eval_results/eval4_chinese_blimp_style/item_scores.csv")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def infer_model(model_run: str) -> tuple[str | None, str | None, int | None]:
    text = model_run.lower()
    if "matched_data" in text or "fixed_data" in text:
        regime = "fixed-data"
    elif "matched_token" in text or "4epoch" in text:
        regime = "matched-token"
    else:
        regime = None

    if "chinese" in text:
        representation = "Chinese-Origin"
    elif "diacritic" in text or "pinyin" in text:
        representation = "Pinyin-Diacritic"
    else:
        representation = None

    seed_match = re.search(r"seed(\d+)", text)
    if seed_match:
        seed = int(seed_match.group(1))
    elif model_run in {"chinese_4epoch", "diacritic_matched_token_4epoch"}:
        seed = 42
    else:
        seed = None
    return representation, regime, seed


def summarize(values: Iterable[float]) -> dict[str, float | int]:
    vals = [v for v in values if not math.isnan(v)]
    if not vals:
        return {"n": 0, "mean": math.nan, "std": math.nan, "min": math.nan, "max": math.nan}
    return {
        "n": len(vals),
        "mean": mean(vals),
        "std": stdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
    }


def aggregate_metric(rows: list[dict[str, object]], group_keys: list[str], value_key: str) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[float]] = defaultdict(list)
    for row in rows:
        key = tuple(row.get(k) for k in group_keys)
        grouped[key].append(float(row[value_key]))
    output = []
    for key, values in sorted(grouped.items()):
        item = dict(zip(group_keys, key))
        item.update(summarize(values))
        output.append(item)
    return output


def accuracy_rows(item_rows: list[dict[str, str]], dataset: str) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in item_rows:
        if row.get("status", "scored") != "scored":
            continue
        model = row.get("model_run", "")
        if not model:
            continue
        scoring = row.get("scoring_mode", "minimal_pair" if dataset == "ZhoBLiMP" else "overall") or "overall"
        grouped[(model, scoring)].append(int(float(row.get("correct", "0") or 0)))
    metrics = []
    for (model, scoring), values in grouped.items():
        representation, regime, seed = infer_model(model)
        if representation is None or seed is None:
            continue
        metrics.append(
            {
                "dataset": dataset,
                "metric": "accuracy",
                "model_run": model,
                "representation": representation,
                "regime": regime,
                "seed": seed,
                "scoring_mode": scoring,
                "value": sum(values) / len(values),
            }
        )
    return metrics


def ppl_rows(summary_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    metrics = []
    for row in summary_rows:
        model = row.get("model") or row.get("model_run") or ""
        representation, regime, seed = infer_model(model)
        if representation is None or seed is None:
            continue
        value_text = row.get("char_ppl") or row.get("per_character_ppl")
        if value_text is None:
            continue
        metrics.append(
            {
                "dataset": "per-character PPL",
                "metric": "char_ppl",
                "model_run": model,
                "representation": representation,
                "regime": regime,
                "seed": seed,
                "scoring_mode": "linelevel",
                "value": float(value_text),
            }
        )
    return metrics


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir

    metric_rows: list[dict[str, object]] = []
    for path_text, loader in [
        (args.ppl_summary, lambda rows: ppl_rows(rows)),
        (args.homophone, lambda rows: accuracy_rows(rows, "Homophone probe")),
        (args.zhoblimp, lambda rows: accuracy_rows(rows, "ZhoBLiMP")),
    ]:
        path = Path(path_text)
        path = path if path.is_absolute() else ROOT / path
        rows = read_csv(path)
        metric_rows.extend(loader(rows))

    matched = [
        row
        for row in metric_rows
        if row.get("regime") == "matched-token" and row.get("seed") in {42, 43, 44}
    ]
    fixed = [row for row in metric_rows if row.get("regime") == "fixed-data"]

    aggregate = aggregate_metric(
        matched,
        ["dataset", "metric", "representation", "regime", "scoring_mode"],
        "value",
    )
    fixed_report = aggregate_metric(
        fixed,
        ["dataset", "metric", "representation", "regime", "scoring_mode"],
        "value",
    )

    write_csv(out_dir / "seed_metric_values.csv", metric_rows)
    write_csv(out_dir / "matched_token_seed42_43_44_summary.csv", aggregate)
    write_csv(out_dir / "fixed_data_control_summary.csv", fixed_report)

    report = [
        "# Robustness Seed Aggregate",
        "",
        "- `matched_token_seed42_43_44_summary.csv` reports mean, std, min, and max across seeds 42/43/44 when available.",
        "- `fixed_data_control_summary.csv` reports the fixed-data Pinyin-Diacritic control separately.",
        "- `seed_metric_values.csv` preserves the per-model metric values used for aggregation.",
        "",
        "This script is intended for after the new seed eval item-score files are merged into the same CSV schemas.",
    ]
    (out_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote aggregate outputs in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
