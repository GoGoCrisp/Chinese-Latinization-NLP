#!/usr/bin/env python3
"""Summarize robustness Eval 1/2/4 outputs with seed means and bootstrap CIs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "eval_results" / "robustness_134m_eval" / "summary"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppl-summary", default="eval_results/robustness_134m_eval/eval1_ppl/summary.csv")
    parser.add_argument("--homophone", default="eval_results/robustness_134m_eval/eval2_homophone/item_scores.csv")
    parser.add_argument("--hard-control", default="eval_results/robustness_134m_eval/eval2_hard_control/item_scores.csv")
    parser.add_argument("--easy-control", default="eval_results/robustness_134m_eval/eval2_easy_control/item_scores.csv")
    parser.add_argument("--zhoblimp", default="eval_results/robustness_134m_eval/eval4_zhoblimp/item_scores.csv")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--matched-token-seeds", default="43,44")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260514)
    return parser.parse_args()


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


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


def truthy(value: str | None) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def infer_model(model_run: str) -> dict[str, object]:
    text = model_run.lower()
    representation = "Chinese-Origin" if "chinese" in text else "Pinyin-Diacritic" if "diacritic" in text else "unknown"
    regime = "matched-data" if "matched_data" in text or "fixed_data" in text else "matched-token" if "matched_token" in text else "unknown"
    seed_match = re.search(r"seed(\d+)", text)
    seed = int(seed_match.group(1)) if seed_match else None
    return {"representation": representation, "regime": regime, "seed": seed}


def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def bootstrap_ci(values: list[float], samples: int, rng: random.Random) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    if samples <= 0:
        value = sum(values) / len(values)
        return value, value

    unique_values = sorted(set(values))
    if len(unique_values) == 1:
        return unique_values[0], unique_values[0]

    np_rng = np.random.default_rng(rng.randrange(2**32))
    n = len(values)
    counts = np.array([sum(1 for value in values if value == unique) for unique in unique_values], dtype=np.int64)
    support = np.array(unique_values, dtype=np.float64)

    if len(unique_values) == 2:
        high_count = int(counts[1])
        high_draws = np_rng.binomial(n, high_count / n, size=samples)
        stats = support[0] + (support[1] - support[0]) * (high_draws / n)
    else:
        sampled_counts = np_rng.multinomial(n, counts / n, size=samples)
        stats = sampled_counts @ support / n
    return float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))


def append_metric(
    rows: list[dict[str, object]],
    dataset: str,
    subset: str,
    metric: str,
    model_run: str,
    scoring_mode: str,
    values: list[float],
    bootstrap_samples: int,
    rng: random.Random,
) -> None:
    meta = infer_model(model_run)
    if not values:
        return
    ci_low, ci_high = bootstrap_ci(values, bootstrap_samples, rng)
    rows.append(
        {
            "dataset": dataset,
            "subset": subset,
            "metric": metric,
            "model_run": model_run,
            "representation": meta["representation"],
            "regime": meta["regime"],
            "seed": meta["seed"],
            "scoring_mode": scoring_mode,
            "n_items": len(values),
            "value": sum(values) / len(values),
            "ci_low": ci_low,
            "ci_high": ci_high,
        }
    )


def summarize_values(values: Iterable[float]) -> dict[str, object]:
    vals = list(values)
    if not vals:
        return {"n_seeds": 0, "mean": math.nan, "std": math.nan, "min": math.nan, "max": math.nan}
    return {
        "n_seeds": len(vals),
        "mean": mean(vals),
        "std": stdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
    }


def add_ppl(rows: list[dict[str, object]], summary_rows: list[dict[str, str]]) -> None:
    for row in summary_rows:
        model_run = row.get("model") or row.get("model_run") or ""
        if not model_run:
            continue
        meta = infer_model(model_run)
        value_text = row.get("char_ppl") or row.get("per_character_ppl")
        if value_text in (None, ""):
            continue
        rows.append(
            {
                "dataset": "Eval1 PPL",
                "subset": "linelevel",
                "metric": "char_ppl",
                "model_run": model_run,
                "representation": meta["representation"],
                "regime": meta["regime"],
                "seed": meta["seed"],
                "scoring_mode": "linelevel",
                "n_items": row.get("scored_lines", ""),
                "value": float(value_text),
                "ci_low": "",
                "ci_high": "",
            }
        )


def add_homophone(rows: list[dict[str, object]], item_rows: list[dict[str, str]], args: argparse.Namespace, rng: random.Random) -> None:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in item_rows:
        grouped[(row.get("model_run", ""), row.get("scoring_mode", "overall") or "overall")].append(row)
    for (model_run, mode), group in grouped.items():
        if not model_run:
            continue
        all_scored = [float(r["correct"]) for r in group if r.get("status") == "scored" and r.get("correct") not in ("", None)]
        noncollapsed = [
            float(r["correct"])
            for r in group
            if r.get("status") == "scored" and not truthy(r.get("collapsed_diacritic")) and r.get("correct") not in ("", None)
        ]
        collapsed_scored = [
            float(r["correct"])
            for r in group
            if r.get("status") == "scored" and truthy(r.get("collapsed_diacritic")) and r.get("correct") not in ("", None)
        ]
        collapsed_count = sum(1 for r in group if truthy(r.get("collapsed_diacritic")) or r.get("status") == "collapsed")
        if all_scored:
            append_metric(rows, "Eval2 Homophone Probe", "resolvable_scored", "accuracy", model_run, mode, all_scored, args.bootstrap_samples, rng)
        if noncollapsed:
            append_metric(rows, "Eval2 Homophone Probe", "noncollapsed", "accuracy", model_run, mode, noncollapsed, args.bootstrap_samples, rng)
        if collapsed_scored:
            append_metric(rows, "Eval2 Homophone Probe", "collapsed_chinese_scored", "accuracy", model_run, mode, collapsed_scored, args.bootstrap_samples, rng)
        if collapsed_count:
            chance_adjusted = [*all_scored, *([0.5] * (collapsed_count if infer_model(model_run)["representation"] == "Pinyin-Diacritic" else 0))]
            if chance_adjusted:
                append_metric(rows, "Eval2 Homophone Probe", "all_chance_adjusted", "accuracy", model_run, mode, chance_adjusted, args.bootstrap_samples, rng)


def add_control(
    rows: list[dict[str, object]],
    item_rows: list[dict[str, str]],
    dataset: str,
    args: argparse.Namespace,
    rng: random.Random,
) -> None:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in item_rows:
        if row.get("status", "scored") != "scored" or row.get("correct") in ("", None):
            continue
        grouped[(row.get("model_run", ""), row.get("scoring_mode", "overall") or "overall")].append(float(row["correct"]))
    for (model_run, mode), values in grouped.items():
        append_metric(rows, dataset, "control", "accuracy", model_run, mode, values, args.bootstrap_samples, rng)


def zho_collapsed(row: dict[str, str]) -> bool:
    flags = row.get("quality_flags") or "[]"
    try:
        parsed = json.loads(flags)
    except json.JSONDecodeError:
        parsed = []
    return "identical_diacritic" in parsed or truthy(row.get("collapsed_diacritic") or row.get("collapsed"))


def add_zhoblimp(rows: list[dict[str, object]], item_rows: list[dict[str, str]], args: argparse.Namespace, rng: random.Random) -> None:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in item_rows:
        grouped[row.get("model_run", "")].append(row)
    for model_run, group in grouped.items():
        if not model_run:
            continue
        finite = [r for r in group if not truthy(r.get("non_finite")) and r.get("correct") not in ("", None)]
        subsets = {
            "overall": finite,
            "noncollapsed": [r for r in finite if not zho_collapsed(r)],
            "collapsed": [r for r in finite if zho_collapsed(r)],
        }
        phenomena = sorted({r.get("phenomenon", "unknown") or "unknown" for r in finite})
        subsets.update({f"phenomenon={ph}": [r for r in finite if (r.get("phenomenon") or "unknown") == ph] for ph in phenomena})
        for subset, subset_rows in subsets.items():
            values = [float(r["correct"]) for r in subset_rows]
            if values:
                append_metric(rows, "Eval4 ZhoBLiMP", subset, "accuracy", model_run, "minimal_pair", values, args.bootstrap_samples, rng)


def aggregate(rows: list[dict[str, object]], seeds: set[int]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[float]] = defaultdict(list)
    keys = ["dataset", "subset", "metric", "representation", "regime", "scoring_mode"]
    for row in rows:
        if row.get("regime") != "matched-token" or row.get("seed") not in seeds:
            continue
        grouped[tuple(row[k] for k in keys)].append(float(row["value"]))
    output = []
    for key, values in sorted(grouped.items()):
        item = dict(zip(keys, key))
        item.update(summarize_values(values))
        output.append(item)
    return output


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    out_dir = project_path(args.out_dir)
    matched_token_seeds = {int(value) for value in args.matched_token_seeds.split(",") if value.strip()}

    metric_rows: list[dict[str, object]] = []
    add_ppl(metric_rows, read_csv(project_path(args.ppl_summary)))
    add_homophone(metric_rows, read_csv(project_path(args.homophone)), args, rng)
    add_control(metric_rows, read_csv(project_path(args.hard_control)), "Eval2 Hard Non-Homophone Control", args, rng)
    add_control(metric_rows, read_csv(project_path(args.easy_control)), "Eval2 Easy Non-Homophone Control", args, rng)
    add_zhoblimp(metric_rows, read_csv(project_path(args.zhoblimp)), args, rng)

    matched_summary = aggregate(metric_rows, matched_token_seeds)
    fixed_data = [row for row in metric_rows if row.get("regime") == "matched-data"]

    write_csv(out_dir / "per_run_metric_values_with_ci.csv", metric_rows)
    write_csv(out_dir / "matched_token_seed_summary.csv", matched_summary)
    write_csv(out_dir / "matched_data_seed42_dedicated_table.csv", fixed_data)
    (out_dir / "README.md").write_text(
        "\n".join(
            [
                "# Robustness 134M Eval Summary",
                "",
                "- `per_run_metric_values_with_ci.csv`: per-model Eval 1/2/4 metrics; Eval 2/4 rows include bootstrap CIs.",
                "- `matched_token_seed_summary.csv`: mean/std/min/max across matched-token seeds.",
                "- `matched_data_seed42_dedicated_table.csv`: Pinyin-Diacritic matched-data seed42 results kept separate.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote summary outputs in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
