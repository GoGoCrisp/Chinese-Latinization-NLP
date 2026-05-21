#!/usr/bin/env python3
"""Bootstrap CIs and paired significance tests for robustness eval item scores."""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "eval_results" / "robustness_ci"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--homophone", default="eval_results/eval2/homophone_probe_v2/item_scores.csv")
    parser.add_argument("--hard-control", default="eval_results/eval2/nonhomophone_control_v2/item_scores.csv")
    parser.add_argument("--easy-control", default="eval_results/eval2/easy_random_control_v2/item_scores.csv")
    parser.add_argument("--zhoblimp", default="eval_results/eval4_chinese_blimp_style/item_scores.csv")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--chinese-model", default="chinese_4epoch")
    parser.add_argument("--pinyin-model", default="diacritic_matched_token_4epoch")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--permutation-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def truthy(value: str | None) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def model_kind(model_run: str, chinese_model: str, pinyin_model: str) -> str | None:
    if model_run == chinese_model:
        return "chinese"
    if model_run == pinyin_model:
        return "pinyin"
    return None


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


def paired_items(
    rows: Iterable[dict[str, str]],
    chinese_model: str,
    pinyin_model: str,
    item_col: str = "id",
) -> list[tuple[str, int, int]]:
    by_item: dict[str, dict[str, int]] = defaultdict(dict)
    for row in rows:
        if row.get("status", "scored") != "scored":
            continue
        kind = model_kind(row.get("model_run", ""), chinese_model, pinyin_model)
        if kind is None:
            continue
        item_id = row.get(item_col, "")
        if not item_id:
            continue
        by_item[item_id][kind] = int(float(row.get("correct", "0") or 0))
    pairs = []
    for item_id, values in by_item.items():
        if "chinese" in values and "pinyin" in values:
            pairs.append((item_id, values["chinese"], values["pinyin"]))
    pairs.sort(key=lambda x: x[0])
    return pairs


def bootstrap_ci(
    pairs: list[tuple[str, int, int]],
    samples: int,
    rng: random.Random,
) -> dict[str, float]:
    n = len(pairs)
    if n == 0:
        return {
            "n_items": 0,
            "chinese_acc": math.nan,
            "pinyin_acc": math.nan,
            "gap": math.nan,
            "chinese_ci_low": math.nan,
            "chinese_ci_high": math.nan,
            "pinyin_ci_low": math.nan,
            "pinyin_ci_high": math.nan,
            "gap_ci_low": math.nan,
            "gap_ci_high": math.nan,
        }
    chinese = [c for _, c, _ in pairs]
    pinyin = [p for _, _, p in pairs]
    obs_c = sum(chinese) / n
    obs_p = sum(pinyin) / n
    obs_gap = obs_c - obs_p
    if samples <= 0:
        boot_c = np.array([obs_c])
        boot_p = np.array([obs_p])
        boot_gap = np.array([obs_gap])
    else:
        np_rng = np.random.default_rng(rng.randrange(2**32))
        # Resampling paired binary outcomes is equivalent to drawing bootstrap
        # counts from the observed four-cell empirical distribution.
        cell_order = [(0, 0), (0, 1), (1, 0), (1, 1)]
        counts = np.array(
            [sum(1 for _, c, p in pairs if (c, p) == cell) for cell in cell_order],
            dtype=np.int64,
        )
        sampled = np_rng.multinomial(n, counts / n, size=samples)
        boot_c = (sampled[:, 2] + sampled[:, 3]) / n
        boot_p = (sampled[:, 1] + sampled[:, 3]) / n
        boot_gap = boot_c - boot_p
    return {
        "n_items": n,
        "chinese_acc": obs_c,
        "pinyin_acc": obs_p,
        "gap": obs_gap,
        "chinese_ci_low": float(np.quantile(boot_c, 0.025)),
        "chinese_ci_high": float(np.quantile(boot_c, 0.975)),
        "pinyin_ci_low": float(np.quantile(boot_p, 0.025)),
        "pinyin_ci_high": float(np.quantile(boot_p, 0.975)),
        "gap_ci_low": float(np.quantile(boot_gap, 0.025)),
        "gap_ci_high": float(np.quantile(boot_gap, 0.975)),
    }


def paired_permutation_pvalue(
    pairs: list[tuple[str, int, int]],
    samples: int,
    rng: random.Random,
) -> float:
    diffs = [c - p for _, c, p in pairs]
    diffs = [d for d in diffs if d != 0]
    if not diffs:
        return 1.0
    observed = abs(sum(diffs))
    np_rng = np.random.default_rng(rng.randrange(2**32))
    positive = np_rng.binomial(len(diffs), 0.5, size=samples)
    stats = np.abs((2 * positive) - len(diffs))
    extreme = int(np.count_nonzero(stats >= observed))
    return (extreme + 1) / (samples + 1)


def mcnemar_optional(pairs: list[tuple[str, int, int]]) -> float | None:
    b = sum(1 for _, c, p in pairs if c == 1 and p == 0)
    c = sum(1 for _, c1, p in pairs if c1 == 0 and p == 1)
    if b + c == 0:
        return 1.0
    try:
        from statsmodels.stats.contingency_tables import mcnemar
    except ImportError:
        return None
    result = mcnemar([[0, b], [c, 0]], exact=False, correction=True)
    return float(result.pvalue)


def chance_adjusted(acc: float, chance: float = 0.5) -> float:
    if math.isnan(acc):
        return math.nan
    return (acc - chance) / (1.0 - chance)


def emit_bootstrap(
    records: list[dict[str, object]],
    dataset: str,
    subset: str,
    scoring_mode: str,
    pairs: list[tuple[str, int, int]],
    args: argparse.Namespace,
    rng: random.Random,
    chance_adjust: bool = False,
) -> None:
    stats = bootstrap_ci(pairs, args.bootstrap_samples, rng)
    row: dict[str, object] = {
        "dataset": dataset,
        "subset": subset,
        "scoring_mode": scoring_mode,
        **stats,
    }
    if chance_adjust:
        row["chinese_chance_adjusted"] = chance_adjusted(float(stats["chinese_acc"]))
        row["pinyin_chance_adjusted"] = chance_adjusted(float(stats["pinyin_acc"]))
        row["gap_chance_adjusted"] = row["chinese_chance_adjusted"] - row["pinyin_chance_adjusted"]
    records.append(row)


def emit_test(
    records: list[dict[str, object]],
    dataset: str,
    subset: str,
    scoring_mode: str,
    pairs: list[tuple[str, int, int]],
    args: argparse.Namespace,
    rng: random.Random,
) -> None:
    b = sum(1 for _, c, p in pairs if c == 1 and p == 0)
    c = sum(1 for _, c1, p in pairs if c1 == 0 and p == 1)
    records.append(
        {
            "dataset": dataset,
            "subset": subset,
            "scoring_mode": scoring_mode,
            "n_items": len(pairs),
            "chinese_only_correct": b,
            "pinyin_only_correct": c,
            "paired_permutation_p": paired_permutation_pvalue(pairs, args.permutation_samples, rng),
            "mcnemar_p": mcnemar_optional(pairs),
        }
    )


def scoring_modes(rows: list[dict[str, str]]) -> list[str]:
    modes = sorted({row.get("scoring_mode", "overall") or "overall" for row in rows})
    return modes or ["overall"]


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
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


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir

    boot_rows: list[dict[str, object]] = []
    test_rows: list[dict[str, object]] = []
    notes: list[str] = []

    probe_specs = [
        ("homophone", Path(args.homophone), "homophone noncollapsed", lambda r: not truthy(r.get("collapsed_diacritic"))),
        ("homophone", Path(args.homophone), "homophone collapsed/chance-adjusted", lambda r: truthy(r.get("collapsed_diacritic"))),
        ("hard_nonhomophone_control", Path(args.hard_control), "hard non-homophone control", lambda r: True),
        ("easy_nonhomophone_control", Path(args.easy_control), "easy non-homophone control", lambda r: True),
    ]
    for dataset, path_text, subset, predicate in probe_specs:
        path = path_text if path_text.is_absolute() else ROOT / path_text
        rows = read_csv(path)
        if not rows:
            notes.append(f"Missing or empty: {path.relative_to(ROOT)}")
            continue
        for mode in scoring_modes(rows):
            filtered = [r for r in rows if (r.get("scoring_mode", "overall") or "overall") == mode and predicate(r)]
            pairs = paired_items(filtered, args.chinese_model, args.pinyin_model)
            emit_bootstrap(
                boot_rows,
                dataset,
                subset,
                mode,
                pairs,
                args,
                rng,
                chance_adjust="chance-adjusted" in subset,
            )
            emit_test(test_rows, dataset, subset, mode, pairs, args, rng)

    zh_path = Path(args.zhoblimp)
    zh_path = zh_path if zh_path.is_absolute() else ROOT / zh_path
    zh_rows = read_csv(zh_path)
    if zh_rows:
        all_phenomena = sorted({row.get("phenomenon", "unknown") or "unknown" for row in zh_rows})
        subsets: list[tuple[str, list[dict[str, str]]]] = [("overall", zh_rows)]
        subsets.extend((f"phenomenon={ph}", [r for r in zh_rows if (r.get("phenomenon") or "unknown") == ph]) for ph in all_phenomena)
        subsets.append(("Anaphor all", [r for r in zh_rows if (r.get("phenomenon") or "").lower() == "anaphor"]))
        if any("collapsed_diacritic" in r or "collapsed" in r for r in zh_rows):
            subsets.append(
                (
                    "Anaphor noncollapsed",
                    [
                        r
                        for r in zh_rows
                        if (r.get("phenomenon") or "").lower() == "anaphor"
                        and not truthy(r.get("collapsed_diacritic") or r.get("collapsed"))
                    ],
                )
            )
        else:
            notes.append("ZhoBLiMP item_scores has no collapsed flag; Anaphor noncollapsed subset was not emitted.")
        for subset, subset_rows in subsets:
            pairs = paired_items(subset_rows, args.chinese_model, args.pinyin_model)
            emit_bootstrap(boot_rows, "ZhoBLiMP", subset, "minimal_pair", pairs, args, rng)
            emit_test(test_rows, "ZhoBLiMP", subset, "minimal_pair", pairs, args, rng)
    else:
        notes.append(f"Missing or empty: {zh_path.relative_to(ROOT)}")

    write_rows(out_dir / "bootstrap_summary.csv", boot_rows)
    write_rows(out_dir / "significance_tests.csv", test_rows)
    report = [
        "# Robustness Bootstrap And Significance Report",
        "",
        f"- Chinese model: `{args.chinese_model}`",
        f"- Pinyin model: `{args.pinyin_model}`",
        f"- Bootstrap samples: {args.bootstrap_samples}",
        f"- Permutation samples: {args.permutation_samples}",
        f"- Bootstrap output: `bootstrap_summary.csv`",
        f"- Significance output: `significance_tests.csv`",
        "",
        "## Notes",
        "",
    ]
    report.extend(f"- {note}" for note in notes)
    if not notes:
        report.append("- No warnings.")
    (out_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote {out_dir / 'bootstrap_summary.csv'}")
    print(f"Wrote {out_dir / 'significance_tests.csv'}")
    print(f"Wrote {out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
