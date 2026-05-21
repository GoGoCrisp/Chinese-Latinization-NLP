#!/usr/bin/env python3
"""Generate an integrated Markdown report for eval2."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = "eval_results/eval2/eval2_integrated_report.md"
PRIMARY_MODE = "candidate_plus_suffix"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an integrated eval2 Markdown report.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--root", default=".")
    return parser.parse_args()


def project_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    return float(value)


def pct(value: Any) -> str:
    number = as_float(value)
    if number is None:
        return ""
    return f"{number * 100:.2f}%"


def pp(value: Any) -> str:
    number = as_float(value)
    if number is None:
        return ""
    return f"{number * 100:+.2f} pp"


def number(value: Any, digits: int = 4) -> str:
    num = as_float(value)
    if num is None:
        return ""
    return f"{num:.{digits}f}"


def csv_row(rows: list[dict[str, str]], **match: str) -> dict[str, str]:
    for row in rows:
        if all(row.get(key) == value for key, value in match.items()):
            return row
    raise KeyError(f"Missing row matching {match}")


def item_score_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def add_table(lines: list[str], headers: list[str], rows: list[list[Any]]) -> None:
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    lines.append("")


def dataset_rows(root: Path) -> list[list[Any]]:
    homophone_meta = read_json(root / "eval_data/homophone_probe_v2/probe_v2_build_meta.json")
    hard_meta = read_json(root / "eval_data/nonhomophone_control_v2/nonhomophone_control_v2_build_meta.json")
    easy_meta = read_json(root / "eval_data/easy_random_control_v2/easy_random_control_v2_build_meta.json")
    return [
        [
            "Homophone Probe v2",
            homophone_meta["n_items"],
            homophone_meta["collapsed_items"],
            "",
            homophone_meta["items_with_embedding_distance"],
            homophone_meta["no_train_data_used"],
        ],
        [
            "Hard Non-Homophone Control v2",
            hard_meta["items_built"],
            "",
            hard_meta["invalid_control_collision_count"],
            hard_meta["distance_summary"]["count"],
            hard_meta["no_train_data_used"],
        ],
        [
            "Easy Random Non-Homophone Control v2",
            easy_meta["items_built"],
            "",
            easy_meta["invalid_control_collision_count"],
            easy_meta["distance_summary"]["count"],
            easy_meta["no_train_data_used"],
        ],
    ]


def build_report(root: Path) -> str:
    homophone = read_csv(root / "eval_results/eval2/homophone_probe_v2/summary_matched_subsets.csv")
    hard_gap = read_csv(root / "eval_results/eval2/nonhomophone_control_v2/homophone_vs_control_gap.csv")
    three_gap = read_csv(root / "eval_results/eval2/easy_random_control_v2/three_probe_gap_comparison.csv")
    hard_summary = read_csv(root / "eval_results/eval2/nonhomophone_control_v2/summary_by_model_and_scoring.csv")
    easy_summary = read_csv(root / "eval_results/eval2/easy_random_control_v2/summary_by_model_and_scoring.csv")

    primary_h_ch = csv_row(homophone, scoring_mode=PRIMARY_MODE, model="chinese_4epoch")
    primary_h_di = csv_row(homophone, scoring_mode=PRIMARY_MODE, model="diacritic_matched_token_4epoch")
    primary_gap = csv_row(three_gap, scoring_mode=PRIMARY_MODE)
    candidate_gap = csv_row(three_gap, scoring_mode="candidate_only")

    hard_ch = csv_row(hard_summary, scoring_mode=PRIMARY_MODE, model_run="chinese_4epoch")
    hard_di = csv_row(hard_summary, scoring_mode=PRIMARY_MODE, model_run="diacritic_matched_token_4epoch")
    easy_ch = csv_row(easy_summary, scoring_mode=PRIMARY_MODE, model_run="chinese_4epoch")
    easy_di = csv_row(easy_summary, scoring_mode=PRIMARY_MODE, model_run="diacritic_matched_token_4epoch")

    score_rows = [
        [
            "Homophone Probe v2",
            item_score_count(root / "eval_results/eval2/homophone_probe_v2/item_scores.csv"),
        ],
        [
            "Hard Non-Homophone Control v2",
            item_score_count(root / "eval_results/eval2/nonhomophone_control_v2/item_scores.csv"),
        ],
        [
            "Easy Random Non-Homophone Control v2",
            item_score_count(root / "eval_results/eval2/easy_random_control_v2/item_scores.csv"),
        ],
    ]

    lines: list[str] = []
    lines.append("# Eval2 整合报告")
    lines.append("")
    lines.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## 范围")
    lines.append("")
    lines.append(
        "Eval2 包括 Homophone Probe v2、Hard Non-Homophone Control v2、"
        "Easy Random Non-Homophone Control v2。主分析使用 `candidate_plus_suffix`；"
        "`candidate_only` 作为辅助诊断。"
    )
    lines.append("")
    lines.append("模型：")
    lines.append("")
    lines.append("- `chinese_4epoch`：Chinese 4epoch 模型。")
    lines.append("- `diacritic_matched_token_4epoch`：Diacritic matched-token 4epoch 模型。")
    lines.append("")

    lines.append("## 数据集构建概览")
    lines.append("")
    add_table(
        lines,
        ["数据集", "条目数", "Collapsed", "无效 control collision", "有距离信息条目", "未使用 train 数据"],
        dataset_rows(root),
    )

    lines.append("## 结果文件检查")
    lines.append("")
    lines.append("每个 1000-item 结果应有 4000 行得分：1000 items x 2 models x 2 scoring modes。")
    lines.append("")
    add_table(lines, ["结果", "得分行数"], score_rows)

    lines.append("## 主结果：candidate_plus_suffix")
    lines.append("")
    add_table(
        lines,
        [
            "Probe",
            "Chinese accuracy",
            "Diacritic accuracy",
            "Chinese-Diacritic gap",
            "Homophone 减 probe gap",
        ],
        [
            [
                "Homophone Probe v2 noncollapsed subset",
                pct(primary_h_ch["noncollapsed_subset_accuracy"]),
                pct(primary_h_di["noncollapsed_subset_accuracy"]),
                pp(primary_gap["homophone_gap"]),
                "",
            ],
            [
                "Hard Non-Homophone Control v2",
                pct(hard_ch["accuracy"]),
                pct(hard_di["accuracy"]),
                pp(primary_gap["hard_control_gap"]),
                pp(primary_gap["homophone_minus_hard_gap"]),
            ],
            [
                "Easy Random Non-Homophone Control v2",
                pct(easy_ch["accuracy"]),
                pct(easy_di["accuracy"]),
                pp(primary_gap["easy_control_gap"]),
                pp(primary_gap["homophone_minus_easy_gap"]),
            ],
        ],
    )

    lines.append("Homophone matched-subset 细节：")
    lines.append("")
    add_table(
        lines,
        [
            "模型",
            "All-item accuracy",
            "Chance-adjusted all accuracy",
            "Noncollapsed accuracy",
            "Collapsed accuracy",
            "Mean margin all scored",
        ],
        [
            [
                "Chinese",
                pct(primary_h_ch["all_item_accuracy"]),
                pct(primary_h_ch["chance_adjusted_all_accuracy"]),
                pct(primary_h_ch["noncollapsed_subset_accuracy"]),
                pct(primary_h_ch["collapsed_subset_accuracy"]),
                number(primary_h_ch["mean_margin_all_scored"]),
            ],
            [
                "Diacritic matched-token",
                pct(primary_h_di["all_item_accuracy"]),
                pct(primary_h_di["chance_adjusted_all_accuracy"]),
                pct(primary_h_di["noncollapsed_subset_accuracy"]),
                pct(primary_h_di["collapsed_subset_accuracy"]),
                number(primary_h_di["mean_margin_all_scored"]),
            ],
        ],
    )

    lines.append("## 辅助诊断：candidate_only")
    lines.append("")
    add_table(
        lines,
        [
            "Probe",
            "Chinese accuracy",
            "Diacritic accuracy",
            "Chinese-Diacritic gap",
            "Homophone 减 probe gap",
        ],
        [
            [
                "Homophone Probe v2 noncollapsed subset",
                pct(candidate_gap["homophone_chinese_noncollapsed_accuracy"]),
                pct(candidate_gap["homophone_diacritic_noncollapsed_accuracy"]),
                pp(candidate_gap["homophone_gap"]),
                "",
            ],
            [
                "Hard Non-Homophone Control v2",
                pct(candidate_gap["hard_control_chinese_accuracy"]),
                pct(candidate_gap["hard_control_diacritic_accuracy"]),
                pp(candidate_gap["hard_control_gap"]),
                pp(candidate_gap["homophone_minus_hard_gap"]),
            ],
            [
                "Easy Random Non-Homophone Control v2",
                pct(candidate_gap["easy_control_chinese_accuracy"]),
                pct(candidate_gap["easy_control_diacritic_accuracy"]),
                pp(candidate_gap["easy_control_gap"]),
                pp(candidate_gap["homophone_minus_easy_gap"]),
            ],
        ],
    )

    lines.append("## 解读")
    lines.append("")
    lines.append(
        f"- 主分析中 Homophone noncollapsed gap 为 {pp(primary_gap['homophone_gap'])}："
        f"Chinese {pct(primary_h_ch['noncollapsed_subset_accuracy'])} vs Diacritic "
        f"{pct(primary_h_di['noncollapsed_subset_accuracy'])}."
    )
    lines.append(
        f"- Hard-control gap 为 {pp(primary_gap['hard_control_gap'])}；"
        f"Homophone 减 hard-control gap 为 {pp(primary_gap['homophone_minus_hard_gap'])}。"
    )
    lines.append(
        f"- Easy-control gap 为 {pp(primary_gap['easy_control_gap'])}；"
        f"Homophone 减 easy-control gap 为 {pp(primary_gap['homophone_minus_easy_gap'])}。"
    )
    lines.append(
        "- Homophone 比较使用两个模型共同的 noncollapsed matched subset，"
        "避免把 Chinese all-item accuracy 和 Diacritic 中不可分辨的 collapsed items 错位比较。"
    )
    lines.append("")

    lines.append("## 来源文件")
    lines.append("")
    lines.append("- `eval_results/eval2/homophone_probe_v2/summary_matched_subsets.csv`")
    lines.append("- `eval_results/eval2/nonhomophone_control_v2/homophone_vs_control_gap.csv`")
    lines.append("- `eval_results/eval2/easy_random_control_v2/three_probe_gap_comparison.csv`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    root = project_path(Path.cwd(), args.root)
    output = project_path(root, args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = build_report(root)
    output.write_text(report + "\n", encoding="utf-8")
    print(f"wrote: {output}")


if __name__ == "__main__":
    main()
