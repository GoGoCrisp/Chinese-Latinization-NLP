#!/usr/bin/env python3
"""Plot seed42 training/evaluation loss curves for the 4-epoch runs."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR.parent
os.environ.setdefault("MPLCONFIGDIR", str(MODEL_DIR / ".matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


@dataclass(frozen=True)
class RunSpec:
    name: str
    log_path: Path
    color: str
    main_offset: float


RUNS = [
    RunSpec(
        "Chinese",
        MODEL_DIR
        / "server_outputs/4epoch/outputs/chinese_125m_b1024_4epoch_seed42/train_log.jsonl",
        "#1f77b4",
        1.2,
    ),
    RunSpec(
        "Pinyin-Diacritic (matched token)",
        MODEL_DIR
        / "server_outputs/4epoch/diacritic_125m_b1024_matched_token_4epoch_seed42_outputs/outputs/diacritic_125m_b1024_matched_token_4epoch_seed42/train_log.jsonl",
        "#ff7f0e",
        0.6,
    ),
    RunSpec(
        "Pinyin-Diacritic (matched data)",
        MODEL_DIR
        / "server_outputs/robustness/unpacked/diacritic_125m_b1024_matched_data_4epoch_seed42/outputs/diacritic_125m_b1024_matched_data_4epoch_seed42/train_log.jsonl",
        "#2ca02c",
        0.0,
    ),
]


def read_log(path: Path) -> dict[str, np.ndarray]:
    steps: list[int] = []
    train_loss: list[float] = []
    eval_steps: list[int] = []
    eval_loss: list[float] = []

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            steps.append(row["step"])
            train_loss.append(row["train_loss"])
            if row.get("eval_loss") is not None:
                eval_steps.append(row["step"])
                eval_loss.append(row["eval_loss"])

    return {
        "steps": np.asarray(steps),
        "train_loss": np.asarray(train_loss),
        "eval_steps": np.asarray(eval_steps),
        "eval_loss": np.asarray(eval_loss),
    }


def add_common_style(ax: plt.Axes) -> None:
    ax.grid(True, which="major", color="#d9d9d9", linewidth=0.8, alpha=0.45)
    ax.spines["top"].set_linewidth(0.8)
    ax.spines["right"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.spines["left"].set_linewidth(0.8)


def plot_loss_curves(output_prefix: Path, zoom_steps: int) -> None:
    data = {run.name: read_log(run.log_path) for run in RUNS}

    fig, ax = plt.subplots(figsize=(9.2, 6.1))
    add_common_style(ax)

    for run in RUNS:
        current = data[run.name]
        ax.plot(
            current["steps"],
            current["train_loss"] + run.main_offset,
            color=run.color,
            linewidth=0.75,
            alpha=0.85,
            zorder=2,
        )
        ax.scatter(
            current["eval_steps"],
            current["eval_loss"] + run.main_offset,
            color=run.color,
            edgecolor="white",
            linewidth=0.45,
            s=24,
            zorder=5,
        )

    ax.set_title("Training and Evaluation Loss over Steps (seed42)")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss (main panel vertically offset)")
    ax.set_ylim(bottom=2.85)
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=run.color,
            linewidth=1.6,
            label=f"{run.name} (+{run.main_offset:.2f})",
        )
        for run in RUNS
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=8,
        title="solid: train; dots: eval; main offsets shown",
        title_fontsize=8,
        frameon=True,
    )

    inset = ax.inset_axes([0.22, 0.42, 0.41, 0.46])
    add_common_style(inset)
    final_window_losses = []

    for run in RUNS:
        current = data[run.name]
        end_step = int(current["steps"][-1])
        mask = current["steps"] >= end_step - zoom_steps + 1
        x_from_end = current["steps"][mask] - end_step
        y = current["train_loss"][mask]
        final_window_losses.extend(y.tolist())
        inset.plot(
            x_from_end,
            y,
            color=run.color,
            linewidth=0.9,
            alpha=0.95,
        )

    y_min, y_max = min(final_window_losses), max(final_window_losses)
    pad = max((y_max - y_min) * 0.12, 0.03)
    inset.set_xlim(-zoom_steps + 1, 0)
    inset.set_ylim(y_min - pad, y_max + pad)
    inset.set_title(f"Final {zoom_steps} steps (end-aligned)", fontsize=8)
    inset.set_xlabel("Steps from run end", fontsize=8)
    inset.set_ylabel("Train loss", fontsize=8)
    inset.tick_params(axis="both", labelsize=7)
    inset.axvline(0, color="#333333", linewidth=0.7, alpha=0.7)

    fig.tight_layout()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot seed42 4-epoch train/eval loss curves for three models."
    )
    parser.add_argument(
        "--zoom-steps",
        type=int,
        default=50,
        help="Number of terminal training steps to show in the inset.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=MODEL_DIR / "figures/loss_curves_seed42_3models_final50",
        help="Output path prefix; .pdf and .png will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.zoom_steps <= 0:
        raise ValueError("--zoom-steps must be positive")
    plot_loss_curves(args.output_prefix, args.zoom_steps)


if __name__ == "__main__":
    main()
