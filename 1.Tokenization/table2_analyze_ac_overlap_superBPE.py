from __future__ import annotations

"""
64K-only A↔C overlap analysis for SuperBPE tokenizers.

A = Chinese superBPE
C = Pinyin-Toned superBPE

Outputs:
  - table2_ac_overlap_superBPE_report.txt
  - table2_ac_overlap_superBPE_summary.csv
  - table2_ac_overlap_superBPE_details.csv
  - table2_ac_overlap_superBPE_plot.png
"""

import csv
import json
import math
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

MPLCONFIGDIR = Path("/tmp") / "mplconfig_codex"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator

try:
    from pypinyin import Style, pinyin

    HAS_PYPINYIN = True
except ImportError:
    HAS_PYPINYIN = False


BASE_DIR = Path(__file__).resolve().parent
TOKENIZERS_DIR = BASE_DIR / "decoded_superTokenizers_2048_subset100k"
DICTS_DIR = BASE_DIR / "dicts"
VOCAB_SIZE = 64000

TOKENIZER_FILES = {
    "A": TOKENIZERS_DIR / f"chinese_origin_subset100k_superbpe_{VOCAB_SIZE}_decoded.json",
    "C": TOKENIZERS_DIR / f"pinyin_toned_subset100k_superbpe_{VOCAB_SIZE}_decoded.json",
}

OUTPUT_DIR = TOKENIZERS_DIR / "table2_ac_overlap_superBPE_outputs"
OUTPUT_REPORT = OUTPUT_DIR / "table2_ac_overlap_superBPE_report.txt"
OUTPUT_SUMMARY_CSV = OUTPUT_DIR / "table2_ac_overlap_superBPE_summary.csv"
OUTPUT_DETAILS_CSV = OUTPUT_DIR / "table2_ac_overlap_superBPE_details.csv"
OUTPUT_PLOT = OUTPUT_DIR / "table2_ac_overlap_superBPE_plot.png"
OUTPUT_PLOT_SVG = OUTPUT_DIR / "table2_ac_overlap_superBPE_plot.svg"
OUTPUT_PLOT_PDF = OUTPUT_DIR / "table2_ac_overlap_superBPE_plot.pdf"

SPECIAL_TOKENS = {"[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"}
SAMPLE_N_VALUES = (3, 20, 22, 41)
SAMPLE_A_TOKEN_COUNT = 3
MAX_EXAMPLE_ROWS_PER_N = 5
SAMPLE_RANDOM_SEED = 20260428
BAR_LABEL_FONTSIZE = 4.4
EXAMPLE_N_LABEL_FONTSIZE = 8.0
EXAMPLE_BOX_FONTSIZE = 8.0
EXAMPLE_BOX_HEIGHT = 0.58
EXAMPLE_BOX_BOTTOM = 0.00
EXAMPLE_ROW_STEP = 0.095
EXAMPLE_COLUMN_LAYOUT = {
    3: {"x": 0.000, "width": 0.310, "arrow": 0.115, "source": 0.142},
    20: {"x": 0.345, "width": 0.195, "arrow": 0.072, "source": 0.098},
    22: {"x": 0.575, "width": 0.195, "arrow": 0.072, "source": 0.098},
    41: {"x": 0.805, "width": 0.195, "arrow": 0.072, "source": 0.098},
}
CJK_FONT_CANDIDATES = [
    Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
    Path("/System/Library/Fonts/Hiragino Sans GB.ttc"),
    Path("/System/Library/Fonts/STHeiti Medium.ttc"),
    Path("/System/Library/Fonts/CJKSymbolsFallback.ttc"),
]

TONE_MARK_TO_BASE_AND_NUM = {
    "ā": ("a", "1"),
    "á": ("a", "2"),
    "ǎ": ("a", "3"),
    "à": ("a", "4"),
    "ē": ("e", "1"),
    "é": ("e", "2"),
    "ě": ("e", "3"),
    "è": ("e", "4"),
    "ī": ("i", "1"),
    "í": ("i", "2"),
    "ǐ": ("i", "3"),
    "ì": ("i", "4"),
    "ō": ("o", "1"),
    "ó": ("o", "2"),
    "ǒ": ("o", "3"),
    "ò": ("o", "4"),
    "ū": ("u", "1"),
    "ú": ("u", "2"),
    "ǔ": ("u", "3"),
    "ù": ("u", "4"),
    "ǖ": ("v", "1"),
    "ǘ": ("v", "2"),
    "ǚ": ("v", "3"),
    "ǜ": ("v", "4"),
    "ń": ("n", "2"),
    "ň": ("n", "3"),
    "ǹ": ("n", "4"),
    "ḿ": ("m", "2"),
}


@dataclass
class AnalysisResult:
    vocab_size: int
    vocab_a_size: int
    vocab_c_size: int
    mapped_a_count: int
    one_to_one_source_count: int
    many_to_one_source_count: int
    many_to_one_pair_count: int
    independent_a_count: int
    independent_c_count: int
    max_n: int
    n_to_entries: dict[int, list[tuple[str, list[str]]]]


class FallbackConverter:
    """Dictionary-based fallback when pypinyin is unavailable."""

    def __init__(self, cedict_path: Path, merged_dict_path: Path) -> None:
        self.word_to_pinyin: dict[str, list[str]] = {}
        self.char_to_pinyin: dict[str, list[str]] = {}
        self.char_to_merged: dict[str, str] = {}
        self.max_word_len = 1
        self._load_cedict(cedict_path)
        self._load_merged_dict(merged_dict_path)

    def _load_cedict(self, cedict_path: Path) -> None:
        if not cedict_path.exists():
            return

        pattern = re.compile(r"(\S+)\s+(\S+)\s+\[(.*?)\]")
        with cedict_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("#"):
                    continue
                match = pattern.match(line)
                if not match:
                    continue
                simplified = match.group(2)
                pinyin_list = [item.lower() for item in match.group(3).split()]
                if not simplified or not pinyin_list:
                    continue
                if len(simplified) == 1:
                    bucket = self.char_to_pinyin.setdefault(simplified, [])
                    for item in pinyin_list:
                        if item not in bucket:
                            bucket.append(item)
                else:
                    self.word_to_pinyin.setdefault(simplified, pinyin_list)
                    self.max_word_len = max(self.max_word_len, len(simplified))

    def _load_merged_dict(self, merged_dict_path: Path) -> None:
        if not merged_dict_path.exists():
            return
        data = json.loads(merged_dict_path.read_text(encoding="utf-8"))
        self.char_to_merged = data.get("data", {})

    @staticmethod
    def _is_chinese_char(ch: str) -> bool:
        return "\u4e00" <= ch <= "\u9fff"

    @staticmethod
    def _tone_marks_to_numbered_syllable(text: str) -> str:
        text = text.lower().replace("u:", "v").replace("ü", "v")
        if re.search(r"[1-5]$", text):
            return text

        tone = ""
        chars = []
        for ch in text:
            if ch in TONE_MARK_TO_BASE_AND_NUM:
                base, tone_num = TONE_MARK_TO_BASE_AND_NUM[ch]
                chars.append(base)
                if not tone:
                    tone = tone_num
            else:
                chars.append(ch)

        return "".join(chars) + tone

    def _diacritic_word_to_numbered(self, text: str) -> str:
        return "".join(
            self._tone_marks_to_numbered_syllable(part)
            for part in text.split()
        )

    def _convert_chunk(self, chunk: str) -> str | None:
        if chunk in self.word_to_pinyin:
            return "".join(self.word_to_pinyin[chunk])
        if len(chunk) == 1 and chunk in self.char_to_pinyin and self.char_to_pinyin[chunk]:
            return self.char_to_pinyin[chunk][0]
        if len(chunk) == 1 and chunk in self.char_to_merged:
            return self._diacritic_word_to_numbered(str(self.char_to_merged[chunk]))
        return None

    def convert(self, text: str) -> str:
        result: list[str] = []
        idx = 0
        length = len(text)

        while idx < length:
            ch = text[idx]
            if not self._is_chinese_char(ch):
                result.append(ch)
                idx += 1
                continue

            found: tuple[int, str] | None = None
            max_end = min(length, idx + self.max_word_len)
            for end in range(max_end, idx, -1):
                chunk = text[idx:end]
                if not all(self._is_chinese_char(c) for c in chunk):
                    continue
                converted = self._convert_chunk(chunk)
                if converted:
                    found = (end, converted)
                    break

            if found:
                end, converted = found
                result.append(converted)
                idx = end
            else:
                result.append(ch)
                idx += 1

        return "".join(result)


def clean_token(token: str) -> str:
    return token.replace("##", "").replace("Ġ", "").strip().replace(" ", "")


def is_special_token(token: str) -> bool:
    return token in SPECIAL_TOKENS or token.startswith("##") or token == "Ġ"


def load_tokenizer_vocab(path: Path) -> dict[str, int]:
    raw_vocab = json.loads(path.read_text(encoding="utf-8"))
    vocab = {}
    for token, token_id in raw_vocab.items():
        if is_special_token(token):
            continue
        cleaned = clean_token(token)
        if cleaned:
            vocab[cleaned] = token_id
    return vocab


def build_converter() -> tuple[callable, str]:
    if HAS_PYPINYIN:
        def convert(text: str) -> str:
            return "".join(
                piece[0] for piece in pinyin(text, style=Style.TONE3, strict=False)
            )

        return convert, "pypinyin.Style.TONE3(strict=False)"

    fallback = FallbackConverter(
        DICTS_DIR / "cedict_ts.u8",
        DICTS_DIR / "merged_pinyin_dict.json",
    )
    return fallback.convert, "fallback_dict_converter"


def analyze_overlap(convert: callable) -> AnalysisResult:
    vocab_a = load_tokenizer_vocab(TOKENIZER_FILES["A"])
    vocab_c = load_tokenizer_vocab(TOKENIZER_FILES["C"])

    mappings: dict[str, str] = {}
    reverse_mappings: dict[str, list[str]] = defaultdict(list)

    for token_a in vocab_a:
        converted = convert(token_a)
        if converted in vocab_c:
            mappings[token_a] = converted
            reverse_mappings[converted].append(token_a)

    n_to_entries: dict[int, list[tuple[str, list[str]]]] = defaultdict(list)
    one_to_one_source_count = 0
    many_to_one_source_count = 0

    for token_c, sources in reverse_mappings.items():
        if len(sources) == 1:
            one_to_one_source_count += 1
        else:
            sorted_sources = sorted(sources)
            many_to_one_source_count += len(sorted_sources)
            n_to_entries[len(sorted_sources)].append((token_c, sorted_sources))

    for entries in n_to_entries.values():
        entries.sort(key=lambda item: (item[0], item[1]))

    many_to_one_pair_count = sum(len(entries) for entries in n_to_entries.values())

    return AnalysisResult(
        vocab_size=VOCAB_SIZE,
        vocab_a_size=len(vocab_a),
        vocab_c_size=len(vocab_c),
        mapped_a_count=len(mappings),
        one_to_one_source_count=one_to_one_source_count,
        many_to_one_source_count=many_to_one_source_count,
        many_to_one_pair_count=many_to_one_pair_count,
        independent_a_count=len(vocab_a) - len(mappings),
        independent_c_count=len(vocab_c) - len(reverse_mappings),
        max_n=max(n_to_entries.keys(), default=1),
        n_to_entries=dict(sorted(n_to_entries.items())),
    )


def write_summary_csv(result: AnalysisResult) -> None:
    with OUTPUT_SUMMARY_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "N",
                "pair_count",
                "source_token_count",
                "a_vocab_size",
                "c_vocab_size",
                "mapped_a_count",
                "one_to_one_source_count",
                "many_to_one_source_count",
                "many_to_one_pair_count",
                "independent_a_count",
                "independent_c_count",
            ]
        )
        for n_value, entries in result.n_to_entries.items():
            writer.writerow(
                [
                    n_value,
                    len(entries),
                    n_value * len(entries),
                    result.vocab_a_size,
                    result.vocab_c_size,
                    result.mapped_a_count,
                    result.one_to_one_source_count,
                    result.many_to_one_source_count,
                    result.many_to_one_pair_count,
                    result.independent_a_count,
                    result.independent_c_count,
                ]
            )


def write_details_csv(result: AnalysisResult) -> None:
    with OUTPUT_DETAILS_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "N",
                "pair_index_within_N",
                "c_token",
                "source_token_count_for_pair",
                "a_tokens_json",
            ]
        )
        for n_value, entries in result.n_to_entries.items():
            for idx, (token_c, sources) in enumerate(entries, start=1):
                writer.writerow(
                    [
                        n_value,
                        idx,
                        token_c,
                        len(sources),
                        json.dumps(sources, ensure_ascii=False),
                    ]
                )


def build_report(result: AnalysisResult, converter_name: str, y_scale_mode: str) -> str:
    lines: list[str] = []
    lines.append("=" * 100)
    lines.append("64K A-C OVERLAP ANALYSIS (SUPERBPE)")
    lines.append("=" * 100)
    lines.append("")
    lines.append("Legend:")
    lines.append("  A = Chinese superBPE")
    lines.append("  C = Pinyin-Toned superBPE")
    lines.append("")
    lines.append("Method:")
    lines.append("  • Only the 64K tokenizer pair is analyzed.")
    lines.append("  • Forward mapping follows the A→C logic from the 9th script.")
    lines.append(f"  • Converter used in this run: {converter_name}")
    lines.append("  • N对1 pair count = number of C tokens that are each mapped from N A tokens.")
    lines.append("  • source_token_count = N × pair_count.")
    lines.append(f"  • Plot Y scale mode: {y_scale_mode}")
    lines.append("")
    lines.append("Files:")
    lines.append(f"  • Output directory: {OUTPUT_DIR.name}")
    lines.append(f"  • Report: {OUTPUT_REPORT.name}")
    lines.append(f"  • Summary CSV: {OUTPUT_SUMMARY_CSV.name}")
    lines.append(f"  • Details CSV: {OUTPUT_DETAILS_CSV.name}")
    lines.append(f"  • Plot: {OUTPUT_PLOT.name}")
    lines.append(f"  • Vector plot: {OUTPUT_PLOT_SVG.name}, {OUTPUT_PLOT_PDF.name}")
    lines.append("")
    lines.append("=" * 100)
    lines.append("SUMMARY")
    lines.append("=" * 100)
    lines.append("")
    lines.append(f"Vocab size: {result.vocab_size}")
    lines.append(f"A vocabulary size: {result.vocab_a_size}")
    lines.append(f"C vocabulary size: {result.vocab_c_size}")
    lines.append(f"Mapped A tokens: {result.mapped_a_count}")
    lines.append(f"1对1 source count: {result.one_to_one_source_count}")
    lines.append(f"N对1 source count: {result.many_to_one_source_count}")
    lines.append(f"N对1 pair count: {result.many_to_one_pair_count}")
    lines.append(f"Independent A: {result.independent_a_count}")
    lines.append(f"Independent C: {result.independent_c_count}")
    lines.append(f"Max N: {result.max_n}")
    lines.append("")
    lines.append("N-to-1 breakdown:")
    for n_value, entries in result.n_to_entries.items():
        lines.append(
            f"  • {n_value}对1: pair_count={len(entries)}, source_token_count={n_value * len(entries)}"
        )
    lines.append("")
    lines.append("Detailed pairs by N:")

    for n_value, entries in result.n_to_entries.items():
        lines.append("")
        lines.append(f"[{n_value}对1] total pairs = {len(entries)}")
        for token_c, sources in entries:
            lines.append(
                f"  C: '{token_c}' <- A({len(sources)}): {json.dumps(sources, ensure_ascii=False)}"
            )

    return "\n".join(lines)


def choose_y_scale(result: AnalysisResult) -> str:
    pair_counts = [len(entries) for entries in result.n_to_entries.values() if len(entries) > 0]
    if not pair_counts:
        return "linear"
    if max(pair_counts) / min(pair_counts) >= 20:
        return "log"
    return "linear"


# Legacy exploratory plotting code is intentionally disabled for the EMNLP-ready
# figure. It used a single annotated line plot with token labels:
#
# def create_plot_legacy(result: AnalysisResult, y_scale_mode: str) -> None:
#     try:
#         plt.style.use("seaborn-v0_8-whitegrid")
#     except OSError:
#         pass
#     n_values = list(result.n_to_entries.keys())
#     pair_counts = [len(result.n_to_entries[n_value]) for n_value in n_values]
#     fig, ax = plt.subplots(figsize=(11.5, 6.5), dpi=160)
#     ax.plot(n_values, pair_counts, color="#1d4ed8", linewidth=2.6, marker="o")
#     annotated_points = [
#         (n_value, len(entries), ", ".join(token_c for token_c, _ in entries))
#         for n_value, entries in result.n_to_entries.items()
#         if len(entries) <= 3
#     ]
#     if y_scale_mode == "log":
#         ax.set_yscale("log")
#     ax.set_title("64K A↔C N-to-1 Overlap Distribution", fontsize=15, pad=14)
#     ax.set_xlabel("N in shared N:1 mapping", fontsize=12)
#     ax.set_ylabel("Pair count", fontsize=24)
#     ax.set_xticks(n_values)
#     ax.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.55)
#     for n_value, pair_count, label in annotated_points:
#         ax.annotate(label, xy=(n_value, pair_count), textcoords="offset points")
#     fig.tight_layout(pad=1.2)
#     fig.savefig(OUTPUT_PLOT, bbox_inches="tight")
#     plt.close(fig)


def _format_count(value: float, _position: int | None = None) -> str:
    if value >= 1000:
        return f"{value / 1000:.0f}k"
    return f"{value:.0f}"


def _format_bar_label(value: int) -> str:
    return f"{value / 1000:.1f}k" if value >= 1000 else str(value)


def _format_axis_label(value: float, _position: int | None = None) -> str:
    if value >= 1000:
        return f"{value / 1000:g}k"
    return f"{value:g}"


def _resolve_plot_font_family() -> list[str]:
    for font_path in CJK_FONT_CANDIDATES:
        if not font_path.exists():
            continue
        font_manager.fontManager.addfont(str(font_path))
        family_name = font_manager.FontProperties(fname=str(font_path)).get_name()
        return [family_name, "DejaVu Sans"]
    return ["DejaVu Sans"]


def _setup_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": _resolve_plot_font_family(),
            "font.size": 7.0,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.0,
            "axes.linewidth": 0.55,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 6.5,
            "legend.handlelength": 1.15,
            "legend.handletextpad": 0.45,
            "legend.columnspacing": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 600,
        }
    )


def _draw_overlap_context(ax: plt.Axes, result: AnalysisResult) -> None:
    colors = {
        "1-to-1 mapped": "#0072B2",
        "N-to-1 mapped": "#D55E00",
        "independent": "#B8B8B8",
    }
    rows = [
        (
            "Chinese superBPE\nA tokens",
            [
                ("1-to-1 mapped", result.one_to_one_source_count),
                ("N-to-1 mapped", result.many_to_one_source_count),
                ("independent", result.independent_a_count),
            ],
            result.vocab_a_size,
        ),
        (
            "Pinyin-Toned superBPE\nC tokens",
            [
                ("1-to-1 mapped", result.one_to_one_source_count),
                ("N-to-1 mapped", result.many_to_one_pair_count),
                ("independent", result.independent_c_count),
            ],
            result.vocab_c_size,
        ),
    ]

    for y_pos, (label, segments, total) in enumerate(rows):
        left = 0
        for segment_label, value in segments:
            ax.barh(
                y_pos,
                value,
                left=left,
                height=0.42,
                color=colors[segment_label],
                edgecolor="white",
                linewidth=0.7,
            )
            if value / total >= 0.08:
                ax.text(
                    left + value / 2,
                    y_pos,
                    _format_bar_label(value),
                    ha="center",
                    va="center",
                    color="white" if segment_label != "independent" else "#333333",
                    fontsize=7.2,
                )
            left += value
        ax.text(
            total + max(result.vocab_a_size, result.vocab_c_size) * 0.015,
            y_pos,
            f"n={total:,}",
            va="center",
            ha="left",
            fontsize=7.4,
            color="#333333",
        )

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([row[0] for row in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Vocabulary entries")
    ax.set_title("(a) Overlap accounting", loc="left", fontweight="bold")
    ax.xaxis.set_major_formatter(FuncFormatter(_format_count))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.grid(axis="x", color="#D8D8D8", linewidth=0.45)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(
        handles=[Patch(facecolor=color, edgecolor="none", label=label) for label, color in colors.items()],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.14),
        ncol=3,
        frameon=False,
        columnspacing=1.2,
        handlelength=1.1,
    )


def _nice_axis_top(value: float) -> float:
    if value <= 0:
        return 1
    magnitude = 10 ** math.floor(math.log10(value))
    for multiplier in (1, 2, 3, 5, 10):
        top = multiplier * magnitude
        if top >= value:
            return top
    return 10 * magnitude


def _shared_y_ticks(y_min: float, y_max: float, y_scale_mode: str) -> list[float]:
    if y_scale_mode == "log":
        ticks = [1, 10, 100, 1000]
        if y_max not in ticks:
            ticks.append(y_max)
        return [tick for tick in ticks if y_min <= tick <= y_max]

    locator = MaxNLocator(nbins=5)
    return [
        tick
        for tick in locator.tick_values(y_min, y_max)
        if y_min <= tick <= y_max
    ]


def _set_shared_y_axes(
    ax_left: plt.Axes,
    ax_right: plt.Axes,
    pair_counts: list[int],
    source_counts: list[int],
    y_scale_mode: str,
) -> None:
    max_value = max(pair_counts + source_counts, default=1)
    if y_scale_mode == "log":
        y_min = 0.75
        y_max = _nice_axis_top(max_value * 1.2)
    else:
        y_min = 0
        y_max = _nice_axis_top(max_value * 1.12)

    ax_left.set_ylim(y_min, y_max)
    ax_right.set_ylim(y_min, y_max)
    ticks = _shared_y_ticks(y_min, y_max, y_scale_mode)
    ax_left.set_yticks(ticks)
    ax_right.set_yticks(ticks)


def _bar_label_y_position(value: int, y_scale_mode: str) -> float:
    if y_scale_mode == "log":
        return max(value * 1.12, 1.15)
    return value * 1.015


def _add_bar_labels(
    ax: plt.Axes,
    bars,
    y_scale_mode: str,
    color: str = "#2B2B2B",
) -> None:
    for bar in bars:
        height = int(bar.get_height())
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            _bar_label_y_position(height, y_scale_mode),
            _format_bar_label(height),
            ha="center",
            va="bottom",
            fontsize=BAR_LABEL_FONTSIZE,
            color=color,
        )


def _sample_a_tokens(n_value: int, token_c: str, sources: list[str]) -> list[str]:
    sample_size = min(SAMPLE_A_TOKEN_COUNT, len(sources))
    rng = random.Random(f"{SAMPLE_RANDOM_SEED}:{n_value}:{token_c}")
    return rng.sample(sources, sample_size)


def _is_phrase_source_token(token: str) -> bool:
    return len(clean_token(token)) > 1


def _select_example_entries(
    n_value: int,
    entries: list[tuple[str, list[str]]],
) -> list[tuple[str, list[str]]]:
    if n_value != 3:
        return entries[:MAX_EXAMPLE_ROWS_PER_N]

    phrase_entries = [
        (token_c, sources)
        for token_c, sources in entries
        if all(_is_phrase_source_token(source) for source in sources)
    ]
    if len(phrase_entries) >= MAX_EXAMPLE_ROWS_PER_N:
        return phrase_entries[:MAX_EXAMPLE_ROWS_PER_N]

    selected = phrase_entries[:]
    selected_keys = {token_c for token_c, _sources in selected}
    for token_c, sources in entries:
        if token_c in selected_keys:
            continue
        selected.append((token_c, sources))
        if len(selected) >= MAX_EXAMPLE_ROWS_PER_N:
            break
    return selected


def _draw_example_box_background(
    ax: plt.Axes,
    x_pos: float,
    width: float,
) -> None:
    bottom = EXAMPLE_BOX_BOTTOM
    height = EXAMPLE_BOX_HEIGHT
    ax.add_patch(
        Rectangle(
            (x_pos, bottom),
            width,
            height,
            transform=ax.transAxes,
            facecolor="#FAFAFA",
            edgecolor="none",
            linewidth=0.0,
            clip_on=False,
        )
    )
    ax.plot(
        [x_pos, x_pos + width],
        [bottom + height, bottom + height],
        transform=ax.transAxes,
        color="#B8C4D0",
        linewidth=0.75,
        solid_capstyle="butt",
        clip_on=False,
    )


def _draw_selected_token_examples(
    ax: plt.Axes,
    result: AnalysisResult,
) -> None:
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    ax.axis("off")
    ax.text(
        0.0,
        0.92,
        "(b) Representative token collisions",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.6,
        fontweight="bold",
        color="#333333",
    )

    box_top = EXAMPLE_BOX_BOTTOM + EXAMPLE_BOX_HEIGHT
    for n_value in SAMPLE_N_VALUES:
        layout = EXAMPLE_COLUMN_LAYOUT[n_value]
        x_pos = layout["x"]
        label_y = box_top + 0.11
        entries = result.n_to_entries.get(n_value, [])
        examples = []
        if entries:
            for token_c, sources in _select_example_entries(n_value, entries):
                sampled_sources = _sample_a_tokens(n_value, token_c, sources)
                examples.append((token_c, ", ".join(sampled_sources)))

        ax.text(
            x_pos,
            label_y,
            f"N={n_value}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=EXAMPLE_N_LABEL_FONTSIZE,
            fontweight="bold",
            color="#333333",
        )

        _draw_example_box_background(ax, x_pos, layout["width"])

        if not examples:
            ax.text(
                x_pos + 0.012,
                box_top - 0.09,
                "No matching C token",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=EXAMPLE_BOX_FONTSIZE,
                color="#2B2B2B",
            )
            continue

        for row_idx, (token_c, source_text) in enumerate(examples):
            row_y = box_top - 0.09 - row_idx * EXAMPLE_ROW_STEP
            ax.text(
                x_pos + 0.012,
                row_y,
                token_c,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=EXAMPLE_BOX_FONTSIZE,
                color="#2B2B2B",
                family="DejaVu Sans Mono",
            )
            ax.text(
                x_pos + layout["arrow"],
                row_y,
                "->",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=EXAMPLE_BOX_FONTSIZE,
                color="#555555",
                family="DejaVu Sans Mono",
            )
            ax.text(
                x_pos + layout["source"],
                row_y,
                source_text,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=EXAMPLE_BOX_FONTSIZE,
                color="#2B2B2B",
            )


def _draw_multiplicity_distribution(
    ax: plt.Axes,
    result: AnalysisResult,
    y_scale_mode: str,
    rare_ax: plt.Axes | None = None,
) -> None:
    n_values = list(result.n_to_entries.keys())
    x_positions = list(range(len(n_values)))
    pair_counts = [len(result.n_to_entries[n_value]) for n_value in n_values]
    source_counts = [n_value * len(result.n_to_entries[n_value]) for n_value in n_values]
    bar_width = 0.32

    c_bars = ax.bar(
        [x_pos - bar_width / 2 for x_pos in x_positions],
        pair_counts,
        width=bar_width,
        color="#D95F02",
        edgecolor="#6B2D00",
        linewidth=0.35,
        alpha=0.92,
    )
    a_bars = ax.bar(
        [x_pos + bar_width / 2 for x_pos in x_positions],
        source_counts,
        width=bar_width,
        color="#4DA3D9",
        edgecolor="#006DA3",
        linewidth=0.35,
        alpha=0.84,
    )

    if y_scale_mode == "log":
        ax.set_yscale("log")

    ax2 = ax.twinx()
    if y_scale_mode == "log":
        ax2.set_yscale("log")
    _set_shared_y_axes(ax, ax2, pair_counts, source_counts, y_scale_mode)

    _add_bar_labels(ax, c_bars, y_scale_mode)
    _add_bar_labels(ax, a_bars, y_scale_mode, color="#005B8F")

    ax.set_xlim(-0.75, len(n_values) - 0.25)
    ax.set_xlabel("Multiplicity N")
    ax.set_ylabel("Count (log scale)")
    ax2.set_ylabel("Count (log scale)")
    ax.tick_params(axis="y", colors="#444444", width=0.55, length=2.6)
    ax2.tick_params(axis="y", colors="#444444", width=0.55, length=2.6)
    ax.spines["left"].set_color("#777777")
    ax2.spines["right"].set_color("#777777")
    ax.set_title("")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(n_value) for n_value in n_values])
    ax.tick_params(axis="x", labelrotation=0, pad=2, width=0.55, length=2.6)
    ax.yaxis.set_major_formatter(FuncFormatter(_format_axis_label))
    ax2.yaxis.set_major_formatter(FuncFormatter(_format_axis_label))
    ax.grid(axis="y", which="major", color="#D5D5D5", linewidth=0.35)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax.spines["bottom"].set_color("#777777")
    ax2.spines["bottom"].set_visible(False)

    legend_handles = [
        Patch(facecolor="#D95F02", edgecolor="#6B2D00", alpha=0.92, label="Pinyin-Toned superBPE"),
        Patch(facecolor="#4DA3D9", edgecolor="#006DA3", alpha=0.84, label="Chinese superBPE"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.12),
        ncol=2,
        frameon=False,
        borderaxespad=0.0,
    )

    if rare_ax is None:
        return
    _draw_selected_token_examples(rare_ax, result)


def create_plot(
    result: AnalysisResult,
    y_scale_mode: str,
) -> None:
    _setup_publication_style()

    fig, (ax_distribution, ax_rare) = plt.subplots(
        2,
        1,
        figsize=(7.2, 4.95),
        dpi=600,
        gridspec_kw={"height_ratios": [2.95, 1.55], "hspace": 0.24},
    )
    fig.patch.set_facecolor("white")
    fig.text(
        0.08,
        0.968,
        "(a) N-to-1 homophone collision profile",
        ha="left",
        va="top",
        fontsize=8.2,
        fontweight="bold",
        color="#111111",
    )

    _draw_multiplicity_distribution(
        ax_distribution,
        result,
        y_scale_mode,
        rare_ax=ax_rare,
    )

    fig.subplots_adjust(left=0.08, right=0.92, bottom=0.06, top=0.89)
    fig.savefig(OUTPUT_PLOT, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUTPUT_PLOT_SVG, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUTPUT_PLOT_PDF, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    convert, converter_name = build_converter()
    result = analyze_overlap(convert)
    y_scale_mode = choose_y_scale(result)

    write_summary_csv(result)
    write_details_csv(result)
    create_plot(result, y_scale_mode)
    report = build_report(result, converter_name, y_scale_mode)
    OUTPUT_REPORT.write_text(report, encoding="utf-8")

    print(report)
    print("")
    print(f"Saved report: {OUTPUT_REPORT}")
    print(f"Saved summary csv: {OUTPUT_SUMMARY_CSV}")
    print(f"Saved details csv: {OUTPUT_DETAILS_CSV}")
    print(f"Saved plot: {OUTPUT_PLOT}")
    print(f"Saved vector plot: {OUTPUT_PLOT_SVG}")
    print(f"Saved vector plot: {OUTPUT_PLOT_PDF}")


if __name__ == "__main__":
    main()
