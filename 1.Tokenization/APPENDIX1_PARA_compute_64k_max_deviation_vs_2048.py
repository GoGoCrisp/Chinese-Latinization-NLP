from __future__ import annotations

import csv
import importlib.util
import json
import re
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
TABLE2_SCRIPT = BASE_DIR / "table2_analyze_ac_overlap_superBPE.py"
SUMMARY_SCRIPT = BASE_DIR / "APPENDIX1_PARA_summarize_64k_transition_robustness.py"

OUTPUT_MD = BASE_DIR / "APPENDIX1_PARA_64k_max_deviation_vs_K2048.md"
OUTPUT_CSV = BASE_DIR / "APPENDIX1_PARA_64k_max_deviation_vs_K2048.csv"

VOCAB_SIZE = 64000
BASELINE_KEY = "K2048"

PARAMETERS = [
    {
        "key": "K1000",
        "label": "K=1000",
        "decoded_dir": BASE_DIR / "decoded_superTokenizers_K1000_64k_subset100k",
        "fertility_key": "K1000",
    },
    {
        "key": "K2048",
        "label": "K=2048",
        "decoded_dir": BASE_DIR / "decoded_superTokenizers_2048_subset100k",
        "fertility_key": "2048_subset100k",
    },
    {
        "key": "K3200",
        "label": "K=3200",
        "decoded_dir": BASE_DIR / "decoded_superTokenizers_0.05",
        "fertility_key": "0.05",
    },
    {
        "key": "K6400",
        "label": "K=6400",
        "decoded_dir": BASE_DIR / "decoded_superTokenizers",
        "fertility_key": "baseline",
    },
]

TOKENIZERS = {
    "Chinese": {
        "type": "origin",
        "file": "chinese_origin_subset100k_superbpe_64000_decoded.json",
    },
    "Diacritic": {
        "type": "diacritic",
        "file": "pinyin_diacritic_subset100k_superbpe_64000_decoded.json",
    },
}

METRICS = [
    {
        "key": "vocab_unique",
        "label": "Vocabulary total, outside parentheses",
        "better_when": None,
    },
    {
        "key": "vocab_raw",
        "label": "Vocabulary total, inside parentheses",
        "better_when": None,
    },
    {
        "key": "overlap_shared",
        "label": "Overlap total shared",
        "better_when": "higher",
    },
    {
        "key": "overlap_independent",
        "label": "Overlap independent",
        "better_when": "lower",
    },
    {
        "key": "fertility",
        "label": "Fertility tokens/original char",
        "better_when": "lower",
    },
]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def whitespace_insensitive_token(token: str) -> str:
    token = token.replace("##", "").replace("Ġ", "")
    return re.sub(r"\s+", "", token)


def vocab_total(decoded_path: Path) -> tuple[int, int]:
    vocab = json.loads(decoded_path.read_text(encoding="utf-8"))
    unique_count = len({whitespace_insensitive_token(token) for token in vocab})
    raw_count = len(vocab)
    return unique_count, raw_count


def load_fertility_rows() -> dict[tuple[str, str], dict]:
    summary = load_module(SUMMARY_SCRIPT, "transition_summary_for_deviation")
    return summary.load_fertility_rows()


def load_table2_module():
    return load_module(TABLE2_SCRIPT, "table2_for_deviation")


def analyze_ad_overlap(decoded_dir: Path) -> dict[str, int]:
    table2 = load_table2_module()
    table2.TOKENIZER_FILES = {
        "A": decoded_dir / f"chinese_origin_subset100k_superbpe_{VOCAB_SIZE}_decoded.json",
        "B": decoded_dir / f"pinyin_toneless_subset100k_superbpe_{VOCAB_SIZE}_decoded.json",
        "C": decoded_dir / f"pinyin_toned_subset100k_superbpe_{VOCAB_SIZE}_decoded.json",
        "D": decoded_dir / f"pinyin_diacritic_subset100k_superbpe_{VOCAB_SIZE}_decoded.json",
    }
    convert, _converter_name = table2.build_converter("D")
    result = table2.analyze_overlap("AD", convert)

    return {
        "Chinese_overlap_shared": result.mapped_a_count,
        "Chinese_overlap_independent": result.independent_a_count,
        "Diacritic_overlap_shared": result.vocab_target_size - result.independent_target_count,
        "Diacritic_overlap_independent": result.independent_target_count,
    }


def collect_values() -> dict[str, dict[str, dict[str, float]]]:
    fertility_rows = load_fertility_rows()
    values: dict[str, dict[str, dict[str, float]]] = {}

    for parameter in PARAMETERS:
        key = parameter["key"]
        decoded_dir = parameter["decoded_dir"]
        values[key] = {"label": parameter["label"]}

        overlap = analyze_ad_overlap(decoded_dir)

        for side, config in TOKENIZERS.items():
            decoded_path = decoded_dir / config["file"]
            vocab_unique, vocab_raw = vocab_total(decoded_path)
            fert = fertility_rows[(parameter["fertility_key"], config["type"])]

            values[key][side] = {
                "vocab_unique": float(vocab_unique),
                "vocab_raw": float(vocab_raw),
                "overlap_shared": float(overlap[f"{side}_overlap_shared"]),
                "overlap_independent": float(overlap[f"{side}_overlap_independent"]),
                "fertility": float(fert["tokens_per_original_char"]),
            }

    return values


def pct_deviation(value: float, baseline: float) -> float:
    if baseline == 0:
        raise ZeroDivisionError("Baseline value is zero")
    return abs(value - baseline) / baseline * 100.0


def signed_pct_change(value: float, baseline: float) -> float:
    if baseline == 0:
        raise ZeroDivisionError("Baseline value is zero")
    return (value - baseline) / baseline * 100.0


def classify_change(signed_pct: float, better_when: str | None) -> str:
    if better_when is None:
        if signed_pct > 0:
            return "higher"
        if signed_pct < 0:
            return "lower"
        return "same"
    if signed_pct == 0:
        return "same"
    if better_when == "higher":
        return "better" if signed_pct > 0 else "worse"
    if better_when == "lower":
        return "worse" if signed_pct > 0 else "better"
    raise ValueError(f"Unsupported better_when value: {better_when}")


def max_deviation(values: dict, side: str, metric: dict) -> dict:
    metric_key = metric["key"]
    baseline = values[BASELINE_KEY][side][metric_key]
    best = None
    for key in ("K1000", "K3200", "K6400"):
        value = values[key][side][metric_key]
        deviation = pct_deviation(value, baseline)
        signed_pct = signed_pct_change(value, baseline)
        candidate = {
            "side": side,
            "metric": metric_key,
            "metric_label": metric["label"],
            "baseline_value": baseline,
            "max_value": value,
            "max_deviation_pct": deviation,
            "signed_pct_change": signed_pct,
            "direction": classify_change(signed_pct, metric["better_when"]),
            "source": values[key]["label"],
        }
        if best is None or candidate["max_deviation_pct"] > best["max_deviation_pct"]:
            best = candidate
    return best


def fmt_int(value: float) -> str:
    return f"{int(round(value)):,}"


def fmt_pct(value: float) -> str:
    return f"{value:.2f}%"


def fmt_signed_pct(value: float) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.2f}%"


def fmt_cell(result: dict) -> str:
    return f"{fmt_signed_pct(result['signed_pct_change'])} ({result['source']}, {result['direction']})"


def build_report(values: dict, results: list[dict]) -> str:
    by_metric_side = {(row["metric"], row["side"]): row for row in results}

    lines = [
        "# Appendix: 64k Max Deviation from K=2048",
        "",
        "This table uses K=2048 as the baseline and compares K=1000, K=3200 (5%), and K=6400 (10%). Each cell is selected by the largest absolute percentage deviation from K=2048, but the displayed percentage is signed. K=0 is intentionally excluded from this robustness table.",
        "",
        "For overlap total shared, higher is marked as better and lower as worse. For overlap independent and fertility, higher is marked as worse and lower as better. Vocabulary-total rows are directional only, because higher/lower vocabulary totals are not treated as intrinsically better or worse here.",
        "",
        "Overlap is computed for the Chinese-Origin to Pinyin-Diacritic pair. For Chinese, total shared is the number of Chinese vocabulary entries mapped to a Diacritic token; for Diacritic, total shared is the number of Diacritic vocabulary entries reached by at least one Chinese entry.",
        "",
        "| Metric | Chinese | Diacritic |",
        "|---|---:|---:|",
    ]

    for metric in METRICS:
        metric_key = metric["key"]
        metric_label = metric["label"]
        chinese = by_metric_side[(metric_key, "Chinese")]
        diacritic = by_metric_side[(metric_key, "Diacritic")]
        lines.append(f"| {metric_label} | {fmt_cell(chinese)} | {fmt_cell(diacritic)} |")

    lines.extend(
        [
            "",
            "## Baseline Values",
            "",
            "| Metric | Chinese K=2048 | Diacritic K=2048 |",
            "|---|---:|---:|",
        ]
    )
    for metric in METRICS:
        metric_key = metric["key"]
        metric_label = metric["label"]
        chinese = values[BASELINE_KEY]["Chinese"][metric_key]
        diacritic = values[BASELINE_KEY]["Diacritic"][metric_key]
        if metric_key == "fertility":
            chinese_value = f"{chinese:.4f}"
            diacritic_value = f"{diacritic:.4f}"
        else:
            chinese_value = fmt_int(chinese)
            diacritic_value = fmt_int(diacritic)
        lines.append(f"| {metric_label} | {chinese_value} | {diacritic_value} |")

    return "\n".join(lines)


def write_csv(results: list[dict]) -> None:
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "metric",
                "metric_label",
                "side",
                "baseline_value",
                "max_value",
                "max_deviation_pct",
                "signed_pct_change",
                "direction",
                "source",
            ],
        )
        writer.writeheader()
        writer.writerows(results)


def main() -> None:
    values = collect_values()
    results = []
    for metric in METRICS:
        for side in ("Chinese", "Diacritic"):
            results.append(max_deviation(values, side, metric))

    OUTPUT_MD.write_text(build_report(values, results) + "\n", encoding="utf-8")
    write_csv(results)

    print(f"Wrote {OUTPUT_MD}")
    print(f"Wrote {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
