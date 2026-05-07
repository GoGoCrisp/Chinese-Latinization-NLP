import csv
import importlib.util
import json
import re
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
EVAL_CSV = BASE_DIR / "APPENDIX1_PARA_tokenizer_evaluation_comparison_3params_16k_64k.csv"
K0_EVAL_CSV = (
    BASE_DIR
    / "decoded_superTokenizers_K0_64k_subset100k"
    / "evaluation_4abcd_superBPE_K0_64k.csv"
)
K1000_EVAL_CSV = (
    BASE_DIR
    / "decoded_superTokenizers_K1000_64k_subset100k"
    / "evaluation_4abcd_superBPE_K1000_64k.csv"
)
TABLE2_SCRIPT = BASE_DIR / "table2_analyze_ac_overlap_superBPE.py"

OUTPUT_MD = BASE_DIR / "APPENDIX1_PARA_64k_transition_robustness_summary.md"
OUTPUT_CSV = BASE_DIR / "APPENDIX1_PARA_64k_transition_robustness_summary.csv"

VOCAB_SIZE = 64000

PARAMETERS = [
    {
        "key": "K0",
        "label": "K=0",
        "decoded_dir": BASE_DIR / "decoded_superTokenizers_K0_64k_subset100k",
    },
    {
        "key": "K1000",
        "label": "K=1000",
        "decoded_dir": BASE_DIR / "decoded_superTokenizers_K1000_64k_subset100k",
    },
    {
        "key": "0.05",
        "label": "5%",
        "decoded_dir": BASE_DIR / "decoded_superTokenizers_0.05",
    },
    {
        "key": "2048_subset100k",
        "label": "K=2048",
        "decoded_dir": BASE_DIR / "decoded_superTokenizers_2048_subset100k",
    },
    {
        "key": "baseline",
        "label": "10%",
        "decoded_dir": BASE_DIR / "decoded_superTokenizers",
    },
]

TOKENIZERS = [
    ("origin", "Chinese Origin", "chinese_origin_subset100k_superbpe_64000_decoded.json"),
    ("toned", "Pinyin-Toned", "pinyin_toned_subset100k_superbpe_64000_decoded.json"),
    ("toneless", "Pinyin-Toneless", "pinyin_toneless_subset100k_superbpe_64000_decoded.json"),
    ("diacritic", "Pinyin-Diacritic", "pinyin_diacritic_subset100k_superbpe_64000_decoded.json"),
]


def load_table2_module():
    spec = importlib.util.spec_from_file_location("table2", TABLE2_SCRIPT)
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
    rows = {}
    with open(EVAL_CSV, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if int(row["size"]) != VOCAB_SIZE:
                continue
            rows[(row["parameter"], row["type"])] = row
    for key, path in [("K0", K0_EVAL_CSV), ("K1000", K1000_EVAL_CSV)]:
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                row_size = int(row["size"]) if row.get("size") else int(row["file"].rsplit("_", 1)[-1])
                if row_size != VOCAB_SIZE:
                    continue
                row["size"] = str(row_size)
                rows[(key, row["type"])] = row
    return rows


def analyze_table2(decoded_dir: Path) -> dict:
    table2 = load_table2_module()
    table2.TOKENIZER_FILES = {
        "A": decoded_dir / f"chinese_origin_subset100k_superbpe_{VOCAB_SIZE}_decoded.json",
        "C": decoded_dir / f"pinyin_toned_subset100k_superbpe_{VOCAB_SIZE}_decoded.json",
    }
    convert, _converter_name = table2.build_converter("C")
    result = table2.analyze_overlap("AC", convert)

    source_gt4 = 0
    target_gt4 = 0
    for n, entries in result.n_to_entries.items():
        if n > 4:
            source_gt4 += n * len(entries)
            target_gt4 += len(entries)

    def pair_counts(n: int) -> tuple[int, int]:
        entries = result.n_to_entries.get(n, [])
        return n * len(entries), len(entries)

    source_2, target_2 = pair_counts(2)
    source_3, target_3 = pair_counts(3)
    source_4, target_4 = pair_counts(4)

    return {
        "shared_1to1_source": result.one_to_one_source_count,
        "shared_1to1_target": result.one_to_one_source_count,
        "shared_2to1_source": source_2,
        "shared_2to1_target": target_2,
        "shared_3to1_source": source_3,
        "shared_3to1_target": target_3,
        "shared_4to1_source": source_4,
        "shared_4to1_target": target_4,
        "shared_gt4to1_source": source_gt4,
        "shared_gt4to1_target": target_gt4,
        "unique_source": result.independent_a_count,
        "unique_target": result.independent_target_count,
        "mapped_source": result.mapped_a_count,
        "source_vocab": result.vocab_a_size,
        "target_vocab": result.vocab_target_size,
    }


def fmt_int(value) -> str:
    return f"{int(value):,}"


def fmt_float(value, digits=4) -> str:
    return f"{float(value):.{digits}f}"


def build_summary() -> tuple[str, list[dict]]:
    fertility = load_fertility_rows()
    rows = []
    table2_by_param = {}

    for parameter in PARAMETERS:
        key = parameter["key"]
        label = parameter["label"]
        decoded_dir = parameter["decoded_dir"]
        table2_by_param[label] = analyze_table2(decoded_dir)

        for token_type, tokenizer_label, filename in TOKENIZERS:
            fert = fertility[(key, token_type)]
            unique_vocab, raw_vocab = vocab_total(decoded_dir / filename)
            rows.append(
                {
                    "parameter": label,
                    "tokenizer": tokenizer_label,
                    "tokens_per_original_char": fert["tokens_per_original_char"],
                    "tokens_per_sample": fert["tokens_per_sample"],
                    "total_test_tokens": fert["total_tokens"],
                    "vocab_unique": unique_vocab,
                    "vocab_raw": raw_vocab,
                    "vocab_total": f"{unique_vocab:,} ({raw_vocab:,})",
                }
            )

    lines = []
    lines.append("# Appendix: 64k SuperBPE Transition Robustness")
    lines.append("")
    lines.append(
        "This appendix compares the 64k SuperBPE tokenizers trained with five transition settings: fixed K=0, fixed K=1000, 5%, fixed K=2048, and 10%. "
        "For 64k vocabularies, 5% and 10% correspond to K=3200 and K=6400. "
        "The goal is not to claim that K=2048 is best on every individual metric, but to show how sensitive the main paper results are to the transition point, including the boundary case where SuperBPE starts immediately at K=0."
    )
    lines.append("")

    lines.append("## Table A. Fertility and Vocabulary Size")
    lines.append("")
    lines.append("| Parameter | Tokenizer | Tokens/original char | Tokens/sample | Total test tokens | Vocabulary total |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| {parameter} | {tokenizer} | {toc} | {tps} | {total} | {vocab} |".format(
                parameter=row["parameter"],
                tokenizer=row["tokenizer"],
                toc=fmt_float(row["tokens_per_original_char"]),
                tps=fmt_float(row["tokens_per_sample"]),
                total=fmt_int(row["total_test_tokens"]),
                vocab=row["vocab_total"],
            )
        )
    lines.append("")
    lines.append(
        "Vocabulary total is formatted as unique whitespace-insensitive content count followed by raw vocabulary-entry count in parentheses, matching the convention used in the vocabulary-composition table."
    )
    lines.append("")

    lines.append("## Table B. Chinese-Origin to Pinyin-Toned Mapping")
    lines.append("")
    lines.append("| Parameter | Mapping type | Chinese source tokens | Pinyin target tokens |")
    lines.append("|---|---|---:|---:|")
    for parameter in PARAMETERS:
        label = parameter["label"]
        t2 = table2_by_param[label]
        mapping_rows = [
            ("Shared 1:1", t2["shared_1to1_source"], t2["shared_1to1_target"]),
            ("Shared 2:1", t2["shared_2to1_source"], t2["shared_2to1_target"]),
            ("Shared 3:1", t2["shared_3to1_source"], t2["shared_3to1_target"]),
            ("Shared 4:1", t2["shared_4to1_source"], t2["shared_4to1_target"]),
            ("Shared >4:1", t2["shared_gt4to1_source"], t2["shared_gt4to1_target"]),
            ("Unique", t2["unique_source"], t2["unique_target"]),
        ]
        for mapping_type, source, target in mapping_rows:
            lines.append(f"| {label} | {mapping_type} | {source:,} | {target:,} |")
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "K=0 is qualitatively different from the nonzero transition settings. "
        "Chinese-Origin remains unchanged at about 0.5448 tokens/original character, and Pinyin-Diacritic remains close to the nonzero settings at about 0.5980. "
        "However, Pinyin-Toned drops to 0.5753 under K=0, compared with 0.5972-0.5989 for K=1000 through 10%; Pinyin-Toneless drops to 0.5574, compared with 0.5703-0.5726 for the nonzero settings. "
        "Thus K=0 does not simply confirm robustness: it changes tokenization efficiency for the numbered-pinyin settings."
    )
    lines.append("")
    lines.append(
        "Vocabulary composition also shifts under K=0. "
        "For Pinyin-Toned, the unique whitespace-insensitive vocabulary total rises to 50,638, far above the 42,448-43,248 range of the nonzero settings. "
        "For Pinyin-Toneless, K=0 rises to 49,794, above the 46,394-46,828 nonzero range. "
        "In contrast, Pinyin-Diacritic falls to 38,336, below the 42,439-43,274 nonzero range. "
        "This indicates that immediately enabling SuperBPE changes how the 64k vocabulary budget is allocated, rather than only making a small perturbation."
    )
    lines.append("")
    lines.append(
        "The Chinese-Origin to Pinyin-Toned mapping table shows the same discontinuity. "
        "K=0 has 41,649 Shared 1:1 tokens, much higher than the 31,748-32,628 range for K=1000 through 10%, and the Chinese-Origin unique count drops to 11,092 rather than about 20k-21k. "
        "The high-order many-to-one structure is still stable: Shared >4:1 remains exactly 3,025 Chinese source tokens to 362 Pinyin target tokens in every setting, and Shared 3:1/4:1 barely move. "
        "So K=0 does not remove the structural many-to-one effect of romanization, but it substantially changes the balance between 1:1 shared tokens and unique tokens."
    )
    lines.append("")
    lines.append(
        "Overall, K=0 should be treated as a boundary-case ablation, not as an interchangeable replacement for K=2048. "
        "The nonzero settings from K=1000 to 10% are stable, but K=0 changes fertility, vocabulary composition, and 1:1 overlap enough that it is less suitable as the main experimental setting. "
        "K=2048 remains more defensible because it preserves an initial ordinary-BPE phase, is fixed rather than percentage-dependent, and sits in the empirically stable nonzero range."
    )

    return "\n".join(lines), rows


def main() -> None:
    markdown, rows = build_summary()
    OUTPUT_MD.write_text(markdown + "\n", encoding="utf-8")

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {OUTPUT_MD}")
    print(f"Wrote {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
