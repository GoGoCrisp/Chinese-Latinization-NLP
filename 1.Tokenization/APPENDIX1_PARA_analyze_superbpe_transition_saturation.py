import csv
import json
import os
import re
from pathlib import Path
from typing import Optional

from tokenizers import Regex, Tokenizer, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel, Split


BASE_DIR = Path(__file__).resolve().parent
STAGE1_ROOT = BASE_DIR / "superTokenizers_BPE"
PINYIN_DICT = BASE_DIR / "dicts" / "merged_pinyin_dict.json"
OUTPUT_DIR = BASE_DIR / "decoded_superTokenizers" / "transition_saturation"

VOCAB_SIZES = [32000, 64000]
REPRESENTATIONS = ["pinyin_toned", "pinyin_toneless", "pinyin_diacritic"]
THRESHOLDS = [0.95, 0.97, 0.99]
CHECKPOINT_KS = [0, 400, 800, 1200, 1600, 1800, 2048, 2400, 3200, 4096, 6400]

STAGE1_REGEX = r"\S+|\s+"


MARK_TO_BASE_TONE = {
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
    "ü": ("v", ""),
}

TONE_MARKS = {
    "a": ["ā", "á", "ǎ", "à"],
    "e": ["ē", "é", "ě", "è"],
    "i": ["ī", "í", "ǐ", "ì"],
    "o": ["ō", "ó", "ǒ", "ò"],
    "u": ["ū", "ú", "ǔ", "ù"],
    "v": ["ǖ", "ǘ", "ǚ", "ǜ"],
}


def normalize_to_numbered(pinyin: str) -> Optional[str]:
    pinyin = pinyin.strip().lower().replace("u:", "v").replace("ü", "v")
    if re.fullmatch(r"[a-zv]+[1-5]", pinyin):
        return pinyin

    base_chars = []
    tone = ""
    for char in pinyin:
        if char in MARK_TO_BASE_TONE:
            base, char_tone = MARK_TO_BASE_TONE[char]
            base_chars.append(base)
            if char_tone:
                tone = char_tone
        elif "a" <= char <= "z":
            base_chars.append(char)
        else:
            return None

    if not tone:
        return None

    base = "".join(base_chars).replace("ü", "v")
    if not re.fullmatch(r"[a-zv]+", base):
        return None
    return base + tone


def numbered_to_diacritic(numbered: str) -> str:
    base, tone = numbered[:-1], numbered[-1]
    if tone == "5":
        return base.replace("v", "ü")

    tone_index = int(tone) - 1
    if "a" in base:
        mark_pos = base.index("a")
    elif "e" in base:
        mark_pos = base.index("e")
    elif "ou" in base:
        mark_pos = base.index("o")
    else:
        vowel_positions = [i for i, char in enumerate(base) if char in "aeiouv"]
        if not vowel_positions:
            return base
        mark_pos = vowel_positions[-1]

    vowel = base[mark_pos]
    marked = TONE_MARKS[vowel][tone_index]
    return base[:mark_pos].replace("v", "ü") + marked + base[mark_pos + 1 :].replace("v", "ü")


def build_pinyin_inventory() -> dict[str, list[str]]:
    with open(PINYIN_DICT, "r", encoding="utf-8") as f:
        raw_data = json.load(f)["data"]

    numbered = set()
    for value in raw_data.values():
        normalized = normalize_to_numbered(value)
        if normalized:
            numbered.add(normalized)

    return {
        "pinyin_toned": sorted(numbered),
        "pinyin_toneless": sorted({item[:-1] for item in numbered}),
        "pinyin_diacritic": sorted({numbered_to_diacritic(item) for item in numbered}),
    }


def read_merges(path: Path) -> list[tuple[str, str]]:
    merges = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            left, right = line.split()
            merges.append((left, right))
    return merges


def build_tokenizer(stage1_dir: Path, k: int) -> Tokenizer:
    with open(stage1_dir / "vocab.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)

    merges = read_merges(stage1_dir / "merges.txt")[:k]
    tokenizer = Tokenizer(BPE(vocab=vocab, merges=merges, unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            Split(pattern=Regex(STAGE1_REGEX), behavior="isolated", invert=False),
            ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=False),
        ]
    )
    return tokenizer


def single_token_coverage(stage1_dir: Path, syllables: list[str], k: int) -> float:
    tokenizer = build_tokenizer(stage1_dir, k)
    single_token_count = 0
    for syllable in syllables:
        if len(tokenizer.encode(syllable).tokens) == 1:
            single_token_count += 1
    return single_token_count / len(syllables)


def find_min_k(stage1_dir: Path, syllables: list[str], threshold: float):
    max_k = len(read_merges(stage1_dir / "merges.txt"))
    lo, hi = 0, max_k
    best_k = None
    best_coverage = 0.0

    while lo <= hi:
        mid = (lo + hi) // 2
        coverage = single_token_coverage(stage1_dir, syllables, mid)
        if coverage >= threshold:
            best_k = mid
            best_coverage = coverage
            hi = mid - 1
        else:
            lo = mid + 1

    return best_k, best_coverage


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    inventory = build_pinyin_inventory()

    rows = []
    checkpoint_rows = []

    for representation in REPRESENTATIONS:
        syllables = inventory[representation]
        for vocab_size in VOCAB_SIZES:
            stage1_dir = STAGE1_ROOT / f"{representation}_subset100k_stage1_{vocab_size}"
            if not stage1_dir.exists():
                print(f"Missing stage1 tokenizer: {stage1_dir}")
                continue

            max_k = len(read_merges(stage1_dir / "merges.txt"))
            max_coverage = single_token_coverage(stage1_dir, syllables, max_k)
            for threshold in THRESHOLDS:
                min_k, coverage = find_min_k(stage1_dir, syllables, threshold)
                rows.append(
                    {
                        "representation": representation,
                        "stage1_vocab_size": vocab_size,
                        "inventory_size": len(syllables),
                        "threshold": threshold,
                        "min_k": min_k,
                        "coverage_at_min_k": coverage,
                        "max_k": max_k,
                        "max_coverage": max_coverage,
                    }
                )

            for k in CHECKPOINT_KS:
                if k > max_k:
                    continue
                coverage = single_token_coverage(stage1_dir, syllables, k)
                checkpoint_rows.append(
                    {
                        "representation": representation,
                        "stage1_vocab_size": vocab_size,
                        "inventory_size": len(syllables),
                        "k": k,
                        "coverage": coverage,
                    }
                )

    summary_path = OUTPUT_DIR / "transition_saturation_summary.csv"
    checkpoints_path = OUTPUT_DIR / "transition_saturation_checkpoints.csv"

    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    with open(checkpoints_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=checkpoint_rows[0].keys())
        writer.writeheader()
        writer.writerows(checkpoint_rows)

    print(f"Wrote {summary_path}")
    print(f"Wrote {checkpoints_path}")
    print()
    print("Minimum K by threshold:")
    for row in rows:
        min_k = row["min_k"] if row["min_k"] is not None else "not reached"
        print(
            f"{row['representation']:18s} stage1={row['stage1_vocab_size']:5d} "
            f"inventory={row['inventory_size']:4d} "
            f"threshold={row['threshold']:.2f} min_k={min_k!s:>11s} "
            f"coverage={row['coverage_at_min_k']:.4f} "
            f"max_coverage={row['max_coverage']:.4f}"
        )


if __name__ == "__main__":
    main()
