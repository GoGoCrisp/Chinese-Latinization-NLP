from __future__ import annotations

"""
Table 2 vocabulary composition analysis for SuperBPE tokenizers.

This script produces mutually exclusive counts for the categories used in the
paper table:

    character-based Chinese BPE vs. Pinyin-Toned BPE
    at 8K / 16K / 32K / 64K vocabulary sizes.

Parent rows such as "SUB-SYLLABLE FRAGMENTS" are reported as subtotals. The
sanity check is computed only over leaf categories, so each tokenizer's leaf
counts must sum to its vocabulary size.

Each table cell is formatted as unique whitespace-insensitive content count, followed by
raw vocabulary-entry count in parentheses. The raw count preserves the original
SuperBPE vocabulary entries, including entries that differ only by whitespace.
"""

import csv
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

try:
    from pypinyin import Style, pinyin
    HAS_PYPINYIN = True
except ImportError:
    HAS_PYPINYIN = False


# ===== Paths / outputs =====

BASE_DIR = Path(__file__).resolve().parent
TOKENIZERS_DIR = BASE_DIR / "decoded_superTokenizers"
DICTS_DIR = BASE_DIR / "dicts"
CEDICT_PATH = DICTS_DIR / "cedict_ts.u8"
MERGED_PINYIN_DICT_PATH = DICTS_DIR / "merged_pinyin_dict.json"

OUTPUT_TXT = TOKENIZERS_DIR / "tokenizer_vocabulary_table2_superBPE.txt"
OUTPUT_CSV = TOKENIZERS_DIR / "tokenizer_vocabulary_table2_superBPE.csv"

VOCAB_SIZES = [8000, 16000, 32000, 64000]

TOKENIZER_FILES = {
    "chinese": {
        size: f"chinese_origin_subset100k_superbpe_{size}_decoded.json"
        for size in VOCAB_SIZES
    },
    "pinyin_toned": {
        size: f"pinyin_toned_subset100k_superbpe_{size}_decoded.json"
        for size in VOCAB_SIZES
    },
}


# ===== Category rows =====

CAT_SINGLE_LETTERS = "SINGLE LETTERS (A-Z)"
CAT_SINGLE_CJK = "SINGLE CJK CHARACTERS"
CAT_DIGITS = "DIGITS"
CAT_PUNCTUATION = "PUNCTUATION"
CAT_SUB_INITIALS = "(initials: zh, ch, sh, ...)"
CAT_SUB_FINALS = "(finals: ong, ang, ian, ...)"
CAT_SUB_OTHER = "(other partial sequences)"
CAT_ONE_SYLLABLE = "1-SYLLABLE / 1-CHAR TOKENS"
CAT_ONE_SYLLABLE_TONE = "1-SYLLABLE + TONE NUMBER"
CAT_TWO = "2-SYLLABLE / 2-CHAR"
CAT_THREE = "3-SYLLABLE / 3-CHAR"
CAT_FOUR = "4-SYLLABLE / 4-CHAR"
CAT_FIVE = "5-SYLLABLE / 5-CHAR"
CAT_SIX_PLUS = "6+-SYLLABLE / 6+-CHAR"
CAT_CROSS_WORD = "CROSS-WORD MERGES"
CAT_MIXED_LATIN_CJK = "MIXED (LATIN + CJK)"
CAT_JK_RARE = "JAPANESE / KOREAN / RARE CJK"
CAT_OTHER = "OTHER / UNKNOWN"

LEAF_CATEGORIES = [
    CAT_SINGLE_LETTERS,
    CAT_SINGLE_CJK,
    CAT_DIGITS,
    CAT_PUNCTUATION,
    CAT_SUB_INITIALS,
    CAT_SUB_FINALS,
    CAT_SUB_OTHER,
    CAT_ONE_SYLLABLE,
    CAT_ONE_SYLLABLE_TONE,
    CAT_TWO,
    CAT_THREE,
    CAT_FOUR,
    CAT_FIVE,
    CAT_SIX_PLUS,
    CAT_CROSS_WORD,
    CAT_MIXED_LATIN_CJK,
    CAT_JK_RARE,
    CAT_OTHER,
]

TABLE_ROWS = [
    ("Base inventory", None),
    (CAT_SINGLE_LETTERS, CAT_SINGLE_LETTERS),
    (CAT_SINGLE_CJK, CAT_SINGLE_CJK),
    (CAT_DIGITS, CAT_DIGITS),
    (CAT_PUNCTUATION, CAT_PUNCTUATION),
    ("Sub-syllable / sub-character units", None),
    ("SUB-SYLLABLE FRAGMENTS", "SUBTOTAL_SUB_SYLLABLE"),
    (CAT_SUB_INITIALS, CAT_SUB_INITIALS),
    (CAT_SUB_FINALS, CAT_SUB_FINALS),
    (CAT_SUB_OTHER, CAT_SUB_OTHER),
    ("Syllable-level / character-level tokens", None),
    (CAT_ONE_SYLLABLE, CAT_ONE_SYLLABLE),
    (CAT_ONE_SYLLABLE_TONE, CAT_ONE_SYLLABLE_TONE),
    ("Multi-syllable / multi-character tokens", None),
    (CAT_TWO, CAT_TWO),
    (CAT_THREE, CAT_THREE),
    (CAT_FOUR, CAT_FOUR),
    (CAT_FIVE, CAT_FIVE),
    (CAT_SIX_PLUS, CAT_SIX_PLUS),
    ("Cross-boundary and mixed tokens", None),
    (CAT_CROSS_WORD, CAT_CROSS_WORD),
    (CAT_MIXED_LATIN_CJK, CAT_MIXED_LATIN_CJK),
    ("Other", None),
    (CAT_JK_RARE, CAT_JK_RARE),
    (CAT_OTHER, CAT_OTHER),
    ("Total", "TOTAL"),
]


# ===== Pinyin definitions =====

VALID_PINYIN = {
    "a", "ai", "an", "ang", "ao", "ba", "bai", "ban", "bang", "bao",
    "bei", "ben", "beng", "bi", "bian", "biao", "bie", "bin", "bing",
    "bo", "bu", "ca", "cai", "can", "cang", "cao", "ce", "cen", "ceng",
    "cha", "chai", "chan", "chang", "chao", "che", "chen", "cheng",
    "chi", "chong", "chou", "chu", "chua", "chuai", "chuan", "chuang",
    "chui", "chun", "chuo", "ci", "cong", "cou", "cu", "cuan", "cui",
    "cun", "cuo", "da", "dai", "dan", "dang", "dao", "de", "dei",
    "deng", "di", "dian", "diao", "die", "ding", "diu", "dong", "dou",
    "du", "duan", "dui", "dun", "duo", "e", "ei", "en", "eng", "er",
    "fa", "fan", "fang", "fei", "fen", "feng", "fo", "fou", "fu", "ga",
    "gai", "gan", "gang", "gao", "ge", "gei", "gen", "geng", "gong",
    "gou", "gu", "gua", "guai", "guan", "guang", "gui", "gun", "guo",
    "ha", "hai", "han", "hang", "hao", "he", "hei", "hen", "heng",
    "hm", "hng", "hong", "hou", "hu", "hua", "huai", "huan", "huang",
    "hui", "hun", "huo", "ji", "jia", "jian", "jiang", "jiao", "jie",
    "jin", "jing", "jiong", "jiu", "ju", "juan", "jue", "jun", "ka",
    "kai", "kan", "kang", "kao", "ke", "kei", "ken", "keng", "kong",
    "kou", "ku", "kua", "kuai", "kuan", "kuang", "kui", "kun", "kuo",
    "la", "lai", "lan", "lang", "lao", "le", "lei", "leng", "li",
    "lia", "lian", "liang", "liao", "lie", "lin", "ling", "liu", "lo",
    "long", "lou", "lu", "lv", "lve", "luan", "lun", "luo", "m", "ma",
    "mai", "man", "mang", "mao", "me", "mei", "men", "meng", "mi",
    "mian", "miao", "mie", "min", "ming", "miu", "mo", "mou", "mu",
    "na", "nai", "nan", "nang", "nao", "ne", "nei", "nen", "neng",
    "ng", "ni", "nian", "niang", "niao", "nie", "nin", "ning", "niu",
    "nong", "nou", "nu", "nv", "nve", "nuan", "nun", "nuo", "o", "ou",
    "pa", "pai", "pan", "pang", "pao", "pei", "pen", "peng", "pi",
    "pian", "piao", "pie", "pin", "ping", "po", "pou", "pu", "qi",
    "qia", "qian", "qiang", "qiao", "qie", "qin", "qing", "qiong",
    "qiu", "qu", "quan", "que", "qun", "ran", "rang", "rao", "re",
    "ren", "reng", "ri", "rong", "rou", "ru", "ruan", "rui", "run",
    "ruo", "sa", "sai", "san", "sang", "sao", "se", "sen", "seng",
    "sha", "shai", "shan", "shang", "shao", "she", "shei", "shen",
    "sheng", "shi", "shou", "shu", "shua", "shuai", "shuan", "shuang",
    "shui", "shun", "shuo", "si", "song", "sou", "su", "suan", "sui",
    "sun", "suo", "ta", "tai", "tan", "tang", "tao", "te", "teng",
    "ti", "tian", "tiao", "tie", "ting", "tong", "tou", "tu", "tuan",
    "tui", "tun", "tuo", "wa", "wai", "wan", "wang", "wei", "wen",
    "weng", "wo", "wu", "xi", "xia", "xian", "xiang", "xiao", "xie",
    "xin", "xing", "xiong", "xiu", "xu", "xuan", "xue", "xun", "ya",
    "yan", "yang", "yao", "ye", "yi", "yin", "ying", "yo", "yong",
    "you", "yu", "yuan", "yue", "yun", "za", "zai", "zan", "zang",
    "zao", "ze", "zei", "zen", "zeng", "zha", "zhai", "zhan", "zhang",
    "zhao", "zhe", "zhei", "zhen", "zheng", "zhi", "zhong", "zhou",
    "zhu", "zhua", "zhuai", "zhuan", "zhuang", "zhui", "zhun", "zhuo",
    "zi", "zong", "zou", "zu", "zuan", "zui", "zun", "zuo",
}

PINYIN_INITIALS = {
    "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h",
    "j", "q", "x", "zh", "ch", "sh", "r", "z", "c", "s", "y", "w",
}

PINYIN_FINALS = {
    "a", "o", "e", "ai", "ei", "ao", "ou", "an", "en", "ang", "eng",
    "ong", "i", "ia", "ie", "iao", "iou", "iu", "ian", "in", "iang",
    "ing", "iong", "u", "ua", "uo", "uai", "uei", "ui", "uan", "uen",
    "un", "uang", "ueng", "v", "ve", "van", "vn", "ü", "üe", "üan",
    "ün", "er",
}

TONE_MARK_TO_BASE_AND_NUM = {
    "ā": ("a", "1"), "á": ("a", "2"), "ǎ": ("a", "3"), "à": ("a", "4"),
    "ē": ("e", "1"), "é": ("e", "2"), "ě": ("e", "3"), "è": ("e", "4"),
    "ī": ("i", "1"), "í": ("i", "2"), "ǐ": ("i", "3"), "ì": ("i", "4"),
    "ō": ("o", "1"), "ó": ("o", "2"), "ǒ": ("o", "3"), "ò": ("o", "4"),
    "ū": ("u", "1"), "ú": ("u", "2"), "ǔ": ("u", "3"), "ù": ("u", "4"),
    "ǖ": ("v", "1"), "ǘ": ("v", "2"), "ǚ": ("v", "3"), "ǜ": ("v", "4"),
    "ń": ("n", "2"), "ň": ("n", "3"), "ǹ": ("n", "4"),
    "ḿ": ("m", "2"),
}

TONE_MARKS = str.maketrans({
    mark: base for mark, (base, _tone) in TONE_MARK_TO_BASE_AND_NUM.items()
} | {"ü": "v"})


# ===== Character / token helpers =====

def strip_token(token: str) -> str:
    return token.replace("##", "").replace("Ġ", "")


def compact_token(token: str) -> str:
    return strip_token(token).strip().replace(" ", "")


def whitespace_insensitive_token(token: str) -> str:
    return re.sub(r"\s+", "", strip_token(token))


def is_standard_cjk_char(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff" and unicodedata.category(ch)[0] != "C"


def count_standard_cjk(token: str) -> int:
    return sum(1 for ch in token if is_standard_cjk_char(ch))


def is_all_standard_cjk(token: str) -> bool:
    return bool(token) and all(is_standard_cjk_char(ch) for ch in token)


def has_latin(token: str) -> bool:
    return bool(re.search(r"[A-Za-z]", token))


def has_standard_cjk(token: str) -> bool:
    return any(is_standard_cjk_char(ch) for ch in token)


def has_japanese_korean_or_rare_cjk(token: str) -> bool:
    if re.search(r"[\u3041-\u3096\u30A1-\u30FF\uAC00-\uD7AF]", token):
        return True
    if re.search(r"[\u3400-\u4DBF]", token):
        return True
    if re.search(r"<0x[0-9A-Fa-f]{2}>", token):
        return True
    for ch in token:
        code = ord(ch)
        if 0x20000 <= code <= 0x2FA1F:
            return True
    return False


def is_punctuation_token(token: str) -> bool:
    if not token:
        return False
    for ch in token:
        if ch.isspace():
            continue
        cat = unicodedata.category(ch)
        if not (cat.startswith("P") or cat.startswith("S")):
            return False
    return True


def normalize_pinyin_base(s: str) -> str:
    s = s.lower().replace("u:", "v").replace("ü", "v")
    s = s.translate(TONE_MARKS)
    s = re.sub(r"[1-5]", "", s)
    s = "".join(
        ch for ch in unicodedata.normalize("NFD", s)
        if unicodedata.category(ch) != "Mn"
    )
    return s


def tone_marks_to_numbered(s: str) -> str:
    """Convert one pinyin syllable with tone marks to tone-number style."""
    s = s.lower().replace("u:", "v").replace("ü", "v")
    if re.search(r"[1-5]$", s):
        return s

    tone = ""
    chars = []
    for ch in s:
        if ch in TONE_MARK_TO_BASE_AND_NUM:
            base, tone = TONE_MARK_TO_BASE_AND_NUM[ch]
            chars.append(base)
        else:
            chars.append(ch)

    numbered = "".join(chars)
    if tone:
        numbered += tone
    return numbered


def normalize_pinyin_toned_syllable(s: str) -> str:
    return tone_marks_to_numbered(s)


def is_valid_pinyin_syllable(s: str) -> bool:
    if not s or any("A" <= ch <= "Z" for ch in s):
        return False
    return normalize_pinyin_base(s) in VALID_PINYIN


def has_tone_number(s: str) -> bool:
    return bool(re.fullmatch(r"[a-züv:]+[1-5]", s.lower()))


def split_pinyin_syllables(token: str) -> list[str] | None:
    """Return pinyin syllables if token is entirely a valid pinyin sequence."""
    clean = strip_token(token)
    if not clean.strip():
        return None

    parts = [part for part in re.split(r"\s+", clean.strip()) if part]
    if len(parts) > 1:
        if all(is_valid_pinyin_syllable(part) for part in parts):
            return [normalize_pinyin_toned_syllable(part) for part in parts]
        return None

    one = parts[0] if parts else ""
    if is_valid_pinyin_syllable(one):
        return [normalize_pinyin_toned_syllable(one)]

    # Handle compact toned tokens such as zhong1guo2 if they occur.
    compact = one.lower().replace("u:", "v").replace("ü", "v")
    matches = re.findall(r"[a-zv]+[1-5]", compact)
    if matches and "".join(matches) == compact:
        if all(is_valid_pinyin_syllable(match) for match in matches):
            return [normalize_pinyin_toned_syllable(match) for match in matches]

    return None


def pinyin_sequence_key(syllables: list[str]) -> tuple[str, ...]:
    return tuple(normalize_pinyin_toned_syllable(s) for s in syllables)


def normalized_content_key(raw_token: str, side: str) -> str:
    """Return the whitespace-insensitive content used for unique counts."""
    return whitespace_insensitive_token(raw_token)


def is_pinyin_sequence_with_spaces(token: str) -> bool:
    clean = strip_token(token)
    return bool(re.search(r"\S\s+\S", clean.strip()))


def classify_latin_fragment(token: str) -> str | None:
    if not re.fullmatch(r"[A-Za-züÜv:]+", token):
        return None

    lower = token.lower().replace("ü", "v").replace("u:", "v")
    if len(lower) == 1:
        return None
    if is_valid_pinyin_syllable(lower):
        return None
    if lower in PINYIN_INITIALS:
        return CAT_SUB_INITIALS
    if lower in PINYIN_FINALS:
        return CAT_SUB_FINALS
    return CAT_SUB_OTHER


def load_vocab(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object vocabulary: {path}")
    return data


def load_merged_char_pinyin() -> dict[str, str]:
    if not MERGED_PINYIN_DICT_PATH.exists():
        return {}
    with MERGED_PINYIN_DICT_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    raw_mapping = data.get("data", {})
    mapping = {}
    for char, py in raw_mapping.items():
        if not char or not py:
            continue
        first = str(py).split()[0]
        numbered = normalize_pinyin_toned_syllable(first)
        if is_valid_pinyin_syllable(numbered):
            mapping[char] = numbered
    return mapping


def load_cedict_sequences() -> set[tuple[str, ...]]:
    sequences = set()
    if not CEDICT_PATH.exists():
        return sequences

    with CEDICT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            match = re.match(r"(\S+)\s+(\S+)\s+\[(.*?)\]", line)
            if not match:
                continue
            simplified = match.group(2)
            pinyin_text = match.group(3)
            if len(simplified) < 2 or not is_all_standard_cjk(simplified):
                continue

            syllables = [
                normalize_pinyin_toned_syllable(part)
                for part in pinyin_text.lower().split()
            ]
            if len(syllables) != len(simplified):
                continue
            if all(is_valid_pinyin_syllable(syllable) for syllable in syllables):
                sequences.add(tuple(syllables))

    return sequences


CEDICT_SEQUENCES = load_cedict_sequences()
MERGED_CHAR_PINYIN = load_merged_char_pinyin()


def build_known_chinese_pinyin_sequences(size: int) -> set[tuple[str, ...]]:
    """Build pinyin signatures for known Chinese tokens at the same vocab size.

    This gives the cross-word heuristic a conservative reference: a pinyin
    multi-syllable token that corresponds to a Chinese vocab token is treated as
    a multi-syllable token, while a valid pinyin sequence not found here is
    treated as a cross-word merge.
    """
    vocab_path = TOKENIZERS_DIR / TOKENIZER_FILES["chinese"][size]
    vocab = load_vocab(vocab_path)
    known = set(CEDICT_SEQUENCES)

    for raw_token in vocab:
        token = compact_token(raw_token)
        if count_standard_cjk(token) < 2 or not is_all_standard_cjk(token):
            continue

        syllables = []
        for char in token:
            py = MERGED_CHAR_PINYIN.get(char)
            if not py:
                syllables = []
                break
            syllables.append(py)

        if len(syllables) == len(token) and all(is_valid_pinyin_syllable(s) for s in syllables):
            known.add(tuple(syllables))

        if HAS_PYPINYIN:
            py = pinyin(token, style=Style.TONE3, strict=False)
            pypinyin_syllables = [
                normalize_pinyin_toned_syllable(item[0])
                for item in py
                if item and item[0]
            ]
            if len(pypinyin_syllables) == len(token):
                known.add(tuple(pypinyin_syllables))

    return known


def classify_token(raw_token: str, side: str, known_sequences: set[tuple[str, ...]]) -> str:
    clean = strip_token(raw_token)
    compact = compact_token(raw_token)

    if not compact:
        return CAT_OTHER

    # Rare scripts first: the table asks for these as an "Other" class.
    if has_japanese_korean_or_rare_cjk(compact):
        return CAT_JK_RARE

    if has_latin(compact) and has_standard_cjk(compact):
        return CAT_MIXED_LATIN_CJK

    # Base inventory.
    if re.fullmatch(r"[A-Za-z]", compact):
        return CAT_SINGLE_LETTERS
    if len(compact) == 1 and is_standard_cjk_char(compact):
        return CAT_SINGLE_CJK
    if re.fullmatch(r"\d+", compact):
        return CAT_DIGITS
    if is_punctuation_token(compact):
        return CAT_PUNCTUATION

    # Pure CJK multi-character tokens. Single CJK has already been consumed by
    # the base inventory row, so 1-CHAR is not counted a second time.
    if is_all_standard_cjk(compact):
        cjk_len = len(compact)
        if cjk_len == 2:
            return CAT_TWO
        if cjk_len == 3:
            return CAT_THREE
        if cjk_len == 4:
            return CAT_FOUR
        if cjk_len == 5:
            return CAT_FIVE
        if cjk_len >= 6:
            return CAT_SIX_PLUS

    # Pinyin syllable-level categories only apply to the pinyin-toned panel.
    if side == "pinyin_toned":
        syllables = split_pinyin_syllables(clean)
        if syllables:
            n_syllables = len(syllables)
            if (
                n_syllables >= 2
                and is_pinyin_sequence_with_spaces(clean)
                and known_sequences
                and pinyin_sequence_key(syllables) not in known_sequences
            ):
                return CAT_CROSS_WORD

            if n_syllables == 1:
                if has_tone_number(syllables[0]):
                    return CAT_ONE_SYLLABLE_TONE
                return CAT_ONE_SYLLABLE
            if n_syllables == 2:
                return CAT_TWO
            if n_syllables == 3:
                return CAT_THREE
            if n_syllables == 4:
                return CAT_FOUR
            if n_syllables == 5:
                return CAT_FIVE
            return CAT_SIX_PLUS

    fragment_category = classify_latin_fragment(compact)
    if fragment_category:
        return fragment_category

    return CAT_OTHER


def analyze_vocab(path: Path, side: str, known_sequences: set[tuple[str, ...]]) -> dict:
    vocab = load_vocab(path)
    counts = Counter()
    unique_keys = defaultdict(set)
    examples = defaultdict(list)

    for raw_token in vocab:
        category = classify_token(raw_token, side, known_sequences)
        counts[category] += 1
        unique_keys[category].add(normalized_content_key(raw_token, side))
        if len(examples[category]) < 8:
            examples[category].append(raw_token)

    total = len(vocab)
    leaf_total = sum(counts[category] for category in LEAF_CATEGORIES)
    unique_total = len(set().union(*(unique_keys[category] for category in LEAF_CATEGORIES)))

    return {
        "total": total,
        "unique_total": unique_total,
        "leaf_total": leaf_total,
        "counts": counts,
        "unique_counts": Counter({
            category: len(unique_keys[category])
            for category in LEAF_CATEGORIES
        }),
        "unique_keys": unique_keys,
        "examples": examples,
    }


def format_dual_count(unique_count: int, raw_count: int) -> str:
    return f"{unique_count} ({raw_count})"


def get_row_value(results: dict, side: str, size: int, key: str | None) -> str:
    if key is None:
        return ""

    result = results[side][size]
    counts = result["counts"]
    unique_counts = result["unique_counts"]

    if key == "TOTAL":
        return format_dual_count(result["unique_total"], result["total"])
    if key == "SUBTOTAL_SUB_SYLLABLE":
        subtotal = counts[CAT_SUB_INITIALS] + counts[CAT_SUB_FINALS] + counts[CAT_SUB_OTHER]
        unique_subtotal = len(
            result["unique_keys"][CAT_SUB_INITIALS]
            | result["unique_keys"][CAT_SUB_FINALS]
            | result["unique_keys"][CAT_SUB_OTHER]
        )
        return format_dual_count(unique_subtotal, subtotal)
    return format_dual_count(unique_counts[key], counts[key])


def format_markdown_table(results: dict) -> str:
    headers = [
        "Token Category",
        "Chinese 8K", "Chinese 16K", "Chinese 32K", "Chinese 64K",
        "Pinyin-Toned 8K", "Pinyin-Toned 16K", "Pinyin-Toned 32K", "Pinyin-Toned 64K",
    ]
    rows = []
    for label, key in TABLE_ROWS:
        values = [label]
        values.extend(get_row_value(results, "chinese", size, key) for size in VOCAB_SIZES)
        values.extend(get_row_value(results, "pinyin_toned", size, key) for size in VOCAB_SIZES)
        rows.append(values)

    widths = [
        max(len(str(row[i])) for row in [headers] + rows)
        for i in range(len(headers))
    ]

    lines = []
    lines.append(" | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    lines.append("-+-".join("-" * widths[i] for i in range(len(headers))))
    for row in rows:
        lines.append(" | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))))
    return "\n".join(lines)


def write_csv(results: dict) -> None:
    headers = [
        "Token Category",
        "Chinese 8K", "Chinese 16K", "Chinese 32K", "Chinese 64K",
        "Pinyin-Toned 8K", "Pinyin-Toned 16K", "Pinyin-Toned 32K", "Pinyin-Toned 64K",
    ]
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for label, key in TABLE_ROWS:
            row = [label]
            row.extend(get_row_value(results, "chinese", size, key) for size in VOCAB_SIZES)
            row.extend(get_row_value(results, "pinyin_toned", size, key) for size in VOCAB_SIZES)
            writer.writerow(row)


def format_examples(results: dict) -> str:
    lines = []
    lines.append("")
    lines.append("SAMPLE TOKENS BY LEAF CATEGORY")
    lines.append("=" * 100)
    for side in ["chinese", "pinyin_toned"]:
        for size in VOCAB_SIZES:
            result = results[side][size]
            lines.append("")
            lines.append(f"{side} {size}")
            lines.append("-" * 80)
            for category in LEAF_CATEGORIES:
                sample = result["examples"].get(category, [])
                if sample:
                    sample_str = ", ".join(repr(x) for x in sample)
                    lines.append(f"{category:<38} {sample_str}")
    return "\n".join(lines)


def format_sanity_checks(results: dict) -> str:
    lines = []
    lines.append("")
    lines.append("SANITY CHECKS")
    lines.append("=" * 100)
    if not HAS_PYPINYIN:
        lines.append(
            "NOTE: pypinyin is not installed. CROSS-WORD MERGES uses CEDICT plus "
            "merged_pinyin_dict.json instead of pypinyin context readings."
        )
    lines.append("Table cells are formatted as: unique whitespace-insensitive content (raw vocab entries).")
    lines.append(f"CEDICT pinyin sequences loaded: {len(CEDICT_SEQUENCES)}")
    lines.append(f"Merged char pinyin entries loaded: {len(MERGED_CHAR_PINYIN)}")

    for side in ["chinese", "pinyin_toned"]:
        for size in VOCAB_SIZES:
            result = results[side][size]
            delta = result["leaf_total"] - result["total"]
            status = "OK" if delta == 0 else f"BAD delta={delta}"
            lines.append(
                f"{side:<13} {size:>5}: leaf_total={result['leaf_total']:<6} "
                f"vocab_size={result['total']:<6} {status}"
            )
    return "\n".join(lines)


def main() -> None:
    print("=" * 100)
    print("TABLE 2 VOCABULARY COMPOSITION ANALYSIS - SUPERBPE")
    print("=" * 100)

    known_sequences_by_size = {}
    for size in VOCAB_SIZES:
        print(f"Building known Chinese pinyin sequence reference for {size}...")
        known_sequences_by_size[size] = build_known_chinese_pinyin_sequences(size)
        print(f"  known sequences: {len(known_sequences_by_size[size])}")

    results = {"chinese": {}, "pinyin_toned": {}}

    for side in ["chinese", "pinyin_toned"]:
        for size in VOCAB_SIZES:
            path = TOKENIZERS_DIR / TOKENIZER_FILES[side][size]
            print(f"Analyzing {side} {size}: {path.name}")
            known_sequences = known_sequences_by_size[size] if side == "pinyin_toned" else set()
            results[side][size] = analyze_vocab(path, side, known_sequences)

    report_parts = [
        "TABLE 2 VOCABULARY COMPOSITION ANALYSIS - SUPERBPE",
        "=" * 100,
        "",
        "Counts are mutually exclusive at the leaf-category level.",
        "Each numeric cell is formatted as unique whitespace-insensitive content count (raw vocab-entry count).",
        "Unique content removes whitespace variants only; otherwise distinct token strings remain distinct.",
        "The SUB-SYLLABLE FRAGMENTS row is a subtotal of initials, finals, and other partial sequences.",
        "SINGLE CJK CHARACTERS consumes 1-character Chinese tokens, so 1-SYLLABLE / 1-CHAR does not recount them.",
        "CROSS-WORD MERGES for pinyin-toned tokenizers are valid multi-syllable pinyin tokens whose syllable sequence is not found among same-size Chinese-origin vocabulary tokens converted to numbered-tone pinyin.",
        "",
        format_markdown_table(results),
        format_sanity_checks(results),
        format_examples(results),
        "",
    ]

    OUTPUT_TXT.write_text("\n".join(report_parts), encoding="utf-8")
    write_csv(results)

    print("")
    print(format_markdown_table(results))
    print(format_sanity_checks(results))
    print("")
    print(f"Report saved to: {OUTPUT_TXT}")
    print(f"CSV saved to:    {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
