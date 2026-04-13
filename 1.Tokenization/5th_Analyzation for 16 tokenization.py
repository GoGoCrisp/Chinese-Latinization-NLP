import os
import csv
from tokenizers import Tokenizer

BASE_DIR = "/Users/crisp/Desktop/code_field/python/Chinese_Latinization_NLP/1.Tokenization"

CORPORA_DIR = os.path.join(BASE_DIR, "corpora")
TOKENIZER_DIR = os.path.join(BASE_DIR, "tokenizers")
DICT_DIR = os.path.join(BASE_DIR, "dicts")

OUTPUT_FILE = os.path.join(BASE_DIR, "evaluation_4abcd.csv")

# ===== DETECT TYPE =====
def detect_type(name):
    name = name.lower()
    if "chinese_origin" in name:
        return "origin"
    elif "pinyin_toned" in name:
        return "toned"
    elif "pinyin_toneless" in name:
        return "toneless"
    elif "pinyin_diacritic" in name:
        return "diacritic"
    return None

# ===== FIND TEST FILE =====
def find_test_file(t_type):
    for file in os.listdir(CORPORA_DIR):
        fname = file.lower()

        if not fname.endswith("test10.txt"):
            continue

        if t_type == "origin" and "chinese_origin" in fname:
            return os.path.join(CORPORA_DIR, file)

        if t_type != "origin" and f"pinyin_{t_type}" in fname:
            return os.path.join(CORPORA_DIR, file)

    raise ValueError(f"No test file found for {t_type}")

# ===== LOAD TEXT =====
def load_texts(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

# ===== LOAD DICTS =====
def load_dict(name):
    path = os.path.join(DICT_DIR, name)
    with open(path, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f)

dict_map = {
    "toned": load_dict("dict_toned.txt"),
    "toneless": load_dict("dict_toneless.txt"),
    "diacritic": load_dict("dict_diacritic.txt"),
}

# ===== ENGLISH VOCAB（baseline）=====
english_vocab = set()

# 用所有 test 构建一个弱 baseline
for file in os.listdir(CORPORA_DIR):
    if "test10" in file:
        texts = load_texts(os.path.join(CORPORA_DIR, file))
        for t in texts[:200]:
            for w in t.split():
                english_vocab.add(w.lower())

# ===== MAIN =====
results = []

for file in os.listdir(TOKENIZER_DIR):
    if not file.endswith(".json"):
        continue

    t_type = detect_type(file)
    if t_type is None:
        continue

    print(f"Running: {file} → {t_type}")

    tokenizer = Tokenizer.from_file(os.path.join(TOKENIZER_DIR, file))

    test_path = find_test_file(t_type)
    texts = load_texts(test_path)

    dictionary = dict_map.get(t_type, None)

    total_tokens = 0
    total_chars = 0
    total_bytes = 0

    valid_tokens = 0
    checked_tokens = 0

    for text in texts:
        enc = tokenizer.encode(text)
        tokens = enc.tokens

        total_tokens += len(tokens)
        total_chars += len(text)
        total_bytes += len(text.encode("utf-8"))

        # ===== 4b =====
        if dictionary:
            for tok in tokens:
                if len(tok) >= 2:
                    checked_tokens += 1
                    if tok in dictionary:
                        valid_tokens += 1

    # ===== 4a =====
    tokens_per_sentence = total_tokens / len(texts)
    tokens_per_char = total_tokens / total_chars

    # ===== 4b =====
    morph_score = (valid_tokens / checked_tokens) if dictionary and checked_tokens > 0 else None

    # ===== 4c =====
    vocab = set(tokenizer.get_vocab().keys())
    overlap = len(vocab & english_vocab) / len(vocab)

    # ===== 4d =====
    chars_per_token = total_chars / total_tokens
    bytes_per_token = total_bytes / total_tokens

    results.append([
        file,
        t_type,
        tokens_per_sentence,
        tokens_per_char,
        morph_score,
        overlap,
        chars_per_token,
        bytes_per_token
    ])

# ===== SAVE =====
with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "tokenizer",
        "type",
        "tokens_per_sentence",
        "tokens_per_char",
        "morphological_coherence",
        "cross_lingual_overlap",
        "chars_per_token",
        "bytes_per_token"
    ])
    writer.writerows(results)

print("✅ DONE:", OUTPUT_FILE)