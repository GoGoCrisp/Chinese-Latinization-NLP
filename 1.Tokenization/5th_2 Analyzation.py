import os
import pandas as pd
from tokenizers import Tokenizer
from tqdm import tqdm

BASE_DIR = "/Users/crisp/Desktop/code_field/python/Chinese_Latinization_NLP/1.Tokenization"

CORPUS_DIR = os.path.join(BASE_DIR, "corpora")
TOKENIZER_DIR = os.path.join(BASE_DIR, "tokenizers")
DICT_PATH = os.path.join(BASE_DIR, "dicts/cedict_ts.u8")

OUTPUT_PATH = os.path.join(BASE_DIR, "evaluation_results.csv")


# =========================
# 固定实验顺序（关键）
# =========================
CORPORA = [
    {
        "name": "chinese_origin",
        "test_file": "chinese_origin_中国_test10.txt",
        "is_chinese": True
    },
    {
        "name": "pinyin_diacritic",
        "test_file": "pinyin_diacritic_zhōngguó_test10.txt",
        "is_chinese": False
    },
    {
        "name": "pinyin_toned",
        "test_file": "pinyin_toned_zhong1guo2_test10.txt",
        "is_chinese": False
    },
    {
        "name": "pinyin_toneless",
        "test_file": "pinyin_toneless_zhongguo_test10.txt",
        "is_chinese": False
    }
]

VOCAB_SIZES = ["8k", "16k", "32k", "64k"]


# =========================
# Load CEDICT
# =========================
def load_cedict(path):
    vocab = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split(" ")
            if len(parts) > 1:
                vocab.add(parts[1])
    return vocab


cedict_vocab = load_cedict(DICT_PATH)


# =========================
# Boundary F1
# =========================
def get_boundaries_from_words(words):
    boundaries = set()
    pos = 0
    for w in words[:-1]:
        pos += len(w)
        boundaries.add(pos)
    return boundaries


def get_boundaries_from_tokens(tokens):
    boundaries = set()
    pos = 0
    for t in tokens[:-1]:
        pos += len(t)
        boundaries.add(pos)
    return boundaries


def boundary_f1_score(words, tokens):
    gold = get_boundaries_from_words(words)
    pred = get_boundaries_from_tokens(tokens)

    if len(pred) == 0 or len(gold) == 0:
        return 0

    tp = len(gold & pred)
    precision = tp / len(pred)
    recall = tp / len(gold)

    if precision + recall == 0:
        return 0

    return 2 * precision * recall / (precision + recall)


# =========================
# Evaluation
# =========================
def evaluate(tokenizer_path, corpus_path, is_chinese=True):
    tokenizer = Tokenizer.from_file(tokenizer_path)

    total_tokens = 0
    total_sentences = 0
    total_chars = 0
    total_bytes = 0

    dict_match_count = 0
    dict_total = 0

    boundary_scores = []

    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=os.path.basename(tokenizer_path)):
            line = line.strip()
            if not line:
                continue

            encoding = tokenizer.encode(line)
            tokens = encoding.tokens

            total_tokens += len(tokens)
            total_sentences += 1
            total_chars += len(line)
            total_bytes += len(line.encode("utf-8"))

            # dict match（仅中文）
            if is_chinese:
                for t in tokens:
                    if len(t) >= 2:
                        dict_total += 1
                        if t in cedict_vocab:
                            dict_match_count += 1

            # boundary F1
            words = line.split()
            f1 = boundary_f1_score(words, tokens)
            boundary_scores.append(f1)

    return {
        "fertility_sent": total_tokens / total_sentences,
        "fertility_char": total_tokens / total_chars,
        "dict_match": dict_match_count / dict_total if dict_total > 0 else None,
        "boundary_f1": sum(boundary_scores) / len(boundary_scores),
        "char_per_token": total_chars / total_tokens,
        "bytes_per_token": total_bytes / total_tokens,
    }


# =========================
# 主循环（严格顺序）
# =========================
results = []

for corpus in CORPORA:
    corpus_name = corpus["name"]
    test_file = corpus["test_file"]
    is_chinese = corpus["is_chinese"]

    corpus_path = os.path.join(CORPUS_DIR, test_file)

    print(f"\n=== Evaluating {corpus_name} ===")

    for vocab in VOCAB_SIZES:
        tokenizer_name = f"{corpus_name}_{vocab}_train90.json"
        tokenizer_path = os.path.join(TOKENIZER_DIR, tokenizer_name)

        if not os.path.exists(tokenizer_path):
            print(f"Missing: {tokenizer_name}")
            continue

        print(f"-> {tokenizer_name}")

        metrics = evaluate(tokenizer_path, corpus_path, is_chinese)

        row = {
            "corpus": corpus_name,
            "vocab": vocab,
            "tokenizer": tokenizer_name,
            **metrics
        }

        results.append(row)


# =========================
# 保存
# =========================
df = pd.DataFrame(results)

# 强制排序（双保险）
df["vocab"] = pd.Categorical(df["vocab"], VOCAB_SIZES, ordered=True)
df = df.sort_values(by=["corpus", "vocab"])

df.to_csv(OUTPUT_PATH, index=False)

print("\nSaved to:", OUTPUT_PATH)