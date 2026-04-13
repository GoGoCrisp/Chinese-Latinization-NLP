import json
from tqdm import tqdm
from pypinyin import pinyin, Style

INPUT_FILE = "./wiki_tokenized.jsonl"
OUTPUT_DIR = "./corpora"

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)


def tokens_to_line(tokens):
    return " ".join(tokens)


# ===== 拼音转换函数 =====

def to_pinyin_toned(tokens):
    result = []
    for word in tokens:
        py = pinyin(word, style=Style.TONE3, strict=False)
        result.append("".join([s[0] for s in py]))
    return result


def to_pinyin_toneless(tokens):
    result = []
    for word in tokens:
        py = pinyin(word, style=Style.NORMAL, strict=False)
        result.append("".join([s[0] for s in py]))
    return result


def to_pinyin_diacritic(tokens):
    result = []
    for word in tokens:
        py = pinyin(word, style=Style.TONE, strict=False)
        result.append("".join([s[0] for s in py]))
    return result


# ===== 输出文件 =====

f_origin = open(f"{OUTPUT_DIR}/chinese_origin_中国.txt", "w", encoding="utf-8")
f_toned = open(f"{OUTPUT_DIR}/pinyin_toned_zhong1guo2.txt", "w", encoding="utf-8")
f_toneless = open(f"{OUTPUT_DIR}/pinyin_toneless_zhongguo.txt", "w", encoding="utf-8")
f_diacritic = open(f"{OUTPUT_DIR}/pinyin_diacritic_zhōngguó.txt", "w", encoding="utf-8")


# ===== 主循环 =====

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in tqdm(f):
        data = json.loads(line)
        tokens = data.get("tokens", [])

        if not tokens:
            continue

        # 1. 原始中文
        f_origin.write(tokens_to_line(tokens) + "\n")

        # 2. 拼音（带数字声调）
        f_toned.write(tokens_to_line(to_pinyin_toned(tokens)) + "\n")

        # 3. 拼音（无声调）
        f_toneless.write(tokens_to_line(to_pinyin_toneless(tokens)) + "\n")

        # 4. 拼音（音标）
        f_diacritic.write(tokens_to_line(to_pinyin_diacritic(tokens)) + "\n")


# ===== 关闭文件 =====

f_origin.close()
f_toned.close()
f_toneless.close()
f_diacritic.close()

print("✅ All corpora generated in:", OUTPUT_DIR)