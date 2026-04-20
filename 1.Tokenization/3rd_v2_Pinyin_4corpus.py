import json
import os
from tqdm import tqdm
from pypinyin import pinyin, Style

INPUT_FILE = "./wiki_tokenized.jsonl"
OUTPUT_DIR = "./corpora"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def tokens_to_line(tokens):
    return " ".join(tokens)


# ===== 拼音转换函数 =====
# 采用“先转化为拼音，再按字加空格”的策略，这样大词库能发挥多音字消歧作用。

def to_pinyin_toned_spaced(tokens):
    result = []
    for word in tokens:
        # pinyin 返回的格式类似 [['zhong1'], ['guo2']]
        py = pinyin(word, style=Style.TONE3, strict=False)
        # 提取每个字的拼音，并用空格连接，而不是直接拼在一起
        result.append(" ".join([s[0] for s in py]))
    return result


def to_pinyin_toneless_spaced(tokens):
    result = []
    for word in tokens:
        py = pinyin(word, style=Style.NORMAL, strict=False)
        result.append(" ".join([s[0] for s in py]))
    return result


def to_pinyin_diacritic_spaced(tokens):
    result = []
    for word in tokens:
        py = pinyin(word, style=Style.TONE, strict=False)
        result.append(" ".join([s[0] for s in py]))
    return result


# ===== 输出文件 =====

# 这里使用了带 "_spaced" 的新后缀，避免覆盖以前的语料
f_toned = open(f"{OUTPUT_DIR}/pinyin_toned_spaced.txt", "w", encoding="utf-8")
f_toneless = open(f"{OUTPUT_DIR}/pinyin_toneless_spaced.txt", "w", encoding="utf-8")
f_diacritic = open(f"{OUTPUT_DIR}/pinyin_diacritic_spaced.txt", "w", encoding="utf-8")


# ===== 主循环 =====

print("开始生成每个字都有空格的拼音语料...")
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in tqdm(f):
        data = json.loads(line)
        tokens = data.get("tokens", [])

        if not tokens:
            continue

        # 1. 拼音（带数字声调，按字切分空格）
        f_toned.write(tokens_to_line(to_pinyin_toned_spaced(tokens)) + "\n")

        # 2. 拼音（无声调，按字切分空格）
        f_toneless.write(tokens_to_line(to_pinyin_toneless_spaced(tokens)) + "\n")

        # 3. 拼音（音标，按字切分空格）
        f_diacritic.write(tokens_to_line(to_pinyin_diacritic_spaced(tokens)) + "\n")


# ===== 关闭文件 =====

f_toned.close()
f_toneless.close()
f_diacritic.close()

print("✅ 新的 Spaced Corpora 生成完毕，保存在:", OUTPUT_DIR)
