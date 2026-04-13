import json
import jieba # cant install pkuseg
from tqdm import tqdm

INPUT_FILE = "./cleaned_wiki.jsonl"
OUTPUT_FILE = "./wiki_tokenized.jsonl"


# =========================
# Step 1: 标点规范化（轻量）
# =========================
def normalize_text(text):
    if not text:
        return ""

    table = {
        "，": ",",
        "。": ".",
        "！": "!",
        "？": "?",
        "；": ";",
        "：": ":",
        "（": "(",
        "）": ")",
        "“": "\"",
        "”": "\"",
        "‘": "'",
        "’": "'",
    }

    for k, v in table.items():
        text = text.replace(k, v)

    return text


# =========================
# Step 2: jieba 分词
# =========================
def tokenize(text):
    # 精确模式（适合语料处理）
    return list(jieba.cut(text, cut_all=False))


# =========================
# Step 3: 单条处理
# =========================
def process_line(obj):

    text = obj.get("text", "")
    text = normalize_text(text)

    tokens = tokenize(text)

    obj["tokens"] = tokens

    return obj


# =========================
# Step 4: 主流程
# =========================
def main():

    with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

        for line in tqdm(fin):
            obj = json.loads(line)

            obj = process_line(obj)

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print("Done ->", OUTPUT_FILE)


if __name__ == "__main__":
    main()