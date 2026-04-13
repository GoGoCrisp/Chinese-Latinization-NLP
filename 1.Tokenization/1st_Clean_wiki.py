import os
import re
import json
from tqdm import tqdm
from opencc import OpenCC

# ========== 初始化 ==========
cc = OpenCC('t2s')

INPUT_DIR = "./extracted/AA"
OUTPUT_FILE = "./cleaned_wiki.jsonl"


# ========== 清洗函数 ==========
def clean_text(text):

    if not text:
        return ""

    # 1. 繁体 -> 简体
    text = cc.convert(text)

    # 2. 去掉引用 [1] [23]
    text = re.sub(r'\[\d+\]', '', text)

    # 3. 去掉 wiki 内部链接 [xxx]
    text = re.sub(r'\[[^\]]*\]', '', text)

    # 4. 去掉“参考文献/外部链接/参见”之后内容（强烈建议截断）
    text = re.split(r'==\s*(参考文献|外部链接|参见|扩展阅读)\s*==', text)[0]

    # 5. 去 HTML / 特殊符号
    text = re.sub(r'[{}【】<>]', '', text)

    # 6. 去列表符号
    text = re.sub(r'^\s*[\-\*].*$', '', text, flags=re.MULTILINE)

    # 7. 合并空白
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# ========== 判断是否有效文本 ==========
def is_valid(text):
    if not text:
        return False
    if len(text) < 50:   # 太短不要
        return False
    if len(set(text)) < 5:  # 极端重复字符
        return False
    return True


# ========== 解析 WikiExtractor JSON ==========
def parse_line(line):
    try:
        return json.loads(line)
    except:
        return None


# ========== 主处理 ==========
def process_file(filepath, fout):
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            obj = parse_line(line)
            if not obj:
                continue

            text = obj.get("text", "")
            title = obj.get("title", "")
            wid = obj.get("id", "")

            clean = clean_text(text)

            if not is_valid(clean):
                continue

            out = {
                "id": wid,
                "title": title,
                "text": clean
            }

            fout.write(json.dumps(out, ensure_ascii=False) + "\n")


# ========== 遍历目录 ==========
def main():
    files = []

    for root, _, fs in os.walk(INPUT_DIR):
        for f in fs:
            files.append(os.path.join(root, f))

    print(f"Found {len(files)} files")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
        for file in tqdm(files):
            process_file(file, fout)

    print("Done ->", OUTPUT_FILE)


if __name__ == "__main__":
    main()