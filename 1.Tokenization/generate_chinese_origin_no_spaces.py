"""
从cleaned_wiki.jsonl重新生成chinese_origin corpora
直接提取原始中文文本，不做分词
流程：cleaned_wiki.jsonl → normalize → 原始中文文本 → split train/test
"""

import json
import os
import random
from tqdm import tqdm

# 配置
INPUT_FILE = "cleaned_wiki.jsonl"
OUTPUT_DIR = "./corpora"
CORPUS_FILE = os.path.join(OUTPUT_DIR, "chinese_origin_中国.txt")
TRAIN_FILE = os.path.join(OUTPUT_DIR, "chinese_origin_中国_train90.txt")
TEST_FILE = os.path.join(OUTPUT_DIR, "chinese_origin_中国_test10.txt")

TRAIN_RATIO = 0.9
SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(SEED)


# =========================
# Step 1: 标点规范化
# =========================
def normalize_text(text):
    """规范化文本中的标点符号"""
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
        """: "\"",
        """: "\"",
        "'": "'",
        "'": "'",
    }

    for k, v in table.items():
        text = text.replace(k, v)

    return text


# =========================
# Step 2: 生成文本（无分词）
# =========================
def process_text(text):
    """
    直接返回规范化后的文本，不进行分词
    保持原始的中文文本样子
    """
    return text


# =========================
# 主流程
# =========================
def main():
    print("=" * 80)
    print("REGENERATING CHINESE_ORIGIN CORPORA FROM cleaned_wiki.jsonl")
    print("=" * 80)
    print()

    # Step 1: 从cleaned_wiki.jsonl读取并处理
    print("Step 1: Reading cleaned_wiki.jsonl and extracting text...")
    print("-" * 80)

    all_lines = []

    if not os.path.exists(INPUT_FILE):
        print(f"✗ Input file not found: {INPUT_FILE}")
        return

    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            line_count = 0
            for line in tqdm(f, desc="Processing"):
                try:
                    data = json.loads(line)
                    text = data.get("text", "").strip()

                    if not text:
                        continue

                    # Normalize标点
                    text = normalize_text(text)

                    # 直接保存（无分词）
                    if text:
                        all_lines.append(text)
                        line_count += 1

                except json.JSONDecodeError:
                    continue

    except Exception as e:
        print(f"✗ Error reading input file: {e}")
        return

    print(f"✓ Processed {line_count} lines")
    print(f"  Total generated lines: {len(all_lines)}")
    print()

    if not all_lines:
        print("✗ No data generated. Exiting.")
        return

    # Step 2: 保存完整的corpus文件
    print("Step 2: Saving complete corpus file...")
    print("-" * 80)

    try:
        with open(CORPUS_FILE, "w", encoding="utf-8") as f:
            for line in all_lines:
                f.write(line + "\n")
        print(f"✓ Saved to: {CORPUS_FILE}")
        print(f"  Total lines: {len(all_lines)}")
    except Exception as e:
        print(f"✗ Error saving corpus file: {e}")
        return

    print()

    # Step 3: 分割为train/test
    print("Step 3: Splitting into train (90%) and test (10%)...")
    print("-" * 80)

    # 随机打乱
    random.shuffle(all_lines)

    split_idx = int(len(all_lines) * TRAIN_RATIO)
    train_lines = all_lines[:split_idx]
    test_lines = all_lines[split_idx:]

    # 保存train文件
    try:
        with open(TRAIN_FILE, "w", encoding="utf-8") as f:
            for line in train_lines:
                f.write(line + "\n")
        print(f"✓ Train file: {TRAIN_FILE}")
        print(f"  Lines: {len(train_lines)}")
    except Exception as e:
        print(f"✗ Error saving train file: {e}")
        return

    # 保存test文件
    try:
        with open(TEST_FILE, "w", encoding="utf-8") as f:
            for line in test_lines:
                f.write(line + "\n")
        print(f"✓ Test file: {TEST_FILE}")
        print(f"  Lines: {len(test_lines)}")
    except Exception as e:
        print(f"✗ Error saving test file: {e}")
        return

    print()
    print("=" * 80)
    print("SUCCESS!")
    print("=" * 80)
    print()
    print("Generated files:")
    print(f"  1. {CORPUS_FILE} ({len(all_lines)} lines)")
    print(f"  2. {TRAIN_FILE} ({len(train_lines)} lines)")
    print(f"  3. {TEST_FILE} ({len(test_lines)} lines)")
    print()
    print("Features:")
    print("  ✓ Raw Chinese text (no tokenization)")
    print("  ✓ Normalized punctuation")
    print("  ✓ No spaces between Chinese characters")
    print()


if __name__ == "__main__":
    main()
