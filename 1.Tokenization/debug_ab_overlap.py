"""
A vs B Overlap 机制调试脚本
不修改原有代码，只是做case study来理解overlap是如何计算的
"""

import json
import os
import re
from tokenizers import Tokenizer as HFTokenizer

# ===== 配置 =====
TOKENIZERS_DIR = "./tokenizers"
DICTS_DIR = "./dicts"

# 加载tokenizers
A_path = os.path.join(TOKENIZERS_DIR, "chinese_origin_64k_train90.json")
B_path = os.path.join(TOKENIZERS_DIR, "pinyin_toneless_64k_train90.json")

print("=" * 100)
print("A vs B OVERLAP MECHANISM CASE STUDY")
print("=" * 100)
print("")

# 加载vocabulary
print("Loading tokenizers...")
tokenizer_A = HFTokenizer.from_file(A_path)
tokenizer_B = HFTokenizer.from_file(B_path)

vocab_A = tokenizer_A.get_vocab()
vocab_B = tokenizer_B.get_vocab()

print(f"✓ Vocab A: {len(vocab_A)} tokens")
print(f"✓ Vocab B: {len(vocab_B)} tokens")
print("")

# 加载CEDICT字典
print("Loading CEDICT dictionary...")
cedict_path = os.path.join(DICTS_DIR, "cedict_ts.u8")
char_to_pinyin = {}
word_to_pinyin = {}  # 🆕 新增词级别的拼音映射

with open(cedict_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        # 格式: 繁体 简体 [拼音] /定义/
        simplified = parts[1]
        pinyin_str = parts[2].strip("[]")
        pinyin_list = pinyin_str.split()
        
        if simplified and pinyin_list:
            # 1. 保存词级别的拼音
            word_to_pinyin[simplified] = pinyin_list
            
            # 2. 如果是单字，也保存到char_to_pinyin（向后兼容）
            if len(simplified) == 1:
                char_to_pinyin[simplified] = pinyin_list

print(f"✓ Loaded {len(word_to_pinyin)} words")
print(f"✓ Loaded {len(char_to_pinyin)} characters")
print("")

# ===== 工具函数 =====

def normalize_token(token):
    """规范化token"""
    return token.replace("##", "").replace("Ġ", "").strip()

def is_special_token(token):
    """检查是否为特殊token"""
    special_tokens = ['[UNK]', '[PAD]', '[CLS]', '[SEP]', '[MASK]']
    return token in special_tokens or token.startswith('##') or token == 'Ġ'

def remove_tone_marks(pinyin):
    """移除拼音中的声调标记"""
    tone_map = {
        'ā': 'a', 'á': 'a', 'ǎ': 'a', 'à': 'a',
        'ē': 'e', 'é': 'e', 'ě': 'e', 'è': 'e',
        'ī': 'i', 'í': 'i', 'ǐ': 'i', 'ì': 'i',
        'ō': 'o', 'ó': 'o', 'ǒ': 'o', 'ò': 'o',
        'ū': 'u', 'ú': 'u', 'ǔ': 'u', 'ù': 'u',
        'ǖ': 'v', 'ǘ': 'v', 'ǚ': 'v', 'ǜ': 'v',
    }
    result = []
    for char in pinyin:
        result.append(tone_map.get(char, char))
    return "".join(result)

def remove_tone_numbers(pinyin):
    """移除拼音中的数字"""
    return re.sub(r'[0-9]', '', pinyin)

def convert_chinese_to_pinyin_toneless(chinese_word):
    """
    将中文词转换为无声调拼音
    改进：先查词级别，再查字级别
    """
    # 🆕 优先查词级别
    if chinese_word in word_to_pinyin:
        py_list = word_to_pinyin[chinese_word]
        # 拼接所有音节，并去掉声调
        converted_parts = []
        for py in py_list:
            py_clean = remove_tone_numbers(remove_tone_marks(py))
            converted_parts.append(py_clean)
        return "".join(converted_parts)
    
    # 降级到字级别转换
    converted_parts = []
    for char in chinese_word:
        if char in char_to_pinyin:
            py = char_to_pinyin[char][0]
            py_clean = remove_tone_numbers(remove_tone_marks(py))
            converted_parts.append(py_clean)
        else:
            converted_parts.append(char)
    return "".join(converted_parts)

# ===== Case Study =====

print("=" * 100)
print("CASE STUDY: Analyzing specific examples")
print("=" * 100)
print("")

# 从A中随机抽取一些多字词
print("🔍 Finding multi-character Chinese tokens in Vocab A...")
print("")

multi_char_tokens_A = []
for token in vocab_A.keys():
    if is_special_token(token):
        continue
    clean = normalize_token(token)
    if len(clean) >= 2 and all('\u4e00' <= c <= '\u9fff' for c in clean):
        multi_char_tokens_A.append(clean)

print(f"Found {len(multi_char_tokens_A)} multi-character Chinese tokens")
print("")

# 选择前1000个进行详细分析
test_tokens = multi_char_tokens_A[:1000]

print(f"Analyzing {len(test_tokens)} test cases:")
print("")

found_count = 0
not_found_count = 0
found_cases = []
not_found_cases = []

# 简化输出格式，每50个显示一次进度
for idx, chinese_token in enumerate(test_tokens):
    if idx % 50 == 0:
        print(f"Progress: {idx}/{len(test_tokens)}")
    
    # 转换为拼音
    pinyin_toneless = convert_chinese_to_pinyin_toneless(chinese_token)
    
    # 检查是否在B中
    in_vocab_b = pinyin_toneless in vocab_B
    
    if in_vocab_b:
        found_count += 1
        found_cases.append((chinese_token, pinyin_toneless))
    else:
        not_found_count += 1
        not_found_cases.append((chinese_token, pinyin_toneless))

print(f"Complete!")
print("")

print("=" * 100)
print("SUMMARY")
print("=" * 100)
print("")
print(f"Test cases analyzed: {len(test_tokens)}")
print(f"✓ Found in vocab B:     {found_count} ({100*found_count/len(test_tokens):.1f}%)")
print(f"✗ Not found in vocab B: {not_found_count} ({100*not_found_count/len(test_tokens):.1f}%)")
print("")

if found_cases:
    print("Examples of SUCCESSFUL overlap (中文 ↔ 拼音):")
    for chinese, pinyin in found_cases[:5]:
        print(f"  ✓ '{chinese}' ↔ '{pinyin}'")
    print("")

if not_found_cases:
    print("Examples of FAILED overlap (中文 ↔ 拼音):")
    for chinese, pinyin in not_found_cases[:5]:
        print(f"  ✗ '{chinese}' → '{pinyin}' [NOT IN VOCAB B]")
    print("")

# 更深入的分析：收集所有A中的词并统计转换后在B中的覆盖率
print("")
print("=" * 100)
print("FULL ANALYSIS: All multi-char tokens from Vocab A")
print("=" * 100)
print("")

print("Scanning all multi-character Chinese tokens in Vocab A...")

total_multi_char = 0
total_found_in_b = 0

for token in vocab_A.keys():
    if is_special_token(token):
        continue
    clean = normalize_token(token)
    
    # 统计多字词
    if len(clean) >= 2 and all('\u4e00' <= c <= '\u9fff' for c in clean):
        total_multi_char += 1
        
        # 转换并检查
        pinyin_toneless = convert_chinese_to_pinyin_toneless(clean)
        if pinyin_toneless in vocab_B:
            total_found_in_b += 1

print("")
print(f"Total multi-character Chinese tokens in A: {total_multi_char}")
print(f"Successfully matched in B:                  {total_found_in_b}")
coverage = 100 * total_found_in_b / total_multi_char if total_multi_char > 0 else 0
print(f"Coverage:                                   {coverage:.1f}%")
print("")

print("=" * 100)
print("KEY INSIGHTS")
print("=" * 100)
print("")
print("The IMPROVED A vs B overlap mechanism works as follows:")
print("1. Take a Chinese token from A, e.g., '巡查'")
print("2. Look up in word_to_pinyin FIRST (word-level)")
print("   - If found: use word's pinyin directly → '巡查' → ['xun2', 'cha2'] → 'xuncha'")
print("   - This solves the multiple-pronunciation problem!")
print("3. If not found in word-level, fall back to character-level")
print("   - Split each character: '巡' → 'xun2', '查' → 'cha2' → 'xuncha'")
print("4. Remove tones: 'xuncha'")
print("5. Check if 'xuncha' exists in Vocab B")
print("")
print("Advantages of word-level lookup:")
print("  ✓ Solves multi-pronunciation issue (多音字)")
print("  ✓ More accurate pinyin conversion")
print("  ✓ Should significantly improve overlap rate")
print("")
print(f"This analysis now uses: {len(word_to_pinyin)} word entries + fallback to {len(char_to_pinyin)} chars")
print("")
