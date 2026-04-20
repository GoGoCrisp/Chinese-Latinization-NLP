"""
显示CEDICT中缺失的字符示例（生冷字、异体字等）
"""
import json
import re
import os
from collections import defaultdict

with open('tokenizers/chinese_origin_64k_train90.json', 'r', encoding='utf-8') as f:
    vocab_a = json.load(f)['model']['vocab']

# 加载CEDICT
char_to_pinyin = {}
cedict_file = 'dicts/cedict_ts.u8'

if os.path.exists(cedict_file):
    with open(cedict_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            try:
                parts = line.split()
                if len(parts) >= 3:
                    simp = parts[1]
                    pinyin_str = parts[2].strip('[]')
                    pinyins = pinyin_str.split('/')
                    for i, char in enumerate(simp):
                        if char not in char_to_pinyin:
                            char_to_pinyin[char] = []
                        if i < len(pinyins):
                            char_to_pinyin[char].append(pinyins[i])
            except:
                pass

print(f"CEDICT加载的字符数: {len(char_to_pinyin):,}")

def is_pure_chinese(token):
    """检查token是否纯粹是中文字符"""
    if not token:
        return False
    return all('\u4e00' <= c <= '\u9fff' for c in token)

chinese_tokens = [t for t in vocab_a.keys() if is_pure_chinese(t)]

# 收集CEDICT缺失的字符
missing_chars = set()
char_not_found_examples = []  # (character, unicode_value, hex_code)

for token in chinese_tokens:
    # 检查这个token中是否有CEDICT没有的字符
    for char in token:
        if char not in char_to_pinyin:
            missing_chars.add(char)
            if len(char_not_found_examples) < 20:
                unicode_val = ord(char)
                hex_code = hex(unicode_val)
                char_not_found_examples.append((char, unicode_val, hex_code))

print("\n" + "=" * 100)
print("CEDICT中缺失的字符示例 (前20个)")
print("=" * 100)

print(f"\n总计缺失字符数: {len(missing_chars):,}")
print(f"\nUnicode范围分析:")

# 按Unicode范围分类
range_counts = defaultdict(int)
for char in missing_chars:
    unicode_val = ord(char)
    if 0x4E00 <= unicode_val <= 0x9FFF:
        range_counts['CJK统一表意文字'] += 1
    elif 0x3400 <= unicode_val <= 0x4DBF:
        range_counts['CJK扩展A'] += 1
    elif 0x20000 <= unicode_val <= 0x2A6DF:
        range_counts['CJK扩展B'] += 1
    elif 0x2A700 <= unicode_val <= 0x2B73F:
        range_counts['CJK扩展C'] += 1
    elif 0x2B740 <= unicode_val <= 0x2B81F:
        range_counts['CJK扩展D'] += 1
    elif 0x2B820 <= unicode_val <= 0x2CEAF:
        range_counts['CJK扩展E'] += 1
    else:
        range_counts['其他'] += 1

for range_name, count in sorted(range_counts.items(), key=lambda x: -x[1]):
    print(f"  • {range_name}: {count:,} 个")

print(f"\n【前20个CEDICT缺失的字符】")
print(f"{'#':<3} {'字符':<4} {'Unicode值':<12} {'16进制':<10} {'字符预览'}")
print("-" * 50)

for i, (char, unicode_val, hex_code) in enumerate(char_not_found_examples, 1):
    print(f"{i:<3} {char:<4} U+{unicode_val:<10} {hex_code:<10} (CJK孤立字)")

print(f"\n【其他缺失字符示例】(随机20个)")
all_missing = sorted(list(missing_chars))
sample_chars = all_missing[20:40] if len(all_missing) > 20 else all_missing[20:]

for i, char in enumerate(sample_chars, 1):
    unicode_val = ord(char)
    hex_code = hex(unicode_val)
    print(f"{i:<3} {char:<4} U+{unicode_val:<10} {hex_code:<10}")

print("\n" + "=" * 100)
print("说明：这些字符主要是生冷字、异体字或CJK扩展区的字符")
print("它们不在常见的CEDICT（Chinese-English Dictionary）中")
print("=" * 100)
