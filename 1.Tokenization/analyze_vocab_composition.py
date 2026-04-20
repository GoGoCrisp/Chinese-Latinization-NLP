#!/usr/bin/env python3
"""
全面分析A_vs_C/D的真实覆盖率问题
"""
import sys
sys.path.insert(0, '/Users/crisp/Desktop/code_field/python/Chinese_Latinization_NLP/1.Tokenization')

from tokenizers import Tokenizer
import re

# 加载tokenizers
print("正在加载tokenizers...\n")
tokenizer_a = Tokenizer.from_file("tokenizers/chinese_origin_16k_train90.json")
tokenizer_c = Tokenizer.from_file("tokenizers/pinyin_toned_16k_train90.json")
tokenizer_d = Tokenizer.from_file("tokenizers/pinyin_diacritic_16k_train90.json")

vocab_a = {tokenizer_a.id_to_token(i): i for i in range(tokenizer_a.get_vocab_size())}
vocab_c = {tokenizer_c.id_to_token(i): i for i in range(tokenizer_c.get_vocab_size())}
vocab_d = {tokenizer_d.id_to_token(i): i for i in range(tokenizer_d.get_vocab_size())}

print(f"Vocab sizes: A={len(vocab_a)}, C={len(vocab_c)}, D={len(vocab_d)}\n")

# 分析A中哪些类型的token
print("="*80)
print("A tokenizer中的token类型分析")
print("="*80)

chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
digit_pattern = re.compile(r'\d')
english_pattern = re.compile(r'[a-zA-Z]')
punctuation_pattern = re.compile(r'[，。！？；：、]')

stats = {
    'pure_chinese': 0,  # 纯汉字
    'pure_ascii': 0,  # 纯ASCII
    'pure_digit': 0,  # 纯数字
    'mixed': 0,  # 混合
    'special': 0,  # 特殊符号
    'other': 0,
}

sample_tokens = {}

for token in list(vocab_a.keys()):
    if token.startswith('['):
        continue
    
    is_chinese = bool(chinese_pattern.search(token))
    is_ascii = bool(english_pattern.search(token))
    is_digit = bool(digit_pattern.search(token))
    is_punct = bool(punctuation_pattern.search(token))
    
    if is_chinese and not is_ascii and not is_digit:
        stats['pure_chinese'] += 1
        if 'pure_chinese' not in sample_tokens:
            sample_tokens['pure_chinese'] = []
        if len(sample_tokens['pure_chinese']) < 5:
            sample_tokens['pure_chinese'].append(token)
    elif is_ascii and not is_chinese and not is_digit:
        stats['pure_ascii'] += 1
        if 'pure_ascii' not in sample_tokens:
            sample_tokens['pure_ascii'] = []
        if len(sample_tokens['pure_ascii']) < 5:
            sample_tokens['pure_ascii'].append(token)
    elif is_digit and not is_chinese and not is_ascii:
        stats['pure_digit'] += 1
        if 'pure_digit' not in sample_tokens:
            sample_tokens['pure_digit'] = []
        if len(sample_tokens['pure_digit']) < 5:
            sample_tokens['pure_digit'].append(token)
    elif is_chinese and (is_ascii or is_digit):
        stats['mixed'] += 1
        if 'mixed' not in sample_tokens:
            sample_tokens['mixed'] = []
        if len(sample_tokens['mixed']) < 5:
            sample_tokens['mixed'].append(token)
    elif is_punct or (not is_chinese and not is_ascii and not is_digit):
        stats['special'] += 1
        if 'special' not in sample_tokens:
            sample_tokens['special'] = []
        if len(sample_tokens['special']) < 5:
            sample_tokens['special'].append(token)
    else:
        stats['other'] += 1

total_non_special = sum(v for k, v in stats.items() if k != 'special')

print(f"\nA tokenizer中的token类型统计:\n")
print(f"  纯汉字: {stats['pure_chinese']:6d} ({100*stats['pure_chinese']/len(vocab_a):.1f}%)")
print(f"  样本: {sample_tokens.get('pure_chinese', [])}")
print()
print(f"  纯ASCII: {stats['pure_ascii']:6d} ({100*stats['pure_ascii']/len(vocab_a):.1f}%)")
print(f"  样本: {sample_tokens.get('pure_ascii', [])}")
print()
print(f"  纯数字: {stats['pure_digit']:6d} ({100*stats['pure_digit']/len(vocab_a):.1f}%)")
print(f"  样本: {sample_tokens.get('pure_digit', [])}")
print()
print(f"  混合(中+ASCII或数字): {stats['mixed']:6d} ({100*stats['mixed']/len(vocab_a):.1f}%)")
print(f"  样本: {sample_tokens.get('mixed', [])}")
print()
print(f"  特殊/其他: {stats['special']:6d} ({100*stats['special']/len(vocab_a):.1f}%)")
print(f"  样本: {sample_tokens.get('special', [])}")

# 关键结论
print("\n" + "="*80)
print("关键洞察")
print("="*80)

print(f"""
结论：

C和D tokenizers是专为拼音设计的：
  - C包含数字拼音token（如'zhong1', 'guo2'）
  - D包含符号拼音token（如'zhōng', 'guó'）

但A tokenizer（中文）包含：
  - 汉字（可以转换为拼音）✓
  - ASCII字母和单词（无法转为拼音）✗
  - 数字（无法转为拼音）✗
  - 混合token（例如"5G"、"COVID"）✗
  - 特殊符号（无法转为拼音）✗

因此，虽然C和D本身相似（75.2%），但：
- A能转换为拼音的只有{stats['pure_chinese']}个纯汉字token（占{100*stats['pure_chinese']/len(vocab_a):.1f}%）
- 其余{len(vocab_a) - stats['pure_chinese']}个token({100*(len(vocab_a) - stats['pure_chinese'])/len(vocab_a):.1f}%)无法转换为拼音

这解释了为什么A_vs_C和A_vs_D的相似度都比较低！
""")

# 验证
print("\n" + "="*80)
print("验证")
print("="*80)

print(f"\nA中可转换为拼音的token比例: {100*stats['pure_chinese']/len(vocab_a):.1f}%")
print(f"A_vs_C的1对1映射比例（当前）: ~47%")
print(f"A_vs_D的1对1映射比例（当前）: ~44%")

print(f"\n从数学上讲，A_vs_C最好的情况是:")
print(f"  如果所有纯汉字都能正确映射到C中的拼音")
print(f"  则1对1（双射）最多 ≈ {100*stats['pure_chinese']/len(vocab_a):.1f}%")
print(f"  但实际结果{47}%还要考虑多音字导致的N对1")
