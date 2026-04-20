#!/usr/bin/env python3
"""
调查为什么A_vs_D的N对1(28.3%)远高于A_vs_C的N对1(18.7%)
"""
import sys
sys.path.insert(0, '/Users/crisp/Desktop/code_field/python/Chinese_Latinization_NLP/1.Tokenization')

# 直接导入同文件中的函数
from compare_tokenizers_v2 import PinyinConverter  # 或者试试其他方式

from tokenizers import Tokenizer

# 加载转换器
converter = PinyinConverter()

# 加载tokenizer
tokenizer_c = Tokenizer.from_file("/Users/crisp/Desktop/code_field/python/Chinese_Latinization_NLP/1.Tokenization/tokenizers/pinyin_toned_16k_train90.json")
tokenizer_d = Tokenizer.from_file("/Users/crisp/Desktop/code_field/python/Chinese_Latinization_NLP/1.Tokenization/tokenizers/pinyin_diacritic_16k_train90.json")

# 获取vocab
vocab_a = {tokenizer_c.id_to_token(i): i for i in range(tokenizer_c.get_vocab_size())}
vocab_c = {tokenizer_c.id_to_token(i): i for i in range(tokenizer_c.get_vocab_size())}
vocab_d = {tokenizer_d.id_to_token(i): i for i in range(tokenizer_d.get_vocab_size())}

# 测试一些中文词
test_words = ['不', '中国', '这个', '说', '长', '行', '了', '会', '要']

print("=== 测试转换差异 ===\n")
for word in test_words:
    c_result = converter.text_to_pinyin_toned(word)
    d_result = converter.text_to_pinyin_diacritic(word)
    
    c_in_vocab = c_result in vocab_c
    d_in_vocab = d_result in vocab_d
    
    print(f"词: {word}")
    print(f"  C格式: {c_result:<20} (在vocab: {c_in_vocab})")
    print(f"  D格式: {d_result:<20} (在vocab: {d_in_vocab})")
    print()

# 分析一些映射到相同D token的多个A词
print("\n=== 查找多个A词映射到同一个D token的例子 ===")
d_to_a_map = {}

# 只检查前10000个vocab_a中的词
for idx, a_token in enumerate(list(vocab_a.keys())[:10000]):
    if a_token in ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '<|endoftext|>', '']:
        continue
    
    try:
        d_result = converter.text_to_pinyin_diacritic(a_token.lower())
        if d_result and d_result in vocab_d:
            if d_result not in d_to_a_map:
                d_to_a_map[d_result] = []
            d_to_a_map[d_result].append(a_token)
    except:
        pass

# 找出N对1的例子（N >= 2）
n_to_1_examples = {d: a_list for d, a_list in d_to_a_map.items() if len(a_list) >= 2}
print(f"发现 {len(n_to_1_examples)} 个N对1的D token")

# 显示前20个例子
for idx, (d_token, a_tokens) in enumerate(sorted(n_to_1_examples.items(), key=lambda x: -len(x[1]))[:20]):
    print(f"  D: '{d_token}' ← A: {a_tokens[:5]}")
