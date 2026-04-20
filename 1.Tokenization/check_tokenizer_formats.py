#!/usr/bin/env python3
"""
验证真正的问题：C和D在tokenizer中被encoding后如何不同
"""
import sys
sys.path.insert(0, '/Users/crisp/Desktop/code_field/python/Chinese_Latinization_NLP/1.Tokenization')

from tokenizers import Tokenizer
from pypinyin import lazy_pinyin, Style

# 加载tokenizer
print("loading tokenizers...")
tokenizer_c = Tokenizer.from_file("tokenizers/pinyin_toned_16k_train90.json")
tokenizer_d = Tokenizer.from_file("tokenizers/pinyin_diacritic_16k_train90.json")

# 重要：明确C是数字声调，D是符号声调
# 从文件名可以看出：pinyin_toned = 数字声调，pinyin_diacritic = 符号声调

# 测试词
test_words = ['中', '国', '中国']

print("\n=== 对比tokenizer中的encoding ===\n")
print(f"{'词':<6} | {'C格式(应该是数字)':<30} | {'D格式(应该是符号)':<30}")
print("-" * 80)

for word in test_words:
    # C格式应该是数字声调
    c_tokens = tokenizer_c.encode(word).tokens
    # D格式应该是符号声调
    d_tokens = tokenizer_d.encode(word).tokens
    
    c_str = " ".join(c_tokens)
    d_str = " ".join(d_tokens)
    
    print(f"{word:<6} | {c_str:<30} | {d_str:<30}")

print("\n\n=== 查看tokenizer的vocab样本 ===\n")

# 采样C tokenizer中的一些tokens
vocab_c = {tokenizer_c.id_to_token(i): i for i in range(tokenizer_c.get_vocab_size())}
# 采样D tokenizer中的一些tokens  
vocab_d = {tokenizer_d.id_to_token(i): i for i in range(tokenizer_d.get_vocab_size())}

# 找出包含'zhong'、'guo'、'zhong1'等的tokens
print("C tokenizer中包含'zhong'或'guo'或数字的token示例:")
c_samples = [t for t in list(vocab_c.keys())[100:500] if any(x in t.lower() for x in ['zhong', 'guo', 'zhōng', 'guó'])]
print(c_samples[:10] if c_samples else "未找到")

print("\nD tokenizer中包含'zhong'或'guo'或符号的token示例:")
d_samples = [t for t in list(vocab_d.keys())[100:500] if any(x in t.lower() for x in ['zhong', 'guo', 'zhōng', 'guó'])]
print(d_samples[:10] if d_samples else "未找到")

# 直接查询
print("\n\nC tokenizer中的'zhong1':", 'zhong1' in vocab_c)
print("C tokenizer中的'zhōng':", 'zhōng' in vocab_c)
print("D tokenizer中的'zhong1':", 'zhong1' in vocab_d)
print("D tokenizer中的'zhōng':", 'zhōng' in vocab_d)

# 查看前100个tokens
print("\n\nC tokenizer的前50个特殊/样本tokens:")
for i in range(50):
    token = tokenizer_c.id_to_token(i)
    if token and not token.startswith('['):
        print(f"  {token}")

print("\n\nD tokenizer的前50个特殊/样本tokens:")
for i in range(50):
    token = tokenizer_d.id_to_token(i)
    if token and not token.startswith('['):
        print(f"  {token}")
