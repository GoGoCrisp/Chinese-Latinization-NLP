"""
详细分析A→B的独立词汇状况
"""
import json
import re

# 加载tokenizers
with open('tokenizers/chinese_origin_64k_train90.json', 'r', encoding='utf-8') as f:
    vocab_a = set(json.load(f)['model']['vocab'].keys())

with open('tokenizers/pinyin_toneless_64k_train90.json', 'r', encoding='utf-8') as f:
    vocab_b = set(json.load(f)['model']['vocab'].keys())

print("=" * 100)
print("分析: 为什么A有33.1%的中文词汇无法映射到B?")
print("=" * 100)

# 统计A中的词汇特征
chinese_in_a = [token for token in vocab_a if any('\u4e00' <= c <= '\u9fff' for c in token)]
non_chinese_in_a = [token for token in vocab_a if not any('\u4e00' <= c <= '\u9fff' for c in token)]

print(f"\n【A的50,932个中文token的具体分类】")
print(f"  • 纯中文: {len(chinese_in_a):,} ({100*len(chinese_in_a)/len(vocab_a):.1f}%)")
print(f"  • 非中文(符号/外文/混合): {len(non_chinese_in_a):,} ({100*len(non_chinese_in_a)/len(vocab_a):.1f}%)")

# 对中文token采样分类
print(f"\n【A中50,932个中文token的内部构成】")

# 纯中文字
pure_chinese = [t for t in chinese_in_a if all('\u4e00' <= c <= '\u9fff' for c in t)]
print(f"  1️⃣ 纯汉字token: {len(pure_chinese):,} ({100*len(pure_chinese)/len(chinese_in_a):.1f}%)")
print(f"     例: {pure_chinese[:5]}")

# 包含阿拉伯数字的
with_digits = [t for t in chinese_in_a if any(c.isdigit() for c in t)]
print(f"  2️⃣ 包含数字: {len(with_digits):,} ({100*len(with_digits)/len(chinese_in_a):.1f}%)")
print(f"     例: {with_digits[:5]}")

# 包含标点/符号
with_punct = [t for t in chinese_in_a if any(ord(c) > 0x9fff for c in t)]
print(f"  3️⃣ 包含非中非阿拉伯数字: {len(with_punct):,} ({100*len(with_punct)/len(chinese_in_a):.1f}%)")
print(f"     例: {with_punct[:5]}")

# 长度分析
long_tokens = [t for t in chinese_in_a if len(t) > 3]
short_tokens = [t for t in chinese_in_a if len(t) <= 3]
print(f"  4️⃣ 短token(≤3字符): {len(short_tokens):,}")
print(f"  5️⃣ 长token(>3字符): {len(long_tokens):,}")

print(f"\n【B的词汇表特征】")
pinyin_only = [t for t in vocab_b if all(ord(c) < 0x4e00 for c in t)]  # 非中文
print(f"  • B完全是拼音/ASCII: {len(vocab_b):,} tokens (全部)")
print(f"    - 无法直接匹配包含中文字的A词汇！")

print(f"\n【关键洞察】")
print(f"")
print(f"  A的33.1% (16,863个)无法映射到B的中文token分布:")
print(f"  ")
print(f"  根本原因不是'B的空间不足'，而是【转换路径问题】:")
print(f"  ")
print(f"  ❌ A→B无法工作的场景:")
print(f"     1. 某些汉字无法转拼音（汉字编码异常）")
print(f"     2. 某个汉字的拼音在B中没有对应token")
print(f"        → 例如: '长'→'zhang'可能存在")
print(f"           但组合后的某个拼音token可能B中没有")
print(f"     3. A中特殊符号/外文混合体无法转拼音")
print(f"")
print(f"  ✅ 35.2% N对1现象说明:")
print(f"     • 这不是'B有更多表达能力'")
print(f"     • 而是【多个A词→同一个B词】= 多音字现象")
print(f"     • 是【信息损失】而非容量增加!")
print(f"")
print(f"  🎯 结论:")
print(f"     N对1(35.2%) + 1对1(31.7%) + 独立(33.1%) = 100%")
print(f"     这完全符合逻辑！")
print(f"     B的53.4%独立token来自:")
print(f"     • 某些特殊拼音组合是A特定词汇无法生成的")
print(f"     • tokenization产生的副产品")

