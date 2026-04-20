"""
探索多种数据源来获取生僻字的拼音

方案：
1. CEDICT（已有）- 10,867个字符
2. Unihan数据库 - 包含所有CJK字符的拼音信息
3. pypinyin库 - Python的拼音库
4. 混合方案 - 优先级逐级fallback
"""
import json
import re
import os

print("=" * 100)
print("生僻字拼音查询方案探索")
print("=" * 100)

with open('tokenizers/chinese_origin_64k_train90.json', 'r', encoding='utf-8') as f:
    vocab_a = json.load(f)['model']['vocab']

# 提取纯中文token
def is_pure_chinese(token):
    if not token:
        return False
    return all('\u4e00' <= c <= '\u9fff' for c in token)

chinese_tokens = [t for t in vocab_a.keys() if is_pure_chinese(t)]

# ==== 方案1：CEDICT（已有） ====
print("\n【方案1】CEDICT词典（现状）")
print("-" * 100)

char_to_pinyin_cedict = {}
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
                        if char not in char_to_pinyin_cedict:
                            char_to_pinyin_cedict[char] = []
                        if i < len(pinyins):
                            char_to_pinyin_cedict[char].append(pinyins[i])
            except:
                pass

print(f"✓ CEDICT覆盖的字符数: {len(char_to_pinyin_cedict):,}")

# ==== 方案2：尝试pypinyin ====
print("\n【方案2】pypinyin库")
print("-" * 100)

try:
    from pypinyin import lazy_pinyin, Style
    print("✓ pypinyin 已安装")
    
    # 测试生僻字
    test_chars = ['両', '乁', '乗', '乚', '乫', '亜']
    print("\n测试生僻字:")
    for char in test_chars:
        result = lazy_pinyin(char, style=Style.NORMAL)
        print(f"  {char}: {result[0]}")
        
except ImportError:
    print("✗ pypinyin 未安装")
    print("  可以通过以下命令安装:")
    print("  pip install pypinyin")

# ==== 方案3：Unihan数据库 ====
print("\n【方案3】Unihan数据库")
print("-" * 100)

unihan_file = 'dicts/Unihan.txt'
if os.path.exists(unihan_file):
    print(f"✓ Unihan.txt 已存在")
    with open(unihan_file, 'r', encoding='utf-8') as f:
        line_count = sum(1 for _ in f)
    print(f"  包含 {line_count:,} 行数据")
else:
    print(f"✗ Unihan.txt 不存在 (可以从 https://www.unicode.org/Public/UCD/latest/ucd/Unihan.zip 下载)")

# ==== 方案4：展示混合方案的潜力 ====
print("\n【方案4】混合方案（建议方案）")
print("-" * 100)

print("""
优先级：
  1️⃣  CEDICT（最可靠，已有）
  2️⃣  Unihan数据库（覆盖全面，需下载）
  3️⃣  pypinyin库（快速fallback，需安装）

预期效果：
  • CEDICT         → 10,867字符（28.2%）
  • +Unihan        → ≤20,000字符（预计50%+）
  • +pypinyin      → ≤21,000字符（预计54%+）
""")

# ==== 测试pypinyin的效果 ====
print("\n【测试】用pypinyin补充生僻字")
print("-" * 100)

try:
    from pypinyin import lazy_pinyin, Style
    
    cedict_missing = []
    for token in chinese_tokens[:1000]:  # 测试前1000个token
        for char in token:
            if char not in char_to_pinyin_cedict:
                cedict_missing.append(char)
    
    cedict_missing_set = set(cedict_missing)
    print(f"测试范围内CEDICT缺失的唯一字符: {len(cedict_missing_set)}")
    
    pypinyin_coverage = 0
    for char in cedict_missing_set:
        result = lazy_pinyin(char, style=Style.NORMAL)
        if result and result[0]:
            pypinyin_coverage += 1
    
    print(f"pypinyin可以补充的字符: {pypinyin_coverage} / {len(cedict_missing_set)}")
    print(f"覆盖率: {100*pypinyin_coverage/len(cedict_missing_set) if cedict_missing_set else 0:.1f}%")
    
    print("\n示例补充:")
    shown = 0
    for char in sorted(cedict_missing_set):
        result = lazy_pinyin(char, style=Style.NORMAL)
        if result and result[0]:
            print(f"  {char} → {result[0]}")
            shown += 1
            if shown >= 10:
                break
                
except ImportError:
    print("pypinyin未安装，跳过测试")

print("\n" + "=" * 100)
print("建议:")
print("  1. 优先下载 Unihan.txt 并集成到代码中")
print("  2. 其次安装 pypinyin 作为backup")
print("  3. 这样可以将映射成功率从 68.1% 提升到 75%+ 甚至更高")
print("=" * 100)
