"""
深入调查：A中哪些中文token无法映射到B？
"""
import json
import re
import sys

sys.path.insert(0, '/Users/crisp/Desktop/code_field/python/Chinese_Latinization_NLP/1.Tokenization')

from pypinyin import pinyin, lazy_pinyin, Style
import os

# 加载tokenizers
with open('tokenizers/chinese_origin_64k_train90.json', 'r', encoding='utf-8') as f:
    vocab_a = json.load(f)['model']['vocab']

with open('tokenizers/pinyin_toneless_64k_train90.json', 'r', encoding='utf-8') as f:
    vocab_b = set(json.load(f)['model']['vocab'].keys())

# 加载CEDICT
cedict_file = 'dicts/cedict_ts.u8'
word_to_pinyin = {}
char_to_pinyin = {}

if os.path.exists(cedict_file):
    with open(cedict_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            try:
                parts = line.split()
                if len(parts) >= 3:
                    trad = parts[0]
                    simp = parts[1]
                    pinyin_str = parts[2].strip('[]')
                    pinyins = pinyin_str.split('/')
                    
                    # 简体词
                    if simp not in word_to_pinyin:
                        word_to_pinyin[simp] = []
                    word_to_pinyin[simp].extend(pinyins)
                    
                    # 逐字处理
                    for i, char in enumerate(simp):
                        if char not in char_to_pinyin:
                            char_to_pinyin[char] = []
                        if i < len(pinyins):
                            char_to_pinyin[char].append(pinyins[i])
            except:
                pass

print("=" * 100)
print("调查：A中有哪些中文token无法映射到B?")
print("=" * 100)
print(f"\n📚 词汇表加载完成:")
print(f"  A中文token: {len(vocab_a):,}")
print(f"  B拼音token: {len(vocab_b):,}")
print(f"  CEDICT词数: {len(word_to_pinyin):,}")
print(f"  CEDICT字数: {len(char_to_pinyin):,}")

# 统计中文token
def is_chinese_token(token):
    return any('\u4e00' <= c <= '\u9fff' for c in token)

chinese_tokens_a = [t for t in vocab_a.keys() if is_chinese_token(t)]

# 尝试映射每个中文token
unmapped = []
mapped = []

for token in chinese_tokens_a:
    success = False
    
    # 方法1：查word_to_pinyin
    if token in word_to_pinyin:
        for p_with_tone in word_to_pinyin[token]:
            p_base = re.sub(r'[0-5]', '', p_with_tone)
            if p_base in vocab_b:
                success = True
                mapped.append((token, p_base))
                break
    
    # 方法2：字级别转换
    if not success:
        try:
            converted = []
            for char in token:
                if char in char_to_pinyin:
                    p_with_tone = char_to_pinyin[char][0]  # 取第一个音
                    p_base = re.sub(r'[0-5]', '', p_with_tone)
                    converted.append(p_base)
                else:
                    # 无法转换这个字
                    converted = None
                    break
            
            if converted and len(converted) > 0:
                result = ''.join(converted)
                if result in vocab_b:
                    success = True
                    mapped.append((token, result))
        except:
            pass
    
    if not success:
        unmapped.append(token)

print(f"\n【映射统计】")
print(f"  ✅ 成功映射: {len(mapped):,} ({100*len(mapped)/len(chinese_tokens_a):.1f}%)")
print(f"  ❌ 无法映射: {len(unmapped):,} ({100*len(unmapped)/len(chinese_tokens_a):.1f}%)")

print(f"\n【无法映射的中文token样本】(显示前100个)")
print(f"  {'Token':<20} | {'长度':<3} | {'特征':<30}")
print(f"  {'-'*70}")

for token in unmapped[:100]:
    # 分析特征
    has_digit = any(c.isdigit() for c in token)
    has_english = any('a' <= c.lower() <= 'z' for c in token)
    has_symbol = any(not ('\u4e00' <= c <= '\u9fff' or c.isdigit() or ('a' <= c.lower() <= 'z')) for c in token)
    has_cedict = all(c in char_to_pinyin for c in token if '\u4e00' <= c <= '\u9fff')
    
    features = []
    if has_digit:
        features.append("×数字")
    if has_english:
        features.append("×英文")
    if has_symbol:
        features.append("×符号")
    if not has_cedict:
        features.append("✗无CEDICT")
    
    feature_str = '|'.join(features) if features else "纯汉字"
    print(f"  {token:<20} | {len(token):<3} | {feature_str:<30}")

# 深入分析无法映射原因
print(f"\n【无法映射原因分类】")

reason_counts = {
    '包含数字': 0,
    '包含英文': 0,
    '包含特殊符号': 0,
    '汉字CEDICT缺失': 0,
    '转换后拼音B中无': 0,
    '其他': 0
}

for token in unmapped:
    if any(c.isdigit() for c in token):
        reason_counts['包含数字'] += 1
    elif any('a' <= c.lower() <= 'z' for c in token):
        reason_counts['包含英文'] += 1
    elif any(not ('\u4e00' <= c <= '\u9fff' or c.isdigit() or ('a' <= c.lower() <= 'z')) for c in token):
        reason_counts['包含特殊符号'] += 1
    elif not all(c in char_to_pinyin for c in token if '\u4e00' <= c <= '\u9fff'):
        reason_counts['汉字CEDICT缺失'] += 1
    else:
        # 尝试转换看看
        try:
            converted = []
            for char in token:
                if char in char_to_pinyin:
                    p_with_tone = char_to_pinyin[char][0]
                    p_base = re.sub(r'[0-5]', '', p_with_tone)
                    converted.append(p_base)
            if converted:
                result = ''.join(converted)
                if result not in vocab_b:
                    reason_counts['转换后拼音B中无'] += 1
                else:
                    reason_counts['其他'] += 1
        except:
            reason_counts['其他'] += 1

for reason, count in reason_counts.items():
    if count > 0:
        pct = 100 * count / len(unmapped)
        print(f"  • {reason}: {count:,} ({pct:.1f}%)")

# 打印一些具体的、转换后拼音在B中无的例子
print(f"\n【特别分析：转换后拼音在B中无的token】")
count = 0
for token in unmapped:
    if count >= 20:
        break
    
    try:
        converted = []
        all_found = True
        for char in token:
            if char in char_to_pinyin:
                p_with_tone = char_to_pinyin[char][0]
                p_base = re.sub(r'[0-5]', '', p_with_tone)
                converted.append(p_base)
            else:
                all_found = False
                break
        
        if all_found and converted:
            result = ''.join(converted)
            if result not in vocab_b:
                print(f"  {token} → {result} (✗ 不在B中)")
                count += 1
    except:
        pass

