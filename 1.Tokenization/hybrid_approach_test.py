"""
混合方案：CEDICT + pypinyin
优先用CEDICT，然后用pypinyin补充生僻字
"""
import json
import re
import os
from pypinyin import lazy_pinyin, Style

with open('tokenizers/chinese_origin_64k_train90.json', 'r', encoding='utf-8') as f:
    vocab_a = json.load(f)['model']['vocab']

with open('tokenizers/pinyin_toneless_64k_train90.json', 'r', encoding='utf-8') as f:
    vocab_b = set(json.load(f)['model']['vocab'].keys())

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

def normalize_pinyin_v2(pinyin_str):
    """正确的转换逻辑"""
    p = pinyin_str.lower()
    p = p.replace('u:', 'v')
    p = p.replace('ü', 'v')
    p = re.sub(r'[0-5]', '', p)
    return p

def is_pure_chinese(token):
    """检查token是否纯粹是中文字符"""
    if not token:
        return False
    return all('\u4e00' <= c <= '\u9fff' for c in token)

chinese_tokens = [t for t in vocab_a.keys() if is_pure_chinese(t)]

# ===== 统计：使用混合方案 =====
mapped_count = 0
unmapped_list = []
cedict_used = 0  # 统计从CEDICT获得的
pypinyin_used = 0  # 统计从pypinyin获得的

for token in chinese_tokens:
    converted = []
    all_found = True
    
    for char in token:
        pinyin_found = None
        
        # 优先1：CEDICT
        if char in char_to_pinyin and char_to_pinyin[char]:
            p_with_tone = char_to_pinyin[char][0]
            if p_with_tone:
                pinyin_found = normalize_pinyin_v2(p_with_tone)
                cedict_used += 1
        
        # 优先2：pypinyin（处理CEDICT缺失的字符或为空的情况）
        if not pinyin_found:
            try:
                result = lazy_pinyin(char, style=Style.NORMAL)
                if result and result[0]:
                    pinyin_found = normalize_pinyin_v2(result[0])
                    pypinyin_used += 1
            except:
                pass
        
        if pinyin_found:
            converted.append(pinyin_found)
        else:
            all_found = False
            break
    
    if all_found and converted:
        result = ''.join(converted)
        if result in vocab_b:
            mapped_count += 1
        else:
            unmapped_list.append((token, f'结果不在B中: {result}'))
    elif not all_found:
        unmapped_list.append((token, '无法找到拼音'))

print("=" * 100)
print("【混合方案结果】CEDICT + pypinyin")
print("=" * 100)

print(f"\n【基础数据】")
print(f"  A总token数: {len(vocab_a):,}")
print(f"  A纯中文token数: {len(chinese_tokens):,}")

print(f"\n【映射统计】")
print(f"  ✅ 映射成功: {mapped_count:,} ({100*mapped_count/len(chinese_tokens):.1f}%)")
print(f"  ❌ 映射失败: {len(unmapped_list):,} ({100*len(unmapped_list)/len(chinese_tokens):.1f}%)")

print(f"\n【数据源使用统计】")
print(f"  • 从CEDICT获得: {cedict_used:,} 个字符")
print(f"  • 从pypinyin获得: {pypinyin_used:,} 个字符（生僻字补充）")
print(f"  • 总字符处理数: {cedict_used + pypinyin_used:,}")

print(f"\n【性能对比】")
print(f"  之前（仅CEDICT）: 33,138 (68.1%)")
print(f"  现在（混合方案）: {mapped_count:,} ({100*mapped_count/len(chinese_tokens):.1f}%)")
improvement = mapped_count - 33138
improvement_pct = 100*improvement/len(chinese_tokens)
print(f"  📈 提升: +{improvement:,} tokens (+{improvement_pct:.1f}%)")

if unmapped_list:
    print(f"\n【映射失败示例】(前15个):")
    for token, reason in unmapped_list[:15]:
        print(f"    {token} | {reason}")

print("\n" + "=" * 100)
