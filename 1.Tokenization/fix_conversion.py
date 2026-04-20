"""
修复转换逻辑，正确处理CEDICT中的拼音格式
"""
import json
import re
import os

# 加载tokenizers
with open('tokenizers/chinese_origin_64k_train90.json', 'r', encoding='utf-8') as f:
    vocab_a = json.load(f)['model']['vocab']

with open('tokenizers/pinyin_toneless_64k_train90.json', 'r', encoding='utf-8') as f:
    vocab_b = set(json.load(f)['model']['vocab'].keys())

# 加载CEDICT并正确处理
def normalize_pinyin(pinyin_str):
    """将CEDICT拼音转换为标准无声调小写拼音
    CEDICT格式可能包含: 大写, 冒号(: → ü), 括号等
    """
    # 替换特殊字符
    p = pinyin_str.replace('ü', 'u:')  # 保持临时冒号形式便于后续处理
    p = p.replace(':', 'v')              # 最后转为v
    p = p.lower()                        # 转小写
    p = re.sub(r'[0-5]', '', p)         # 移除声调数字
    p = re.sub(r'[^a-z]', '', p)        # 移除其他非字母
    return p

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
                    
                    # 逐字处理
                    for i, char in enumerate(simp):
                        if char not in char_to_pinyin:
                            char_to_pinyin[char] = []
                        if i < len(pinyins):
                            char_to_pinyin[char].append(pinyins[i])
            except:
                pass

print("=" * 100)
print("修复转换逻辑后的映射测试")
print("=" * 100)

# 测试一些examples
test_chars = ['仉', '侣', '侴', '吕', '尧', '屡', '履', '廖']
print(f"\n【测试特定字符的转换】")
for char in test_chars:
    if char in char_to_pinyin:
        original_pinyins = char_to_pinyin[char]
        normalized = [normalize_pinyin(p) for p in original_pinyins]
        in_b = any(p in vocab_b for p in normalized)
        
        print(f"  {char}:")
        print(f"    原始拼音: {original_pinyins}")
        print(f"    标准化后: {normalized}")
        print(f"    在B中: {in_b}")

# 重新统计映射成功率
print(f"\n【重新统计映射成功率】")
chinese_tokens = [t for t in vocab_a.keys() if any('\u4e00' <= c <= '\u9fff' for c in t)]

mapped_count = 0
unmapped_with_reason = {
    '包含数字': [],
    '包含英文': [],
    'CEDICT缺失': [],
    '转换后仍未找到': [],
    '其他': []
}

for token in chinese_tokens:
    success = False
    
    # 尝试逐字转换
    if all(c in char_to_pinyin or not ('\u4e00' <= c <= '\u9fff') for c in token):
        try:
            converted = []
            for char in token:
                if char in char_to_pinyin:
                    p_with_tone = char_to_pinyin[char][0]
                    p_normalized = normalize_pinyin(p_with_tone)
                    converted.append(p_normalized)
                else:
                    converted.append(char)  # 非汉字原样保留
            
            result = ''.join(converted)
            if result and result in vocab_b:
                success = True
                mapped_count += 1
        except:
            pass
    
    # 分类失败原因
    if not success:
        if any(c.isdigit() for c in token):
            unmapped_with_reason['包含数字'].append(token)
        elif any('a' <= c.lower() <= 'z' for c in token):
            unmapped_with_reason['包含英文'].append(token)
        elif not all(c in char_to_pinyin or not ('\u4e00' <= c <= '\u9fff') for c in token):
            unmapped_with_reason['CEDICT缺失'].append(token)
        else:
            unmapped_with_reason['转换后仍未找到'].append(token)

print(f"  总中文token: {len(chinese_tokens):,}")
print(f"  ✅ 映射成功: {mapped_count:,} ({100*mapped_count/len(chinese_tokens):.1f}%)")
print(f"  ❌ 映射失败: {len(chinese_tokens)-mapped_count:,}")

print(f"\n【失败原因分类】")
for reason, tokens in unmapped_with_reason.items():
    if tokens:
        pct = 100 * len(tokens) / (len(chinese_tokens) - mapped_count)
        print(f"  • {reason}: {len(tokens):,} ({pct:.1f}%)")
        print(f"    例: {tokens[:5]}")

