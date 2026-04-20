"""
测试正确的拼音转换逻辑
"""
import json
import re

with open('tokenizers/pinyin_toneless_64k_train90.json', 'r', encoding='utf-8') as f:
    vocab_b = set(json.load(f)['model']['vocab'].keys())

def normalize_pinyin_correct(pinyin_str):
    """正确的转换逻辑
    CEDICT中: lu:3 表示 lü3（ü用冒号表示）
    目标: 转换为 lv (无声调)
    """
    # 第一步：移除声调数字
    p = re.sub(r'[0-5]', '', pinyin_str)
    
    # 第二步：将ü相关的表示转为lv
    # CEDICT用: 表示ü，所以lu:就是lü
    p = p.replace('ü', 'v')   # ü直接换成v (在某些CEDICT版本中)
    p = p.replace(':', 'v')   # : 也换成v (冒号表示)
    p = p.replace('Ü', 'V')   # 大写
    
    # 第三步：转小写
    p = p.lower()
    
    return p

# 测试
test_cases = [
    ('lu:3', 'lv'),      # lu:3 → lv
    ('nv3', 'nv'),       # nv3 → nv
    ('lü3', 'lv'),       # lü3 → lv
    ('Zhang3', 'zhang'), # Zhang3 → zhang
    ('Chou3', 'chou'),   # Chou3 → chou
]

print("=" * 100)
print("测试正确的转换逻辑")
print("=" * 100)

for input_py, expected in test_cases:
    result = normalize_pinyin_correct(input_py)
    status = "✓" if result == expected else "✗"
    in_b = result in vocab_b
    print(f"{status} '{input_py}' → '{result}' (expected: {expected}, in B: {in_b})")

# 现在重新检查那些无法映射的字
print(f"\n【重新检查那些含ü的字】")

char_to_pinyin_data = {
    '侣': ['lu:3'],
    '吕': ['Lu:3', 'lu:3'],
    '屡': ['lu:3'],
    '履': ['lu:3'],
    '女': ['nv3'],
}

for char, pinyins in char_to_pinyin_data.items():
    print(f"\n{char}:")
    for p_with_tone in pinyins:
        normalized = normalize_pinyin_correct(p_with_tone)
        in_b = normalized in vocab_b
        print(f"  '{p_with_tone}' → '{normalized}' (在B中: {in_b})")

