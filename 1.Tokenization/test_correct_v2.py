"""
正确的拼音转换逻辑 v2
"""
import json
import re

with open('tokenizers/pinyin_toneless_64k_train90.json', 'r', encoding='utf-8') as f:
    vocab_b = set(json.load(f)['model']['vocab'].keys())

def normalize_pinyin_v2(pinyin_str):
    """正确的转换逻辑 v2
    关键: 识别 u: 为一个单元，表示ü
    """
    # 第一步：移除声调数字
    p = re.sub(r'[0-5]', '', pinyin_str)
    
    # 第二步：处理ü的各种表示
    # u: → v (CEDICT用冒号表示ü)
    p = p.replace('u:', 'v')
    # ü → v (某些版本中直接使用ü)
    p = p.replace('ü', 'v')
    # 大写
    p = p.replace('U:', 'V')
    p = p.replace('Ü', 'V')
    
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
print("测试正确的转换逻辑 v2")
print("=" * 100)

for input_py, expected in test_cases:
    result = normalize_pinyin_v2(input_py)
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
        normalized = normalize_pinyin_v2(p_with_tone)
        in_b = normalized in vocab_b
        print(f"  '{p_with_tone}' → '{normalized}' (在B中: {in_b})")

