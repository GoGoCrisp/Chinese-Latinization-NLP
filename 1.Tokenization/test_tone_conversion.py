"""
处理Unihan中的声调符号问题
Unihan使用声调符号（ā á ǎ à），需要替换为数字声调，然后处理
"""
import re

# 声调符号到数字声调的映射
TONE_MARKS = {
    'ā': 'a1', 'á': 'a2', 'ǎ': 'a3', 'à': 'a4',
    'ē': 'e1', 'é': 'e2', 'ě': 'e3', 'è': 'e4',
    'ī': 'i1', 'í': 'i2', 'ǐ': 'i3', 'ì': 'i4',
    'ō': 'o1', 'ó': 'o2', 'ǒ': 'o3', 'ò': 'o4',
    'ū': 'u1', 'ú': 'u2', 'ǔ': 'u3', 'ù': 'u4',
    'ǖ': 'v1', 'ǘ': 'v2', 'ǚ': 'v3', 'ǜ': 'v4',
    'ń': 'n2', 'ň': 'n3', 'ǹ': 'n4',
    'ḿ': 'm2',
}

test_pinyins = ['yī', 'dīng', 'kǎo', 'qī', 'shàng', 'xià', 'wàn', 'zhàng', 'sān']

print("=" * 100)
print("测试Unihan拼音转换")
print("=" * 100)

def convert_tone_marks_to_numbers(pinyin_str):
    """将声调符号转换为数字声调"""
    p = pinyin_str
    for mark, replacement in TONE_MARKS.items():
        p = p.replace(mark, replacement)
    return p

def normalize_pinyin_v3(pinyin_str):
    """改进版：处理BOTH声调符号和数字声调"""
    # 第1步：转小写（安全起见）
    p = pinyin_str.lower()
    # 第2步：将声调符号转换为数字
    p = convert_tone_marks_to_numbers(p)
    # 第3步：处理ü表示
    p = p.replace('u:', 'v')
    p = p.replace('ü', 'v')
    # 第4步：移除数字声调
    p = re.sub(r'[0-5]', '', p)
    return p

print("\n【测试Unihan格式的拼音转换】")
for pinyin_with_tone in test_pinyins:
    normalized = normalize_pinyin_v3(pinyin_with_tone)
    print(f"  {pinyin_with_tone:15} → {normalized:15}")

# 现在测试新的normalize函数
with open('tokenizers/pinyin_toneless_64k_train90.json', 'r') as f:
    import json
    vocab_b = set(json.load(f)['model']['vocab'].keys())

print("\n【验证转换结果是否在B中】")
test_results = [
    ('yī', 'yi'),
    ('dīng', 'ding'),
    ('kǎo', 'kao'),
    ('qī', 'qi'),
    ('shàng', 'shang'),
]

for pinyin_tone, expected in test_results:
    normalized = normalize_pinyin_v3(pinyin_tone)
    in_b = normalized in vocab_b
    match = "✓" if normalized == expected else "✗"
    print(f"  {match} {pinyin_tone:10} → {normalized:10} (期望: {expected:10}) | 在B中: {in_b}")

print("\n" + "=" * 100)
