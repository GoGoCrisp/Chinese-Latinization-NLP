import sys
sys.path.insert(0, '.')

from tokeni zers import Tokenizer
from pypinyin import lazy_pinyin, Style

# 模拟转换
def _tone_numbers_to_marks(pinyin: str) -> str:
    tone_marks = {
        'a': {'1': 'ā', '2': 'á', '3': 'ǎ', '4': 'à'},
        'e': {'1': 'ē', '2': 'é', '3': 'ě', '4': 'è'},
        'i': {'1': 'ī', '2': 'í', '3': 'ǐ', '4': 'ì'},
        'o': {'1': 'ō', '2': 'ó', '3': 'ǒ', '4': 'ò'},
        'u': {'1': 'ū', '2': 'ú', '3': 'ǔ', '4': 'ù'},
        'ü': {'1': 'ǖ', '2': 'ǘ', '3': 'ǚ', '4': 'ǜ'},
        'v': {'1': 'ǖ', '2': 'ǘ', '3': 'ǚ', '4': 'ǜ'},
    }
    
    if not pinyin or not pinyin[-1].isdigit():
        return pinyin
    
    tone_num = pinyin[-1]
    pinyin_base = pinyin[:-1]
    
    if tone_num in ['0', '5']:
        return pinyin_base
    
    result = []
    matched = False
    
    for char in pinyin_base:
        if not matched and char in 'ae':
            result.append(tone_marks.get(char, {}).get(tone_num, char))
            matched = True
        else:
            result.append(char)
    
    if not matched:
        for i in range(len(result) - 1, -1, -1):
            char = result[i]
            if char in 'iouüv':
                result[i] = tone_marks.get(char, {}).get(tone_num, char)
                break
    
    return "".join(result)

# 测试例子
test_cases = [
    ('zhong1', 'zhōng'),
    ('guo2', 'guó'),
    ('fan1', 'fān'),
    ('bing4', 'bìng'),
]

print("测试 _tone_numbers_to_marks:")
for input_val, expected in test_cases:
    result = _tone_numbers_to_marks(input_val)
    status = "✓" if result == expected else "✗"
    print(f"  {status} {input_val} → {result} (expected: {expected})")
