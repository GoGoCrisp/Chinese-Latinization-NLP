#!/usr/bin/env python3
"""
找出导致A_vs_D N对1偏高的具体词
"""
import sys
sys.path.insert(0, '/Users/crisp/Desktop/code_field/python/Chinese_Latinization_NLP/1.Tokenization')

from tokenizers import Tokenizer

# 加载各个tokenizer的vocab
print("loading vocabs...")
tokenizer_c = Tokenizer.from_file("tokenizers/pinyin_toned_16k_train90.json")
tokenizer_d = Tokenizer.from_file("tokenizers/pinyin_diacritic_16k_train90.json")

vocab_c = {tokenizer_c.id_to_token(i): i for i in range(tokenizer_c.get_vocab_size())}
vocab_d = {tokenizer_d.id_to_token(i): i for i in range(tokenizer_d.get_vocab_size())}

print(f"C vocab size: {len(vocab_c)}")
print(f"D vocab size: {len(vocab_d)}")

# 检查常见词的转换
from pypinyin import lazy_pinyin, Style

test_words = [
    # 多音字词
    '长大', '长期', '说话', '同意', '同行', '读书', '读音',
    '给我', '给了', '只有', '只能', '还要', '还是', '还有',
    # 通常的词
    '中国', '不要', '要说', '现在', '就是', '一样', '但是',
    # 一些特殊的，可能在vocab中不同格式出现
    '行业', '银行', '音乐', '音调',
]

def tone_numbers_to_marks_simple(pinyin: str) -> str:
    """简化版转换"""
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
    
    result = list(pinyin_base)
    matched = False
    
    for i, char in enumerate(result):
        if not matched and char in 'ae':
            result[i] = tone_marks.get(char, {}).get(tone_num, char)
            matched = True
    
    if not matched:
        for i in range(len(result) - 1, -1, -1):
            char = result[i]
            if char in 'iouüv':
                result[i] = tone_marks.get(char if char != 'v' else 'ü', {}).get(tone_num, char)
                break
    
    return "".join(result)

def convert_pinyin_toned(chars: str) -> str:
    """C格式模拟"""
    result = []
    for char in chars:
        py_candidates = lazy_pinyin(char, style=Style.TONE, errors='default')
        if py_candidates and py_candidates[0]:
            result.append(py_candidates[0])
    return "".join(result)

def convert_pinyin_diacritic_from_toned(toned: str) -> str:
    """将toned格式转为diacritic格式"""
    # 需要逐个拼音处理，不能搞成一个字符串
    # 因为'zhong1guo2'这样的会被_tone_numbers_to_marks错误处理
    
    # 提取所有拼音
    import re
    pinyins = re.findall(r'[a-zü]+\d', toned)
    if not pinyins or ''.join(pinyins) != toned:
        # 不是标准格式
        return "转换失败"
    
    return "".join([tone_numbers_to_marks_simple(py) for py in pinyins])

print("\n词转换对比:\n")
print(f"{'词':<6} | {'C格式':<15} | {'D格式':<15} | {'一致性'}")
print("-" * 60)

for word in test_words:
    c_result = convert_pinyin_toned(word)
    d_from_manual = convert_pinyin_diacritic_from_toned(c_result)
    
    # 也用各tokenizer查询
    c_in_vocab = 'Y' if c_result in vocab_c else 'N'
    d_in_vocab = (d_from_manual != "转换失败") and ('Y' if d_from_manual in vocab_d else 'N') or 'N'
    
    match = "✓" if d_from_manual != "转换失败" else "✗"
    
    print(f"{word:<6} | {c_result:<15} | {d_from_manual:<15} | {match} ({c_in_vocab}/{d_in_vocab})")

print("\n\n分析：")
print("问题分析:")
print("1. 如果C和D的转换结果都不在vocab中，那么A→C和A→D的'独立'会增加")
print("2. 如果某些多音字的处理不同步，可能导致C和D的映射不对应")
print("3. 需要检查word_to_pinyin是否为所有词都有正确的拼音")
