#!/usr/bin/env python3
"""
深入调查A_vs_D N对1过高的原因
"""
import sys
sys.path.insert(0, '/Users/crisp/Desktop/code_field/python/Chinese_Latinization_NLP/1.Tokenization')

# 直接在这个脚本中实现转换
import os
import re
from pypinyin import lazy_pinyin, Style

class QuickConverter:
    def __init__(self, cedict_path):
        self.char_to_pinyin = {}
        self.word_to_pinyin = {}
        self.load_cedict(cedict_path)
    
    def load_cedict(self, cedict_path):
        if not os.path.exists(cedict_path):
            print(f"CEDICT not found: {cedict_path}")
            return
        
        with open(cedict_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                
                match = re.match(r'(\S+)\s+(\S+)\s+\[(.*?)\]', line)
                if not match:
                    continue
                
                simplified = match.group(2)
                pinyin_str = match.group(3)
                
                pinyin_list = [p.lower() for p in pinyin_str.split()]
                if simplified and pinyin_list:
                    if len(simplified) == 1:
                        if simplified not in self.char_to_pinyin:
                            self.char_to_pinyin[simplified] = []
                        for py in pinyin_list:
                            if py not in self.char_to_pinyin[simplified]:
                                self.char_to_pinyin[simplified].append(py)
                    else:
                        if simplified not in self.word_to_pinyin:
                            self.word_to_pinyin[simplified] = pinyin_list
        
        print(f"Loaded CEDICT: {len(self.char_to_pinyin)} chars, {len(self.word_to_pinyin)} words")
    
    def _tone_numbers_to_marks(self, pinyin: str) -> str:
        """C格式(数字) → D格式(符号)"""
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
    
    def text_to_pinyin_toned(self, text: str) -> str:
        """A → C (数字声调)"""
        if text in self.word_to_pinyin:
            return "".join(self.word_to_pinyin[text])
        
        result = []
        for char in text:
            if char in self.char_to_pinyin:
                result.append(self.char_to_pinyin[char][0])
            else:
                py_candidates = lazy_pinyin(char, style=Style.TONE, errors='default')
                if py_candidates and py_candidates[0]:
                    result.append(py_candidates[0])
                else:
                    result.append(char)
        return "".join(result)
    
    def text_to_pinyin_diacritic(self, text: str) -> str:
        """A → D (符号声调)"""
        if text in self.word_to_pinyin:
            pinyin_list = self.word_to_pinyin[text]
            return "".join([self._tone_numbers_to_marks(py) for py in pinyin_list])
        
        result = []
        for char in text:
            if char in self.char_to_pinyin:
                pinyin = self.char_to_pinyin[char][0]
                result.append(self._tone_numbers_to_marks(pinyin))
            else:
                py_candidates = lazy_pinyin(char, style=Style.TONE, errors='default')
                if py_candidates and py_candidates[0]:
                    result.append(self._tone_numbers_to_marks(py_candidates[0]))
                else:
                    result.append(char)
        return "".join(result)


# 初始化转换器
converter = QuickConverter('/Users/crisp/Desktop/code_field/python/Chinese_Latinization_NLP/1.Tokenization/dicts/cedict_ts.u8')

print("\n=== 测试转换 ===\n")

# 测试用例
test_words = ['中', '国', '中国', '不', '要', '说', '长', '行', '读', '给', '约', '能', '只', '还']

print(f"{'词':<5} | {'C格式(数字)':<15} | {'D格式(符号)':<15} | {'一致性'}")
print("-" * 60)

for word in test_words:
    c_result = converter.text_to_pinyin_toned(word)
    d_result = converter.text_to_pinyin_diacritic(word)
    
    # 检查一致性：D应该是C的符号版本
    d_from_c = converter._tone_numbers_to_marks(c_result) if any(ch.isdigit() for ch in c_result) else c_result
    
    match = "✓" if d_from_c == d_result else "✗ MISMATCH"
    print(f"{word:<5} | {c_result:<15} | {d_result:<15} | {match}")


# 检查问题：是否存在不同的字生成相同的D token的情况
print("\n\n=== 检查N对1问题 ===\n")

# 收集字符对应关系
char_to_c = {}
char_to_d = {}
c_to_chars = {}
d_to_chars = {}

# 只检查常用汉字（前2000个）
import sys
sys.path.insert(0, '/Users/crisp/Desktop/code_field/python/Chinese_Latinization_NLP/1.Tokenization')

common_chars_file = '/Users/crisp/Desktop/code_field/python/Chinese_Latinization_NLP/1.Tokenization/dicts/cedict_ts.u8'
test_chars = list(converter.char_to_pinyin.keys())[:500]  # 测试前500个

print(f"检查 {len(test_chars)} 个汉字...")

for char in test_chars:
    c_result = converter.text_to_pinyin_toned(char)
    d_result = converter.text_to_pinyin_diacritic(char)
    
    char_to_c[char] = c_result
    char_to_d[char] = d_result
    
    if c_result not in c_to_chars:
        c_to_chars[c_result] = []
    c_to_chars[c_result].append(char)
    
    if d_result not in d_to_chars:
        d_to_chars[d_result] = []
    d_to_chars[d_result].append(char)

# 查找N对1的情况
print(f"\nC格式中的N对1 (N>=2):")
c_n_to_1 = {c: chars for c, chars in c_to_chars.items() if len(chars) >= 2}
print(f"  发现 {len(c_n_to_1)} 个")
for c, chars in sorted(c_n_to_1.items(), key=lambda x: -len(x[1]))[:10]:
    print(f"    '{c}' ← {chars[:5]}")

print(f"\nD格式中的N对1 (N>=2):")
d_n_to_1 = {d: chars for d, chars in d_to_chars.items() if len(chars) >= 2}
print(f"  发现 {len(d_n_to_1)} 个")
for d, chars in sorted(d_n_to_1.items(), key=lambda x: -len(x[1]))[:10]:
    print(f"    '{d}' ← {chars[:5]}")

# 统计
print(f"\n统计汇总:")
print(f"  C格式 - 映射到多个字的token: {len(c_n_to_1)} 个")
print(f"  D格式 - 映射到多个字的token: {len(d_n_to_1)} 个 ← 差异(如果远大于C)")
