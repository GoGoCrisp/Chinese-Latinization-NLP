#!/usr/bin/env python3
"""
调查"中国"为什么出现MISMATCH
"""
import sys
sys.path.insert(0, '/Users/crisp/Desktop/code_field/python/Chinese_Latinization_NLP/1.Tokenization')

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
        print(f"  [C转换] 输入: '{text}'")
        
        if text in self.word_to_pinyin:
            result = "".join(self.word_to_pinyin[text])
            print(f"    → 找到词级别映射: {self.word_to_pinyin[text]} = '{result}'")
            return result
        
        print(f"    → 未找到词级别映射，逐字转换")
        result = []
        for char in text:
            if char in self.char_to_pinyin:
                py = self.char_to_pinyin[char][0]
                print(f"      '{char}' → '{py}' (字级别)")
                result.append(py)
            else:
                py_candidates = lazy_pinyin(char, style=Style.TONE, errors='default')
                if py_candidates and py_candidates[0]:
                    print(f"      '{char}' → '{py_candidates[0]}' (pypinyin)")
                    result.append(py_candidates[0])
                else:
                    print(f"      '{char}' → '{char}' (保留原字)")
                    result.append(char)
        
        final = "".join(result)
        print(f"    最终结果: '{final}'")
        return final
    
    def text_to_pinyin_diacritic(self, text: str) -> str:
        """A → D (符号声调)"""
        print(f"  [D转换] 输入: '{text}'")
        
        if text in self.word_to_pinyin:
            pinyin_list = self.word_to_pinyin[text]
            result = "".join([self._tone_numbers_to_marks(py) for py in pinyin_list])
            print(f"    → 找到词级别映射: {pinyin_list}")
            print(f"    → 转换为符号: {[self._tone_numbers_to_marks(py) for py in pinyin_list]} = '{result}'")
            return result
        
        print(f"    → 未找到词级别映射，逐字转换")
        result = []
        for char in text:
            if char in self.char_to_pinyin:
                pinyin = self.char_to_pinyin[char][0]
                py_mark = self._tone_numbers_to_marks(pinyin)
                print(f"      '{char}' → '{pinyin}' → '{py_mark}' (字级别)")
                result.append(py_mark)
            else:
                py_candidates = lazy_pinyin(char, style=Style.TONE, errors='default')
                if py_candidates and py_candidates[0]:
                    py_mark = self._tone_numbers_to_marks(py_candidates[0])
                    print(f"      '{char}' → '{py_candidates[0]}' → '{py_mark}' (pypinyin)")
                    result.append(py_mark)
                else:
                    print(f"      '{char}' → '{char}' (保留原字)")
                    result.append(char)
        
        final = "".join(result)
        print(f"    最终结果: '{final}'")
        return final


# 初始化转换器
converter = QuickConverter('/Users/crisp/Desktop/code_field/python/Chinese_Latinization_NLP/1.Tokenization/dicts/cedict_ts.u8')

print("\n=== 调查'中国'的MISMATCH ===\n")

# 查看word_to_pinyin中的映射
if '中国' in converter.word_to_pinyin:
    print(f"word_to_pinyin['中国'] = {converter.word_to_pinyin['中国']}")
else:
    print(f"word_to_pinyin中未包含'中国'")

print()

# 逐步转换
c_result = converter.text_to_pinyin_toned('中国')
print()
d_result = converter.text_to_pinyin_diacritic('中国')
print()

# 对比
d_from_c = converter._tone_numbers_to_marks(c_result) if any(ch.isdigit() for ch in c_result) else c_result

print(f"\n对比结果:")
print(f"  C格式: {c_result}")
print(f"  D格式: {d_result}")
print(f"  从C转换的D: {d_from_c}")
print(f"  一致性: {'✓' if d_from_c == d_result else '✗ MISMATCH'}")
print()

# 逐个字符转换
print(f"按单字转换:")
for char in '中国':
    c = converter.text_to_pinyin_toned(char)
    d = converter.text_to_pinyin_diacritic(char)
    print(f"  {char}: C={c}, D={d}")
