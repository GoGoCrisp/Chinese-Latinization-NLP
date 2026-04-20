#!/usr/bin/env python3
"""
详细输出AB、AC、AD的转换过程
"""
import sys
sys.path.insert(0, '/Users/crisp/Desktop/code_field/python/Chinese_Latinization_NLP/1.Tokenization')

import os
import re
from pypinyin import lazy_pinyin, Style

HAS_PYPINYIN = True

class DebugConverter:
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
        
        print(f"✓ Loaded: {len(self.char_to_pinyin)} chars, {len(self.word_to_pinyin)} words\n")
    
    def _tone_numbers_to_marks(self, pinyin: str) -> str:
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
    
    def remove_tone_numbers(self, pinyin: str) -> str:
        return re.sub(r'[0-9]', '', pinyin)
    
    def remove_tone_marks(self, pinyin: str) -> str:
        tone_map = {
            'ā': 'a', 'á': 'a', 'ǎ': 'a', 'à': 'a',
            'ē': 'e', 'é': 'e', 'ě': 'e', 'è': 'e',
            'ī': 'i', 'í': 'i', 'ǐ': 'i', 'ì': 'i',
            'ō': 'o', 'ó': 'o', 'ǒ': 'o', 'ò': 'o',
            'ū': 'u', 'ú': 'u', 'ǔ': 'u', 'ù': 'u',
            'ǖ': 'v', 'ǘ': 'v', 'ǚ': 'v', 'ǜ': 'v',
        }
        result = []
        for char in pinyin:
            result.append(tone_map.get(char, char))
        return "".join(result)
    
    def text_to_pinyin_toned(self, text: str, debug=False) -> str:
        if debug: print(f"    text_to_pinyin_toned('{text}')")
        
        # 1️⃣ 词级别
        if text in self.word_to_pinyin:
            result = "".join(self.word_to_pinyin[text])
            if debug: print(f"      → 找到词级别: {self.word_to_pinyin[text]} = '{result}'")
            return result
        
        # 2️⃣ 字级别
        if debug: print(f"      → 逐字处理")
        result = []
        for char in text:
            if char in self.char_to_pinyin:
                py = self.char_to_pinyin[char][0]
                result.append(py)
                if debug: print(f"         '{char}' → char[0] = '{py}'")
            else:
                # 3️⃣ pypinyin fallback
                if HAS_PYPINYIN:
                    py_candidates = lazy_pinyin(char, style=Style.TONE, errors='default')
                    if py_candidates and py_candidates[0]:
                        result.append(py_candidates[0])
                        if debug: print(f"         '{char}' → pypinyin = '{py_candidates[0]}'")
                    else:
                        result.append(char)
                        if debug: print(f"         '{char}' → 保留")
                else:
                    result.append(char)
        
        final = "".join(result)
        if debug: print(f"      → 最终: '{final}'")
        return final
    
    def text_to_pinyin_diacritic(self, text: str, debug=False) -> str:
        if debug: print(f"    text_to_pinyin_diacritic('{text}')")
        
        # 1️⃣ 词级别
        if text in self.word_to_pinyin:
            pinyin_list = self.word_to_pinyin[text]
            converted = [self._tone_numbers_to_marks(py) for py in pinyin_list]
            result = "".join(converted)
            if debug: print(f"      → 找到词级别: {pinyin_list}")
            if debug: print(f"      → 转为符号: {converted} = '{result}'")
            return result
        
        # 2️⃣ 字级别
        if debug: print(f"      → 逐字处理")
        result = []
        for char in text:
            if char in self.char_to_pinyin:
                pinyin = self.char_to_pinyin[char][0]
                py_mark = self._tone_numbers_to_marks(pinyin)
                result.append(py_mark)
                if debug: print(f"         '{char}' → char[0] = '{pinyin}' → marks = '{py_mark}'")
            else:
                # 3️⃣ pypinyin fallback
                if HAS_PYPINYIN:
                    py_candidates = lazy_pinyin(char, style=Style.TONE, errors='default')
                    if py_candidates and py_candidates[0]:
                        py_mark = self._tone_numbers_to_marks(py_candidates[0])
                        result.append(py_mark)
                        if debug: print(f"         '{char}' → pypinyin = '{py_candidates[0]}' → marks = '{py_mark}'")
                    else:
                        result.append(char)
                        if debug: print(f"         '{char}' → 保留")
                else:
                    result.append(char)
        
        final = "".join(result)
        if debug: print(f"      → 最终: '{final}'")
        return final
    
    def convert_to_b(self, text: str, debug=False) -> str:
        """A → B转换"""
        if debug: print(f"    A→B转换('{text}')")
        
        converted_parts = []
        for char in text:
            if char in self.char_to_pinyin:
                py = self.char_to_pinyin[char][0]
                py_clean = self.remove_tone_numbers(self.remove_tone_marks(py))
                converted_parts.append(py_clean)
                if debug: print(f"      '{char}' → char[0] = '{py}' → remove all = '{py_clean}'")
            else:
                converted_parts.append(char)
                if debug: print(f"      '{char}' → 保留")
        
        converted = "".join(converted_parts)
        if debug: print(f"      → 最终: '{converted}'")
        return converted


# 初始化
converter = DebugConverter('/Users/crisp/Desktop/code_field/python/Chinese_Latinization_NLP/1.Tokenization/dicts/cedict_ts.u8')

# 测试用例
test_words = ['中', '国', '中国', '长', '长大', '说', '说话']

print("="*80)
print("详细转换过程对比")
print("="*80)

for word in test_words:
    print(f"\n【{word}】")
    print("-" * 40)
    
    ab = converter.convert_to_b(word, debug=True)
    print(f"  → A→B结果: '{ab}'")
    print()
    
    ac = converter.text_to_pinyin_toned(word, debug=True)
    print(f"  → A→C结果: '{ac}'")
    print()
    
    ad = converter.text_to_pinyin_diacritic(word, debug=True)
    print(f"  → A→D结果: '{ad}'")
    print()
    
    # 检查一致性
    print(f"  一致性检查:")
    print(f"    AB: {ab}")
    print(f"    AC: {ac}")
    print(f"    AD: {ad}")
    
    # 从AC推导AD
    ac_to_ad = "".join([converter._tone_numbers_to_marks(py) for py in re.findall(r'[a-zü]+\d', ac)])
    print(f"    AC→AD推导: {ac_to_ad}")
    print(f"    AD一致性: {'✓' if ac_to_ad == ad else '✗'}")

print("\n" + "="*80)
