"""
完整版：9th_compare_tokenizers_overlap.py 的增强版
集成三层方案：Unihan + CEDICT + pypinyin
特别改进了拼音normalize函数处理Unihan格式
"""

import json
import os
import re
import random
from tokenizers import Tokenizer as HFTokenizer
from itertools import combinations
from tqdm import tqdm
import unicodedata

# 新增：pypinyin用于fallback
try:
    from pypinyin import lazy_pinyin, Style
    HAS_PYPINYIN = True
except ImportError:
    HAS_PYPINYIN = False
    print("Warning: pypinyin not installed.")

# ===== 配置 =====
TOKENIZERS_DIR = "./tokenizers"
DICTS_DIR = "./dicts"
OUTPUT_FILE = "tokenizer_overlap_analysis_enhanced.txt"

TOKENIZERS_64K = {
    "A_chinese_origin": "chinese_origin_64k_train90.json",
    "B_pinyin_toneless": "pinyin_toneless_64k_train90.json",
    "C_pinyin_toned": "pinyin_toned_64k_train90.json",
    "D_pinyin_diacritic": "pinyin_diacritic_64k_train90.json",
}

TOKENIZER_PAIRS = list(combinations(sorted(TOKENIZERS_64K.keys()), 2))


# ===== 增强版：拼音转换工具 =====

class PinyinConverterEnhanced:
    """增强版拼音转换工具：三层方案（Unihan + CEDICT + pypinyin）"""
    
    def __init__(self, cedict_path: str, merged_dict_path: str = None):
        """初始化"""
        self.word_to_pinyin = {}
        self.char_to_pinyin = {}
        self.char_to_pinyin_merged = {}  # 新增：Unihan + CEDICT合并
        
        self.load_cedict(cedict_path)
        if merged_dict_path and os.path.exists(merged_dict_path):
            self.load_merged_dict(merged_dict_path)
    
    def load_merged_dict(self, merged_dict_path: str):
        """加载Unihan + CEDICT合并字典"""
        try:
            with open(merged_dict_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.char_to_pinyin_merged = data.get('data', {})
            print(f"✓ Loaded merged dict: {len(self.char_to_pinyin_merged)} chars")
        except Exception as e:
            print(f"✗ Error loading merged dict: {e}")
    
    def load_cedict(self, cedict_path: str):
        """从CC-CEDICT加载"""
        if not os.path.exists(cedict_path):
            print(f"⊘ CEDICT not found: {cedict_path}")
            return
        
        try:
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
            
            print(f"✓ Loaded CEDICT: {len(self.char_to_pinyin)} chars, {len(self.word_to_pinyin)} words")
        except Exception as e:
            print(f"✗ Error loading CEDICT: {e}")
    
    def normalize_pinyin_advanced(self, pinyin: str) -> str:
        """
        高级normalize函数：处理多种格式
        - 数字声调 (zhang1)
        - 声调符号 (zhāng)
        - Unihan格式 (可能包含声调符号)
        - u:表示ü的情况
        """
        p = pinyin.lower()
        
        # 第1步：处理声调符号（来自Unihan）
        tone_marks = {
            'ā': 'a1', 'á': 'a2', 'ǎ': 'a3', 'à': 'a4',
            'ē': 'e1', 'é': 'e2', 'ě': 'e3', 'è': 'e4',
            'ī': 'i1', 'í': 'i2', 'ǐ': 'i3', 'ì': 'i4',
            'ō': 'o1', 'ó': 'o2', 'ǒ': 'o3', 'ò': 'o4',
            'ū': 'u1', 'ú': 'u2', 'ǔ': 'u3', 'ù': 'u4',
            'ǖ': 'v1', 'ǘ': 'v2', 'ǚ': 'v3', 'ǜ': 'v4',
            'ń': 'n2', 'ň': 'n3', 'ǹ': 'n4',
            'ḿ': 'm2',
        }
        
        for mark, replacement in tone_marks.items():
            p = p.replace(mark, replacement)
        
        # 第2步：处理u:表示ü的情况（CEDICT格式）
        p = p.replace('u:', 'v')
        p = p.replace('ü', 'v')
        
        # 第3步：移除所有声调数字
        p = re.sub(r'[0-5]', '', p)
        
        return p
    
    def get_pinyin_toneless(self, char: str) -> str:
        """三层优先级获取拼音"""
        # 优先1：CEDICT
        if char in self.char_to_pinyin and self.char_to_pinyin[char]:
            py = self.char_to_pinyin[char][0]
            return self.normalize_pinyin_advanced(py)
        
        # 优先2：合并字典（Unihan + CEDICT）
        if char in self.char_to_pinyin_merged:
            py = self.char_to_pinyin_merged[char]
            if py:
                return self.normalize_pinyin_advanced(str(py))
        
        # 优先3：pypinyin
        if HAS_PYPINYIN:
            try:
                result = lazy_pinyin(char, style=Style.NORMAL)
                if result and result[0]:
                    return result[0].lower()
            except:
                pass
        
        return ""
    
    def text_to_pinyin_toneless(self, text: str) -> str:
        """文本到无声调拼音"""
        # 词级别优先
        if text in self.word_to_pinyin:
            pinyin_list = self.word_to_pinyin[text]
            converted = "".join(pinyin_list)
            return self.normalize_pinyin_advanced(converted)
        
        # 字级别
        result = []
        for char in text:
            py = self.get_pinyin_toneless(char)
            if py:
                result.append(py)
            else:
                result.append(char)
        
        return "".join(result)


# ===== 使用新转换器进行完整分析 =====

def analyze_tokenizers_enhanced():
    """使用增强版转换器进行完整分析"""
    
    print("=" * 100)
    print("TOKENIZER OVERLAP ANALYSIS - ENHANCED VERSION (Unihan + CEDICT + pypinyin)")
    print("=" * 100)
    
    # 初始化增强版转换器
    cedict_path = os.path.join(DICTS_DIR, "cedict_ts.u8")
    merged_dict_path = os.path.join(DICTS_DIR, "merged_pinyin_dict.json")
    
    converter = PinyinConverterEnhanced(cedict_path, merged_dict_path)
    
    print(f"\n【数据源】")
    print(f"  • CEDICT字符数: {len(converter.char_to_pinyin):,}")
    print(f"  • 合并字典字符数: {len(converter.char_to_pinyin_merged):,}")
    
    # 加载tokenizer
    tokenizers = {}
    for name, filename in TOKENIZERS_64K.items():
        path = os.path.join(TOKENIZERS_DIR, filename)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            tokenizers[name] = set(data['model']['vocab'].keys())
    
    print(f"\n【Tokenizer加载】")
    for name, vocab in tokenizers.items():
        print(f"  • {name}: {len(vocab):,} tokens")
    
    # ===== A→B映射分析 =====
    print(f"\n" + "=" * 100)
    print("A → B 映射分析（使用增强版拼音转换）")
    print("=" * 100)
    
    vocab_a = tokenizers["A_chinese_origin"]
    vocab_b = tokenizers["B_pinyin_toneless"]
    
    # 提取纯中文token
    def is_pure_chinese(token):
        if not token:
            return False
        return all('\u4e00' <= c <= '\u9fff' for c in token)
    
    chinese_tokens = [t for t in vocab_a if is_pure_chinese(t)]
    
    print(f"\n【中文Token】")
    print(f"  • A总token数: {len(vocab_a):,}")
    print(f"  • A纯中文token数: {len(chinese_tokens):,}")
    
    # 统计映射
    mapped = 0
    unmapped = []
    
    for token in tqdm(chinese_tokens, desc="Mapping A→B"):
        converted = converter.text_to_pinyin_toneless(token)
        if converted and converted in vocab_b:
            mapped += 1
        else:
            unmapped.append((token, converted))
    
    print(f"\n【映射结果】")
    print(f"  ✅ 映射成功: {mapped:,} ({100*mapped/len(chinese_tokens):.1f}%)")
    print(f"  ❌ 映射失败: {len(unmapped):,} ({100*len(unmapped)/len(chinese_tokens):.1f}%)")
    
    # 失败示例
    if unmapped:
        print(f"\n【失败示例】(前20个):")
        for token, converted in unmapped[:20]:
            in_b = converted in vocab_b if converted else "N/A"
            print(f"    {token:<20} → {converted:<30} | 在B中: {in_b}")
    
    return mapped, len(unmapped), len(chinese_tokens)


if __name__ == "__main__":
    mapped, unmapped, total = analyze_tokenizers_enhanced()
    
    print(f"\n" + "=" * 100)
    print(f"【最终统计】")
    print(f"  总token: {total:,}")
    print(f"  成功: {mapped:,} ({100*mapped/total:.1f}%)")
    print(f"  失败: {unmapped:,} ({100*unmapped/total:.1f}%)")
    print("=" * 100)
