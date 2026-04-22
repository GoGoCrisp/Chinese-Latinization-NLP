"""
比较不同64k tokenizers之间的词语映射和重叠
使用CC-CEDICT建立汉字→拼音的映射

A = chinese_origin (中文)
B = pinyin_toneless (无声调拼音, e.g., "zhongwen")
C = pinyin_toned (带数字声调, e.g., "zhong1wen2")
D = pinyin_diacritic (带声调符号, e.g., "zhōngwén")

比较规则:
- AB: A转换为B格式（查字典），然后比较
- AC: A转换为C格式（查字典），然后比较
- AD: A转换为D格式（查字典），然后比较
- BC: C去掉数字转换为B格式，然后比较
- BD: D去掉声调转换为B格式，然后比较
- CD: D去掉声调转换为C格式，然后比较
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
    from pypinyin import pinyin, lazy_pinyin, Style
    HAS_PYPINYIN = True
except ImportError:
    HAS_PYPINYIN = False
    print("Warning: pypinyin not installed. Fallback will be disabled.")

# ===== 配置 =====
TOKENIZERS_DIR = "decoded_superTokenizers"
DICTS_DIR = "./dicts"
OUTPUT_FILE = "tokenizer_overlap_analysis_superBPE.txt"

# 4个64k tokenizers映射
TOKENIZERS_64K = {
    "A_chinese_origin": "chinese_origin_subset100k_superbpe_64000_decoded.json",
    "B_pinyin_toneless": "pinyin_toneless_subset100k_superbpe_64000_decoded.json",
    "C_pinyin_toned": "pinyin_toned_subset100k_superbpe_64000_decoded.json",
    "D_pinyin_diacritic": "pinyin_diacritic_subset100k_superbpe_64000_decoded.json",
}

# 生成所有的tokenizer对
TOKENIZER_PAIRS = list(combinations(sorted(TOKENIZERS_64K.keys()), 2))


# ===== 拼音转换工具 =====

class PinyinConverter:
    """拼音转换和映射工具 (混合方案：CEDICT + pypinyin fallback)"""
    
    def __init__(self, cedict_path: str, merged_dict_path: str = None):
        """初始化，加载CC-CEDICT字典和Unihan合并字典"""
        self.word_to_pinyin = {}  # 词 → 拼音列表
        self.char_to_pinyin = {}  # 汉字 → 拼音列表
        self.char_to_pinyin_merged = {}  # 新增：Unihan + CEDICT合并字典
        self.load_cedict(cedict_path)
        if merged_dict_path:
            self.load_merged_dict(merged_dict_path)
    
    def load_cedict(self, cedict_path: str):
        """从CC-CEDICT加载词级别和字级别的拼音映射
        
        核心改进：
        1. 对于多音字，收集所有可能的读音（不只是第一个）
        2. 在转换时优先使用最常见的（第一个）读音
        3. 这样可以处理多音字问题
        
        例如"约"字：
          约 约 [yao1] /约定/...
          约 约 [yue1] /约束/...
        应该保存为: char_to_pinyin['约'] = ['yao1', 'yue1']
        转换时使用第一个 yao1
        但当需要时可以尝试其他读音
        """
        if not os.path.exists(cedict_path):
            print(f"⊘ CEDICT not found: {cedict_path}")
            return
        
        try:
            with open(cedict_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("#"):
                        continue
                    
                    # 格式: 繁体 简体 [拼音1 拼音2 ...] /定义/
                    # 使用正则表达式提取括号内的拼音
                    match = re.match(r'(\S+)\s+(\S+)\s+\[(.*?)\]', line)
                    if not match:
                        continue
                    
                    simplified = match.group(2)  # 简体
                    pinyin_str = match.group(3)  # 括号内的拼音
                    
                    # 保存拼音列表（转换为小写）
                    pinyin_list = [p.lower() for p in pinyin_str.split()]
                    if simplified and pinyin_list:
                        if len(simplified) == 1:
                            # 单字：收集所有读音而不是覆盖
                            if simplified not in self.char_to_pinyin:
                                self.char_to_pinyin[simplified] = []
                            for py in pinyin_list:
                                if py not in self.char_to_pinyin[simplified]:
                                    self.char_to_pinyin[simplified].append(py)
                        else:
                            # 多字词：保留第一个条目（最常见）
                            if simplified not in self.word_to_pinyin:
                                self.word_to_pinyin[simplified] = pinyin_list
            
            print(f"✓ Loaded CEDICT: {len(self.char_to_pinyin)} chars (multi-pronunciation support), {len(self.word_to_pinyin)} words")
        except Exception as e:
            print(f"✗ Error loading CEDICT: {e}")
    
    def load_merged_dict(self, merged_dict_path: str):
        """加载Unihan + CEDICT合并字典"""
        if not os.path.exists(merged_dict_path):
            return
        try:
            with open(merged_dict_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.char_to_pinyin_merged = data.get('data', {})
            print(f"✓ Loaded merged dict (Unihan+CEDICT): {len(self.char_to_pinyin_merged)} chars")
        except Exception as e:
            print(f"⊘ Error loading merged dict: {e}")
    
    def _pinyin_to_toneless(self, pinyin: str) -> str:
        """移除拼音中的声调标记和数字（改进版，支持Unihan格式）"""
        p = pinyin.lower()
        
        # 第1步：处理声调符号（Unihan格式）
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
    
    def _tone_numbers_to_marks(self, pinyin: str) -> str:
        """
        将数字声调转换为声调符号
        例如: "zhong1" -> "zhōng", "guo2" -> "guó"
        """
        # 声调符号映射表
        tone_marks = {
            'a': {'1': 'ā', '2': 'á', '3': 'ǎ', '4': 'à'},
            'e': {'1': 'ē', '2': 'é', '3': 'ě', '4': 'è'},
            'i': {'1': 'ī', '2': 'í', '3': 'ǐ', '4': 'ì'},
            'o': {'1': 'ō', '2': 'ó', '3': 'ǒ', '4': 'ò'},
            'u': {'1': 'ū', '2': 'ú', '3': 'ǔ', '4': 'ù'},
            'ü': {'1': 'ǖ', '2': 'ǘ', '3': 'ǚ', '4': 'ǜ'},
            'v': {'1': 'ǖ', '2': 'ǘ', '3': 'ǚ', '4': 'ǜ'},
        }
        
        # 提取末尾的数字（1-4表示第几声，5表示轻声）
        tone_num = ''
        if pinyin and pinyin[-1].isdigit():
            tone_num = pinyin[-1]
            pinyin_base = pinyin[:-1]
        else:
            return pinyin
        
        # 轻声直接返回
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
        
        # 如果a/e没有匹配，处理其他元音
        if not matched:
            for i in range(len(result) - 1, -1, -1):
                char = result[i]
                if char in 'iouüv':
                    result[i] = tone_marks.get(char if char != 'v' else 'ü', {}).get(tone_num, char)
                    break
        
        return "".join(result)
    
    def text_to_pinyin_toneless(self, text: str) -> str:
        """
        将中文文本转换为无声调拼音
        使用pypinyin直接转换（Style.NORMAL）
        """
        py_result = pinyin(text, style=Style.NORMAL, strict=False)
        return "".join([p[0] for p in py_result])
    
    def text_to_pinyin_toned(self, text: str) -> str:
        """
        将中文文本转换为带数字声调的拼音
        使用pypinyin直接转换（Style.TONE3）
        """
        py_result = pinyin(text, style=Style.TONE3, strict=False)
        return "".join([p[0] for p in py_result])
    
    def text_to_pinyin_diacritic(self, text: str) -> str:
        """
        将中文文本转换为带声调符号的拼音
        使用pypinyin直接转换（Style.TONE）
        """
        py_result = pinyin(text, style=Style.TONE, strict=False)
        return "".join([p[0] for p in py_result])
    
    def remove_tone_numbers(self, pinyin: str) -> str:
        """移除拼音中的数字声调 (e.g., "zhong1guo2" -> "zhongguo")"""
        return re.sub(r'[0-9]', '', pinyin)
    
    def remove_tone_marks(self, pinyin: str) -> str:
        """移除拼音中的声调标记 (e.g., "zhōngguó" -> "zhongguo")"""
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
    
    def _tone_numbers_to_marks_full(self, pinyin_str: str) -> str:
        """
        将拼音字符串中的数字声调转换为符号
        例如: "zhong1guo2" -> "zhōngguó"
        处理多音节的情况（通过正则表达式分割音节）
        """
        # 使用正则表达式提取音节：[a-z]+[0-5]?
        # 这样可以识别 "zhang" "zhang1" "zho" 等
        pattern = r'([a-zü]+)([0-4]?)'
        
        def convert_syllable(match):
            syllable = match.group(1)
            tone_digit = match.group(2) if match.group(2) else ''
            
            if tone_digit:
                # 有声调数字，转换为符号
                return self._tone_numbers_to_marks(syllable + tone_digit)
            else:
                # 无声调
                return syllable
        
        # 替换所有音节
        result = re.sub(r'([a-zü]+)([0-4])', convert_syllable, pinyin_str, flags=re.IGNORECASE)
        return result
    
    def align_d_to_c_structure(self, c_token: str, d_token: str) -> str:
        """
        使用C的音节结构来转换D (带声调符号) 为带数字的格式
        
        目标：D (例如 lǐhài) 通过与C (例如 li4hai4) 对齐，转换为 li3hai4 的格式
        
        步骤：
        1. 从C中提取音节结构 (li, 4), (hai, 4)
        2. 从D中去掉声调得到基础拼音 lihai
        3. 验证D基础拼音与C相符
        4. 从D中的声调符号提取数字声调
        5. 按C的音节边界重组D，转为数字声调格式
        
        强约束：C中的数字声调个数必须等于D中的声调符号个数
        （例如：'ke4'有1个声调，只能映射到'kè'，不能映射到'ke'）
        """
        import re
        
        c_lower = c_token.lower()
        d_lower = d_token.lower()
        
        # 步骤0：强约束检查 - D中的声调符号个数必须等于C中的数字声调个数
        tone_marks = {'ā', 'á', 'ǎ', 'à', 'ē', 'é', 'ě', 'è', 'ī', 'í', 'ǐ', 'ì', 
                     'ō', 'ó', 'ǒ', 'ò', 'ū', 'ú', 'ǔ', 'ù', 'ǖ', 'ǘ', 'ǚ', 'ǜ'}
        
        # 计算C中的数字声调个数（1-4为声调，5和0为轻声）
        c_tone_count = sum(1 for ch in c_lower if ch.isdigit() and ch in '1234')
        
        # 计算D中的声调符号个数
        d_tone_count = sum(1 for ch in d_lower if ch in tone_marks)
        
        # 强约束：声调个数必须匹配
        if c_tone_count != d_tone_count:
            return None
        
        # 步骤1：从C提取音节结构 (拼音基础, 声调数字)
        c_pattern = r'([a-z]+)([0-5])'
        c_syllables = re.findall(c_pattern, c_lower)
        if not c_syllables:
            return None
        
        # 步骤2&3：从D去掉声调并验证基础拼音
        d_base = self._pinyin_to_toneless(d_lower)
        c_base = self.remove_tone_numbers(c_lower)
        
        if d_base != c_base:
            # 基础拼音不匹配，说明C和D不是对应的
            return None
        
        # 步骤4&5：从D中提取声调，按C的边界重组
        tone_to_number = {
            'ā': '1', 'á': '2', 'ǎ': '3', 'à': '4',
            'ē': '1', 'é': '2', 'ě': '3', 'è': '4',
            'ī': '1', 'í': '2', 'ǐ': '3', 'ì': '4',
            'ō': '1', 'ó': '2', 'ǒ': '3', 'ò': '4',
            'ū': '1', 'ú': '2', 'ǔ': '3', 'ù': '4',
            'ǖ': '1', 'ǘ': '2', 'ǚ': '3', 'ǜ': '4',
        }
        
        # 提取D中每个位置的声调信息
        # 关键修复：处理带声调符号形式（如èxìng）时，声调符号本身就是元音字母的替代品
        # 需要可靠地对应d_base中的位置
        d_tones = {}  # 位置 -> 声调数字 (在d_base中的位置)
        
        # 方法：通过对齐D和d_base来提取声调
        # d_base已经是去掉声调的版本，其中每个元音已经转换为无声调形式
        d_pos = 0  # D中的当前位置
        d_base_pos = 0  # d_base中的当前位置
        
        while d_pos < len(d_lower) and d_base_pos < len(d_base):
            d_char = d_lower[d_pos]
            base_char = d_base[d_base_pos]
            
            if d_char in tone_to_number:
                # 当前D字符是个带声调的元音
                # 找到对应的无声调元音（应该是相同的字母，只是没有声调符号）
                base_version = self._pinyin_to_toneless(d_char)
                if base_version == base_char:
                    # 匹配：这个声调符号对应d_base中这个位置
                    d_tones[d_base_pos] = tone_to_number[d_char]
                    d_pos += 1
                    d_base_pos += 1
                else:
                    # 不匹配的情况（不应该发生）
                    return None
            else:
                # D中是普通字符
                if d_char == base_char:
                    # 匹配
                    d_pos += 1
                    d_base_pos += 1
                else:
                    # 不匹配
                    return None
        
        # 检查是否全部扫描完毕
        if d_pos != len(d_lower) or d_base_pos != len(d_base):
            return None
        
        # 按C的音节边界分割D，提取每个音节的声调
        # 关键修复：严格验证声调数量，确保每个音节都有完整的声调信息
        result_parts = []
        base_pos = 0
        
        for c_base_py, c_tone_digit in c_syllables:
            # 这个音节在d_base中对应的部分
            segment = d_base[base_pos:base_pos + len(c_base_py)]
            
            if segment != c_base_py:
                # 音节不匹配
                return None
            
            # 在这个音节范围内查找所有声调符号
            # 对于多元音的音节（如'iao'），可能有多个声调符号
            syllable_tones = {}
            for i in range(len(c_base_py)):
                char_pos = base_pos + i
                if char_pos in d_tones:
                    syllable_tones[i] = d_tones[char_pos]
            
            # 确定这个音节的声调
            # 对于多元音，第一个找到的声调就是这个音节的声调
            if syllable_tones:
                d_tone = syllable_tones[min(syllable_tones.keys())]
            else:
                d_tone = None
            
            # 严格验证：D中必须有声调，且与C中的声调一致
            # 如果C有声调数字但D中没有对应的声调符号，说明不匹配
            if d_tone is None:
                # D中这个音节没有声调符号
                # 检查C中这个音节是否有声调数字
                if c_tone_digit != '5':
                    # C中有明确的声调，但D中没有找到，说明不匹配
                    return None
                else:
                    # C中是轻声，D中也没有声调，使用轻声
                    d_tone = '5'
            else:
                # D中找到了声调符号
                # 如果C中是无声调形式（轻声），但D中有声调，也是不匹配
                if c_tone_digit == '5' and d_tone != '5':
                    # C中是轻声，但D中有声调，不匹配
                    return None
            
            result_parts.append(c_base_py + d_tone)
            base_pos += len(c_base_py)
        
        return "".join(result_parts)
    
    def add_tone_numbers_to_toneless(self, toneless: str) -> list:
        """
        给无声调拼音添加数字声调
        例如: "zhongguo" -> ["zhong1guo2", "zhong2guo2", ...]
        返回可能的带数字拼音列表
        
        这是个简化实现：实际中文字很少有多音字，我们先假设最常见的声调
        """
        # 这里我们返回可能的组合
        # 简单起见，返回原文 + 尝试常见的声调模式
        results = [toneless]  # 直接保留无声调形式
        
        # 尝试添加1,2,4声的组合（3声相对较少）
        for tone_pattern in self._generate_tone_patterns(len([c for c in toneless if c.isalpha()])):
            results.append(tone_pattern)
        
        return results[0:1]  # 现在先只返回无声调版本本身
    
    def _generate_tone_patterns(self, n_syllables: int) -> list:
        """生成n个音节的声调模式（简化版）"""
        # 这是个占位符
        return []


def load_tokenizer_vocab(tokenizer_path: str, is_pinyin: bool = False) -> dict:
    """加载tokenizer的vocabulary"""
    try:
        with open(tokenizer_path, 'r', encoding='utf-8') as f:
            raw_vocab = json.load(f)
            
        vocab = {}
        for k, v in raw_vocab.items():
            clean_k = k.replace("##", "").replace("Ġ", "").strip()
            clean_k = clean_k.replace(" ", "")
            if not clean_k:
                continue
            vocab[clean_k] = v
            
        return vocab
    except Exception as e:
        print(f"✗ Error loading {tokenizer_path}: {e}")
        return {}


def is_special_token(token: str) -> bool:
    """检查是否为特殊token"""
    special_tokens = ['[UNK]', '[PAD]', '[CLS]', '[SEP]', '[MASK]']
    return token in special_tokens or token.startswith('##') or token == 'Ġ'


def normalize_token(token: str) -> str:
    """规范化token"""
    return token.replace("##", "").replace("Ġ", "").strip()


def is_chinese_token(token: str) -> bool:
    """判断token是否全部为中文字符（纯汉字，不包含混合）
    
    正确判断：
    - '中' → True (纯汉字)
    - '中国' → True (纯汉字)
    - 'hello世界' → False (混合，不算纯中文)
    - '世界123' → False (混合，不算纯中文)
    - 'abc' → False (没有汉字)
    """
    if not token:  # 空token不算中文
        return False
    
    for char in token:
        # 每个字符都必须是汉字
        if not ('\u4e00' <= char <= '\u9fff'):  # CJK Unified Ideographs range
            return False
    
    return True


def compare_tokenizer_pair(vocab_a: dict, vocab_b: dict, vocab_c: dict, vocab_d: dict,
                          pair_name: str, converter: PinyinConverter) -> dict:
    """
    比较两对tokenizer，收集详细映射关系
    pair_name 格式: "A_vs_B", "A_vs_C", 等等
    """
    # 解析对名称
    parts = pair_name.split("_vs_")
    name1 = parts[0]  # A, B, C, or D
    name2 = parts[1]
    
    name_map = {"A": (vocab_a, "chinese"), "B": (vocab_b, "toneless"), 
                "C": (vocab_c, "toned"), "D": (vocab_d, "diacritic")}
    
    vocab1, type1 = name_map[name1]
    vocab2, type2 = name_map[name2]
    
    # 1->N 映射（token1 -> [matching tokens in vocab2]）
    mappings = {}  # token1 -> list of matches
    
    total_tokens_1 = len([t for t in vocab1.keys() if not is_special_token(t)])
    token_list_1 = [t for t in vocab1.keys() if not is_special_token(t)]
    
    # 用进度条扫描vocab1
    with tqdm(total=total_tokens_1, desc=f"Comparing {pair_name}", leave=True) as pbar:
        for token1 in token_list_1:
            pbar.update(1)
            
            token1_clean = normalize_token(token1)
            candidates_in_vocab2 = set()
            
            # 根据比较对类型选择转换策略
            if name1 == "A" and name2 == "B":
                # A(中文) → B(无声调拼音)
                # 关键改进: 先查词级别word_to_pinyin，再查字级别char_to_pinyin
                # 这样才能正确处理多音词（如"长大"、"说话"等）
                
                converted_parts = []
                
                # 1️⃣ 首先检查是否存在词级别的映射
                if token1_clean in converter.word_to_pinyin:
                    # 词级别的拼音列表
                    pinyin_list = converter.word_to_pinyin[token1_clean]
                    for py in pinyin_list:
                        # 去掉声调数字和符号
                        py_clean = converter.remove_tone_numbers(converter.remove_tone_marks(py))
                        converted_parts.append(py_clean)
                else:
                    # 2️⃣ 没有词级别映射，逐字查字级别（三层优先级）
                    for char in token1_clean:
                        py_found = None
                        
                        # 优先2.1: CEDICT
                        if char in converter.char_to_pinyin and converter.char_to_pinyin[char]:
                            py = converter.char_to_pinyin[char][0]
                            py_found = converter.remove_tone_numbers(converter.remove_tone_marks(py))
                        
                        # 优先2.2: 合并字典（Unihan + CEDICT）
                        elif char in converter.char_to_pinyin_merged:
                            py = converter.char_to_pinyin_merged[char]
                            if py:
                                py_found = converter._pinyin_to_toneless(str(py))
                        
                        # 优先2.3: pypinyin fallback  
                        elif HAS_PYPINYIN:
                            try:
                                py_candidates = lazy_pinyin(char, style=Style.NORMAL, errors='default')
                                if py_candidates and py_candidates[0]:
                                    py_found = py_candidates[0].lower()
                            except:
                                pass
                        
                        if py_found:
                            converted_parts.append(py_found)
                        else:
                            converted_parts.append(char)
                
                converted = "".join(converted_parts)
                candidates_in_vocab2.add(converted)
                
            elif name1 == "A" and name2 == "C":
                # A(中文) → C(带数字拼音) - 使用混合字典方案
                converted = converter.text_to_pinyin_toned(token1_clean)
                candidates_in_vocab2.add(converted)
                
            elif name1 == "A" and name2 == "D":
                # A(中文) → D(带声调符号拼音) - 使用混合字典方案
                converted = converter.text_to_pinyin_diacritic(token1_clean)
                candidates_in_vocab2.add(converted)
                
            elif name1 == "B" and name2 == "C":
                #pass
                # B(无声调) → C(带数字)
                # 直接保留token以备反向查找
                candidates_in_vocab2.add(token1_clean)
                
            elif name1 == "B" and name2 == "D":
                #pass
                # B(无声调) → D(带声调符号)
                # 直接保留token以备反向查找
                candidates_in_vocab2.add(token1_clean)
                
            elif name1 == "C" and name2 == "D":
                # C(带数字) → D(带声调符号)
                # 注意：CD的映射会在后面的专门逻辑中处理（使用align_d_to_c_structure）
                # 这里不添加候选，避免干扰约束
                pass
                
            else:
                candidates_in_vocab2.add(token1_clean)
            
            # 收集所有在vocab2中的匹配
            matched_tokens = []
            for candidate in candidates_in_vocab2:
                if candidate in vocab2:
                    matched_tokens.append(candidate)
            
            if matched_tokens:
                mappings[token1_clean] = matched_tokens
    
    # 对于BC和BD，添加反向查找逻辑（从vocab2去掉声调后查找vocab1）
    # 重要：允许多个C/D映射到同一个B（这是正常的语言现象）
    # 只是排除掉不符合格式的token
    if name1 == "B" and name2 in ["C", "D"]:
        token_list_2 = [t for t in vocab2.keys() if not is_special_token(t)]
        
        for token2 in token_list_2:
            token2_clean = normalize_token(token2)
            
            if name2 == "C":
                # C(带数字) → B(无声调)：去掉数字声调
                # 关键修复1：排除C中包含连续数字的token（不是有效拼音）
                import re
                if re.search(r'\d\d', token2_clean):
                    # 两个或以上数字连在一起，不是有效拼音，跳过
                    continue
                
                # 关键修复2：排除以数字开头的token（如'2K', '3rd', '1st'）
                if token2_clean and token2_clean[0].isdigit():
                    continue
                
                # 验证：C应该长于B（因为C多了数字）
                if len(token2_clean) <= 0:
                    continue
                converted = converter.remove_tone_numbers(token2_clean)
                # 验证：转换后长度应该 < 原长度（证明确实去掉了数字）
                if len(converted) >= len(token2_clean):
                    continue
                
                # 关键修复3：排除converted后以数字开头的情况（原B不应该以数字开头）
                if converted and converted[0].isdigit():
                    continue
                
                # 关键修复4：排除包含大写字母的token（拼音都是小写）
                if any(c.isupper() for c in converted):
                    # 拼音应该都是小写，有大写说明不是拼音
                    continue
                
                # 关键修复5：排除没有元音的token（所有有效拼音都必须包含元音aeioü）
                # 这可以过滤掉km2, cm3, SO4等编码/单位记号，同时保留gu2这样的有效拼音
                vowels = set('aeioüu')  # ü也是元音，u作为备选
                letter_part = re.sub(r'\d', '', converted).lower()
                if not any(v in letter_part for v in vowels):
                    # 没有任何元音，不是有效拼音
                    continue
                    
            elif name2 == "D":
                # D(带声调符号) → B(无声调)：去掉声调符号转为无声调
                # 关键修复：排除以数字开头的D token
                if token2_clean and token2_clean[0].isdigit():
                    continue
                
                # 验证：D去掉声调后应该 = B（都是字母，声调是替换而非添加）
                converted = converter._pinyin_to_toneless(token2_clean)
                # 验证：长度应该相等（说明确实只是替换了声调）
                if len(converted) != len(token2_clean):
                    # 长度不等，说明有字符被添加/删除，不应该映射
                    continue
                
                # 关键修复：排除包含大写字母的D token
                if any(c.isupper() for c in token2_clean):
                    # 拼音中不应该有大写字母
                    continue
                
                # 关键修复：排除没有元音的token（所有有效拼音都必须包含元音aeioü）
                # 这可以过滤掉km2, cm3, SO4等编码/单位记号，同时保留gu2这样的有效拼音
                vowels = set('aeioüu')  # ü也是元音，u作为备选
                if not any(v in converted.lower() for v in vowels):
                    # 没有任何元音，不是有效拼音
                    continue
                
                # 对D也应用同样的字母部分检查
                # 排除长度过短且字母部分不足的token
                if len(converted) < 2:
                    # 长度少于2个字符，不是有效拼音
                    continue
                
                # 排除包含大写字母的token（拼音都是小写）
                if any(c.isupper() for c in token2_clean):
                    # 拼音应该都是小写（D中的字母都应该是小写）
                    continue
            else:
                continue
            
            # 检查converted是否在vocab1中
            if converted in vocab1:
                # 允许多个C/D映射到同一个B
                if converted not in mappings:
                    mappings[converted] = [token2_clean]
                elif token2_clean not in mappings[converted]:
                    mappings[converted].append(token2_clean)

                # 修复N:1映射问题：当逆向映射成功时（确认为带有音调/符号的拼音），
                # 应该删除原先由于直接前向字符串相等而产生的映射。
                # 比如 C的 'yi1' 逆向映射到 B的 'yi'，则 B的 'yi1' 不应该映射到 C的 'yi1'
                if token2_clean in mappings and token2_clean in mappings[token2_clean]:
                    mappings[token2_clean].remove(token2_clean)
                    if not mappings[token2_clean]:
                        del mappings[token2_clean]
    
    # 对于CD，使用改进的精确音调匹配逻辑
    # 关键修复：D的不同声调符号对应不同的数字声调
    # 例如：ì对应4声，í对应2声，所以必须进行精确转换而非宽泛匹配
    # 优化：建立base形式的索引，避免O(n²)复杂度
    if name1 == "C" and name2 == "D":
        token_list_2 = [t for t in vocab2.keys() if not is_special_token(t)]
        
        # 建立C的base形式索引（去掉数字声调）
        c_base_map = {}  # base -> [c_token, ...]
        for c_token_raw in token_list_1:
            c_token = normalize_token(c_token_raw)
            c_base = converter.remove_tone_numbers(c_token)
            if c_base not in c_base_map:
                c_base_map[c_base] = []
            c_base_map[c_base].append(c_token)
        
        # 遍历D token
        for d_token_raw in token_list_2:
            d_token = normalize_token(d_token_raw)
            
            # 获取D的base形式（去掉声调符号）
            d_base = converter.remove_tone_marks(d_token)
            
            # 只查找匹配的base形式的C tokens
            if d_base in c_base_map:
                # 对匹配base的C token进行精确转换验证
                for c_token in c_base_map[d_base]:
                    d_converted_to_c = converter.align_d_to_c_structure(c_token, d_token)
                    
                    if d_converted_to_c == c_token:
                        # 精确匹配：D转换后完全等于C
                        if c_token not in mappings:
                            mappings[c_token] = []
                        if d_token not in mappings[c_token]:
                            mappings[c_token].append(d_token)
            
    
    # 统计映射关系的类型（按1对N和N对1分类）
    stats = {}  # 映射类型 -> [(token1, [matches]), ...]
    
    # 对于A_vs_X的情况，统计中文token的映射
    chinese_token_stats = {
        'total': 0,
        '1对1': 0,
        '1对N': 0,
        'N对1': 0,
        '独立': 0,
    }
    
    # 建立反向映射：vocab2中的token <- [vocab1中的tokens]
    reverse_mappings = {}  # token2 -> [tokens1, ...]
    for token1, matches in mappings.items():
        for token2 in matches:
            if token2 not in reverse_mappings:
                reverse_mappings[token2] = []
            reverse_mappings[token2].append(token1)
    
    # 统计正向映射（1对N）并追踪中文token
    for token1, matches in mappings.items():
        mapping_type = f"1对{len(matches)}"
        if mapping_type not in stats:
            stats[mapping_type] = []
        stats[mapping_type].append((token1, matches))
        
        # 如果是A_vs_X，统计中文token
        if name1 == "A" and is_chinese_token(token1):
            chinese_token_stats['total'] += 1
            if len(matches) == 1:
                # 这是1对1的候选，需要检查反向是否也是1对1
                token2 = matches[0]
                if len(reverse_mappings.get(token2, [])) == 1:
                    # 确实是1对1
                    chinese_token_stats['1对1'] += 1
                else:
                    # 是N对1
                    chinese_token_stats['N对1'] += 1
            else:
                # 是1对N
                chinese_token_stats['1对N'] += 1
    
    # 统计反向映射（N对1）
    reverse_stats = {}  # 映射类型 -> [(token2, [tokens1]), ...]
    for token2, sources in reverse_mappings.items():
        if len(sources) > 1:  # 只有多个来源才记录为"多对1"
            mapping_type = f"{len(sources)}对1"
            if mapping_type not in reverse_stats:
                reverse_stats[mapping_type] = []
            reverse_stats[mapping_type].append((token2, sources))
    
    # 统计vocab1中的独立词语
    # 独立 = token无法找到任何对应（完全不在mappings中）
    # 包括所有参与映射的token（1对1、1对2、2对1等都算作非独立）
    mapped_vocab1 = set(mappings.keys())  # 所有作为源的vocab1 token
    independent1 = total_tokens_1 - len(mapped_vocab1)
    
    # 统计vocab2中的独立词语（通过反向检查）
    # 独立 = token无法被任何token映射指向
    # 包括所有被任何映射覆盖的token（1对1、被1对2覆盖、被N对1映射都算作非独立）
    token_list_2 = [t for t in vocab2.keys() if not is_special_token(t)]
    covered_vocab2 = set()
    for matches in mappings.values():
        covered_vocab2.update(matches)
    independent2 = len(token_list_2) - len(covered_vocab2)
    
    # 验证：非独立token数 = 所有被统计在任何映射类别中的token数
    # independent1应该等于：总数 - (1对1中的vocab1 + 1对N中的vocab1 + N对1中的vocab1源)
    # 这三类应该覆盖所有在mappings.keys()中的token，因为mappings就是所有正向映射的集合
    
    # 对于A类，添加中文token的独立统计
    if name1 == "A":
        chinese_token_stats['独立'] = len([t for t in token_list_1 if is_chinese_token(t) and t not in mappings])
    
    results = {
        "pair": pair_name,
        "name1": name1,
        "name2": name2,
        "vocab1_size": total_tokens_1,
        "vocab2_size": len(token_list_2),
        "stats": stats,  # 正向映射统计（1对N）
        "reverse_stats": reverse_stats,  # 反向映射统计（N对1）
        "independent1": independent1,
        "independent2": independent2,
        "mappings": mappings,  # 保存完整的映射用于输出
        "reverse_mappings": reverse_mappings,  # 反向映射
        "chinese_token_stats": chinese_token_stats,  # 中文token统计
        "mapped_vocab1_count": len(mappings.keys()),  # 在mappings中的A token数（用于调试）
    }
    
    return results


# ===== 主流程 =====

def main():
    print("=" * 100)
    print("TOKENIZER OVERLAP ANALYSIS (64K VOCABULARY)")
    print("=" * 100)
    print("")
    
    # 初始化拼音转换器（三层方案：Unihan + CEDICT + pypinyin）
    cedict_path = os.path.join(DICTS_DIR, "cedict_ts.u8")
    merged_dict_path = os.path.join(DICTS_DIR, "merged_pinyin_dict.json")
    converter = PinyinConverter(cedict_path, merged_dict_path)
    print("")
    
    # 加载所有tokenizers
    print("Loading tokenizers...")
    print("-" * 80)
    
    vocab_A = load_tokenizer_vocab(os.path.join(TOKENIZERS_DIR, TOKENIZERS_64K["A_chinese_origin"]), is_pinyin=False)
    vocab_B = load_tokenizer_vocab(os.path.join(TOKENIZERS_DIR, TOKENIZERS_64K["B_pinyin_toneless"]), is_pinyin=True)
    vocab_C = load_tokenizer_vocab(os.path.join(TOKENIZERS_DIR, TOKENIZERS_64K["C_pinyin_toned"]), is_pinyin=True)
    vocab_D = load_tokenizer_vocab(os.path.join(TOKENIZERS_DIR, TOKENIZERS_64K["D_pinyin_diacritic"]), is_pinyin=True)
    
    print("")
    
    if not all([vocab_A, vocab_B, vocab_C, vocab_D]):
        print("✗ Unable to load all 4 tokenizers. Exiting.")
        return
    
    # 比较每一对tokenizer
    print("Comparing tokenizer pairs with progress...")
    print("")
    
    all_results = {}
    
    # 按照用户需要的顺序比较：AB, AC, AD, BC, BD, CD
    pair_order = ["A_vs_B", "A_vs_C", "A_vs_D", "B_vs_C", "B_vs_D", "C_vs_D"]
    
    for pair_name in pair_order:
        result = compare_tokenizer_pair(vocab_A, vocab_B, vocab_C, vocab_D, pair_name, converter)
        all_results[pair_name] = result
        print("")
    
    # 生成报告
    print("\n" + "=" * 100)
    print("Generating summary report...")
    print("=" * 100)
    
    report = generate_detailed_report(all_results, pair_order)
    
    # 保存报告
    output_path = os.path.join(TOKENIZERS_DIR, OUTPUT_FILE)
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n✓ Report saved to: {output_path}")
    except Exception as e:
        print(f"✗ Error saving report: {e}")
    
    print("\n" + report)
    
    print("\n" + "=" * 100)
    print("ANALYSIS COMPLETE!")
    print("=" * 100)


def generate_detailed_report(all_results: dict, pair_order: list) -> str:
    """生成详细的比较报告，包含映射示例"""
    report = []
    report.append("=" * 100)
    report.append("TOKENIZER OVERLAP ANALYSIS (64K VOCABULARY)")
    report.append("=" * 100)
    report.append("")
    report.append("Legend:")
    report.append("  A = chinese_origin (中文)")
    report.append("  B = pinyin_toneless (无声调拼音)")
    report.append("  C = pinyin_toned (带数字声调拼音)")
    report.append("  D = pinyin_diacritic (带声调符号拼音)")
    report.append("")
    
    # 总体统计表
    report.append("=" * 100)
    report.append("SUMMARY TABLE")
    report.append("=" * 100)
    report.append("")
    
    report.append("表列说明 (完全互斥分类):")
    report.append("  • 1对1: A词只映射B的1个token，且B的这个token也只被该A词映射（双射、完全独占）")
    report.append("  • 1对N: A词映射B的多个token（N>1）")
    report.append("  • N对1: A词只映射B的1个token，但B的这个token被多个A词映射（多对一、多音现象）")
    report.append("  • 独立1: A词无B词对应")
    report.append("  • 独立2: B词无A词映射")
    report.append("验证: 1对1 + 1对N + N对1 + 独立1 = 100%（来自vocab1，完全互斥）")
    report.append("")
    
    # 调试信息：显示mappings中的实际A token数
    report.append("DEBUG: mapped_vocab1_count for each pair (for validation):")
    for pair_name in pair_order:
        result = all_results[pair_name]
        report.append(f"  {result['pair']}: {result['mapped_vocab1_count']}")
    report.append("")
    
    report.append(f"{'Pair':<15} | {'1对1':<20} | {'1对N':<20} | {'N对1':<20} | {'独立1':<15} | {'独立2':<15}")
    report.append("-" * 125)
    
    for pair_name in pair_order:
        result = all_results[pair_name]
        
        # 识别真正的1对1（双射）、1对N、N对1
        one_to_one = 0      # A词只映射1个B词，且B词也只被该A词映射
        one_to_many = 0     # A词映射多个B词
        many_to_one = 0     # A词只映射1个B词，但B词被多个A词映射
        
        # 遍历所有mappings
        for token1, matches in result['stats'].get('1对1', []):
            # 这个A词映射1个B词
            if len(matches) == 1:
                b_token = matches[0]
                # 检查这个B词是否只被该A词映射（需要查reverse_mappings）
                if b_token in result['reverse_mappings'] and len(result['reverse_mappings'][b_token]) == 1:
                    # 确实是双射
                    one_to_one += 1
                else:
                    # B词被多个A词映射，这是N对1
                    many_to_one += 1
        
        # 1对N映射（A词映射多个B词）
        one_to_many = sum(len(v) for k, v in result['stats'].items() if k.startswith('1对') and k != '1对1')
        
        independent1 = result['independent1']
        independent2 = result['independent2']
        
        # 验证互斥性：这四类应该覆盖所有A词
        verify_sum = one_to_one + one_to_many + many_to_one + independent1
        
        # 计算各个百分比
        pct_1to1 = 100 * one_to_one / result['vocab1_size']
        pct_1tomany = 100 * one_to_many / result['vocab1_size']
        pct_nto1 = 100 * many_to_one / result['vocab1_size']
        pct_indep1 = 100 * independent1 / result['vocab1_size']
        pct_indep2 = 100 * independent2 / result['vocab2_size']
        
        one_to_one_str = f"{one_to_one}({pct_1to1:.1f}%)"
        one_to_many_str = f"{one_to_many}({pct_1tomany:.1f}%)"
        many_to_one_str = f"{many_to_one}({pct_nto1:.1f}%)"
        independent1_str = f"{independent1}({pct_indep1:.1f}%)"
        independent2_str = f"{independent2}({pct_indep2:.1f}%)"
        
        row = f"{result['pair']:<15} | {one_to_one_str:<20} | {one_to_many_str:<20} | {many_to_one_str:<20} | {independent1_str:<15} | {independent2_str:<15}"
        report.append(row)
    
    report.append("")
    
    # 添加中文token的统计信息（仅针对A_vs_B、A_vs_C、A_vs_D）
    report.append("=" * 100)
    report.append("CHINESE TOKEN MAPPING STATISTICS (For A_vs_B/C/D)")
    report.append("=" * 100)
    report.append("")
    report.append("📊 表格说明:")
    report.append("  • 【关键】所有百分比都相对于 A 中的总中文token数计算")
    report.append("  • A中总中文token = 参与映射的中文 + 独立的中文")
    report.append("  • 百分比加起来 = 100%")
    report.append("")
    
    report.append(f"{'Pair':<10} | {'A中文总数':<12} | {'中文1对1':<15} | {'中文1对N':<15} | {'中文N对1':<15} | {'中文独立':<15}")
    report.append("-" * 105)
    
    for pair_name in pair_order:
        result = all_results[pair_name]
        
        # 仅对A_vs_B/C/D统计中文token
        if result['name1'] == 'A':
            chinese_stats = result['chinese_token_stats']
            total_mapped = chinese_stats['total']  # 参与映射的中文token
            total_independent = chinese_stats['独立']  # 不参与映射的中文token
            total_all_chinese = total_mapped + total_independent  # 所有中文token
            
            if total_all_chinese > 0:
                # 所有百分比都相对于总中文token计算，这样加起来=100%
                pct_1to1 = 100 * chinese_stats['1对1'] / total_all_chinese
                pct_1toN = 100 * chinese_stats['1对N'] / total_all_chinese
                pct_Nto1 = 100 * chinese_stats['N对1'] / total_all_chinese
                pct_indep = 100 * total_independent / total_all_chinese
                
                col1 = f"{chinese_stats['1对1']}({pct_1to1:.1f}%)"
                col2 = f"{chinese_stats['1对N']}({pct_1toN:.1f}%)"
                col3 = f"{chinese_stats['N对1']}({pct_Nto1:.1f}%)"
                col4 = f"{total_independent}({pct_indep:.1f}%)"
                
                row = f"{result['pair']:<10} | {total_all_chinese:<12} | {col1:<15} | {col2:<15} | {col3:<15} | {col4:<15}"
            else:
                row = f"{result['pair']:<10} | {'0':<12} | N/A | N/A | N/A | N/A"
            
            report.append(row)
    
    report.append("")
    
    # 详细分析每一对
    report.append("\n" + "=" * 100)
    report.append("DETAILED ANALYSIS FOR EACH PAIR")
    report.append("=" * 100)
    
    for pair_name in pair_order:
        result = all_results[pair_name]
        
        report.append("")
        report.append(f"{'=' * 100}")
        report.append(f"PAIR: {result['name1']} ↔ {result['name2']}")
        report.append(f"{'=' * 100}")
        report.append("")
        
        independent1 = result['independent1']
        independent2 = result['independent2']
        
        report.append(f"Vocabulary size: {result['name1']} = {result['vocab1_size']}, {result['name2']} = {result['vocab2_size']}")
        report.append(f"")
        
        # 如果是A开头的对，输出中文token统计
        if result['pair'].startswith('A_vs_') and result['chinese_token_stats']:
            stats = result['chinese_token_stats']
            report.append(f"【中文Token统计】")
            total_chinese = stats['total']
            bijection_chinese = stats['1对1']
            one_to_many_chinese = stats['1对N']
            many_to_one_chinese = stats['N对1']
            
            report.append(f"  • 参与映射的中文token总数: {total_chinese:,}")
            report.append(f"    - 1对1 (双射): {bijection_chinese:,} ({100*bijection_chinese/total_chinese:.1f}%)" if total_chinese > 0 else "    - 1对1 (双射): 0")
            report.append(f"    - 1对N: {one_to_many_chinese:,} ({100*one_to_many_chinese/total_chinese:.1f}%)" if total_chinese > 0 else "    - 1对N: 0")
            report.append(f"    - N对1: {many_to_one_chinese:,} ({100*many_to_one_chinese/total_chinese:.1f}%)" if total_chinese > 0 else "    - N对1: 0")
            report.append(f"")
        
        report.append(f"Mapping Summary (Detailed breakdown by type):")
        
        # 计算新的互斥分类统计
        one_to_one = 0      # 1对1（双射）
        one_to_many = 0     # 1对多
        many_to_one = 0     # 多对1
        
        # 遍历所有mappings来精确分类
        for token1, matches in result['stats'].get('1对1', []):
            if len(matches) == 1:
                b_token = matches[0]
                if b_token in result['reverse_mappings'] and len(result['reverse_mappings'][b_token]) == 1:
                    # 双射
                    one_to_one += 1
                else:
                    # N对1
                    many_to_one += 1
        
        # 1对多（A词映射多个B词）
        one_to_many = sum(len(v) for k, v in result['stats'].items() if k.startswith('1对') and k != '1对1')
        
        independent1 = result['independent1']
        independent2 = result['independent2']
        
        # 显示新的互斥分类统计
        report.append(f"  ┌─ 1对1 (双射): {result['name1']} ↔ {result['name2']}")
        pct = 100 * one_to_one / result['vocab1_size']
        report.append(f"  │  • {one_to_one:<8} ({pct:.1f}%)")
        
        report.append(f"  ┌─ 1对N: {result['name1']} → {result['name2']}")
        pct = 100 * one_to_many / result['vocab1_size']
        report.append(f"  │  • {one_to_many:<8} ({pct:.1f}%)")
        
        report.append(f"  ┌─ N对1: 多个{result['name1']} → {result['name2']}")
        pct = 100 * many_to_one / result['vocab1_size']
        report.append(f"  │  • {many_to_one:<8} ({pct:.1f}%)")
        
        report.append(f"  └─ Independent: {result['name1']}: {independent1:<8} ({100*independent1/result['vocab1_size']:.1f}%)")
        report.append(f"  └─ Independent: {result['name2']}: {independent2:<8} ({100*independent2/result['vocab2_size']:.1f}%)")
        report.append("")
        
        # 显示正向映射（1对多）的详细类型分布
        sorted_stats = sorted(result['stats'].items(), key=lambda x: -len(x[1]))
        if len(sorted_stats) > 1:  # 如果不仅仅是1对1
            report.append("Forward Mapping Type Details:")
            for map_type, items in sorted_stats:
                percentage = 100 * len(items) / result['vocab1_size']
                report.append(f"  • {map_type}: {len(items):<8} ({percentage:.1f}%)")
            report.append("")
        
        # 显示反向映射（N对1）的详细类型分布
        sorted_reverse_stats = sorted(result['reverse_stats'].items(), key=lambda x: -len(x[1]))
        if sorted_reverse_stats:
            report.append("Reverse Mapping Type Details (N对1):")
            for map_type, items in sorted_reverse_stats[:10]:  # 显示前10种
                percentage = 100 * len(items) / result['vocab2_size']
                report.append(f"  • {map_type}: {len(items):<8} ({percentage:.1f}%)")
            if len(sorted_reverse_stats) > 10:
                report.append(f"  ... and {len(sorted_reverse_stats) - 10} more types")
            report.append("")
        
        # 显示示例
        report.append("-" * 100)
        report.append(f"Mapping Examples:")
        report.append("-" * 100)
        for map_type, items in sorted_stats[:3]:  # 显示前3种类型
            if items:
                report.append(f"\n{map_type} (showing 5 samples):")
                samples = random.sample(items, min(5, len(items)))
                for token1, matches in samples:
                    if len(matches) == 1:
                        report.append(f"  {result['name1']}: '{token1}' → {result['name2']}: '{matches[0]}'")
                    else:
                        report.append(f"  {result['name1']}: '{token1}' → {result['name2']}: {matches}")
        report.append("")
        
        # 显示反向映射（N对1）的示例
        if sorted_reverse_stats:
            report.append("-" * 100)
            report.append(f"Reverse Mapping Examples (N对1: 多个{result['name1']} → 同一个{result['name2']}):")
            report.append(f"(This shows homonymy/多音字 phenomena)")
            report.append("-" * 100)
            for map_type, items in sorted_reverse_stats[:3]:  # 显示前3种类型
                if items:
                    report.append(f"\n{map_type} (showing 5 samples):")
                    samples = random.sample(items, min(5, len(items)))
                    for token2, sources in samples:
                        report.append(f"  {result['name2']}: '{token2}' ← {result['name1']}: {sources}")
            report.append("")
    
    return "\n".join(report)


if __name__ == "__main__":
    main()

