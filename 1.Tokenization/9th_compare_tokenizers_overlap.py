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

# ===== 配置 =====
TOKENIZERS_DIR = "./tokenizers"
DICTS_DIR = "./dicts"
OUTPUT_FILE = "tokenizer_overlap_analysis.txt"

# 4个64k tokenizers映射
TOKENIZERS_64K = {
    "A_chinese_origin": "chinese_origin_64k_train90.json",
    "B_pinyin_toneless": "pinyin_toneless_64k_train90.json",
    "C_pinyin_toned": "pinyin_toned_64k_train90.json",
    "D_pinyin_diacritic": "pinyin_diacritic_64k_train90.json",
}

# 生成所有的tokenizer对
TOKENIZER_PAIRS = list(combinations(sorted(TOKENIZERS_64K.keys()), 2))


# ===== 拼音转换工具 =====

class PinyinConverter:
    """拼音转换和映射工具"""
    
    def __init__(self, cedict_path: str):
        """初始化，加载CC-CEDICT字典建立汉字→拼音映射"""
        self.char_to_pinyin = {}  # 汉字 → 拼音列表
        self.load_cedict(cedict_path)
    
    def load_cedict(self, cedict_path: str):
        """从CC-CEDICT加载汉字→拼音映射"""
        if not os.path.exists(cedict_path):
            print(f"⊘ CEDICT not found: {cedict_path}")
            return
        
        try:
            with open(cedict_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("#"):
                        continue
                    
                    parts = line.split()
                    if len(parts) < 3:
                        continue
                    
                    # 格式: 繁体 简体 [拼音] /定义/
                    simplified = parts[1]
                    pinyin_str = parts[2].strip("[]")
                    
                    # 保存拼音列表
                    pinyin_list = pinyin_str.split()
                    if simplified and pinyin_list:
                        self.char_to_pinyin[simplified] = pinyin_list
            
            print(f"✓ Loaded {len(self.char_to_pinyin)} characters from CEDICT")
        except Exception as e:
            print(f"✗ Error loading CEDICT: {e}")
    
    def text_to_pinyin_toneless(self, text: str) -> str:
        """将中文文本转换为无声调拼音"""
        result = []
        for char in text:
            if char in self.char_to_pinyin:
                # 获取第一个拼音（主要拼音）
                pinyin = self.char_to_pinyin[char][0]
                # 移除数字和声调标记
                pinyin_clean = re.sub(r'[0-9āáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜ]', '', pinyin)
                result.append(pinyin_clean)
            else:
                result.append(char)
        return "".join(result)
    
    def text_to_pinyin_toned(self, text: str) -> str:
        """将中文文本转换为带数字声调的拼音"""
        result = []
        for char in text:
            if char in self.char_to_pinyin:
                # 获取第一个拼音
                pinyin = self.char_to_pinyin[char][0]
                result.append(pinyin)
            else:
                result.append(char)
        return "".join(result)
    
    def text_to_pinyin_diacritic(self, text: str) -> str:
        """将中文文本转换为带声调符号的拼音"""
        # 从CEDICT直接获取的拼音中已经带有声调标记
        result = []
        for char in text:
            if char in self.char_to_pinyin:
                pinyin = self.char_to_pinyin[char][0]
                result.append(pinyin)
            else:
                result.append(char)
        return "".join(result)
    
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


def load_tokenizer_vocab(tokenizer_path: str) -> dict:
    """加载tokenizer的vocabulary"""
    try:
        tokenizer = HFTokenizer.from_file(tokenizer_path)
        vocab = tokenizer.get_vocab()
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
                converted_parts = []
                for char in token1_clean:
                    if char in converter.char_to_pinyin:
                        py = converter.char_to_pinyin[char][0]
                        py_clean = converter.remove_tone_numbers(converter.remove_tone_marks(py))
                        converted_parts.append(py_clean)
                    else:
                        converted_parts.append(char)
                converted = "".join(converted_parts)
                candidates_in_vocab2.add(converted)
                
            elif name1 == "A" and name2 == "C":
                # A(中文) → C(带数字拼音)
                converted_parts = []
                for char in token1_clean:
                    if char in converter.char_to_pinyin:
                        py = converter.char_to_pinyin[char][0]
                        # 保留数字，移除声调标记
                        py_clean = converter.remove_tone_marks(py)
                        converted_parts.append(py_clean)
                    else:
                        converted_parts.append(char)
                converted = "".join(converted_parts)
                candidates_in_vocab2.add(converted)
                
            elif name1 == "A" and name2 == "D":
                # A(中文) → D(带声调符号拼音)
                converted_parts = []
                for char in token1_clean:
                    if char in converter.char_to_pinyin:
                        py = converter.char_to_pinyin[char][0]
                        converted_parts.append(py)
                    else:
                        converted_parts.append(char)
                converted = "".join(converted_parts)
                candidates_in_vocab2.add(converted)
                
            elif name1 == "B" and name2 == "C":
                # B(无声调) → C(带数字)
                # 尝试添加1-4声调
                base_b = token1_clean
                for i in range(1, 5):
                    candidates_in_vocab2.add(base_b + str(i))
                
            elif name1 == "B" and name2 == "D":
                # B(无声调) → D(带声调符号)
                candidates_in_vocab2.add(token1_clean)
                
            elif name1 == "C" and name2 == "D":
                # C(带数字) → D(带声调符号)
                base_c = converter.remove_tone_numbers(token1_clean)
                candidates_in_vocab2.add(base_c)
                
            else:
                candidates_in_vocab2.add(token1_clean)
            
            # 收集所有在vocab2中的匹配
            matched_tokens = []
            for candidate in candidates_in_vocab2:
                if candidate in vocab2:
                    matched_tokens.append(candidate)
            
            if matched_tokens:
                mappings[token1_clean] = matched_tokens
    
    # 统计映射关系的类型（按1对N和N对1分类）
    stats = {}  # 映射类型 -> [(token1, [matches]), ...]
    
    # 建立反向映射：vocab2中的token <- [vocab1中的tokens]
    reverse_mappings = {}  # token2 -> [tokens1, ...]
    for token1, matches in mappings.items():
        for token2 in matches:
            if token2 not in reverse_mappings:
                reverse_mappings[token2] = []
            reverse_mappings[token2].append(token1)
    
    # 统计正向映射（1对N）
    for token1, matches in mappings.items():
        mapping_type = f"1对{len(matches)}"
        if mapping_type not in stats:
            stats[mapping_type] = []
        stats[mapping_type].append((token1, matches))
    
    # 统计反向映射（N对1）
    reverse_stats = {}  # 映射类型 -> [(token2, [tokens1]), ...]
    for token2, sources in reverse_mappings.items():
        if len(sources) > 1:  # 只有多个来源才记录为"多对1"
            mapping_type = f"{len(sources)}对1"
            if mapping_type not in reverse_stats:
                reverse_stats[mapping_type] = []
            reverse_stats[mapping_type].append((token2, sources))
    
    # 统计vocab1中的独立词语
    independent1 = total_tokens_1 - len(mappings)
    
    # 统计vocab2中的独立词语（通过反向检查）
    token_list_2 = [t for t in vocab2.keys() if not is_special_token(t)]
    covered_vocab2 = set()
    for matches in mappings.values():
        covered_vocab2.update(matches)
    independent2 = len(token_list_2) - len(covered_vocab2)
    
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
    }
    
    return results


# ===== 主流程 =====

def main():
    print("=" * 100)
    print("TOKENIZER OVERLAP ANALYSIS (64K VOCABULARY)")
    print("=" * 100)
    print("")
    
    # 初始化拼音转换器
    cedict_path = os.path.join(DICTS_DIR, "cedict_ts.u8")
    converter = PinyinConverter(cedict_path)
    print("")
    
    # 加载所有tokenizers
    print("Loading tokenizers...")
    print("-" * 80)
    
    vocab_A = load_tokenizer_vocab(os.path.join(TOKENIZERS_DIR, TOKENIZERS_64K["A_chinese_origin"]))
    vocab_B = load_tokenizer_vocab(os.path.join(TOKENIZERS_DIR, TOKENIZERS_64K["B_pinyin_toneless"]))
    vocab_C = load_tokenizer_vocab(os.path.join(TOKENIZERS_DIR, TOKENIZERS_64K["C_pinyin_toned"]))
    vocab_D = load_tokenizer_vocab(os.path.join(TOKENIZERS_DIR, TOKENIZERS_64K["D_pinyin_diacritic"]))
    
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
    
    report.append(f"{'Pair':<15} | {'1对1':<8} | {'1对多':<10} | {'多对1':<10} | {'独立(1)':<8} | {'总覆盖':<8}")
    report.append("-" * 100)
    
    for pair_name in pair_order:
        result = all_results[pair_name]
        one_to_one = len(result['stats'].get('1对1', []))
        one_to_many = sum(len(v) for k, v in result['stats'].items() if k.startswith('1对') and k != '1对1')
        many_to_one = sum(len(v) for k, v in result['reverse_stats'].items())
        independent1 = result['independent1']
        total_coverage = one_to_one + one_to_many + many_to_one
        
        row = f"{result['pair']:<15} | {one_to_one:<8} | {one_to_many:<10} | {many_to_one:<10} | {independent1:<8} | {total_coverage:<8}"
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
        report.append(f"Mapping Summary (Detailed breakdown by type):")
        
        # 按映射类型排序（正向）
        sorted_stats = sorted(result['stats'].items(), key=lambda x: -len(x[1]))
        sorted_reverse_stats = sorted(result['reverse_stats'].items(), key=lambda x: -len(x[1]))
        total_mapped = sum(len(v) for v in result['stats'].values())
        
        # 显示正向映射统计
        report.append(f"  ┌─ Forward (1对N): {result['name1']} → {result['name2']}")
        for map_type, items in sorted_stats:
            percentage = 100 * len(items) / result['vocab1_size']
            report.append(f"  │  • {map_type}: {len(items):<8} ({percentage:.1f}%)")
        
        # 显示反向映射统计
        if sorted_reverse_stats:
            report.append(f"  ┌─ Reverse (N对1): {result['name2']} ← {result['name1']} (多对一映射)")
            for map_type, items in sorted_reverse_stats:
                percentage = 100 * len(items) / result['vocab2_size']
                report.append(f"  │  • {map_type}: {len(items):<8} ({percentage:.1f}%)")
        
        report.append(f"  └─ Independent: {result['name1']}: {independent1:<8} ({100*independent1/result['vocab1_size']:.1f}%)")
        report.append(f"  └─ Independent: {result['name2']}: {independent2:<8} ({100*independent2/result['vocab2_size']:.1f}%)")
        report.append("")
        
        # 显示正向映射（1对N）的示例
        report.append("-" * 100)
        report.append(f"Forward Mapping Examples (1对N: {result['name1']} → {result['name2']}):")
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

