"""
综合分析AB、AC、AD的overlap关系
A = chinese_origin (中文)
B = pinyin_toneless (无声调拼音)
C = pinyin_toned (带数字声调)
D = pinyin_diacritic (带声调符号)
"""

import json
import os
import re
from tqdm import tqdm

try:
    from pypinyin import pinyin, lazy_pinyin, Style
    HAS_PYPINYIN = True
except ImportError:
    HAS_PYPINYIN = False


# ===== 配置 =====
DICTS_DIR = "./dicts"
TOKENIZERS_DIR = "./tokenizers"
OUTPUT_FILES = {
    "AB": "overlap_analysis_AB.txt",
    "AC": "overlap_analysis_AC.txt",
    "AD": "overlap_analysis_AD.txt",
}


# ===== PinyinConverter 类定义（完整三层方案） =====

class PinyinConverter:
    """
    拼音转换工具（完整三层方案）
    支持：无声调B、数字声调C、符号声调D
    """
    
    def __init__(self, cedict_path: str, merged_dict_path: str = None):
        self.word_to_pinyin = {}
        self.char_to_pinyin = {}
        self.char_to_pinyin_merged = {}
        self.load_cedict(cedict_path)
        if merged_dict_path:
            self.load_merged_dict(merged_dict_path)
    
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
    
    def _pinyin_to_toneless(self, pinyin: str) -> str:
        """处理声调标记，转换为无声调拼音"""
        p = pinyin.lower()
        
        # 处理声调符号
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
        
        p = p.replace('u:', 'v')
        p = p.replace('ü', 'v')
        p = re.sub(r'[0-5]', '', p)
        
        return p
    
    def _tone_numbers_to_marks(self, pinyin: str) -> str:
        """将数字声调转换为声调符号"""
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
        """转换为无声调拼音（B格式）
        使用pypinyin直接转换（Style.NORMAL）
        """
        py_result = pinyin(text, style=Style.NORMAL, strict=False)
        return "".join([p[0] for p in py_result])
    
    def text_to_pinyin_toned(self, text: str) -> str:
        """转换为带数字声调拼音（C格式）
        使用pypinyin直接转换（Style.TONE3）
        """
        py_result = pinyin(text, style=Style.TONE3, strict=False)
        return "".join([p[0] for p in py_result])
    
    def text_to_pinyin_diacritic(self, text: str) -> str:
        """转换为带声调符号拼音（D格式）
        使用pypinyin直接转换（Style.TONE）
        """
        py_result = pinyin(text, style=Style.TONE, strict=False)
        return "".join([p[0] for p in py_result])
    
    def _tone_numbers_to_marks(self, pinyin: str) -> str:
        """将数字声调拼音转换为符号拼音
        例如：zhong1guo2 → zhōngguó
        """
        # 声调符号映射表
        tone_marks = {
            'a': ['', 'ā', 'á', 'ǎ', 'à'],
            'e': ['', 'ē', 'é', 'ě', 'è'],
            'i': ['', 'ī', 'í', 'ǐ', 'ì'],
            'o': ['', 'ō', 'ó', 'ǒ', 'ò'],
            'u': ['', 'ū', 'ú', 'ǔ', 'ù'],
            'ü': ['', 'ǖ', 'ǘ', 'ǚ', 'ǜ'],
            'v': ['', 'ǖ', 'ǘ', 'ǚ', 'ǜ'],
        }
        
        p = pinyin.lower()
        result = list(p)
        
        # 找所有 "元音+数字" 的位置
        i = 0
        while i < len(p):
            if p[i] in 'aeiouü' and i + 1 < len(p) and p[i + 1] in '12345':
                tone_num = int(p[i + 1])
                
                if tone_num not in [0, 5]:  # 有效的声调
                    # 检查优先级：a/e > o > i/u/ü
                    target_idx = i
                    
                    # 如果当前是i/u/ü，检查前面是否有o
                    if p[i] in 'iuü':
                        for j in range(i - 1, -1, -1):
                            if p[j] == 'o':
                                target_idx = j
                                break
                            elif p[j] not in 'aeiouü':
                                break
                    
                    # 如果当前是o或i/u/ü，检查前面是否有a/e
                    if p[i] in 'oiuü':
                        for j in range(i - 1, -1, -1):
                            if p[j] in 'ae':
                                target_idx = j
                                break
                            elif p[j] not in 'aeiouü':
                                break
                    
                    # 添加声调标记
                    if target_idx < len(result):
                        target_char = result[target_idx]
                        if target_char in tone_marks:
                            result[target_idx] = tone_marks[target_char][tone_num]
                
                # 移除数字
                result[i + 1] = ''
                i += 2
            else:
                i += 1
        
        final = ''.join(char for char in result if char != '')
        return final.replace('v', 'ü')
    

def is_pure_chinese(token):
    """纯中文token判断"""
    if not token:
        return False
    return all('\u4e00' <= c <= '\u9fff' for c in token)

def load_vocab(path):
    """加载tokenizer词汇表"""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return set(data['model']['vocab'].keys())

def analyze_pair(pair_name, vocab_a, vocab_x, converter, conversion_func):
    """分析A→X的映射关系"""
    
    # 提取纯中文token
    chinese_tokens = [t for t in vocab_a if is_pure_chinese(t)]
    
    # 分析映射
    mapped = []
    independent = []
    
    for token in tqdm(chinese_tokens, desc=f"Analyzing A→{pair_name}"):
        converted = conversion_func(converter, token)
        
        if converted and converted in vocab_x:
            mapped.append((token, converted))
        else:
            independent.append((token, converted))
    
    return chinese_tokens, mapped, independent

def generate_report(pair_name, vocab_sizes, chinese_tokens, mapped, independent, output_file):
    """生成分析报告"""
    
    # 统计信息
    mapped_count = len(mapped)
    independent_count = len(independent)
    total = len(chinese_tokens)
    
    if total == 0:
        print(f"⚠ No Chinese tokens found for {pair_name}")
        return
    
    mapped_pct = 100 * mapped_count / total
    independent_pct = 100 * independent_count / total
    
    # 独立词语特征分析
    single_char_indep = sum(1 for t, _ in independent if len(t) == 1)
    multi_char_indep = independent_count - single_char_indep
    
    empty_pinyin_indep = sum(1 for t, p in independent if not p or p in [t, ''])
    has_pinyin_not_in_x = independent_count - empty_pinyin_indep
    
    # 生成报告
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write(f"TOKENIZER OVERLAP ANALYSIS - A→{pair_name}\n")
        f.write("Using: Unihan + CEDICT + pypinyin (Three-layer approach)\n")
        f.write("=" * 100 + "\n\n")
        
        f.write("SUMMARY\n")
        f.write("-" * 100 + "\n")
        f.write(f"A Chinese Pure Tokens: {total:,}\n")
        f.write(f"  ✅ Mapped to {pair_name}: {mapped_count:,} ({mapped_pct:.1f}%)\n")
        f.write(f"  ❌ Independent (not in {pair_name}): {independent_count:,} ({independent_pct:.1f}%)\n\n")
        
        f.write(f"Vocab Sizes:\n")
        f.write(f"  • A (chinese_origin): {vocab_sizes['A']:,}\n")
        f.write(f"  • {pair_name} ({pair_name}_pinyin): {vocab_sizes[pair_name]:,}\n\n")
        
        f.write("INDEPENDENT TOKENS ANALYSIS\n")
        f.write("-" * 100 + "\n")
        f.write(f"Total Independent: {independent_count:,}\n")
        f.write(f"  • Single-char: {single_char_indep:,} ({100*single_char_indep/independent_count:.1f}%)\n")
        f.write(f"  • Multi-char: {multi_char_indep:,} ({100*multi_char_indep/independent_count:.1f}%)\n")
        f.write(f"  • No pinyin result: {empty_pinyin_indep:,} ({100*empty_pinyin_indep/independent_count:.1f}%)\n")
        f.write(f"  • Pinyin not in {pair_name}: {has_pinyin_not_in_x:,} ({100*has_pinyin_not_in_x/independent_count:.1f}%)\n\n")
        
        f.write(f"TOP 1000 INDEPENDENT TOKENS\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'#':<5} {'Token':<20} {'Converted {pair_name}':<30} {'Length':<8}\n".format(pair_name=pair_name))
        f.write("-" * 100 + "\n")
        
        for i, (token, pinyin) in enumerate(independent[:1000], 1):
            f.write(f"{i:<5} {token:<20} {pinyin:<30} {len(token):<8}\n")
        
        if len(independent) > 1000:
            f.write(f"\n... and {len(independent) - 1000} more independent tokens\n")
    
    print(f"✓ Report saved to {output_file}")
    print(f"  {pair_name}: {mapped_count:,} mapped ({mapped_pct:.1f}%), {independent_count:,} independent ({independent_pct:.1f}%)")
    
    return {
        'pair': pair_name,
        'mapped': mapped_count,
        'independent': independent_count,
        'mapped_pct': mapped_pct,
        'independent_pct': independent_pct,
    }

def analyze():
    """执行综合分析"""
    
    print("=" * 100)
    print("COMPREHENSIVE OVERLAP ANALYSIS: AB, AC, AD")
    print("=" * 100)
    
    # 初始化转换器
    cedict_path = os.path.join(DICTS_DIR, "cedict_ts.u8")
    merged_dict_path = os.path.join(DICTS_DIR, "merged_pinyin_dict.json")
    
    converter = PinyinConverter(cedict_path, merged_dict_path)
    
    # 加载tokenizers
    print("\nLoading vocabularies...")
    vocab_a = load_vocab(os.path.join(TOKENIZERS_DIR, "chinese_origin_64k_train90.json"))
    vocab_b = load_vocab(os.path.join(TOKENIZERS_DIR, "pinyin_toneless_64k_train90.json"))
    vocab_c = load_vocab(os.path.join(TOKENIZERS_DIR, "pinyin_toned_64k_train90.json"))
    vocab_d = load_vocab(os.path.join(TOKENIZERS_DIR, "pinyin_diacritic_64k_train90.json"))
    
    print(f"✓ Loaded vocabularies:")
    print(f"  • A (chinese_origin): {len(vocab_a):,}")
    print(f"  • B (pinyin_toneless): {len(vocab_b):,}")
    print(f"  • C (pinyin_toned): {len(vocab_c):,}")
    print(f"  • D (pinyin_diacritic): {len(vocab_d):,}")
    
    # 分析AB、AC、AD
    results = []
    
    print("\n" + "=" * 100)
    print("ANALYZING AB (A→B: Chinese to Toneless Pinyin)")
    print("=" * 100)
    chinese_tokens_ab, mapped_ab, independent_ab = analyze_pair(
        "B", vocab_a, vocab_b, converter, 
        lambda conv, token: conv.text_to_pinyin_toneless(token)
    )
    result_ab = generate_report(
        "B", 
        {'A': len(vocab_a), 'B': len(vocab_b)},
        chinese_tokens_ab, mapped_ab, independent_ab,
        OUTPUT_FILES["AB"]
    )
    results.append(result_ab)
    
    print("\n" + "=" * 100)
    print("ANALYZING AC (A→C: Chinese to Toned Pinyin with Numbers)")
    print("=" * 100)
    chinese_tokens_ac, mapped_ac, independent_ac = analyze_pair(
        "C", vocab_a, vocab_c, converter,
        lambda conv, token: conv.text_to_pinyin_toned(token)
    )
    result_ac = generate_report(
        "C",
        {'A': len(vocab_a), 'C': len(vocab_c)},
        chinese_tokens_ac, mapped_ac, independent_ac,
        OUTPUT_FILES["AC"]
    )
    results.append(result_ac)
    
    print("\n" + "=" * 100)
    print("ANALYZING AD (A→D: Chinese to Toned Pinyin with Marks)")
    print("=" * 100)
    chinese_tokens_ad, mapped_ad, independent_ad = analyze_pair(
        "D", vocab_a, vocab_d, converter,
        lambda conv, token: conv.text_to_pinyin_diacritic(token)
    )
    result_ad = generate_report(
        "D",
        {'A': len(vocab_a), 'D': len(vocab_d)},
        chinese_tokens_ad, mapped_ad, independent_ad,
        OUTPUT_FILES["AD"]
    )
    results.append(result_ad)
    
    # 最终摘要
    print("\n" + "=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)
    for result in results:
        print(f"\n{result['pair']}:")
        print(f"  Mapped:      {result['mapped']:,} ({result['mapped_pct']:.1f}%)")
        print(f"  Independent: {result['independent']:,} ({result['independent_pct']:.1f}%)")
    
    print("\n" + "=" * 100)

if __name__ == "__main__":
    analyze()
