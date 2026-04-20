"""
运行增强版分析并统计中文独立token
"""

import json
import os
import re
from tqdm import tqdm

try:
    from pypinyin import lazy_pinyin, Style
    HAS_PYPINYIN = True
except ImportError:
    HAS_PYPINYIN = False

# ===== 配置 =====
DICTS_DIR = "./dicts"
TOKENIZERS_DIR = "./tokenizers"
OUTPUT_FILE = "tokenizer_overlap_analysis_v3.txt"


# ===== PinyinConverter 类定义（复制自主代码） =====

class PinyinConverter:
    """拼音转换工具（三层方案）"""
    
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
        """改进版normalize：处理Unihan格式"""
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
    
    def text_to_pinyin_toneless(self, text: str) -> str:
        """三层优先级查询"""
        # 1️⃣ 词级别
        if text in self.word_to_pinyin:
            pinyin_list = self.word_to_pinyin[text]
            converted = "".join(pinyin_list)
            return self._pinyin_to_toneless(converted)
        
        # 2️⃣ 字级别
        result = []
        for char in text:
            py_found = None
            
            # 2.1 CEDICT
            if char in self.char_to_pinyin and self.char_to_pinyin[char]:
                py = self.char_to_pinyin[char][0]
                py_found = self._pinyin_to_toneless(py)
            
            # 2.2 合并字典
            elif char in self.char_to_pinyin_merged:
                py = self.char_to_pinyin_merged[char]
                if py:
                    py_found = self._pinyin_to_toneless(str(py))
            
            # 2.3 pypinyin fallback
            elif HAS_PYPINYIN:
                try:
                    py_candidates = lazy_pinyin(char, style=Style.NORMAL, errors='default')
                    if py_candidates and py_candidates[0]:
                        py_found = py_candidates[0].lower()
                except:
                    pass
            
            if py_found:
                result.append(py_found)
            else:
                result.append(char)
        
        return "".join(result)
    

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

def analyze():
    """执行分析"""
    
    print("=" * 100)
    print("ENHANCED ANALYSIS V3: Unihan + CEDICT + pypinyin")
    print("=" * 100)
    
    # 初始化转换器
    cedict_path = os.path.join(DICTS_DIR, "cedict_ts.u8")
    merged_dict_path = os.path.join(DICTS_DIR, "merged_pinyin_dict.json")
    
    converter = PinyinConverter(cedict_path, merged_dict_path)
    
    # 加载tokenizers
    vocab_a = load_vocab(os.path.join(TOKENIZERS_DIR, "chinese_origin_64k_train90.json"))
    vocab_b = load_vocab(os.path.join(TOKENIZERS_DIR, "pinyin_toneless_64k_train90.json"))
    
    # 提取纯中文token
    chinese_tokens = [t for t in vocab_a if is_pure_chinese(t)]
    
    print(f"\n【数据】")
    print(f"  A总token: {len(vocab_a):,}")
    print(f"  A纯中文token: {len(chinese_tokens):,}")
    print(f"  B总token: {len(vocab_b):,}")
    
    # 分析映射
    mapped = []
    independent = []  # 独立词语（无法映射到B）
    
    for token in tqdm(chinese_tokens, desc="Analyzing A→B"):
        converted = converter.text_to_pinyin_toneless(token)
        
        if converted and converted in vocab_b:
            mapped.append((token, converted))
        else:
            independent.append((token, converted))
    
    print(f"\n【A→B映射结果】")
    print(f"  ✅ 映射成功: {len(mapped):,} ({100*len(mapped)/len(chinese_tokens):.1f}%)")
    print(f"  ❌ 映射失败: {len(independent):,} ({100*len(independent)/len(chinese_tokens):.1f}%)")
    
    # 分析独立词语的特征
    print(f"\n【独立词语特性分析】")
    
    # 1. 单字vs多字
    single_char_indep = sum(1 for t, _ in independent if len(t) == 1)
    multi_char_indep = len(independent) - single_char_indep
    
    print(f"  • 单字独立: {single_char_indep:,} ({100*single_char_indep/len(independent):.1f}%)")
    print(f"  • 多字独立: {multi_char_indep:,} ({100*multi_char_indep/len(independent):.1f}%)")
    
    # 2. 拼音是否为空
    empty_pinyin_indep = sum(1 for t, p in independent if not p or p in [t, ''])
    has_pinyin_notin_b = len(independent) - empty_pinyin_indep
    
    print(f"  • 无拼音结果: {empty_pinyin_indep:,} ({100*empty_pinyin_indep/len(independent):.1f}%)")
    print(f"  • 拼音不在B中: {has_pinyin_notin_b:,} ({100*has_pinyin_notin_b/len(independent):.1f}%)")
    
    # 生成输出报告
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("ENHANCED TOKENIZER OVERLAP ANALYSIS - V3\n")
        f.write("Using: Unihan + CEDICT + pypinyin (Three-layer approach)\n")
        f.write("=" * 100 + "\n\n")
        
        f.write("SUMMARY\n")
        f.write("-" * 100 + "\n")
        f.write(f"A Chinese Pure Tokens: {len(chinese_tokens):,}\n")
        f.write(f"  ✅ Mapped to B: {len(mapped):,} ({100*len(mapped)/len(chinese_tokens):.1f}%)\n")
        f.write(f"  ❌ Independent (not in B): {len(independent):,} ({100*len(independent)/len(chinese_tokens):.1f}%)\n\n")
        
        f.write("INDEPENDENT TOKENS ANALYSIS\n")
        f.write("-" * 100 + "\n")
        f.write(f"Total Independent: {len(independent):,}\n")
        f.write(f"  • Single-char: {single_char_indep:,} ({100*single_char_indep/len(independent):.1f}%)\n")
        f.write(f"  • Multi-char: {multi_char_indep:,} ({100*multi_char_indep/len(independent):.1f}%)\n")
        f.write(f"  • No pinyin result: {empty_pinyin_indep:,} ({100*empty_pinyin_indep/len(independent):.1f}%)\n")
        f.write(f"  • Pinyin not in B: {has_pinyin_notin_b:,} ({100*has_pinyin_notin_b/len(independent):.1f}%)\n\n")
        
        f.write("TOP 1000 INDEPENDENT TOKENS\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'#':<5} {'Token':<20} {'Converted Pinyin':<30} {'Length':<8}\n")
        f.write("-" * 100 + "\n")
        
        for i, (token, pinyin) in enumerate(independent[:1000], 1):
            f.write(f"{i:<5} {token:<20} {pinyin:<30} {len(token):<8}\n")
        
        if len(independent) > 1000:
            f.write(f"\n... and {len(independent) - 1000} more independent tokens\n")
    
    print(f"\n✓ Report saved to {OUTPUT_FILE}")
    
    # 示例输出
    print(f"\n【独立词语示例】(前50个)")
    for i, (token, pinyin) in enumerate(independent[:50], 1):
        print(f"  {i}. {token:<20} → {pinyin:<30}")
    
    return len(independent), 100*len(independent)/len(chinese_tokens)

if __name__ == "__main__":
    indep_count, indep_pct = analyze()
    print(f"\n" + "=" * 100)
    print(f"【最终统计】")
    print(f"  独立词语数量: {indep_count:,}")
    print(f"  独立词语占比: {indep_pct:.1f}%")
    print("=" * 100)
