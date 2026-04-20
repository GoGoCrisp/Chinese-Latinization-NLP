"""
测试CD对比是否可以直接用Style转换
C = 带数字拼音 (e.g., zhong1guo2)
D = 带声调符号拼音 (e.g., zhōngguó)

当前的9th代码使用了复杂的align_d_to_c_structure()来精确匹配D→C的转换
这个测试看看能不能简化
"""

import os
import re
from pypinyin import pinyin, Style
from tokenizers import Tokenizer as HFTokenizer

def load_tokenizer_vocab(tokenizer_path: str) -> dict:
    try:
        tokenizer = HFTokenizer.from_file(tokenizer_path)
        return tokenizer.get_vocab()
    except:
        return {}

def is_special_token(token: str) -> bool:
    return (token in ['[UNK]', '[PAD]', '[CLS]', '[SEP]', '[MASK]'] 
            or token.startswith('##') or token == 'Ġ')

def normalize_token(token: str) -> str:
    return token.replace("##", "").replace("Ġ", "").strip()

def remove_tone_marks(pinyin_text: str) -> str:
    """去掉声调符号，转为无声调"""
    tone_map = {
        'ā': 'a', 'á': 'a', 'ǎ': 'a', 'à': 'a',
        'ē': 'e', 'é': 'e', 'ě': 'e', 'è': 'e',
        'ī': 'i', 'í': 'i', 'ǐ': 'i', 'ì': 'i',
        'ō': 'o', 'ó': 'o', 'ǒ': 'o', 'ò': 'o',
        'ū': 'u', 'ú': 'u', 'ǔ': 'u', 'ù': 'u',
        'ǖ': 'v', 'ǘ': 'v', 'ǚ': 'v', 'ǜ': 'v',
    }
    return "".join([tone_map.get(c, c) for c in pinyin_text])

def direct_d_to_c(d_token: str) -> str:
    """
    简单方案：D→C的直接转换
    思路：D已经是无声调→带符号的转换，我们把符号转回数字声调
    """
    # 先去掉声调符号得到无声调
    d_base = remove_tone_marks(d_token)
    
    # 然后用pypinyin对d_base再转一遍得到带数字声调
    # 这样可以获得可能的声调组合
    # 但问题是我们不知道原始的拼音是什么...
    
    # 其实更好的办法是：对D中的声调符号进行转换而不是去掉
    # 这需要建立声调符号→数字的映射
    
    tone_to_number = {
        'ā': '1', 'á': '2', 'ǎ': '3', 'à': '4',
        'ē': '1', 'é': '2', 'ě': '3', 'è': '4',
        'ī': '1', 'í': '2', 'ǐ': '3', 'ì': '4',
        'ō': '1', 'ó': '2', 'ǒ': '3', 'ò': '4',
        'ū': '1', 'ú': '2', 'ǔ': '3', 'ù': '4',
        'ǖ': '1', 'ǘ': '2', 'ǚ': '3', 'ǜ': '4',
    }
    
    result = []
    syllable_buffer = []
    
    for char in d_token:
        syllable_buffer.append(char)
        
        # 检查是否遇到了声调符号
        if char in tone_to_number:
            # 这是一个带声调的元音，说明这个音节结束
            # 需要提取这个音节的声调并转换
            # 实际上这个逻辑很复杂...
            pass
    
    # 这个方案太复杂了，换个思路
    return None

def test_cd_conversion():
    """
    测试方案：通过比较tokenizer中的C和D来看是否能建立映射
    """
    print("=" * 80)
    print("测试CD转换的可行性")
    print("=" * 80)
    print()
    
    tokenizers_dir = "./Chinese_Latinization_NLP/1.Tokenization/tokenizers"
    
    vocab_c = load_tokenizer_vocab(os.path.join(tokenizers_dir, "pinyin_toned_64k_train90.json"))
    vocab_d = load_tokenizer_vocab(os.path.join(tokenizers_dir, "pinyin_diacritic_64k_train90.json"))
    
    print(f"C词汇数: {len(vocab_c)}")
    print(f"D词汇数: {len(vocab_d)}")
    print()
    
    # 测试策略：从真实的中文词汇出发，观察C和D如何映射
    test_cases = [
        "中国",
        "北京",
        "上海",
        "广州",
        "深圳",
        "音乐",
        "电影",
        "书籍",
        "学生",
        "老师",
        "朝代",
        "历史",
        "长大",
        "行走",
    ]
    
    print("【从中文出发，观察C和D的映射关系】\n")
    
    for text in test_cases:
        # 用pypinyin生成C和D格式
        py_c = pinyin(text, style=Style.TONE3, strict=False)
        py_d = pinyin(text, style=Style.TONE, strict=False)
        
        c_str = "".join([p[0] for p in py_c])
        d_str = "".join([p[0] for p in py_d])
        
        # 检查是否在tokenizer中
        c_in = c_str in vocab_c
        d_in = d_str in vocab_d
        
        status_c = "✓" if c_in else "✗"
        status_d = "✓" if d_in else "✗"
        
        print(f"{text}:")
        print(f"  C {status_c} {c_str}")
        print(f"  D {status_d} {d_str}")
        
        # 关键问题：如果两个都在tokenizer中，能不能建立可靠的映射？
        # 理论上，D中的每个声调符号都对应一个声调数字
        # 所以应该可以建立一个确定的映射
        
    print()
    print("=" * 80)
    print("【分析】")
    print("=" * 80)
    print()
    
    # 分析声调符号和数字的对应关系
    tone_symbol_map = {
        '1': {'ā', 'ē', 'ī', 'ō', 'ū', 'ǖ'},
        '2': {'á', 'é', 'í', 'ó', 'ú', 'ǘ'},
        '3': {'ǎ', 'ě', 'ǐ', 'ǒ', 'ǔ', 'ǚ'},
        '4': {'à', 'è', 'ì', 'ò', 'ù', 'ǜ'},
    }
    
    print("声调符号与数字的对应关系：")
    for tone_num, symbols in tone_symbol_map.items():
        print(f"  {tone_num}声: {', '.join(sorted(symbols))}")
    
    print()
    print("【关键发现】")
    print()
    print("可以建立精确的D→C映射的方式：")
    print("  1. D中的每个声调符号都对应一个唯一的数字声调")
    print("  2. 去掉声调得到的拼音是相同的")
    print("  3. 因此可以用保留声调→替换为数字的方式进行转换")
    print()
    print("但问题是：")
    print("  • 需要处理多音节拼音的音节边界判断")
    print("  • 当前的align_d_to_c_structure()就是在做这个判断")
    print("  • 简化方案可能会失去精确性")
    print()
    print("建议：")
    print("  • 保留CD的原有逻辑（align_d_to_c_structure）")
    print("  • 或者创建一个更轻量的版本只做声调符号→数字的替换")
    print("  • 不过由于AB/AC/AD已经简化了，整体代码量已经大幅降低")

if __name__ == "__main__":
    test_cd_conversion()
