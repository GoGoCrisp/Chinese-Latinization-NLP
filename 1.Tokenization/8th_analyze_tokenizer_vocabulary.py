"""
分析Tokenizer Vocabulary构成
统计各类型token的数量和分布
"""

import json
import os
import re
import random
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

# ===== 配置 =====
TOKENIZERS_DIR = "./tokenizers"

# 配置：分别分析 Chinese Origin 和 Pinyin Toneless
CONFIGS = {
    "chinese_origin": {
        "tokenizers": [
            "chinese_origin_8k_train90.json",
            "chinese_origin_16k_train90.json",
            "chinese_origin_32k_train90.json",
            "chinese_origin_64k_train90.json",
        ],
        "output_file": "tokenizer_vocabulary_analysis_chinese_origin.txt",
        "title": "CHINESE ORIGIN TOKENIZERS - VOCABULARY COMPOSITION ANALYSIS",
    },
    "pinyin_toneless": {
        "tokenizers": [
            "pinyin_toneless_8k_train90.json",
            "pinyin_toneless_16k_train90.json",
            "pinyin_toneless_32k_train90.json",
            "pinyin_toneless_64k_train90.json",
        ],
        "output_file": "tokenizer_vocabulary_analysis_pinyin_toneless.txt",
        "title": "PINYIN TONELESS TOKENIZERS - VOCABULARY COMPOSITION ANALYSIS",
    },
}


# ===== 分类函数 =====

def is_latin(token: str) -> bool:
    """检查是否为拉丁字母"""
    return bool(re.match(r'^[a-zA-Z]+$', token))


def is_latin_digit(token: str) -> bool:
    """检查是否为拉丁字母+数字"""
    return bool(re.match(r'^[a-zA-Z0-9]+$', token))


def is_punctuation(token: str) -> bool:
    """检查是否为标点符号"""
    return bool(re.match(r'^[^\w\s\u4e00-\u9fff]+$', token, re.UNICODE))


def is_japanese(token: str) -> bool:
    """检查是否为日语（平假名、片假名）"""
    # 平假名: \u3041-\u3096
    # 片假名: \u30A1-\u30FC
    return bool(re.search(r'[\u3041-\u3096\u30A1-\u30FC]', token))


def is_korean(token: str) -> bool:
    """检查是否为韩语（韩文字母）"""
    # 韩文字母范围: \uAC00-\uD7AF
    return bool(re.search(r'[\uAC00-\uD7AF]', token))


def is_rare_chinese_or_utf8_byte(token: str) -> bool:
    """
    检查是否为生僻中文字或UTF-8字节碎片
    包括CJK扩展和特殊的字节编码
    """
    # 1. 检查是否为特殊的字节表示 <0xXX>
    if re.match(r'^<0x[0-9A-Fa-f]{2}>$', token):
        return True
    
    # 2. 检查是否为中文扩展区域的字符
    # 中日韩统一表意文字扩展A: U+3400-U+4DBF
    if re.search(r'[\u3400-\u4DBF]', token):
        return True
    
    # 3. 检查单个字符且为高Unicode码点（可能是组合字符或特殊字符）
    if len(token) == 1 and ord(token) > 0xE000:
        return True
    
    return False


def count_chinese_chars(token: str) -> int:
    """
    统计token中中文字符的数量（标准CJK范围）
    排除无效的边界字符（如U+9FFF等）
    """
    count = 0
    for char in token:
        # 检查是否在标准CJK范围
        if '\u4e00' <= char <= '\u9fff':
            # 排除某些无效的边界字符
            # U+9FFF, U+FFFE, U+FFFF 等
            if ord(char) not in [0x9fff, 0xfffe, 0xffff]:
                # 检查是否是真实的字符（不是控制字符）
                try:
                    cat = unicodedata.category(char)
                    # 排除控制字符 (Cc) 和格式字符 (Cf)
                    if cat not in ['Cc', 'Cf', 'Co', 'Cs', 'Cn']:
                        count += 1
                except:
                    pass
    return count


def get_token_type(token: str) -> str:
    """
    对token进行分类
    返回分类标签
    """
    # 移除特殊前缀
    clean_token = token.replace("##", "").replace("Ġ", "").replace(" ", "")
    
    if not clean_token:
        return "EMPTY"
    
    # 统计中文字符
    chinese_count = count_chinese_chars(clean_token)
    
    # 1. 纯中文token（标准CJK范围）
    if chinese_count > 0 and len(clean_token) == chinese_count:
        return f"CHINESE_{chinese_count}CHAR"
    
    # 2. 包含中文但不纯中文的token
    if chinese_count > 0:
        return "MIXED_WITH_CHINESE"
    
    # 3. 拉丁字母
    if is_latin(clean_token):
        return "LATIN_LETTER"
    
    # 4. 拉丁字母+数字
    if is_latin_digit(clean_token):
        return "LATIN_DIGIT"
    
    # 5. 数字
    if clean_token.isdigit():
        return "DIGIT"
    
    # 6. 标点符号
    if is_punctuation(clean_token):
        return "PUNCTUATION"
    
    # ===== OTHER分类 =====
    # 7. 日语
    if is_japanese(clean_token):
        return "OTHER_JAPANESE"
    
    # 8. 韩语
    if is_korean(clean_token):
        return "OTHER_KOREAN"
    
    # 9. 生僻中文或UTF-8字节碎片
    if is_rare_chinese_or_utf8_byte(clean_token):
        return "OTHER_RARE_CHINESE_UTF8"
    
    # 10. 其他
    return "OTHER_UNKNOWN"


# ===== 主分析函数 =====

def analyze_tokenizer(vocab_path: str) -> dict:
    """
    分析单个tokenizer的vocabulary
    """
    print(f"\nAnalyzing: {vocab_path}")
    print("-" * 80)
    
    # 加载vocabulary
    try:
        with open(vocab_path, "r", encoding="utf-8") as f:
            # huggingface tokenizer JSON格式
            data = json.load(f)
            
            # 获取vocabulary (model.vocab 或其他格式)
            if "model" in data and "vocab" in data["model"]:
                vocab = data["model"]["vocab"]
            elif "vocab" in data:
                vocab = data["vocab"]
            else:
                # 尝试直接使用token_to_id
                vocab = {}
                print("✗ Could not find vocabulary in file")
                return {}
    except Exception as e:
        print(f"✗ Error loading tokenizer: {e}")
        return {}
    
    print(f"✓ Total tokens: {len(vocab)}")
    
    # 统计各类型token
    type_counter = Counter()
    detailed_stats = defaultdict(list)
    
    for token_str in vocab.keys():
        token_type = get_token_type(token_str)
        type_counter[token_type] += 1
        detailed_stats[token_type].append(token_str)
    
    # 整理结果
    results = {
        "total_vocab_size": len(vocab),
        "type_distribution": dict(type_counter),
        "detailed_stats": dict(detailed_stats),
    }
    
    return results


def sort_tokenizers_by_vocab_size(tokenizer_names):
    """
    按vocab大小排序tokenizer
    从名字中提取 8k, 16k, 32k, 64k 等
    """
    def extract_vocab_size(name):
        match = re.search(r'_(\d+)k_', name)
        if match:
            return int(match.group(1)) * 1000
        return 0
    
    return sorted(tokenizer_names, key=extract_vocab_size)


def extract_vocab_size_from_name(name):
    """从tokenizer名字中提取vocab大小用于排序"""
    match = re.search(r'_(\d+)k_', name)
    if match:
        return int(match.group(1)) * 1000
    return 0


def format_results(results_dict: dict, title: str = "") -> str:
    """
    将分析结果格式化为表格
    """
    report = []
    report.append("=" * 100)
    report.append(title or "TOKENIZERS - VOCABULARY COMPOSITION ANALYSIS")
    report.append("=" * 100)
    report.append("")
    report.append("Analyzing: Chinese characters, Latin letters, punctuation, and other languages (Japanese, Korean)")
    report.append("Special focus: UTF-8 byte fragments (represented as 'half characters')")
    report.append("Note: Pinyin tokens like 'zhishao', 'jinwang' are correct (correspond to Chinese words like '至少', '金王')")
    report.append("")
    
    # 提取所有可能的类型（按顺序排列）
    all_types = set()
    for results in results_dict.values():
        all_types.update(results.get("type_distribution", {}).keys())
    
    # 排序类型
    type_order = [
        "LATIN_LETTER",
        "LATIN_DIGIT", 
        "DIGIT",
        "PUNCTUATION",
        "CHINESE_1CHAR",
        "CHINESE_2CHAR",
        "CHINESE_3CHAR",
        "CHINESE_4CHAR",
        "CHINESE_5CHAR",
        "CHINESE_6CHAR",
        "MIXED_WITH_CHINESE",
        "OTHER_JAPANESE",
        "OTHER_KOREAN",
        "OTHER_RARE_CHINESE_UTF8",
        "OTHER_UNKNOWN",
        "EMPTY",
    ]
    
    sorted_types = [t for t in type_order if t in all_types]
    sorted_types.extend(sorted([t for t in all_types if t not in type_order]))
    
    # 生成表格标题
    header = "Token Type".ljust(35)
    for name in sort_tokenizers_by_vocab_size(list(results_dict.keys())):
        header += f" | {name[:13]:>13}"
    header += " | TOTAL"
    
    report.append(header)
    report.append("=" * len(header))
    
    # 生成表格内容
    totals = {name: 0 for name in results_dict.keys()}
    sorted_names = sort_tokenizers_by_vocab_size(list(results_dict.keys()))
    
    for token_type in sorted_types:
        row = token_type.ljust(35)
        type_total = 0
        
        for name in sorted_names:
            count = results_dict[name]["type_distribution"].get(token_type, 0)
            row += f" | {count:>13}"
            totals[name] += count
            type_total += count
        
        row += f" | {type_total:>5}"
        report.append(row)
    
    # 总和行
    report.append("-" * len(header))
    row = "TOTAL".ljust(35)
    grand_total = 0
    for name in sorted_names:
        total = results_dict[name]["total_vocab_size"]
        row += f" | {total:>13}"
        grand_total += total
    row += f" | {grand_total:>5}"
    report.append(row)
    
    report.append("")
    report.append("=" * 100)
    report.append("CATEGORY EXPLANATIONS")
    report.append("=" * 100)
    report.append("")
    report.append("LATIN_LETTER       - Pure Latin letters (a-zA-Z)")
    report.append("LATIN_DIGIT        - Latin letters and digits combined")
    report.append("DIGIT              - Pure digits (0-9)")
    report.append("PUNCTUATION        - Punctuation marks and symbols")
    report.append("")
    report.append("CHINESE_NCHAR      - N pure Chinese characters (Standard CJK: U+4E00-U+9FFF)")
    report.append("MIXED_WITH_CHINESE - Tokens containing Chinese chars mixed with other content")
    report.append("")
    report.append("OTHER_JAPANESE     - Japanese characters (Hiragana/Katakana)")
    report.append("OTHER_KOREAN       - Korean characters (Hangul: U+AC00-U+D7AF)")
    report.append("OTHER_RARE_CHINESE_UTF8 - Rare Chinese chars or UTF-8 byte fragments")
    report.append("                     (CJK Extension: U+3400-U+4DBF, or byte notation <0xXX>)")
    report.append("OTHER_UNKNOWN      - Other unclassified tokens")
    report.append("")
    
    return "\n".join(report)


def generate_detailed_report(results_dict: dict) -> str:
    """
    生成详细报告
    """
    report = []
    report.append("")
    report.append("=" * 100)
    report.append("DETAILED BREAKDOWN BY TOKENIZER")
    report.append("=" * 100)
    report.append("")
    
    for name, results in sorted(results_dict.items(), key=lambda x: extract_vocab_size_from_name(x[0])):
        report.append(f"\n{'=' * 100}")
        report.append(f"TOKENIZER: {name}")
        report.append(f"{'=' * 100}")
        report.append(f"Total Vocabulary Size: {results['total_vocab_size']}")
        report.append("")
        
        # 按数量排序类型
        sorted_types = sorted(
            results["type_distribution"].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        report.append(f"{'Type':<45} {'Count':>10} {'Percentage':>12}")
        report.append("-" * 70)
        
        for token_type, count in sorted_types:
            percentage = (count / results["total_vocab_size"]) * 100
            report.append(f"{token_type:<45} {count:>10} {percentage:>11.2f}%")
        
        # 示例token - 显示所有类型
        report.append("")
        report.append("Sample tokens by type:")
        report.append("-" * 70)
        
        # 动态获取该tokenizer的所有token类型，按照出现数量排序
        all_token_types = sorted(
            results["detailed_stats"].keys(),
            key=lambda t: len(results["detailed_stats"].get(t, [])),
            reverse=True
        )
        
        for token_type in all_token_types:
            tokens = results["detailed_stats"].get(token_type, [])
            if tokens:
                # 采样策略：前3个 + 最后2个 + 随机3个
                samples = []
                
                # 前3个
                samples.extend(tokens[:min(3, len(tokens))])
                
                # 后2个（如果足够多）
                if len(tokens) > 5:
                    samples.extend(tokens[-2:])
                
                # 随机采样3个（从中间部分）
                if len(tokens) > 10:
                    random_samples = random.sample(tokens[3:-2], min(3, len(tokens)-5))
                    samples.extend(random_samples)
                
                # 去重并保持顺序
                seen = set()
                unique_samples = []
                for s in samples:
                    if s not in seen:
                        seen.add(s)
                        unique_samples.append(s)
                
                samples_str = ", ".join([repr(t) for t in unique_samples[:10]])
                report.append(f"  {token_type:<40}: {samples_str}")
        
        report.append("")
    
    # 添加说明
    report.append("")
    report.append("=" * 100)
    report.append("CATEGORY EXPLANATIONS")
    report.append("=" * 100)
    report.append("")
    report.append("LATIN_LETTER       - Pure Latin letters (a-zA-Z)")
    report.append("LATIN_DIGIT        - Latin letters and digits combined")
    report.append("DIGIT              - Pure digits (0-9)")
    report.append("PUNCTUATION        - Punctuation marks and symbols")
    report.append("")
    report.append("CHINESE_NCHAR      - N pure Chinese characters (Standard CJK: U+4E00-U+9FFF)")
    report.append("MIXED_WITH_CHINESE - Tokens containing Chinese chars mixed with other content")
    report.append("")
    report.append("OTHER_JAPANESE     - Japanese characters (Hiragana/Katakana)")
    report.append("OTHER_KOREAN       - Korean characters (Hangul: U+AC00-U+D7AF)")
    report.append("OTHER_RARE_CHINESE_UTF8 - Rare Chinese chars or UTF-8 byte fragments")
    report.append("                     (CJK Extension: U+3400-U+4DBF, or byte notation <0xXX>)")
    report.append("OTHER_UNKNOWN      - Other unclassified tokens")
    report.append("")
    
    return "\n".join(report)


# ===== 主流程 =====

def analyze_config(config_name: str, config: dict):
    """
    分析单个配置（中文或拼音）的所有tokenizer
    """
    print(f"\n{'='*100}")
    print(f"ANALYZING: {config_name.upper()}")
    print(f"{'='*100}")
    
    results_dict = {}
    tokenizers_to_analyze = config["tokenizers"]
    
    # 分析每个tokenizer
    for tokenizer_name in tokenizers_to_analyze:
        tokenizer_path = os.path.join(TOKENIZERS_DIR, tokenizer_name)
        
        if not os.path.exists(tokenizer_path):
            print(f"✗ File not found: {tokenizer_path}")
            continue
        
        results = analyze_tokenizer(tokenizer_path)
        results_dict[tokenizer_name] = results
    
    if not results_dict:
        print(f"✗ No tokenizers analyzed for {config_name}. Skipping.")
        return
    
    # 生成报告
    summary_report = format_results(results_dict, title=config["title"])
    detailed_report = generate_detailed_report(results_dict)
    
    # 完整报告
    full_report = summary_report + detailed_report
    
    # 保存到文件
    output_path = os.path.join(TOKENIZERS_DIR, config["output_file"])
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(full_report)
        print(f"\n✓ Report saved to: {output_path}")
    except Exception as e:
        print(f"✗ Error saving report: {e}")
    
    # 打印到控制台（摘要部分）
    print("\n" + summary_report)


def main():
    print("=" * 100)
    print("TOKENIZER VOCABULARY ANALYSIS - DUAL ANALYSIS")
    print("=" * 100)
    print("\nThis analysis will compare both Chinese Origin and Pinyin Toneless tokenizers")
    print("and save results to separate files for easy comparison.\n")
    
    # 处理每个配置
    for config_name, config in CONFIGS.items():
        analyze_config(config_name, config)
    
    print(f"\n{'='*100}")
    print("ALL ANALYSES COMPLETE!")
    print(f"{'='*100}")
    print("\nReports saved to:")
    for config_name, config in CONFIGS.items():
        output_path = os.path.join(TOKENIZERS_DIR, config["output_file"])
        print(f"  • {config_name}: {output_path}")


if __name__ == "__main__":
    main()
