# 2026-04-22 更新说明
# 本脚本评估 superTokenizers_BPE/*_subset100k_superbpe_*/tokenizer.json。
# 每个 tokenizer 会根据 detect_type() + find_test_file() 匹配同类型 test10 文件：
# • chinese_origin_*     -> chinese_origin_中国_test10.txt
# • pinyin_toned_*       -> pinyin_toned_spaced_test10.txt
# • pinyin_toneless_*    -> pinyin_toneless_spaced_test10.txt
# • pinyin_diacritic_*   -> pinyin_diacritic_spaced_test10.txt
#
# 指标说明：
# 4a Fertility
# • tokens/sample：每行测试样本平均 token 数。这里的 sample 是 corpus line，不严格等于 sentence。
# • tokens/surface char：token 数 / 当前表示文本字符数。可用于同一文字系统内部比较。
# • tokens/original Chinese char：token 数 / 原始中文 test10 字符数。用于跨汉字/拼音表示的公平比较。
#
# 4b Morphological Coherence Proxy
# • 当前实现是 dictionary hit rate，不是严格的 BPE merge 形态学分析。
# • SuperBPE 使用 byte-level token 显示形式，4b 必须先用 ByteLevel decoder 解码 token id。
#   例如 æĸ° / xÄ«n 这类内部 token 字符串不能直接查字典。
# • origin 使用 CEDICT 简体词表。
# • pinyin_toned / pinyin_toneless / pinyin_diacritic 使用对应拼音词典。
# • 拼音 token 同时检查原 token 和去掉内部空格后的 compact token，避免 spaced pinyin 漏判。
# • 报告 checked / valid / skipped punctuation / invalid examples，方便判断低分来自 tokenizer 还是字典覆盖。
#
# 4c Cross-lingual Overlap
# • 当前仍不启用。原先用 test split 构造 english vocab 的方法不成立。
#
# 4d Compression Efficiency
# • 主指标保持 chars/token、bytes/token、chars/byte。
# • Summary 中 bytes/token 按越大越紧凑解释，因此选 max(bytes_per_token)。
# • 额外报告 bytes/original Chinese char，辅助判断表达同一批中文内容时的字节成本。

import os
import csv
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel
from tqdm import tqdm
BASE_DIR = "/Users/crisp/Desktop/code_field/python/Chinese_Latinization_NLP/1.Tokenization"

CORPORA_DIR = os.path.join(BASE_DIR, "corpora")
TOKENIZER_DIR = os.path.join(BASE_DIR, "superTokenizers_BPE")
DICT_DIR = os.path.join(BASE_DIR, "dicts")

OUTPUT_FILE = os.path.join(BASE_DIR, "evaluation_4abcd_superBPE.csv")
REPORT_FILE = os.path.join(BASE_DIR, "tokenizer_evaluation_report_superBPE.txt")

# ===== DETECT TYPE =====
def detect_type(name):
    name = name.lower()
    if "chinese_origin" in name:
        return "origin"
    elif "pinyin_toned" in name:
        return "toned"
    elif "pinyin_toneless" in name:
        return "toneless"
    elif "pinyin_diacritic" in name:
        return "diacritic"
    return None

# ===== FIND TEST FILE =====
def find_test_file(t_type):
    for file in os.listdir(CORPORA_DIR):
        fname = file.lower()

        if t_type == "origin" and fname == "chinese_origin_中国_test10.txt":
            return os.path.join(CORPORA_DIR, file)

        if t_type == "diacritic" and fname == "pinyin_diacritic_spaced_test10.txt":
            return os.path.join(CORPORA_DIR, file)

        if t_type == "toneless" and fname == "pinyin_toneless_spaced_test10.txt":
            return os.path.join(CORPORA_DIR, file)

        if t_type == "toned" and fname == "pinyin_toned_spaced_test10.txt":
            return os.path.join(CORPORA_DIR, file)

    raise ValueError(f"No test file found for {t_type}")


# ===== LOAD TEXT =====
def load_texts(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

# ===== LOAD DICTS =====
def load_dict(name):
    path = os.path.join(DICT_DIR, name)
    with open(path, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f)

def load_cedict_simplified():
    words = set()
    path = os.path.join(DICT_DIR, "cedict_ts.u8")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                words.add(parts[1])
    return words

dict_map = {
    "origin": load_cedict_simplified(),
    "toned": load_dict("dict_toned.txt"),
    "toneless": load_dict("dict_toneless.txt"),
    "diacritic": load_dict("dict_diacritic.txt"),
}

# ===== ENGLISH VOCAB (Commented Out for 4c) =====
# english_vocab = set()
# for file in os.listdir(CORPORA_DIR):
#     if "test10" in file:
#         texts = load_texts(os.path.join(CORPORA_DIR, file))
#         for t in texts[:200]:
#             for w in t.split():
#                 english_vocab.add(w.lower())

import unicodedata
import re

VALID_PINYIN = set(['a', 'ai', 'an', 'ang', 'ao', 'ba', 'bai', 'ban', 'bang', 'bao', 'bei', 'ben', 'beng', 'bi', 'bian', 'biao', 'bie', 'bin', 'bing', 'bo', 'bu', 'ca', 'cai', 'can', 'cang', 'cao', 'ce', 'cen', 'ceng', 'cha', 'chai', 'chan', 'chang', 'chao', 'che', 'chen', 'cheng', 'chi', 'chong', 'chou', 'chu', 'chua', 'chuai', 'chuan', 'chuang', 'chui', 'chun', 'chuo', 'ci', 'cong', 'cou', 'cu', 'cuan', 'cui', 'cun', 'cuo', 'da', 'dai', 'dan', 'dang', 'dao', 'de', 'dei', 'deng', 'di', 'dian', 'diao', 'die', 'ding', 'diu', 'dong', 'dou', 'du', 'duan', 'dui', 'dun', 'duo', 'e', 'ei', 'en', 'eng', 'er', 'fa', 'fan', 'fang', 'fei', 'fen', 'feng', 'fo', 'fou', 'fu', 'ga', 'gai', 'gan', 'gang', 'gao', 'ge', 'gei', 'gen', 'geng', 'gong', 'gou', 'gu', 'gua', 'guai', 'guan', 'guang', 'gui', 'gun', 'guo', 'ha', 'hai', 'han', 'hang', 'hao', 'he', 'hei', 'hen', 'heng', 'hm', 'hng', 'hong', 'hou', 'hu', 'hua', 'huai', 'huan', 'huang', 'hui', 'hun', 'huo', 'ji', 'jia', 'jian', 'jiang', 'jiao', 'jie', 'jin', 'jing', 'jiong', 'jiu', 'ju', 'juan', 'jue', 'jun', 'ka', 'kai', 'kan', 'kang', 'kao', 'ke', 'kei', 'ken', 'keng', 'kong', 'kou', 'ku', 'kua', 'kuai', 'kuan', 'kuang', 'kui', 'kun', 'kuo', 'la', 'lai', 'lan', 'lang', 'lao', 'le', 'lei', 'leng', 'li', 'lia', 'lian', 'liang', 'liao', 'lie', 'lin', 'ling', 'liu', 'lo', 'long', 'lou', 'lu', 'lv', 'luan', 'lue', 'lve', 'lun', 'luo', 'm', 'ma', 'mai', 'man', 'mang', 'mao', 'me', 'mei', 'men', 'meng', 'mi', 'mian', 'miao', 'mie', 'min', 'ming', 'miu', 'mo', 'mou', 'mu', 'na', 'nai', 'nan', 'nang', 'nao', 'ne', 'nei', 'nen', 'neng', 'ng', 'ni', 'nian', 'niang', 'niao', 'nie', 'nin', 'ning', 'niu', 'nong', 'nou', 'nu', 'nv', 'nuan', 'nue', 'nve', 'nun', 'nuo', 'o', 'ou', 'pa', 'pai', 'pan', 'pang', 'pao', 'pei', 'pen', 'peng', 'pi', 'pian', 'piao', 'pie', 'pin', 'ping', 'po', 'pou', 'pu', 'qi', 'qia', 'qian', 'qiang', 'qiao', 'qie', 'qin', 'qing', 'qiong', 'qiu', 'qu', 'quan', 'que', 'qun', 'ran', 'rang', 'rao', 're', 'ren', 'reng', 'ri', 'rong', 'rou', 'ru', 'ruan', 'rui', 'run', 'ruo', 'sa', 'sai', 'san', 'sang', 'sao', 'se', 'sen', 'seng', 'sha', 'shai', 'shan', 'shang', 'shao', 'she', 'shei', 'shen', 'sheng', 'shi', 'shou', 'shu', 'shua', 'shuai', 'shuan', 'shuang', 'shui', 'shun', 'shuo', 'si', 'song', 'sou', 'su', 'suan', 'sui', 'sun', 'suo', 'ta', 'tai', 'tan', 'tang', 'tao', 'te', 'teng', 'ti', 'tian', 'tiao', 'tie', 'ting', 'tong', 'tou', 'tu', 'tuan', 'tui', 'tun', 'tuo', 'wa', 'wai', 'wan', 'wang', 'wei', 'wen', 'weng', 'wo', 'wu', 'xi', 'xia', 'xian', 'xiang', 'xiao', 'xie', 'xin', 'xing', 'xiong', 'xiu', 'xu', 'xuan', 'xue', 'xun', 'ya', 'yan', 'yang', 'yao', 'ye', 'yi', 'yin', 'ying', 'yo', 'yong', 'you', 'yu', 'yuan', 'yue', 'yun', 'za', 'zai', 'zan', 'zang', 'zao', 'ze', 'zei', 'zen', 'zeng', 'zha', 'zhai', 'zhan', 'zhang', 'zhao', 'zhe', 'zhei', 'zhen', 'zheng', 'zhi', 'zhong', 'zhou', 'zhu', 'zhua', 'zhuai', 'zhuan', 'zhuang', 'zhui', 'zhun', 'zhuo', 'zi', 'zong', 'zou', 'zu', 'zuan', 'zui', 'zun', 'zuo'])

def is_valid_pinyin_syllable(s: str) -> bool:
    if any('A' <= c <= 'Z' for c in s):
        return False
    s = s.replace('ü', 'v').replace('u:', 'v')
    s = re.sub(r'[1-5]', '', s)
    s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    return s in VALID_PINYIN

def is_chinese_chars(s: str) -> bool:
    if not s: return False
    return all('一' <= c <= '鿿' for c in s)

def decode_token_id(tokenizer, token_id, fallback_token, cache):
    if token_id not in cache:
        try:
            cache[token_id] = tokenizer.decode([token_id])
        except Exception:
            cache[token_id] = fallback_token
    return cache[token_id]

# ===== ANALYSIS STRUCTURE =====
class TokenizerAnalysisCollector:
    """收集和存储分析结果"""
    def __init__(self):
        self.results = []
        self.details = {}  # 存储详细的分析结果
    
    def add_result(self, analysis_dict, details_dict=None):
        self.results.append(analysis_dict)
        if details_dict:
            key = analysis_dict.get('file', '')
            self.details[key] = details_dict


def generate_report(collector, output_file=None):
    """
    生成详细的分析报告，参考7th的generate_report实现方式
    """
    if output_file is None:
        output_file = REPORT_FILE
    
    print("\n" + "=" * 100)
    print("GENERATING COMPREHENSIVE REPORT...")
    print("=" * 100)
    
    report = []
    report.append("=" * 100)
    report.append("TOKENIZER EVALUATION REPORT: SuperBPE Tokenizers Analysis (Chinese Origin + Pinyin Systems)")
    report.append("=" * 100)
    report.append("")
    
    # ===== TEST DATA SUMMARY =====
    report.append("TEST DATA SUMMARY BY TEST FILE")
    report.append("-" * 100)

    test_files = {}
    for result in sorted(collector.results, key=lambda x: (x.get('type', ''), x.get('file', ''))):
        test_file = result.get('test_file')
        if test_file:
            test_files.setdefault(test_file, set()).add(result.get('file', 'UNKNOWN'))

    for test_file, tokenizer_files in sorted(test_files.items()):
        test_texts = load_texts(test_file)
        total_test_chars = sum(len(t) for t in test_texts)
        avg_sample_length = total_test_chars / len(test_texts) if test_texts else 0

        report.append(f"Test file: {test_file}")
        report.append(f"Matched tokenizers: {len(tokenizer_files)}")
        report.append(f"Number of test samples: {len(test_texts)}")
        report.append(f"Total characters: {total_test_chars}")
        report.append(f"Average sample length: {avg_sample_length:.2f} chars")
        report.append("Sample texts:")
        for i, text in enumerate(test_texts[:3]):
            report.append(f"  {i+1}. {text}")
        report.append("")
    report.append("")
    
    # ===== 4A: FERTILITY ANALYSIS =====
    report.append("=" * 100)
    report.append("4A. FERTILITY ANALYSIS (Tokens-per-sample & Tokens-per-character)")
    report.append("=" * 100)
    report.append("")
    
    for result in sorted(collector.results, key=lambda x: x.get('file', '')):
        report.append(f"{result.get('file', 'UNKNOWN').upper()}")
        report.append("-" * 50)
        report.append(f"  Type: {result.get('type', 'N/A')}")
        report.append(f"  Test file: {result.get('test_file', 'N/A')}")
        report.append(f"  Tokens per sample: {result.get('tokens_per_sample', 'N/A'):.4f}")
        report.append(f"  Tokens per surface character: {result.get('tokens_per_surface_char', 'N/A'):.4f}")
        report.append(f"  Tokens per original Chinese character: {result.get('tokens_per_original_char', 'N/A'):.4f}")
        report.append(f"  Total test tokens: {result.get('total_tokens', 'N/A')}")
        report.append(f"  Total surface characters: {result.get('total_chars', 'N/A')}")
        report.append(f"  Total original Chinese characters: {result.get('total_original_chars', 'N/A')}")
        report.append("")
    
    # ===== 4B: MORPHOLOGICAL COHERENCE ANALYSIS =====
    report.append("=" * 100)
    report.append("4B. MORPHOLOGICAL COHERENCE PROXY (Dictionary Hit Rate)")
    report.append("=" * 100)
    report.append("")
    
    for result in sorted(collector.results, key=lambda x: x.get('file', '')):
        report.append(f"{result.get('file', 'UNKNOWN').upper()}")
        report.append("-" * 50)
        report.append(f"  Type: {result.get('type', 'N/A')}")
        report.append(f"  Test file: {result.get('test_file', 'N/A')}")
        report.append(f"  Dictionary hit rate: {result.get('morph_score', 'N/A'):.4f}")
        report.append(f"  Valid tokens checked: {result.get('checked_tokens', 'N/A')}")
        report.append(f"  Valid tokens found: {result.get('valid_tokens', 'N/A')}")
        report.append(f"  Invalid dictionary misses: {result.get('invalid_tokens', 'N/A')}")
        report.append(f"  Punctuation/non-word tokens skipped: {result.get('skipped_punctuation_tokens', 'N/A')}")
        
        details = collector.details.get(result.get('file', ''), {})
        if 'sample_valid_tokens' in details:
            report.append(f"  Sample valid tokens: {details['sample_valid_tokens'][:10]}")
        if 'sample_invalid_tokens' in details:
            report.append(f"  Sample invalid tokens: {details['sample_invalid_tokens'][:10]}")
        report.append("")
    
    # ===== 4C: CROSS-LINGUAL OVERLAP =====
    # report.append("=" * 100)
    # report.append("4C. CROSS-LINGUAL OVERLAP ANALYSIS")
    # report.append("=" * 100)
    # report.append("")
    
    # for result in sorted(collector.results, key=lambda x: x.get('file', '')):
    #     report.append(f"{result.get('file', 'UNKNOWN').upper()}")
    #     report.append("-" * 50)
    #     report.append(f"  Cross-lingual overlap ratio: {result.get('overlap', 'N/A'):.4f}")
    #     report.append("")
    
    # ===== 4D: COMPRESSION EFFICIENCY ANALYSIS =====
    report.append("=" * 100)
    report.append("4D. COMPRESSION EFFICIENCY ANALYSIS (Chars/Token & Bytes/Token)")
    report.append("=" * 100)
    report.append("")
    
    for result in sorted(collector.results, key=lambda x: x.get('file', '')):
        report.append(f"{result.get('file', 'UNKNOWN').upper()}")
        report.append("-" * 50)
        report.append(f"  Type: {result.get('type', 'N/A')}")
        report.append(f"  Test file: {result.get('test_file', 'N/A')}")
        report.append(f"  Compression ratio (chars/token): {result.get('chars_per_token', 'N/A'):.4f}")
        report.append(f"  Bytes per token: {result.get('bytes_per_token', 'N/A'):.4f}")
        report.append(f"  Information efficiency (chars per UTF-8 byte): {result.get('chars_per_byte', 'N/A'):.4f}")
        report.append(f"  Bytes per original Chinese character: {result.get('bytes_per_original_char', 'N/A'):.4f}")
        report.append("")
    
    # ===== SUMMARY & FINDINGS =====
    report.append("=" * 100)
    report.append("SUMMARY & FINDINGS")
    report.append("=" * 100)
    report.append("")
    
    if collector.results:
        # Best compression
        best_compression = max(
            collector.results,
            key=lambda x: x.get('chars_per_token', 0)
        )
        report.append(f"🏆 Best compression ratio (chars/token):")
        report.append(f"   {best_compression.get('file', 'N/A')} - {best_compression.get('chars_per_token', 'N/A'):.4f}")
        report.append("")
        
        # Highest dictionary hit rate
        best_morph = max(
            collector.results,
            key=lambda x: x.get('morph_score', 0) if x.get('morph_score') is not None else 0
        )
        if best_morph.get('morph_score') is not None:
            report.append(f"🏆 Highest dictionary hit rate:")
            report.append(f"   {best_morph.get('file', 'N/A')} - {best_morph.get('morph_score', 'N/A'):.4f}")
            report.append("")
        
        # Highest bytes per token means each token covers more input bytes.
        best_bytes = max(
            collector.results,
            key=lambda x: x.get('bytes_per_token', 0)
        )
        report.append(f"🏆 Highest bytes per token:")
        report.append(f"   {best_bytes.get('file', 'N/A')} - {best_bytes.get('bytes_per_token', 'N/A'):.4f}")
        report.append("")
    
    report.append("")
    report.append("=" * 100)
    report.append("END OF REPORT")
    report.append("=" * 100)
    
    # ===== SAVE REPORT =====
    report_text = "\n".join(report)
    
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"\n✓ Report saved to: {output_file}")
    except Exception as e:
        print(f"✗ Error saving report: {e}")
        print("Report content:")
        print(report_text)
    
    return report_text


# ===== MAIN =====
collector = TokenizerAnalysisCollector()

origin_test_path = find_test_file("origin")
origin_texts = load_texts(origin_test_path)
total_original_chars = sum(len(t) for t in origin_texts)

folders = [folder for folder in os.listdir(TOKENIZER_DIR) if "_superbpe_" in folder]

for folder in tqdm(folders, desc="Tokenizers"):
    # Only pick superbpe directories
    if "_superbpe_" not in folder:
        continue

    t_type = detect_type(folder)
    if t_type is None:
        continue
        
    tokenizer_path = os.path.join(TOKENIZER_DIR, folder, "tokenizer.json")
    if not os.path.exists(tokenizer_path):
        continue

    print(f"Running: {folder} → {t_type}")

    tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer.decoder = ByteLevel()
    file = folder  # Map 'file' variable to folder name so results table uses it

    test_path = find_test_file(t_type)
    texts = load_texts(test_path)

    dictionary = dict_map.get(t_type, None)
    decoded_token_cache = {}

    total_tokens = 0
    total_chars = 0
    total_bytes = 0

    valid_tokens = 0
    checked_tokens = 0
    skipped_punctuation_tokens = 0
    sample_valid_tokens = []
    sample_invalid_tokens = []
    
    for text in tqdm(texts, desc=f"Processing {folder}", leave=False):
        enc = tokenizer.encode(text)
        tokens = enc.tokens

        total_tokens += len(tokens)
        total_chars += len(text)
        total_bytes += len(text.encode("utf-8"))

        # ===== 4b =====
        
        for token_id, raw_tok in zip(enc.ids, tokens):
            decoded_tok = decode_token_id(
                tokenizer, token_id, raw_tok, decoded_token_cache
            )
            clean_tok = decoded_tok.replace("##", "").replace("Ġ", "").strip()
            compact_tok = re.sub(r"\s+", "", clean_tok)
            
            # Skip purely punctuation or non-alphanumeric forms
            if not clean_tok or re.match(r'^[^\w一-鿿]+$', clean_tok):
                skipped_punctuation_tokens += 1
                continue
                
            if len(clean_tok) >= 2:
                checked_tokens += 1
                is_valid = False
                
                # Check 1: Dictionary Match
                if dictionary and (clean_tok in dictionary or compact_tok in dictionary):
                    is_valid = True
                
                # Check 2: Chinese Origin pure CJK
                elif t_type == "origin" and is_chinese_chars(clean_tok):
                    is_valid = True
                
                # Check 3: Valid Pinyin Combinations (with optional numbers)
                elif t_type != "origin":
                    pinyin_parts = [p.strip() for p in clean_tok.split(" ") if p.strip()]
                    if pinyin_parts:
                        all_parts_valid = True
                        for p in pinyin_parts:
                            if not (p.isdigit() or is_valid_pinyin_syllable(p)):
                                all_parts_valid = False
                                break
                        if all_parts_valid:
                            is_valid = True
                
                if is_valid:
                    valid_tokens += 1
                    if len(sample_valid_tokens) < 20:
                        sample_valid_tokens.append(clean_tok)
                elif len(sample_invalid_tokens) < 20:
                    sample_invalid_tokens.append(clean_tok)

    # ===== 4a =====
    tokens_per_sample = total_tokens / len(texts) if len(texts) > 0 else 0
    tokens_per_surface_char = total_tokens / total_chars if total_chars > 0 else 0
    tokens_per_original_char = total_tokens / total_original_chars if total_original_chars > 0 else 0

    # ===== 4b =====
    morph_score = (valid_tokens / checked_tokens) if checked_tokens > 0 else None
    invalid_tokens = checked_tokens - valid_tokens

    # ===== 4c =====
    # vocab = set(tokenizer.get_vocab().keys())
    # overlap = len(vocab & english_vocab) / len(vocab)
    overlap = 0.0  # Placeholder

    # ===== 4d =====
    chars_per_token = total_chars / total_tokens if total_tokens > 0 else 0
    bytes_per_token = total_bytes / total_tokens if total_tokens > 0 else 0
    chars_per_byte = total_chars / total_bytes if total_bytes > 0 else 0
    bytes_per_original_char = total_bytes / total_original_chars if total_original_chars > 0 else 0

    # Store in collector for report
    collector.add_result(
        {
            'file': file,
            'type': t_type,
            'test_file': test_path,
            'tokens_per_sample': tokens_per_sample,
            'tokens_per_surface_char': tokens_per_surface_char,
            'tokens_per_original_char': tokens_per_original_char,
            'total_tokens': total_tokens,
            'total_chars': total_chars,
            'total_original_chars': total_original_chars,
            'morph_score': morph_score,
            'valid_tokens': valid_tokens,
            'checked_tokens': checked_tokens,
            'invalid_tokens': invalid_tokens,
            'skipped_punctuation_tokens': skipped_punctuation_tokens,
            'overlap': overlap,
            'chars_per_token': chars_per_token,
            'bytes_per_token': bytes_per_token,
            'chars_per_byte': chars_per_byte,
            'bytes_per_original_char': bytes_per_original_char,
        },
        {
            'sample_valid_tokens': sample_valid_tokens,
            'sample_invalid_tokens': sample_invalid_tokens,
        }
    )

# ===== GENERATE DETAILED REPORT =====
generate_report(collector)

print("\n" + "=" * 100)
print("ANALYSIS COMPLETE!")
print("=" * 100)
