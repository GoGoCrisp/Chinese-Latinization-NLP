"""
Tokenizer Comparison: 2048 SuperBPE 64K Custom Tokenizers vs. Production LLM Tokenizers
对比内容：
- 4a. Fertility (肥沃度): tokens-per-sentence, tokens-per-character
- 4b. Morphological coherence (形态学连贯性): BPE merges是否对应有意义的词素
- 4d. Compression efficiency (压缩效率): bits-per-character, bytes-per-token

2026-04-29 更新说明：
- 临时只跑 4b Morphological Coherence，以便快速迭代该指标；4a Fertility、
  4d Compression Efficiency 和 AI tokenizer 加载暂时在 run_full_analysis() 中停用。
- 4b 已从旧版 test-set token occurrence dictionary hit rate 改成 vocabulary-level
  token type 质量评估。目标不再是“这个 tokenizer 在测试文本中切出了多少词典词”，
  而是“这个 tokenizer 的 vocab 里有多少 token 本身是语义/形态上合理的单位”。
- 4b 分四类：SV=strong_valid，WV=weak_valid，IV=invalid，EX=excluded。
  分母是 SV+WV+IV；EX 只用于 coverage，不进入 valid/invalid 率。
- 主要输出：
    strict_valid_rate = SV / (SV + WV + IV)
    inclusive_valid_rate = (SV + WV) / (SV + WV + IV)
    invalid_rate = IV / (SV + WV + IV)
    coverage_over_vocab = (SV + WV + IV) / vocab_size
- 中文判断思路：CEDICT 命中、单字词素/功能词、专名/数字时间表达算 SV；完整词+
  的/了/着/过/地/得、方位词/范围词附着、多个完整词构成的自然短语算 WV；词中
  残片或不自然跨界算 IV。比如“中华人民共和国”是 SV，“中华人民共和国的”是 WV，
  “华人民共和国”虽然没跨边界，但只是“中华人民共和国”的词中残片，因此算 IV。
- 拼音判断思路：不能要求唯一还原成汉字；只要存在合理中文解释即可。先检查是否是
  完整合法拼音音节序列，再用拼音化词典、功能词、可解释短语和数字时间规则分类。
  例如 zui hou yi 很可能是“最后一”，dui huo de 可理解为“对获得”，huai yi shi 是
  “怀疑是”，wang zi de 是“王子的”，jiang zi ji 是“将自己”，tui te shang 是“推特上”，
  jin zhong jiang 可能是“仅中奖/金钟奖”等歧义形式；这些歧义不自动判 IV。
- 拼音数字时间特例：di 1 可看作“第1”这类可解释表达；但 yu 2013 少了“年”等后续
  时间单位，nian 6 yue 2 ri 少了前面的年份数字，属于残缺时间片段，算 IV。
- 这套 SV/WV/IV 仍是可复现的启发式，不是假装有唯一语言学金标准。4b 会额外输出
  逐 token TSV，便于后续人工抽查、修正规则和写文章时回忆这些取舍。
- 4b 现在还会生成 manual_audit_sample.html / manual_audit_sample.csv：对每个
  tokenizer 的自动 SV/WV/IV 各随机抽最多 100 个，做成可单选 SV/WV/IV/UNK 的人工
  复核表。目的不是只找 IV 里的漏判，也检查 SV/WV 是否被规则高估。
"""

import csv
import json
import os
import re
import ast
import unicodedata
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import numpy as np
from pathlib import Path
import random
from tqdm import tqdm

# Try importing AI tokenizers
try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False
    print("Warning: tiktoken not installed. Install with: pip install tiktoken")

try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers not installed. Install with: pip install transformers")

try:
    from pypinyin import pinyin as pypinyin_pinyin, lazy_pinyin, Style as PinyinStyle
    HAS_PYPINYIN = True
except ImportError:
    HAS_PYPINYIN = False
    print("Warning: pypinyin not installed. Enhanced pinyin lexicon will be limited.")

import json
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.decoders import ByteLevel

# HuggingFace Token配置 - 使用环境变量 HF_TOKEN (在 ~/.huggingface/token 或环境变量中设置)
# 不在代码中硬编码敏感信息
HF_TOKEN = os.environ.get("HF_TOKEN", None)


class TokenizerComparison:
    """对比多个tokenizer的性能指标"""

    def __init__(self, base_dir: str, test_file: str = None):
        self.base_dir = base_dir
        self.tokenizers_dir = os.path.join(base_dir, "tokenizers")
        self.corpora_dir = os.path.join(base_dir, "corpora")
        self.test_file = test_file or os.path.join(
            self.corpora_dir, "chinese_origin_中国_test10.txt"
        )
        self.dicts_dir = os.path.join(base_dir, "dicts")

        self.tokenizers = {}  # 存储加载的tokenizer
        self.results = {}  # 存储分析结果
        self.morphology_token_details = {}
        self.morphology_sample_seed = 20260429
        self.test_data = []  # 测试数据
        self.test_data_by_file = {}
        self.test_files_by_tokenizer = {}
        self.custom_test_files = {
            "chinese_origin_64k": os.path.join(
                self.corpora_dir, "chinese_origin_中国_test10.txt"
            ),
            "pinyin_diacritic_64k": os.path.join(
                self.corpora_dir, "pinyin_diacritic_spaced_test10.txt"
            ),
            "pinyin_toned_64k": os.path.join(
                self.corpora_dir, "pinyin_toned_spaced_test10.txt"
            ),
            "pinyin_toneless_64k": os.path.join(
                self.corpora_dir, "pinyin_toneless_spaced_test10.txt"
            ),
        }

    def load_custom_tokenizers(self):
        """加载本地训练的2048 SuperBPE 64k tokenizers"""
        print("=" * 80)
        print("Loading Custom 2048 SuperBPE 64k Tokenizers...")
        print("=" * 80)

        tokenizer_dir = os.path.join(
            self.base_dir, "superTokenizers_BPE_2048_subset100k"
        )
        tokenizer_paths = {
            "chinese_origin_64k": os.path.join(
                tokenizer_dir,
                "chinese_origin_subset100k_superbpe_64000",
                "tokenizer.json",
            ),
            "pinyin_diacritic_64k": os.path.join(
                tokenizer_dir,
                "pinyin_diacritic_subset100k_superbpe_64000",
                "tokenizer.json",
            ),
            "pinyin_toned_64k": os.path.join(
                tokenizer_dir,
                "pinyin_toned_subset100k_superbpe_64000",
                "tokenizer.json",
            ),
            "pinyin_toneless_64k": os.path.join(
                tokenizer_dir,
                "pinyin_toneless_subset100k_superbpe_64000",
                "tokenizer.json",
            ),
        }

        for key, path in tokenizer_paths.items():
            if os.path.exists(path):
                try:
                    tokenizer = HFTokenizer.from_file(path)
                    tokenizer.decoder = ByteLevel()
                    tokenizer_key = f"custom_{key}"
                    self.tokenizers[tokenizer_key] = tokenizer
                    self.test_files_by_tokenizer[tokenizer_key] = self.custom_test_files.get(
                        key, self.test_file
                    )
                    print(f"✓ Loaded: {key}")
                except Exception as e:
                    print(f"✗ Failed to load {path}: {e}")
            else:
                print(f"✗ Not found: {path}")

    def load_ai_tokenizers(self):
        """加载AI模型的tokenizers"""
        print("\n" + "=" * 80)
        print("Loading AI Tokenizers...")
        print("=" * 80)

        # GPT-4 cl100k_base
        if HAS_TIKTOKEN:
            try:
                self.tokenizers["gpt4_cl100k"] = tiktoken.get_encoding("cl100k_base")
                self.test_files_by_tokenizer["gpt4_cl100k"] = self.test_file
                print("✓ Loaded: GPT-4 (cl100k_base)")
            except Exception as e:
                print(f"✗ Failed to load GPT-4 tokenizer: {e}")
        else:
            print("⊘ Skipped: GPT-4 (tiktoken not installed)")

        # Llama-3
        if HAS_TRANSFORMERS:
            llama_models = [
                ("NousResearch/Meta-Llama-3-8B", "NousResearch Meta-Llama-3-8B"),
                ("meta-llama/Llama-3.2-1B", "Llama-3.2-1B"),
                ("meta-llama/Llama-3.1-8B", "Llama-3.1-8B"),
                ("mistralai/Mistral-7B-v0.1", "Mistral-7B"),
            ]
            
            for model_path, model_name in llama_models:
                try:
                    print(f"  Attempting to load {model_name}...")
                    llama_tokenizer = AutoTokenizer.from_pretrained(
                        model_path, 
                        trust_remote_code=True,
                        token=HF_TOKEN
                    )
                    self.tokenizers["llama"] = llama_tokenizer
                    self.test_files_by_tokenizer["llama"] = self.test_file
                    print(f"✓ Loaded: {model_name}")
                    break
                except Exception as e:
                    print(f"  ⊘ {model_name} failed: {str(e)[:80]}...")
                    continue
            
            # 如果都失败了
            if "llama" not in self.tokenizers:
                print("⊘ All Llama models unavailable.")
                print("  Consider:")
                print("  1. Accepting license at https://huggingface.co/meta-llama/Llama-2-7b-hf")
                print("  2. Running: huggingface-cli login")

        # Qwen
        if HAS_TRANSFORMERS:
            try:
                qwen_tokenizer = AutoTokenizer.from_pretrained(
                    "Qwen/Qwen-7B", 
                    trust_remote_code=True,
                    token=HF_TOKEN
                )
                self.tokenizers["qwen"] = qwen_tokenizer
                self.test_files_by_tokenizer["qwen"] = self.test_file
                print("✓ Loaded: Qwen (with auth token)")
            except Exception as e:
                print(f"⊘ Qwen not available: {e}")
                print("  Note: Qwen models may require specific access.")

    def _load_text_file(self, path: str, max_lines: int = None) -> List[str]:
        texts = []
        if not os.path.exists(path):
            print(f"✗ Test file not found: {path}")
            return texts

        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        texts.append(line)
                        if max_lines and len(texts) >= max_lines:
                            break
        except Exception as e:
            print(f"✗ Error loading test data from {path}: {e}")

        return texts

    def load_test_data(self, max_lines: int = None):
        """
        加载测试数据。中文 tokenizer 和外部 AI tokenizer 使用中文 test；
        拼音 tokenizer 使用各自对应的拼音 test。
        """
        print("\n" + "=" * 80)
        print("Loading tokenizer-specific test data")
        print("=" * 80)

        unique_files = sorted(set(self.test_files_by_tokenizer.values()) or {self.test_file})
        for path in unique_files:
            texts = self._load_text_file(path, max_lines=max_lines)
            self.test_data_by_file[path] = texts
            print(f"✓ Loaded {len(texts)} test sentences from: {path}")
            if texts:
                print(f"  Sample: {texts[0][:50]}...")

        self.test_data = self.test_data_by_file.get(self.test_file, [])

        print("\nTokenizer → test file mapping:")
        for tokenizer_key in sorted(self.tokenizers.keys()):
            print(
                f"  {tokenizer_key}: "
                f"{self.test_files_by_tokenizer.get(tokenizer_key, self.test_file)}"
            )

    def get_test_data_for_tokenizer(self, tokenizer_key: str) -> List[str]:
        path = self.test_files_by_tokenizer.get(tokenizer_key, self.test_file)
        return self.test_data_by_file.get(path, [])

    def get_test_file_for_tokenizer(self, tokenizer_key: str) -> str:
        return self.test_files_by_tokenizer.get(tokenizer_key, self.test_file)

    def tokenize_text(self, text: str, tokenizer_key: str) -> List[int]:
        """对文本进行tokenize"""
        tokenizer = self.tokenizers.get(tokenizer_key)
        if tokenizer is None:
            return []

        try:
            if "custom_" in tokenizer_key:
                # 自定义tokenizer (HuggingFace Tokenizer)
                tokens = tokenizer.encode(text).ids
            elif tokenizer_key == "gpt4_cl100k":
                # tiktoken tokenizer
                tokens = tokenizer.encode(text)
            else:
                # transformers tokenizer
                tokens = tokenizer.encode(text)

            return tokens
        except Exception as e:
            print(f"Error tokenizing with {tokenizer_key}: {e}")
            return []

    def calculate_fertility(self) -> Dict:
        """
        4a. Fertility分析
        计算tokens-per-sentence和tokens-per-character
        """
        print("\n" + "=" * 80)
        print("4a. FERTILITY ANALYSIS")
        print("=" * 80)

        results = {}

        for tokenizer_key in self.tokenizers.keys():
            test_data = self.get_test_data_for_tokenizer(tokenizer_key)
            tokens_per_sentence = []
            tokens_per_char = []
            total_chars = 0
            total_tokens = 0

            for text in tqdm(
                test_data,
                desc=f"Fertility {tokenizer_key}",
                unit="sent",
                leave=True,
            ):
                tokens = self.tokenize_text(text, tokenizer_key)
                char_count = len(text)

                if char_count > 0:
                    tokens_per_sentence.append(len(tokens))
                    tokens_per_char.append(len(tokens) / char_count)
                    total_chars += char_count
                    total_tokens += len(tokens)

            if tokens_per_sentence:
                avg_tps = np.mean(tokens_per_sentence)
                std_tps = np.std(tokens_per_sentence)
                avg_tpc = np.mean(tokens_per_char)

                results[tokenizer_key] = {
                    "avg_tokens_per_sentence": round(avg_tps, 4),
                    "std_tokens_per_sentence": round(std_tps, 4),
                    "avg_tokens_per_char": round(avg_tpc, 4),
                    "total_tokens": total_tokens,
                    "total_chars": total_chars,
                    "compression_ratio": round(
                        total_chars / total_tokens, 4
                    ),  # chars per token
                }

                print(
                    f"\n{tokenizer_key}:"
                    f"\n  Avg tokens/sentence: {avg_tps:.2f} (±{std_tps:.2f})"
                    f"\n  Avg tokens/char: {avg_tpc:.4f}"
                    f"\n  Compression ratio (chars/token): {results[tokenizer_key]['compression_ratio']:.4f}"
                )

        return results

    def load_cedict(self) -> set:
        """加载CC-CEDICT中文词典，用于检查形态学连贯性"""
        cedict_path = os.path.join(self.dicts_dir, "cedict_ts.u8")
        chinese_words = set()

        if os.path.exists(cedict_path):
            try:
                with open(cedict_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("#"):
                            continue
                        parts = line.split()
                        if len(parts) >= 2:
                            # 第一个是繁体，第二个是简体
                            simplified = parts[1]
                            if len(simplified) > 0:
                                chinese_words.add(simplified)
                print(
                    f"✓ Loaded {len(chinese_words)} Chinese words from CEDICT"
                )
            except Exception as e:
                print(f"✗ Error loading CEDICT: {e}")
        else:
            print(f"⊘ CEDICT not found at {cedict_path}")

        return chinese_words

    def load_word_dict(self, name: str) -> set:
        path = os.path.join(self.dicts_dir, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                return set(line.strip() for line in f if line.strip())
        except Exception as e:
            print(f"✗ Error loading dictionary {path}: {e}")
            return set()

    def load_valid_pinyin_syllables(self) -> set:
        """
        Reuse the VALID_PINYIN syllable inventory from the 5th analysis script,
        which is the historical implementation used by the SuperBPE 4b metric.
        """
        path = os.path.join(self.base_dir, "5th_Analyzation for 16 tokenization.py")
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            match = re.search(r"VALID_PINYIN\s*=\s*set\((\[.*?\])\)", text, re.S)
            if not match:
                print(f"⊘ VALID_PINYIN not found in {path}")
                return set()
            return set(ast.literal_eval(match.group(1)))
        except Exception as e:
            print(f"⊘ Error loading VALID_PINYIN from {path}: {e}")
            return set()

    def detect_custom_type(self, tokenizer_key: str) -> str:
        key = tokenizer_key.lower()
        if "chinese_origin" in key:
            return "origin"
        if "pinyin_toned" in key:
            return "toned"
        if "pinyin_toneless" in key:
            return "toneless"
        if "pinyin_diacritic" in key:
            return "diacritic"
        return "unknown"

    @staticmethod
    def is_chinese_chars(token: str) -> bool:
        return bool(token) and all("一" <= ch <= "鿿" for ch in token)

    @staticmethod
    def normalize_pinyin_base(s: str) -> str:
        s = s.replace("ü", "v").replace("u:", "v")
        s = re.sub(r"[1-5]", "", s)
        return "".join(
            ch
            for ch in unicodedata.normalize("NFD", s)
            if unicodedata.category(ch) != "Mn"
        )

    @staticmethod
    def normalize_pinyin_preserve_numbers(s: str) -> str:
        s = s.replace("ü", "v").replace("u:", "v")
        s = re.sub(r"(?<=[A-Za-züÜv:])[1-5]\b", "", s)
        return "".join(
            ch
            for ch in unicodedata.normalize("NFD", s)
            if unicodedata.category(ch) != "Mn"
        )

    def is_valid_pinyin_syllable(self, s: str, valid_pinyin: set) -> bool:
        if any("A" <= ch <= "Z" for ch in s):
            return False
        return self.normalize_pinyin_base(s) in valid_pinyin

    @staticmethod
    def decode_token_id(tokenizer, token_id, fallback_token, cache):
        if token_id not in cache:
            try:
                cache[token_id] = tokenizer.decode([token_id])
            except Exception:
                cache[token_id] = fallback_token
        return cache[token_id]

    @staticmethod
    def clean_decoded_token(token: str) -> str:
        return token.replace("##", "").replace("Ġ", " ").strip()

    @staticmethod
    def compact_token(token: str) -> str:
        return re.sub(r"\s+", "", token)

    @staticmethod
    def is_punctuation_or_symbol(token: str) -> bool:
        if not token:
            return False
        for ch in token:
            if ch.isspace():
                continue
            cat = unicodedata.category(ch)
            if not (cat.startswith("P") or cat.startswith("S")):
                return False
        return True

    @staticmethod
    def has_cjk(token: str) -> bool:
        return any("一" <= ch <= "鿿" for ch in token)

    @staticmethod
    def has_latin(token: str) -> bool:
        return bool(re.search(r"[A-Za-z]", token))

    @staticmethod
    def is_chinese_numeric_time(token: str) -> bool:
        return bool(
            re.fullmatch(
                r"[第前后上下]?[0-9０-９一二三四五六七八九十百千万亿两〇零]+"
                r"(年|月|日|号|天|周|星期|世纪|年代|个|次|届|名|人|元|美元|%|％)?",
                token,
            )
            or re.fullmatch(
                r"[0-9０-９]{2,4}年([0-9０-９]{1,2}月([0-9０-９]{1,2}[日号])?)?",
                token,
            )
        )

    @staticmethod
    def build_chinese_substring_set(dictionary: set) -> set:
        substrings = set()
        for word in dictionary:
            if len(word) < 3:
                continue
            for start in range(len(word)):
                for end in range(start + 2, len(word) + 1):
                    if start == 0 and end == len(word):
                        continue
                    substrings.add(word[start:end])
        return substrings

    def can_segment_chinese_phrase(self, token: str, dictionary: set) -> bool:
        productive_single_morphemes = {
            "本", "篇", "系", "区", "道", "营", "省", "市", "县", "州",
            "镇", "村", "岛", "山", "河", "湖", "海", "港", "站", "线",
            "路", "桥", "场", "馆", "院", "校", "部", "局", "厅", "府",
            "会", "社", "队", "团", "派", "党", "军", "城", "国", "人",
            "者", "家", "员", "长", "师", "生", "书", "门", "下", "上",
            "中", "内", "外", "前", "后", "处", "里", "地", "位", "罪",
            "术", "型", "期", "卷", "集", "章", "节", "项", "目", "课",
        }
        function_prefixes = {
            "也", "亦", "并", "又", "再", "则", "但", "而", "且", "或",
            "和", "与", "及", "同", "向", "至", "于", "在", "为", "由",
            "被", "将", "以", "其", "该", "此", "每", "各", "全", "共",
            "可", "曾", "已", "未", "最", "更", "约", "近", "第",
        }
        function_suffixes = {
            "的", "了", "着", "过", "地", "得", "上", "下", "中",
            "里", "内", "外", "前", "后", "间", "者", "们", "为",
            "是", "有", "和", "与", "及", "或", "一个", "一种",
            "以来", "以上", "以下", "之间", "之一", "之中",
        }
        n = len(token)
        dp = [False] * (n + 1)
        pieces = [0] * (n + 1)
        dp[0] = True

        for i in range(n):
            if not dp[i]:
                continue
            for j in range(i + 1, n + 1):
                piece = token[i:j]
                is_final_function = j == n and piece in function_suffixes
                is_dictionary_piece = len(piece) >= 2 and piece in dictionary
                is_productive_single = len(piece) == 1 and piece in productive_single_morphemes
                is_function_prefix = piece in function_prefixes
                is_numeric_time = self.is_chinese_numeric_time(piece)
                if (
                    is_dictionary_piece
                    or is_final_function
                    or is_productive_single
                    or is_function_prefix
                    or is_numeric_time
                ):
                    if not dp[j] or pieces[i] + 1 < pieces[j]:
                        dp[j] = True
                        pieces[j] = pieces[i] + 1

        return dp[n] and pieces[n] >= 2

    def can_segment_chinese_lexical_compound(self, token: str, dictionary: set) -> bool:
        """
        Conservative SV expansion: accept compounds that are built from dictionary
        words plus productive bound morphemes, but not function-word phrases.
        """
        productive_single_morphemes = {
            "本", "篇", "系", "区", "道", "营", "省", "市", "县", "州",
            "镇", "村", "岛", "山", "河", "湖", "海", "港", "站", "线",
            "路", "桥", "场", "馆", "院", "校", "部", "局", "厅", "府",
            "会", "社", "队", "团", "派", "党", "军", "城", "国", "人",
            "者", "家", "员", "长", "师", "生", "书", "门", "下", "上",
            "中", "内", "外", "前", "后", "处", "里", "地", "位", "罪",
            "术", "型", "期", "卷", "集", "章", "节", "项", "目", "课",
        }
        if len(token) < 2 or len(token) > 8:
            return False

        n = len(token)
        dp = [False] * (n + 1)
        has_productive_single = [False] * (n + 1)
        pieces = [0] * (n + 1)
        dp[0] = True

        for i in range(n):
            if not dp[i]:
                continue
            for j in range(i + 1, n + 1):
                piece = token[i:j]
                is_dict_piece = len(piece) >= 2 and piece in dictionary
                is_productive_piece = len(piece) == 1 and piece in productive_single_morphemes
                if not (is_dict_piece or is_productive_piece):
                    continue
                candidate_pieces = pieces[i] + 1
                candidate_has_single = has_productive_single[i] or is_productive_piece
                if not dp[j] or candidate_pieces < pieces[j]:
                    dp[j] = True
                    pieces[j] = candidate_pieces
                    has_productive_single[j] = candidate_has_single

        return dp[n] and pieces[n] >= 2 and has_productive_single[n]

    def classify_chinese_vocab_token(
        self,
        clean_tok: str,
        dictionary: set,
        chinese_substrings: set,
    ) -> Tuple[str, str]:
        compact = self.compact_token(clean_tok)

        if not compact or self.is_punctuation_or_symbol(clean_tok):
            return "EX", "punctuation/symbol/empty token"

        if compact.isdigit():
            return "EX", "pure numeric inventory token"

        if self.is_chinese_numeric_time(compact):
            return "SV", "complete numeric/time expression"

        if not self.has_cjk(compact):
            return "EX", "non-CJK token excluded from Chinese morphology"

        if self.has_latin(compact):
            return "IV", "mixed Latin+CJK token"

        if compact in dictionary:
            return "SV", "CEDICT hit"

        if len(compact) == 1 and self.is_chinese_chars(compact):
            return "SV", "single CJK morpheme/character"

        function_suffixes = {"的", "了", "着", "过", "地", "得"}
        locative_suffixes = {"上", "下", "中", "里", "内", "外", "前", "后", "间"}
        if len(compact) >= 2:
            prefix, suffix = compact[:-1], compact[-1]
            if suffix in function_suffixes and (
                prefix in dictionary
                or len(prefix) == 1
                or prefix in {"是", "在", "为", "有", "与", "和", "及", "或"}
            ):
                return "WV", "content unit plus grammatical particle"
            if suffix in locative_suffixes and prefix in dictionary:
                return "WV", "content unit plus locative/range suffix"

            phrase_prefixes = {"的", "是", "在", "为", "有", "与", "和", "及", "或", "于"}
            phrase_suffixes = {"为", "是", "有", "于", "到", "至", "中", "上", "内", "外"}
            for prefix_marker in sorted(phrase_prefixes, key=len, reverse=True):
                rest = compact[len(prefix_marker):]
                if compact.startswith(prefix_marker) and rest:
                    if (
                        rest in dictionary
                        or self.is_chinese_numeric_time(rest)
                        or self.can_segment_chinese_phrase(rest, dictionary)
                    ):
                        return "WV", "function/predicate prefix plus known unit"
            for suffix_marker in sorted(phrase_suffixes, key=len, reverse=True):
                rest = compact[:-len(suffix_marker)]
                if compact.endswith(suffix_marker) and rest:
                    if rest in dictionary or self.can_segment_chinese_phrase(rest, dictionary):
                        return "WV", "known unit plus predicate/function suffix"

        if self.can_segment_chinese_phrase(compact, dictionary):
            return "WV", "decomposable into known Chinese units"

        if compact in chinese_substrings:
            return "IV", "proper substring of a longer dictionary word"

        return "IV", "not a dictionary unit, licensed phrase, or known morpheme"

    def parse_pinyin_syllables(self, clean_tok: str, valid_pinyin: set) -> List[str]:
        parts = [part.strip() for part in clean_tok.lower().split() if part.strip()]
        if parts and all(re.fullmatch(r"[a-zA-ZüÜ:]+[1-5]?", part) for part in parts):
            bases = [self.normalize_pinyin_base(part.lower()) for part in parts]
            if all(base in valid_pinyin for base in bases):
                return bases
            return []

        compact = self.normalize_pinyin_base(self.compact_token(clean_tok.lower()))
        if not compact or not re.fullmatch(r"[a-zv]+", compact):
            return []

        syllables = []
        i = 0
        max_len = max(len(s) for s in valid_pinyin) if valid_pinyin else 0
        while i < len(compact):
            best = None
            for j in range(min(len(compact), i + max_len), i, -1):
                candidate = compact[i:j]
                if candidate in valid_pinyin:
                    best = candidate
                    break
            if best is None:
                return []
            syllables.append(best)
            i += len(best)
        return syllables

    def classify_pinyin_numeric_token(self, clean_tok: str) -> Tuple[str, str]:
        normalized = self.normalize_pinyin_preserve_numbers(clean_tok.lower())
        normalized = re.sub(r"\s+", " ", normalized).strip()
        parts = normalized.split()

        if re.fullmatch(r"di\s*\d+", normalized):
            return "SV", "ordinal expression such as di 1"

        numeric_measure_units = {
            "ri", "hao", "yue", "nian", "shi", "shiji", "shi ji",
            "pingfangmi", "ping fang mi", "pingfangqianmi", "ping fang qian mi",
            "pingfanggongli", "ping fang gong li", "qianmi", "qian mi",
            "gongli", "gong li", "mi", "ren", "renci", "ren ci", "wei",
            "ci", "ge", "jie", "yuan", "wan", "yi",
        }
        if parts and any(part.isdigit() for part in parts):
            for idx, part in enumerate(parts):
                if not part.isdigit():
                    continue
                following = parts[idx + 1:]
                for length in range(1, min(4, len(following)) + 1):
                    unit = " ".join(following[:length])
                    compact_unit = "".join(following[:length])
                    if unit in numeric_measure_units or compact_unit in numeric_measure_units:
                        return "WV", "numeric expression followed by a complete measure/time unit"

        if re.fullmatch(
            r"\d{2,4}\s*nian(\s+\d{1,2}\s*yue(\s+\d{1,2}\s*(ri|hao))?)?",
            normalized,
        ):
            return "SV", "complete year/date expression"

        if re.fullmatch(r"\d+\s*(nian|yue|ri|hao|wan|yi|ge|ci|jie|ren|yuan)", normalized):
            return "SV", "complete numeric classifier expression"

        if re.fullmatch(r"yu\s+\d+", normalized):
            return "IV", "incomplete time phrase like yu 2013"

        if re.match(r"^nian\s+\d+", normalized):
            return "IV", "date fragment missing the leading year number"

        return "IV", "mixed digit+pinyin token without a complete numeric pattern"

    @staticmethod
    def looks_like_tone_numbered_pinyin(clean_tok: str) -> bool:
        if not re.search(r"\d", clean_tok):
            return False
        parts = [part.strip().lower() for part in clean_tok.split() if part.strip()]
        if not parts:
            return False
        if any(part.isdigit() for part in parts):
            return False
        if any(digit not in "12345" for digit in re.findall(r"\d", clean_tok)):
            return False
        return all(re.fullmatch(r"[a-zü:]+[1-5]?", part) for part in parts)

    def pinyin_sequence_has_known_decomposition(
        self,
        syllables: List[str],
        dictionary_base: set,
        function_syllables: set,
        locative_syllables: set,
    ) -> bool:
        n = len(syllables)
        dp = [False] * (n + 1)
        pieces = [0] * (n + 1)
        dp[0] = True
        for i in range(n):
            if not dp[i]:
                continue
            for j in range(i + 1, n + 1):
                piece = "".join(syllables[i:j])
                is_known = piece in dictionary_base
                is_final_function = j == n and piece in function_syllables
                is_final_locative = j == n and piece in locative_syllables
                if is_known or is_final_function or is_final_locative:
                    if not dp[j] or pieces[i] + 1 < pieces[j]:
                        dp[j] = True
                        pieces[j] = pieces[i] + 1
        return dp[n] and pieces[n] >= 2

    def pinyin_variants_for_chinese_word(self, word: str) -> Dict[str, str]:
        if not HAS_PYPINYIN or not word or not self.is_chinese_chars(word):
            return {}
        try:
            toned = "".join(
                item[0].lower()
                for item in pypinyin_pinyin(word, style=PinyinStyle.TONE3, strict=False)
            )
            toneless = "".join(lazy_pinyin(word, style=PinyinStyle.NORMAL, strict=False))
            diacritic = "".join(
                item[0].lower()
                for item in pypinyin_pinyin(word, style=PinyinStyle.TONE, strict=False)
            )
            return {
                "toned": toned,
                "toneless": toneless,
                "diacritic": diacritic,
            }
        except Exception:
            return {}

    def build_enhanced_chinese_lexicon(self, base_dictionary: set) -> Tuple[set, set]:
        enhanced = set(base_dictionary)
        added = set()
        chinese_keys = [
            key
            for key in self.tokenizers
            if "custom_" in key and self.detect_custom_type(key) == "origin"
        ]

        for tokenizer_key in chinese_keys:
            tokenizer = self.tokenizers[tokenizer_key]
            for raw_tok, token_id in tokenizer.get_vocab().items():
                decoded = self.decode_token_id(tokenizer, token_id, raw_tok, {})
                clean = self.clean_decoded_token(decoded)
                compact = self.compact_token(clean)
                if (
                    compact
                    and self.is_chinese_chars(compact)
                    and compact not in enhanced
                    and self.can_segment_chinese_lexical_compound(compact, base_dictionary)
                ):
                    enhanced.add(compact)
                    added.add(compact)

        print(
            f"✓ Enhanced Chinese lexicon: {len(base_dictionary)} base + "
            f"{len(added)} vocab-derived compounds = {len(enhanced)}"
        )
        return enhanced, added

    def classify_pinyin_vocab_token(
        self,
        clean_tok: str,
        dictionary_exact: set,
        dictionary_base: set,
        valid_pinyin: set,
    ) -> Tuple[str, str]:
        compact_exact = self.compact_token(clean_tok)
        compact_base = self.normalize_pinyin_base(compact_exact.lower())

        if not compact_exact or self.is_punctuation_or_symbol(clean_tok):
            return "EX", "punctuation/symbol/empty token"

        if re.fullmatch(r"\d+", compact_exact):
            return "EX", "pure numeric inventory token"

        if re.fullmatch(r"[a-zA-Z]", compact_exact):
            return "EX", "single Latin inventory letter"

        if re.search(r"\d", clean_tok) and not self.looks_like_tone_numbered_pinyin(clean_tok):
            if compact_exact in dictionary_exact or compact_base in dictionary_base:
                return "SV", "pinyin dictionary hit with digits"
            return self.classify_pinyin_numeric_token(clean_tok)

        if re.search(r"[A-Z]", clean_tok):
            return "EX", "foreign/cased Latin token excluded from pinyin morphology"

        if not re.fullmatch(r"[a-zA-ZüÜ:āáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜńňǹḿ1-5\s]+", clean_tok):
            return "EX", "contains non-pinyin characters"

        if compact_exact in dictionary_exact or compact_base in dictionary_base:
            return "SV", "pinyin dictionary hit"

        syllables = self.parse_pinyin_syllables(clean_tok, valid_pinyin)
        if not syllables:
            return "IV", "cannot parse into complete legal pinyin syllables"

        function_syllables = {
            "de", "le", "zhe", "guo", "di", "ge", "men", "zai", "he",
            "yu", "ji", "qi", "ba", "bei", "shi", "bu",
        }
        locative_syllables = {
            "shang", "xia", "zhong", "li", "nei", "wai", "qian", "hou", "jian",
        }

        if len(syllables) == 1 and syllables[0] in function_syllables:
            return "SV", "single function/common syllable"

        if self.pinyin_sequence_has_known_decomposition(
            syllables, dictionary_base, function_syllables, locative_syllables
        ):
            return "WV", "complete syllable sequence with plausible known-unit decomposition"

        if len(syllables) <= 4:
            return "WV", "complete legal pinyin syllable sequence with plausible reading"

        return "IV", "long legal syllable sequence without dictionary/decomposition evidence"

    def calculate_morphological_coherence(self) -> Dict:
        """
        4b. Vocabulary-level Morphological Coherence分析。
        对 tokenizer vocab 中每个 token type 分类为：
        SV strong_valid, WV weak_valid, IV invalid, EX excluded。
        分母为 SV + WV + IV；EX 只用于 coverage。
        """
        print("\n" + "=" * 80)
        print("4b. VOCABULARY-LEVEL MORPHOLOGICAL COHERENCE ANALYSIS")
        print("=" * 80)

        results = {}
        base_chinese_dictionary = self.load_cedict()
        enhanced_chinese_dictionary, enhanced_added_words = (
            self.build_enhanced_chinese_lexicon(base_chinese_dictionary)
        )
        dict_map = {
            "origin": enhanced_chinese_dictionary,
            "toned": self.load_word_dict("dict_toned.txt"),
            "toneless": self.load_word_dict("dict_toneless.txt"),
            "diacritic": self.load_word_dict("dict_diacritic.txt"),
        }
        enhanced_pinyin_additions = {"toned": 0, "toneless": 0, "diacritic": 0}
        for word in enhanced_added_words:
            variants = self.pinyin_variants_for_chinese_word(word)
            for key, value in variants.items():
                if value and value not in dict_map[key]:
                    dict_map[key].add(value)
                    enhanced_pinyin_additions[key] += 1

        print(
            "✓ Enhanced pinyin lexicons from same Chinese compounds: "
            + ", ".join(
                f"{key}+{count}" for key, count in enhanced_pinyin_additions.items()
            )
        )
        chinese_substrings = self.build_chinese_substring_set(dict_map["origin"])
        pinyin_base_dict_map = {
            key: {self.normalize_pinyin_base(item.lower()) for item in value}
            for key, value in dict_map.items()
            if key != "origin"
        }
        valid_pinyin = self.load_valid_pinyin_syllables()

        # 只分析自定义的tokenizers
        for tokenizer_key in self.tokenizers.keys():
            if "custom_" not in tokenizer_key:
                continue

            print(f"\nAnalyzing: {tokenizer_key}")

            tokenizer = self.tokenizers[tokenizer_key]
            tokenizer_type = self.detect_custom_type(tokenizer_key)
            dictionary = dict_map.get(tokenizer_type, set())
            dictionary_base = pinyin_base_dict_map.get(tokenizer_type, set())

            if tokenizer_type == "unknown":
                print(f"  Unknown tokenizer type, skipped: {tokenizer_key}")
                continue

            decoded_token_cache = {}
            label_counts = Counter()
            reason_counts = Counter()
            unit_counts = Counter()
            details = []

            vocab_items = sorted(tokenizer.get_vocab().items(), key=lambda x: x[1])

            for raw_tok, token_id in tqdm(
                vocab_items,
                desc=f"Vocab morphology {tokenizer_key}",
                unit="tok",
                leave=True,
            ):
                decoded_tok = self.decode_token_id(
                    tokenizer, token_id, raw_tok, decoded_token_cache
                )
                clean_tok = self.clean_decoded_token(decoded_tok)

                if tokenizer_type == "origin":
                    label, reason = self.classify_chinese_vocab_token(
                        clean_tok, dictionary, chinese_substrings
                    )
                    unit_count = len(self.compact_token(clean_tok))
                else:
                    label, reason = self.classify_pinyin_vocab_token(
                        clean_tok, dictionary, dictionary_base, valid_pinyin
                    )
                    syllables = self.parse_pinyin_syllables(clean_tok, valid_pinyin)
                    unit_count = len(syllables) if syllables else 0

                label_counts[label] += 1
                reason_counts[f"{label}: {reason}"] += 1
                unit_counts[f"{label}_unit_{unit_count}"] += 1

                details.append(
                    {
                        "token_id": token_id,
                        "raw_token": raw_tok,
                        "decoded_token": decoded_tok,
                        "clean_token": clean_tok,
                        "label": label,
                        "reason": reason,
                        "unit_count": unit_count,
                    }
                )

            samples = {}
            for label in ["SV", "WV", "IV", "EX"]:
                label_tokens = [
                    row["clean_token"]
                    for row in details
                    if row["label"] == label and row["clean_token"]
                ]
                rng = random.Random(
                    f"{self.morphology_sample_seed}:{tokenizer_key}:{label}"
                )
                if len(label_tokens) > 20:
                    samples[label] = rng.sample(label_tokens, 20)
                else:
                    samples[label] = label_tokens

            strong_valid_tokens = label_counts["SV"]
            weak_valid_tokens = label_counts["WV"]
            invalid_tokens = label_counts["IV"]
            excluded_tokens = label_counts["EX"]
            total_vocab_tokens = len(vocab_items)
            classified_tokens = strong_valid_tokens + weak_valid_tokens + invalid_tokens

            strict_valid_rate = (
                strong_valid_tokens / classified_tokens if classified_tokens else 0
            )
            inclusive_valid_rate = (
                (strong_valid_tokens + weak_valid_tokens) / classified_tokens
                if classified_tokens
                else 0
            )
            invalid_rate = invalid_tokens / classified_tokens if classified_tokens else 0
            coverage_over_vocab = (
                classified_tokens / total_vocab_tokens if total_vocab_tokens else 0
            )

            results[tokenizer_key] = {
                "type": tokenizer_type,
                "analysis_level": "vocabulary_token_types",
                "dictionary_source": "CEDICT + vocab-derived lexical compounds; pinyin derived with pypinyin",
                "enhanced_chinese_added_words": len(enhanced_added_words),
                "total_vocab_tokens": total_vocab_tokens,
                "classified_tokens": classified_tokens,
                "strong_valid_tokens": strong_valid_tokens,
                "weak_valid_tokens": weak_valid_tokens,
                "invalid_tokens": invalid_tokens,
                "excluded_tokens": excluded_tokens,
                "coherence_ratio": round(strict_valid_rate, 4),
                "strict_valid_rate": round(strict_valid_rate, 4),
                "inclusive_valid_rate": round(inclusive_valid_rate, 4),
                "invalid_rate": round(invalid_rate, 4),
                "coverage_over_vocab": round(coverage_over_vocab, 4),
                "top_reasons": reason_counts.most_common(12),
                "unit_bucket_counts": dict(sorted(unit_counts.items())),
                "sample_strong_valid_tokens": samples["SV"],
                "sample_weak_valid_tokens": samples["WV"],
                "sample_invalid_tokens": samples["IV"],
                "sample_excluded_tokens": samples["EX"],
            }
            self.morphology_token_details[tokenizer_key] = details

            print(
                f"  Type: {tokenizer_type}"
                f"\n  Total vocab tokens: {total_vocab_tokens}"
                f"\n  Classified tokens (SV+WV+IV): {classified_tokens}"
                f"\n  Strong valid tokens: {strong_valid_tokens}"
                f"\n  Weak valid tokens: {weak_valid_tokens}"
                f"\n  Invalid tokens: {invalid_tokens}"
                f"\n  Excluded tokens: {excluded_tokens}"
                f"\n  Strict valid rate: {strict_valid_rate:.2%}"
                f"\n  Inclusive valid rate: {inclusive_valid_rate:.2%}"
                f"\n  Coverage over vocab: {coverage_over_vocab:.2%}"
            )

        return results

    def calculate_compression_efficiency(self) -> Dict:
        """
        4d. Compression Efficiency分析
        计算bits-per-character和bytes-per-token
        """
        print("\n" + "=" * 80)
        print("4d. COMPRESSION EFFICIENCY ANALYSIS")
        print("=" * 80)

        results = {}

        for tokenizer_key in self.tokenizers.keys():
            total_tokens = 0
            total_chars = 0
            token_sizes = []
            test_data = self.get_test_data_for_tokenizer(tokenizer_key)

            for text in tqdm(
                test_data,
                desc=f"Compression {tokenizer_key}",
                unit="sent",
                leave=True,
            ):
                tokens = self.tokenize_text(text, tokenizer_key)
                total_tokens += len(tokens)
                total_chars += len(text)
                token_sizes.extend(tokens)

            if total_tokens > 0 and total_chars > 0:
                # 根据tokenizer_key提取vocab大小
                if "8k" in tokenizer_key:
                    vocab_size = 8000
                elif "16k" in tokenizer_key:
                    vocab_size = 16000
                elif "32k" in tokenizer_key:
                    vocab_size = 32000
                elif "64k" in tokenizer_key:
                    vocab_size = 64000
                else:
                    vocab_size = 32768  # 默认值
                
                # Bits per character
                bits_per_token = np.ceil(np.log2(vocab_size))
                bits_per_char = (bits_per_token * total_tokens) / total_chars

                # Bytes per token
                bytes_per_token = bits_per_token / 8

                results[tokenizer_key] = {
                    "total_tokens": total_tokens,
                    "total_chars": total_chars,
                    "vocab_size": vocab_size,
                    "bits_per_token": round(bits_per_token, 2),
                    "bits_per_char": round(bits_per_char, 4),
                    "bytes_per_token": round(bytes_per_token, 2),
                    "avg_token_id": round(np.mean(token_sizes), 2),
                }

                print(
                    f"\n{tokenizer_key}:"
                    f"\n  Vocab size: {vocab_size}"
                    f"\n  Bits per token: {bits_per_token:.2f}"
                    f"\n  Bits per character: {bits_per_char:.4f}"
                    f"\n  Bytes per token: {bytes_per_token:.2f}"
                )

        return results

    def compare_chinese_vs_english(self) -> Dict:
        """
        对比中文和英文的tokenization效率
        """
        print("\n" + "=" * 80)
        print("CHINESE vs ENGLISH COMPARISON")
        print("=" * 80)

        # 中文测试数据
        chinese_texts = self.test_data

        # 英文测试数据
        english_texts = [
            "This is a test sentence.",
            "China is a great country.",
            "Natural language processing is important.",
            "The Art of War is an ancient Chinese military treatise.",
        ]

        results = {}

        for tokenizer_key in self.tokenizers.keys():
            chinese_tokens = []
            english_tokens = []

            for text in chinese_texts:
                tokens = self.tokenize_text(text, tokenizer_key)
                chinese_tokens.extend(tokens)

            for text in english_texts:
                try:
                    tokens = self.tokenize_text(text, tokenizer_key)
                    english_tokens.extend(tokens)
                except:
                    pass

            if chinese_tokens and english_tokens:
                results[tokenizer_key] = {
                    "chinese_tokens": len(chinese_tokens),
                    "english_tokens": (
                        len(english_tokens) if english_tokens else 0
                    ),
                    "chinese_avg_token_length": round(
                        sum(
                            len(self.test_data[i])
                            for i in range(len(self.test_data))
                        )
                        / len(chinese_tokens),
                        4,
                    ),
                }

        return results

    def get_decoded_output_dir(self) -> str:
        return os.path.join(self.base_dir, "decoded_superTokenizers_2048_subset100k")

    @staticmethod
    def write_text(path: str, text: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    @staticmethod
    def write_json(path: str, data):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    @staticmethod
    def write_metrics_csv(path: str, results: Dict):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        scalar_keys = []
        for metrics in results.values():
            for key, value in metrics.items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    if key not in scalar_keys:
                        scalar_keys.append(key)

        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["tokenizer"] + scalar_keys)
            for tokenizer_key, metrics in sorted(results.items()):
                writer.writerow([tokenizer_key] + [metrics.get(key, "") for key in scalar_keys])

    @staticmethod
    def write_metrics_table(path: str, results: Dict):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not results:
            with open(path, "w", encoding="utf-8") as f:
                f.write("No results were generated in this run.\n")
            return

        preferred_keys = [
            "type",
            "total_vocab_tokens",
            "classified_tokens",
            "strong_valid_tokens",
            "weak_valid_tokens",
            "invalid_tokens",
            "excluded_tokens",
            "strict_valid_rate",
            "inclusive_valid_rate",
            "invalid_rate",
            "coverage_over_vocab",
            "avg_tokens_per_sentence",
            "avg_tokens_per_char",
            "compression_ratio",
            "bits_per_char",
            "bytes_per_token",
        ]
        scalar_keys = []
        for key in preferred_keys:
            if any(key in metrics for metrics in results.values()):
                scalar_keys.append(key)
        for metrics in results.values():
            for key, value in metrics.items():
                if key in scalar_keys:
                    continue
                if isinstance(value, (str, int, float, bool)) or value is None:
                    scalar_keys.append(key)

        headers = ["tokenizer"] + scalar_keys
        rows = []
        for tokenizer_key, metrics in sorted(results.items()):
            row = [tokenizer_key] + [metrics.get(key, "") for key in scalar_keys]
            rows.append([str(value) for value in row])

        widths = [
            max(len(header), *(len(row[i]) for row in rows))
            for i, header in enumerate(headers)
        ]
        sep = "+".join("-" * (width + 2) for width in widths)

        lines = []
        lines.append(sep)
        lines.append(
            "|".join(f" {header:<{widths[i]}} " for i, header in enumerate(headers))
        )
        lines.append(sep)
        for row in rows:
            lines.append(
                "|".join(f" {value:<{widths[i]}} " for i, value in enumerate(row))
            )
        lines.append(sep)

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def build_section_report(self, title: str, results: Dict) -> str:
        lines = [title, "=" * len(title), ""]
        if not results:
            lines.append("No results were generated in this run.")
            return "\n".join(lines) + "\n"

        for tokenizer_key, metrics in sorted(results.items()):
            lines.append(tokenizer_key.upper())
            lines.append("-" * 80)
            for key, value in metrics.items():
                if key.startswith("sample_"):
                    lines.append(f"{key}:")
                    for item in value[:20]:
                        lines.append(f"  - {item}")
                elif key == "top_reasons":
                    lines.append("top_reasons:")
                    for reason, count in value[:20]:
                        lines.append(f"  - {count}: {reason}")
                elif key == "unit_bucket_counts":
                    lines.append("unit_bucket_counts:")
                    for bucket, count in sorted(value.items()):
                        lines.append(f"  - {bucket}: {count}")
                else:
                    lines.append(f"{key}: {value}")
            lines.append("")
        return "\n".join(lines)

    def section_readme(self, section: str) -> str:
        if section == "4a":
            return """# 4.3 Fertility

This folder stores the standalone 4.3 fertility outputs in addition to the full comparison report.

Method memory:
- Fertility is an occurrence-level tokenization efficiency metric.
- The script tokenizes each tokenizer-specific test file and reports tokens per sample, tokens per surface character, total tokens, total characters, and chars/token.
- Earlier discussion noted that pinyin and Chinese surface strings have different character lengths, so 4.3 is useful for within-representation efficiency and should be read carefully for cross-representation claims.
- In the current fast 4B iteration run, 4.3 is intentionally disabled in run_full_analysis(); this folder may therefore contain only this README unless 4A is re-enabled.
"""
        if section == "4b":
            return """# 4.4 Morphological Coherence

This folder stores the standalone 4.4 vocabulary-level morphological coherence outputs.

Method memory:
- We moved from test-set dictionary hit rate to vocabulary-level token type quality.
- The research question is now: how many tokens in the tokenizer vocabulary are linguistically reasonable units?
- Labels:
  - SV = strong_valid: complete and natural lexical/morphemic token.
  - WV = weak_valid: understandable and linguistically plausible, but not an ideal standalone lexical token.
  - IV = invalid: arbitrary fragment, broken syllable, incomplete numeric/time phrase, or uninterpretable unit.
  - EX = excluded: punctuation, empty/control/symbol tokens, or out-of-scope foreign inventory items.
- Denominator for rates is SV + WV + IV, not vocab_size. Coverage reports (SV+WV+IV)/vocab_size.
- Main metrics:
  - strict_valid_rate = SV / (SV + WV + IV)
  - inclusive_valid_rate = (SV + WV) / (SV + WV + IV)
  - invalid_rate = IV / (SV + WV + IV)
  - coverage_over_vocab = (SV + WV + IV) / vocab_size

Chinese memory:
- CEDICT hits, single CJK morphemes, and complete numeric/time expressions are SV.
- Content unit + 的/了/着/过/地/得, locative/range suffixes, and natural multi-unit phrases are WV.
- Proper substrings of longer dictionary words are not automatically IV: if they can be naturally decomposed,
  they are WV; otherwise they are IV.
- Current large-dictionary trial uses CEDICT as the base lexicon, then adds tokenizer-vocab compounds
  that can be conservatively segmented into dictionary words plus productive bound morphemes. Examples:
  战俘营 = 战俘 + 营, 系教授 = 系 + 教授, 区道 = 区 + 道, 本篇 = 本 + 篇.
- The same enhanced Chinese compound source is converted with pypinyin into toned/toneless/diacritic
  pinyin entries, so Chinese and pinyin systems receive the same lexical-source expansion.
- Example decisions:
  - 中华人民共和国 -> SV
  - 中华人民共和国的 -> WV
  - 华人民共和国 -> IV, because it is a residual substring of 中华人民共和国.
  - 人民共和国 -> WV, because it can be decomposed as 人民 + 共和国 and appears naturally in XX人民共和国.
  - 的一个 / 是在 / 面积为 -> WV, because they are function/predicate combinations with a known unit.

Pinyin memory:
- We decided not to force pinyin tokens to uniquely map back to Chinese characters.
- Pinyin tokens are judged by complete syllable integrity plus plausible linguistic evidence.
- Ambiguity is acceptable if at least one reasonable Chinese reading exists.
- Single Latin inventory letters such as m/p/q/t/w are EX even if they appear in the pinyin dictionary,
  because they are better treated as base inventory symbols than morphological pinyin units.
- Example decisions from manual review:
  - zui hou yi likely 最后一; the label can be WV even if a naive splitter proposes the wrong split.
  - dui huo de can be 对获得.
  - huai yi shi is 怀疑是.
  - wang zi de is 王子的.
  - jiang zi ji is 将自己.
  - tui te shang is 推特上.
  - jin zhong jiang may be 仅中奖 or 金钟奖, so ambiguity alone is not invalid.
  - di 1 can be 第1 and is acceptable.
  - 1 ri ping jun shang xia che ren ci wei, zai 20 shi ji, and
    50 ping fang qian mi are WV because the number is followed by a complete
    time/measure unit such as 日、世纪、平方千米.
  - yu 2013 is incomplete because it lacks 年 or another time unit, so IV.
  - nian 6 yue 2 ri is incomplete because it lacks the leading year number, so IV.

This is a reproducible heuristic, not a claim that morphology has a single universal tokenizer-validity standard.
The TSV files in this folder are meant for manual inspection and future rule tightening.
The metrics_table.txt file is the human-readable table version of metrics.csv.

Manual audit memory:
- manual_audit_sample.csv and manual_audit_sample.html sample up to 100 tokens from
  each automatic SV/WV/IV group for each tokenizer.
- With four custom tokenizers, the expected maximum is 100 * 3 * 4 = 1200 rows.
- The HTML table gives a single-select manual label control: SV, WV, IV, or UNK.
- UNK is for cases that are plausible but genuinely uncertain and should not be
  forced into a hard SV/WV/IV decision during a quick pass.
- We sample all three automatic groups because the heuristic can miss valid tokens
  in IV, but it can also overestimate SV/WV. This audit is meant to estimate both
  false negatives and false positives.
- The HTML page stores edits in browser localStorage; use Download CSV to export
  the reviewed labels.
"""
        if section == "4d":
            return """# 4.5 Compression Efficiency

This folder stores the standalone 4.5 compression efficiency outputs in addition to the full comparison report.

Method memory:
- The current script calls this function compression_efficiency; in the paper outline it corresponds to 4.5.
- It reports vocab-size-derived bits/token, bits/character, bytes/token, and average token id over each tokenizer-specific test file.
- Earlier discussion separated this from fertility: fertility counts tokenizer output length, while compression efficiency translates that length through an assumed fixed-width token id cost.
- In the current fast 4B iteration run, 4.5 is intentionally disabled in run_full_analysis(); this folder may therefore contain only this README unless 4D is re-enabled.
"""
        return ""

    def write_standalone_metric_outputs(self):
        output_root = self.get_decoded_output_dir()
        sections = [
            (
                "4a",
                "4.3_fertility",
                "4.3 FERTILITY",
                self.results.get("fertility", {}),
            ),
            (
                "4b",
                "4.4_morphological_coherence",
                "4.4 MORPHOLOGICAL COHERENCE",
                self.results.get("morphological_coherence", {}),
            ),
            (
                "4d",
                "4.5_compression_efficiency",
                "4.5 COMPRESSION EFFICIENCY",
                self.results.get("compression_efficiency", {}),
            ),
        ]

        for section_key, dirname, title, section_results in sections:
            section_dir = os.path.join(output_root, dirname)
            os.makedirs(section_dir, exist_ok=True)
            self.write_text(
                os.path.join(section_dir, "README.md"),
                self.section_readme(section_key),
            )
            self.write_text(
                os.path.join(section_dir, "report.txt"),
                self.build_section_report(title, section_results),
            )
            self.write_json(
                os.path.join(section_dir, "metrics.json"),
                section_results,
            )
            self.write_metrics_table(
                os.path.join(section_dir, "metrics_table.txt"),
                section_results,
            )
            if section_results:
                self.write_metrics_csv(
                    os.path.join(section_dir, "metrics.csv"),
                    section_results,
                )

        morph_dir = os.path.join(output_root, "4.4_morphological_coherence")
        for tokenizer_key, rows in sorted(self.morphology_token_details.items()):
            tsv_path = os.path.join(morph_dir, f"{tokenizer_key}_token_classification.tsv")
            with open(tsv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "token_id",
                        "raw_token",
                        "decoded_token",
                        "clean_token",
                        "label",
                        "reason",
                        "unit_count",
                    ],
                    delimiter="\t",
                )
                writer.writeheader()
                writer.writerows(rows)

        self.write_manual_morphology_audit(morph_dir)

    def write_manual_morphology_audit(self, morph_dir: str, sample_per_label: int = 100):
        labels = ["SV", "WV", "IV"]
        audit_rows = []

        for tokenizer_key, rows in sorted(self.morphology_token_details.items()):
            for label in labels:
                candidates = [row for row in rows if row["label"] == label]
                rng = random.Random(
                    f"{self.morphology_sample_seed}:manual-audit:{tokenizer_key}:{label}"
                )
                selected = (
                    rng.sample(candidates, sample_per_label)
                    if len(candidates) > sample_per_label
                    else candidates
                )
                for row in selected:
                    audit_rows.append(
                        {
                            "tokenizer": tokenizer_key,
                            "sample_group": label,
                            "token_id": row["token_id"],
                            "clean_token": row["clean_token"],
                            "auto_label": row["label"],
                            "manual_label": row["label"],
                            "reason": row["reason"],
                            "unit_count": row["unit_count"],
                            "comment": "",
                        }
                    )

        csv_path = os.path.join(morph_dir, "manual_audit_sample.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "tokenizer",
                    "sample_group",
                    "token_id",
                    "clean_token",
                    "auto_label",
                    "manual_label",
                    "reason",
                    "unit_count",
                    "comment",
                ],
            )
            writer.writeheader()
            writer.writerows(audit_rows)

        html_rows = []
        label_options = ["SV", "WV", "IV", "UNK"]
        for idx, row in enumerate(audit_rows, 1):
            options = "".join(
                f'<option value="{option}" {"selected" if option == row["auto_label"] else ""}>{option}</option>'
                for option in label_options
            )
            html_rows.append(
                "<tr>"
                f"<td>{idx}</td>"
                f"<td>{row['tokenizer']}</td>"
                f"<td>{row['sample_group']}</td>"
                f"<td>{row['token_id']}</td>"
                f"<td class=\"token\">{self.html_escape(str(row['clean_token']))}</td>"
                f"<td>{row['auto_label']}</td>"
                f"<td><select data-row=\"{idx}\" data-field=\"manual_label\">{options}</select></td>"
                f"<td>{self.html_escape(str(row['reason']))}</td>"
                f"<td>{row['unit_count']}</td>"
                f"<td><input data-row=\"{idx}\" data-field=\"comment\" type=\"text\" /></td>"
                "</tr>"
            )

        html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>4.4 Morphological Coherence Manual Audit</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #d0d7de; padding: 6px 8px; vertical-align: top; }}
    th {{ position: sticky; top: 0; background: #f6f8fa; z-index: 1; }}
    .token {{ font-size: 16px; font-weight: 600; min-width: 120px; }}
    select, input {{ width: 100%; box-sizing: border-box; }}
    .controls {{ display: flex; gap: 8px; align-items: center; margin: 12px 0 18px; }}
    button {{ padding: 6px 10px; }}
    .note {{ color: #57606a; max-width: 920px; line-height: 1.4; }}
  </style>
</head>
<body>
  <h1>4.4 Morphological Coherence Manual Audit</h1>
  <p class="note">
    This table samples up to {sample_per_label} tokens from each automatic SV/WV/IV group for each tokenizer.
    The dropdown defaults to the automatic label. Use UNK for plausible but uncertain tokens that should not be forced into SV/WV/IV.
    Changes are stored in this browser via localStorage. Use Download CSV to export your labels.
  </p>
  <div class="controls">
    <button onclick="downloadCsv()">Download CSV</button>
    <button onclick="clearSaved()">Clear saved choices</button>
    <span id="status"></span>
  </div>
  <table id="audit-table">
    <thead>
      <tr>
        <th>#</th><th>tokenizer</th><th>sample_group</th><th>token_id</th><th>token</th>
        <th>auto</th><th>manual</th><th>reason</th><th>units</th><th>comment</th>
      </tr>
    </thead>
    <tbody>
      {''.join(html_rows)}
    </tbody>
  </table>
  <script>
    const storageKey = "morphology_manual_audit_2048_64k_v1";
    const baseRows = {json.dumps(audit_rows, ensure_ascii=False)};

    function loadState() {{
      const saved = JSON.parse(localStorage.getItem(storageKey) || "{{}}");
      document.querySelectorAll("[data-row]").forEach(el => {{
        const row = el.dataset.row;
        const field = el.dataset.field;
        if (saved[row] && saved[row][field] !== undefined) {{
          el.value = saved[row][field];
        }}
      }});
      updateStatus();
    }}

    function saveField(el) {{
      const saved = JSON.parse(localStorage.getItem(storageKey) || "{{}}");
      const row = el.dataset.row;
      const field = el.dataset.field;
      saved[row] = saved[row] || {{}};
      saved[row][field] = el.value;
      localStorage.setItem(storageKey, JSON.stringify(saved));
      updateStatus();
    }}

    function currentRows() {{
      const saved = JSON.parse(localStorage.getItem(storageKey) || "{{}}");
      return baseRows.map((row, i) => {{
        const key = String(i + 1);
        return {{
          ...row,
          manual_label: saved[key]?.manual_label ?? row.manual_label,
          comment: saved[key]?.comment ?? row.comment
        }};
      }});
    }}

    function csvEscape(value) {{
      const s = String(value ?? "");
      return /[",\\n]/.test(s) ? '"' + s.replaceAll('"', '""') + '"' : s;
    }}

    function downloadCsv() {{
      const rows = currentRows();
      const headers = Object.keys(rows[0]);
      const csv = [headers.join(",")]
        .concat(rows.map(row => headers.map(h => csvEscape(row[h])).join(",")))
        .join("\\n");
      const blob = new Blob([csv + "\\n"], {{ type: "text/csv;charset=utf-8" }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "manual_audit_labels.csv";
      a.click();
      URL.revokeObjectURL(url);
    }}

    function clearSaved() {{
      if (confirm("Clear all saved manual labels and comments?")) {{
        localStorage.removeItem(storageKey);
        location.reload();
      }}
    }}

    function updateStatus() {{
      const rows = currentRows();
      const changed = rows.filter(row => row.manual_label !== row.auto_label || row.comment).length;
      document.getElementById("status").textContent = `${{changed}} edited / ${{rows.length}} rows`;
    }}

    document.querySelectorAll("[data-row]").forEach(el => {{
      el.addEventListener("change", () => saveField(el));
      el.addEventListener("input", () => saveField(el));
    }});
    loadState();
  </script>
</body>
</html>
"""
        self.write_text(os.path.join(morph_dir, "manual_audit_sample.html"), html)

    @staticmethod
    def html_escape(value: str) -> str:
        return (
            value.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    def generate_report(
        self,
        output_file: str = "decoded_superTokenizers_2048_subset100k/tokenizer_comparison_report.txt",
    ):
        """生成完整的对比报告"""
        print("\n" + "=" * 80)
        print("GENERATING COMPREHENSIVE REPORT...")
        print("=" * 80)

        report = []
        report.append("=" * 100)
        report.append(
            "TOKENIZER COMPARISON REPORT: 2048 SuperBPE 64K Custom Tokenizers vs. Production LLM Tokenizers"
        )
        report.append("=" * 100)
        report.append("")

        # 基本信息
        report.append("TEST DATA SUMMARY BY TOKENIZER")
        report.append("-" * 100)
        for tokenizer_key in sorted(self.tokenizers.keys()):
            test_file = self.get_test_file_for_tokenizer(tokenizer_key)
            texts = self.get_test_data_for_tokenizer(tokenizer_key)
            total_chars = sum(len(t) for t in texts)
            avg_sentence_length = total_chars / len(texts) if texts else 0
            report.append(f"{tokenizer_key}:")
            report.append(f"  Test file: {test_file}")
            report.append(f"  Number of test sentences: {len(texts)}")
            report.append(f"  Total characters: {total_chars}")
            report.append(f"  Average sentence length: {avg_sentence_length:.2f} chars")
        report.append("")

        # Fertility分析
        report.append("=" * 100)
        report.append("4A. FERTILITY ANALYSIS (Tokens-per-sentence & Tokens-per-character)")
        report.append("=" * 100)
        report.append("")

        fertility_results = self.results.get("fertility", {})
        for tokenizer_key, metrics in sorted(fertility_results.items()):
            report.append(f"{tokenizer_key.upper()}")
            report.append("-" * 50)
            for key, value in metrics.items():
                report.append(f"  {key}: {value}")
            report.append("")

        # Morphological Coherence分析
        report.append("=" * 100)
        report.append("4B. MORPHOLOGICAL COHERENCE ANALYSIS")
        report.append("=" * 100)
        report.append("")

        morpho_results = self.results.get("morphological_coherence", {})
        for tokenizer_key, metrics in sorted(morpho_results.items()):
            report.append(f"{tokenizer_key.upper()}")
            report.append("-" * 50)
            for key, value in metrics.items():
                if key in {
                    "sample_meaningful_tokens",
                    "sample_valid_tokens",
                    "sample_strong_valid_tokens",
                    "sample_weak_valid_tokens",
                    "sample_invalid_tokens",
                    "sample_excluded_tokens",
                }:
                    report.append(f"  {key}:")
                    report.append(f"    {value[:10]}")
                elif key in {"top_reasons"}:
                    report.append(f"  {key}:")
                    for reason, count in value[:10]:
                        report.append(f"    {count}: {reason}")
                else:
                    report.append(f"  {key}: {value}")
            report.append("")

        # Compression Efficiency分析
        report.append("=" * 100)
        report.append("4D. COMPRESSION EFFICIENCY ANALYSIS")
        report.append("=" * 100)
        report.append("")

        compression_results = self.results.get("compression_efficiency", {})
        for tokenizer_key, metrics in sorted(compression_results.items()):
            report.append(f"{tokenizer_key.upper()}")
            report.append("-" * 50)
            for key, value in metrics.items():
                report.append(f"  {key}: {value}")
            report.append("")

        # 总结
        report.append("=" * 100)
        report.append("SUMMARY & FINDINGS")
        report.append("=" * 100)
        report.append("")

        if fertility_results:
            best_compression = max(
                fertility_results.items(),
                key=lambda x: x[1].get("compression_ratio", 0),
            )
            report.append(f"Best compression ratio: {best_compression[0]}")
            report.append(f"  Compression ratio (chars/token): {best_compression[1].get('compression_ratio', 'N/A')}")
            report.append("")

        if morpho_results:
            best_coherence = max(
                morpho_results.items(),
                key=lambda x: x[1].get("coherence_ratio", 0),
            )
            report.append(f"Best morphological coherence: {best_coherence[0]}")
            report.append(f"  Coherence ratio: {best_coherence[1].get('coherence_ratio', 'N/A')}")
            report.append("")

        report.append("")
        report.append("=" * 100)
        report.append("END OF REPORT")
        report.append("=" * 100)

        # 保存到文件
        report_text = "\n".join(report)

        output_path = os.path.join(
            os.path.dirname(self.test_file), "..", output_file
        )

        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report_text)
            print(f"\n✓ Report saved to: {output_path}")
            self.write_standalone_metric_outputs()
            print(f"✓ Standalone metric outputs saved under: {self.get_decoded_output_dir()}")
        except Exception as e:
            print(f"✗ Error saving report: {e}")
            print("Report content:")
            print(report_text)

        return report_text

    def run_full_analysis(self, max_test_lines: int = None):
        """运行完整的分析流程"""
        print("\n")
        print("╔" + "=" * 98 + "╗")
        print("║" + " " * 98 + "║")
        print("║" + "2048 SUPERBPE 64K TOKENIZER COMPARISON ANALYSIS - FULL PIPELINE".center(98) + "║")
        print("║" + " " * 98 + "║")
        print("╚" + "=" * 98 + "╝")

        # 1. 加载tokenizers
        self.load_custom_tokenizers()
        # 2026-04-29: 4b currently only evaluates custom_* tokenizers, so skip
        # AI tokenizer loading while iterating on Morphological Coherence.
        # self.load_ai_tokenizers()

        # 2. 加载测试数据
        self.load_test_data(max_lines=max_test_lines)

        if len(self.test_data) == 0:
            print("\n✗ No test data loaded. Exiting.")
            return

        # 3. 运行分析
        # 2026-04-29: temporarily disable 4a/4d to speed up 4b iteration.
        # self.results["fertility"] = self.calculate_fertility()
        self.results["morphological_coherence"] = (
            self.calculate_morphological_coherence()
        )
        # self.results["compression_efficiency"] = (
        #     self.calculate_compression_efficiency()
        # )

        # 4. 生成报告
        report = self.generate_report()

        print("\n" + "=" * 100)
        print("ANALYSIS COMPLETE!")
        print("=" * 100)


def main():
    """主函数"""
    base_dir = "/Users/crisp/Desktop/code_field/python/Chinese_Latinization_NLP/1.Tokenization"

    # 初始化对比器
    comparison = TokenizerComparison(
        base_dir=base_dir,
        test_file=os.path.join(
            base_dir, "corpora", "chinese_origin_中国_test10.txt"
        ),
    )

    # 运行完整分析（不限制，使用全部测试数据）
    comparison.run_full_analysis(max_test_lines=None)


if __name__ == "__main__":
    main()
