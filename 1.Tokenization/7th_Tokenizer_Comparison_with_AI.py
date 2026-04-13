"""
Tokenizer Comparison: 32K Custom Tokenizers vs. Production LLM Tokenizers
对比内容：
- 4a. Fertility (肥沃度): tokens-per-sentence, tokens-per-character
- 4b. Morphological coherence (形态学连贯性): BPE merges是否对应有意义的词素
- 4c. Compression efficiency (压缩效率): bits-per-character, bytes-per-token
"""

import json
import os
import re
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import numpy as np
from pathlib import Path

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

import json
from tokenizers import Tokenizer as HFTokenizer

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
        self.test_data = []  # 测试数据

    def load_custom_tokenizers(self):
        """加载本地训练的32k tokenizers"""
        print("=" * 80)
        print("Loading Custom 32k Tokenizers...")
        print("=" * 80)

        tokenizer_names = [
            "chinese_origin_32k_train90.json",
            "pinyin_diacritic_32k_train90.json",
            "pinyin_toned_32k_train90.json",
            "pinyin_toneless_32k_train90.json",
        ]

        for name in tokenizer_names:
            path = os.path.join(self.tokenizers_dir, name)
            if os.path.exists(path):
                try:
                    tokenizer = HFTokenizer.from_file(path)
                    key = name.replace("_32k_train90.json", "")
                    self.tokenizers[f"custom_{key}"] = tokenizer
                    print(f"✓ Loaded: {key}")
                except Exception as e:
                    print(f"✗ Failed to load {name}: {e}")
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
                print("✓ Loaded: GPT-4 (cl100k_base)")
            except Exception as e:
                print(f"✗ Failed to load GPT-4 tokenizer: {e}")
        else:
            print("⊘ Skipped: GPT-4 (tiktoken not installed)")

        # Llama-3
        if HAS_TRANSFORMERS:
            llama_models = [
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
                print("✓ Loaded: Qwen (with auth token)")
            except Exception as e:
                print(f"⊘ Qwen not available: {e}")
                print("  Note: Qwen models may require specific access.")

    def load_test_data(self, max_lines: int = None):
        """
        加载测试数据
        由于测试文件很大，按行读取
        """
        print("\n" + "=" * 80)
        print(f"Loading test data from: {self.test_file}")
        print("=" * 80)

        if not os.path.exists(self.test_file):
            print(f"✗ Test file not found: {self.test_file}")
            print("  Creating sample test data...")
            # 创建示例数据
            self.test_data = [
                "这是一个测试句子。",
                "中国是一个伟大的国家。",
                "自然语言处理很重要。",
                "孙子兵法是中国古代军事理论著作。",
            ]
        else:
            try:
                with open(self.test_file, "r", encoding="utf-8") as f:
                    for i, line in enumerate(f):
                        line = line.strip()
                        if line:
                            self.test_data.append(line)
                            if max_lines and i >= max_lines - 1:
                                break
                print(f"✓ Loaded {len(self.test_data)} test sentences")
                if len(self.test_data) > 0:
                    print(f"  Sample: {self.test_data[0][:50]}...")
            except Exception as e:
                print(f"✗ Error loading test data: {e}")

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
            tokens_per_sentence = []
            tokens_per_char = []
            total_chars = 0
            total_tokens = 0

            for text in self.test_data:
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

    def calculate_morphological_coherence(self) -> Dict:
        """
        4b. Morphological Coherence分析
        检查BPE合并是否对应有意义的中文词素
        """
        print("\n" + "=" * 80)
        print("4b. MORPHOLOGICAL COHERENCE ANALYSIS")
        print("=" * 80)

        results = {}
        chinese_words = self.load_cedict()

        # 只分析自定义的tokenizers
        for tokenizer_key in self.tokenizers.keys():
            if "custom_" not in tokenizer_key:
                continue

            print(f"\nAnalyzing: {tokenizer_key}")

            tokenizer = self.tokenizers[tokenizer_key]

            # 获取vocabulary
            try:
                vocab = tokenizer.get_vocab()
                print(f"  Vocabulary size: {len(vocab)}")
            except Exception as e:
                print(f"  Error getting vocabulary: {e}")
                continue

            # 分析multi-character tokens
            multi_char_tokens = []
            meaningful_tokens = 0
            total_multi_char = 0

            for token_str, token_id in vocab.items():
                # 移除特殊字符前缀（##等）
                clean_token = token_str.replace("##", "").replace("Ġ", "")

                if len(clean_token) > 1:
                    total_multi_char += 1

                    # 检查是否是有意义的中文词
                    if clean_token in chinese_words:
                        meaningful_tokens += 1
                        multi_char_tokens.append(token_str)

            if total_multi_char > 0:
                coherence_ratio = meaningful_tokens / total_multi_char
            else:
                coherence_ratio = 0

            results[tokenizer_key] = {
                "total_tokens": len(vocab),
                "multi_char_tokens": total_multi_char,
                "meaningful_tokens": meaningful_tokens,
                "coherence_ratio": round(coherence_ratio, 4),
                "sample_meaningful_tokens": multi_char_tokens[:20],
            }

            print(
                f"  Multi-char tokens: {total_multi_char}"
                f"\n  Meaningful tokens: {meaningful_tokens}"
                f"\n  Morphological coherence: {coherence_ratio:.2%}"
            )

        return results

    def calculate_compression_efficiency(self) -> Dict:
        """
        4c. Compression Efficiency分析
        计算bits-per-character和bytes-per-token
        """
        print("\n" + "=" * 80)
        print("4c. COMPRESSION EFFICIENCY ANALYSIS")
        print("=" * 80)

        results = {}

        for tokenizer_key in self.tokenizers.keys():
            total_tokens = 0
            total_chars = 0
            token_sizes = []

            for text in self.test_data:
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

    def generate_report(self, output_file: str = "tokenizer_comparison_report.txt"):
        """生成完整的对比报告"""
        print("\n" + "=" * 80)
        print("GENERATING COMPREHENSIVE REPORT...")
        print("=" * 80)

        report = []
        report.append("=" * 100)
        report.append(
            "TOKENIZER COMPARISON REPORT: 32K Custom Tokenizers vs. Production LLM Tokenizers"
        )
        report.append("=" * 100)
        report.append("")

        # 基本信息
        report.append("TEST DATA SUMMARY")
        report.append("-" * 100)
        report.append(f"Test file: {self.test_file}")
        report.append(f"Number of test sentences: {len(self.test_data)}")
        if len(self.test_data) > 0:
            total_chars = sum(len(t) for t in self.test_data)
            avg_sentence_length = total_chars / len(self.test_data)
            report.append(f"Total characters: {total_chars}")
            report.append(f"Average sentence length: {avg_sentence_length:.2f} chars")
            report.append(f"Sample sentences:")
            for i, text in enumerate(self.test_data[:3]):
                report.append(f"  {i+1}. {text}")
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
                if key == "sample_meaningful_tokens":
                    report.append(f"  {key}:")
                    report.append(f"    {value[:10]}")
                else:
                    report.append(f"  {key}: {value}")
            report.append("")

        # Compression Efficiency分析
        report.append("=" * 100)
        report.append("4C. COMPRESSION EFFICIENCY ANALYSIS")
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
        print("║" + "32K TOKENIZER COMPARISON ANALYSIS - FULL PIPELINE".center(98) + "║")
        print("║" + " " * 98 + "║")
        print("╚" + "=" * 98 + "╝")

        # 1. 加载tokenizers
        self.load_custom_tokenizers()
        self.load_ai_tokenizers()

        # 2. 加载测试数据
        self.load_test_data(max_lines=max_test_lines)

        if len(self.test_data) == 0:
            print("\n✗ No test data loaded. Exiting.")
            return

        # 3. 运行分析
        self.results["fertility"] = self.calculate_fertility()
        self.results["morphological_coherence"] = (
            self.calculate_morphological_coherence()
        )
        self.results["compression_efficiency"] = (
            self.calculate_compression_efficiency()
        )

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
