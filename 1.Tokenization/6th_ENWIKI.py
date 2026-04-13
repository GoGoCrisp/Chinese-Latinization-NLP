
import bz2
import xml.etree.ElementTree as ET
import re
import mwparserfromhell
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFKC
import os
from tqdm import tqdm





# =========================
# 清洗 Wikipedia 数据
# =========================
def clean_wiki_text(text):
    """Clean Wikipedia markup"""
    if not text:
        return ""

    # remove templates / links / formatting
    try:
        wikicode = mwparserfromhell.parse(text)
        text = wikicode.strip_code()
    except:
        pass

    # remove refs like [1], [12]
    text = re.sub(r"\[[0-9]+\]", " ", text)

    # remove leftover markup artifacts
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def extract_enwiki_text(xml_bz2_path, output_txt_path, max_chars=None):
    """
    Stream Wikipedia dump and extract clean text, while limiting the number of characters
    """
    print("Reading:", xml_bz2_path)

    context = ET.iterparse(bz2.open(xml_bz2_path, "rb"), events=("end",))
    _, root = next(context)

    count = 0
    total_chars = 0  # 计算总字符数

    with open(output_txt_path, "w", encoding="utf-8") as out:
        for event, elem in tqdm(context, desc="Extracting Wikipedia"):

            if elem.tag.endswith("text"):
                if elem.text:
                    text = clean_wiki_text(elem.text)

                    # filter too short or noisy docs
                    if len(text) > 200:
                        out.write(text + "\n")
                        total_chars += len(text)

                        if max_chars and total_chars >= max_chars:
                            print(f"Reached max characters: {max_chars}")
                            break

            if elem.tag.endswith("page"):
                count += 1
                root.clear()

    print(f"Finished extracting {count} pages, saved to {output_txt_path}")


# =========================
# 训练 BPE tokenizer
# =========================
def train_bpe_tokenizer(corpus_path, vocab_size, save_path):
    """
    Train BPE tokenizer on English Wikipedia text
    """
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.normalizer = NFKC()
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"]
    )

    print(f"Training BPE vocab_size={vocab_size}")
    tokenizer.train([corpus_path], trainer)
    tokenizer.save(save_path)
    print("Saved:", save_path)


# =========================
# 主流程：从 bz2 提取并训练 tokenizers
# =========================
BASE_DIR = "/Users/crisp/Desktop/code_field/python/Chinese_Latinization_NLP/1.Tokenization"

RAW_XML = os.path.join(BASE_DIR, "enwiki-20260401-pages-articles-multistream.xml.bz2")  # 你的 enwiki 文件路径
CLEAN_TXT = os.path.join(BASE_DIR, "corpora/enwiki_clean.txt")

TOKENIZER_DIR = os.path.join(BASE_DIR, "tokenizers")

VOCAB_SIZES = [8000, 16000, 32000, 64000]

# Step 1: 提取和清洗
if not os.path.exists(CLEAN_TXT):
    extract_enwiki_text(
        xml_bz2_path=RAW_XML,
        output_txt_path=CLEAN_TXT,
          # 限制英文字符数与中文数据量一致
    )

# Step 2: 训练 BPE tokenizer
for vocab in VOCAB_SIZES:
    save_path = os.path.join(TOKENIZER_DIR, f"enwiki_bpe_{vocab}.json")
    train_bpe_tokenizer(
        corpus_path=CLEAN_TXT,
        vocab_size=vocab,
        save_path=save_path
    )