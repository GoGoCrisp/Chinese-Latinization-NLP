from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import os

# ===== 路径配置 =====

CORPUS_DIR = "./corpora"
OUTPUT_DIR = "./tokenizers"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===== 训练函数 =====

def train_bpe(corpus_path, vocab_size, output_path):
    tokenizer = Tokenizer(models.BPE())

    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]"]
    )

    print(f"Training on {corpus_path}")
    print(f"Vocab size: {vocab_size}")

    tokenizer.train([corpus_path], trainer)

    tokenizer.save(output_path)

    print(f"Saved → {output_path}\n")


# ===== corpus（只用 train90） =====

corpora = {
     "chinese_origin": f"{CORPUS_DIR}/chinese_origin_中国_train90.txt",
     "pinyin_toned": f"{CORPUS_DIR}/pinyin_toned_zhong1guo2_train90.txt",
     "pinyin_toneless": f"{CORPUS_DIR}/pinyin_toneless_zhongguo_train90.txt",
     "pinyin_diacritic": f"{CORPUS_DIR}/pinyin_diacritic_zhōngguó_train90.txt"
}

vocab_sizes = [8000, 16000, 32000, 64000]


# ===== 批量训练 =====

for name, path in corpora.items():
    if not os.path.exists(path):
        print(f"⚠️ Missing file: {path}")
        continue

    for vocab_size in vocab_sizes:
        output_path = os.path.join(
            OUTPUT_DIR,
            f"{name}_{vocab_size//1000}k_train90.json"
        )

        train_bpe(path, vocab_size, output_path)

print("✅ Tokenizer training complete!")