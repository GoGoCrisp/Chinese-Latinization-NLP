from tokenizers import Tokenizer
import os

# =========================
# CONFIG
# =========================

VOCAB_SIZE_CHECK = 64000   # 👈 改这里
TOKENIZER_DIR = "./tokenizers"
MAX_VOCAB_SAMPLE = 20

# =========================
# 4种输入（严格按你的定义）
# =========================

TEST_TEXTS = {
    "chinese_origin": "在不更换接口的情况下更换四种收数器",

    # toned (numbers)
    "pinyin_toned": "zai4bu4geng1huan4jie1tou2deqing1kuang4xia4geng1huan4si4zhong3shou1shu4qi4",

    # toneless
    "pinyin_toneless": "zaibugenghuanjietoudeqingkuangxiagenghuansizhongshoushuqi",

    # diacritic (correct full form, NO spaces, NO syllable split)
    "pinyin_diacritic": "zàibùgēnghuànjiētóudeqíngkuàngxiàgēnghuànsìzhǒngshōushùqì"
}

# =========================
# CHECK FUNCTION
# =========================

def check_tokenizer(path, test_text):
    print("\n" + "="*90)
    print(f"Loading: {path}")

    tokenizer = Tokenizer.from_file(path)

    # 1. vocab size
    print("Vocab size:", tokenizer.get_vocab_size())

    # 2. sample vocab
    vocab = tokenizer.get_vocab()
    print("\nSample vocab:")
    for i, (token, idx) in enumerate(vocab.items()):
        if i >= MAX_VOCAB_SAMPLE:
            break
        print(f"{token} -> {idx}")

    # 3. encode test
    encoding = tokenizer.encode(test_text)

    print("\nTest text:")
    print(test_text)

    print("\nTokens:", encoding.tokens)
    print("Token count:", len(encoding.tokens))
    print("IDs:", encoding.ids)

    # 4. model type
    print("\nModel type:", type(tokenizer.model))


# =========================
# MAIN LOOP
# =========================

def main():
    print(f"\n🔍 Checking tokenizers at vocab = {VOCAB_SIZE_CHECK}\n")

    for name, text in TEST_TEXTS.items():

        file_path = os.path.join(
            TOKENIZER_DIR,
            f"{name}_{VOCAB_SIZE_CHECK//1000}k_train90.json"
        )

        if not os.path.exists(file_path):
            print(f"\n⚠️ Missing: {file_path}")
            continue

        check_tokenizer(file_path, text)


if __name__ == "__main__":
    main()