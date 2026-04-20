from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel

def peek_tokens(path, name):
    print(f"\n--- {name} ---")
    tk = Tokenizer.from_file(path)
    tk.decoder = ByteLevel()
    vocab = tk.get_vocab()
    # Sort to see mid-frequency tokens
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    for token, tid in sorted_vocab[4000:4005]:
        decoded = tk.decode([tid])
        print(f"Raw: {repr(token)} | Decoded: {repr(decoded)} | Stripped: {repr(decoded.strip())}")

base = "/Users/crisp/Desktop/code_field/python/Chinese_Latinization_NLP/1.Tokenization/superTokenizers_BPE/"
peek_tokens(base + "pinyin_toned_subset100k_superbpe_8000/tokenizer.json", "Toned")
peek_tokens(base + "chinese_origin_subset100k_superbpe_8000/tokenizer.json", "Chinese")
