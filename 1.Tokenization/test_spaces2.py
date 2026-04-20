from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel

def peek_tokens(path, name):
    print(f"\n--- {name} ---")
    tk = Tokenizer.from_file(path)
    tk.decoder = ByteLevel()
    vocab = {v: k for k, v in tk.get_vocab().items()}
    # look at some very high frequency tokens (which are usually words with prefix space)
    for i in range(300, 305):
        raw = vocab[i]
        decoded = tk.decode([i])
        print(f"ID: {i} | Raw: {repr(raw)} | Decoded: {repr(decoded)} | Stripped: {repr(decoded.strip())}")

base = "/Users/crisp/Desktop/code_field/python/Chinese_Latinization_NLP/1.Tokenization/superTokenizers_BPE/"
peek_tokens(base + "pinyin_toned_subset100k_superbpe_8000/tokenizer.json", "Toned")
peek_tokens(base + "chinese_origin_subset100k_superbpe_8000/tokenizer.json", "Chinese")
