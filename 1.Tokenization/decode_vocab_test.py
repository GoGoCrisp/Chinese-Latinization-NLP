from tokenizers import Tokenizer
import json

path = "/Users/crisp/Desktop/code_field/python/Chinese_Latinization_NLP/1.Tokenization/superTokenizers_BPE/pinyin_diacritic_subset100k_superbpe_8000/tokenizer.json"
tk = Tokenizer.from_file(path)
vocab = tk.get_vocab()
sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])

with open("/Users/crisp/Desktop/code_field/python/Chinese_Latinization_NLP/1.Tokenization/readable_diacritic_vocab.txt", "w", encoding="utf-8") as f:
    for token, tid in sorted_vocab[-100:]:  # last 100
        readable = tk.decode([tid])
        f.write(f"ID: {tid} | Raw Token: {token} | Readable: {readable}\n")
