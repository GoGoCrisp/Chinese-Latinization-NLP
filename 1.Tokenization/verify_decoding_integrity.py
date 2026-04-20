from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel

def test_tokenizer(name, tk_path, sample_text):
    print(f"\n[{name}] Testing encoding/decoding integrity:")
    try:
        tk = Tokenizer.from_file(tk_path)
        tk.decoder = ByteLevel()
        
        # Encode string into integers
        encoded = tk.encode(sample_text)
        print(f"Original Text: '{sample_text}'")
        print(f"Token IDs:     {encoded.ids}")
        
        # Decode integers back to string
        decoded = tk.decode(encoded.ids)
        print(f"Decoded Text:  '{decoded}'")
        
        if sample_text == decoded:
            print("✅ Status: SUCCESS (Perfect matching)")
        else:
            print("❌ Status: FAILED (Information lost)")
    except Exception as e:
        print(f"Error: {e}")

# Diacritic Test (pinyin with tones)
diacritic_tk = "/Users/crisp/Desktop/code_field/python/Chinese_Latinization_NLP/1.Tokenization/superTokenizers_BPE/pinyin_diacritic_subset100k_superbpe_16000/tokenizer.json"
diacritic_text = "zhōng guó shì wéng xīng yán fù zé de !"
test_tokenizer("Pinyin Diacritic", diacritic_tk, diacritic_text)

# Chinese Origin Test
chinese_tk = "/Users/crisp/Desktop/code_field/python/Chinese_Latinization_NLP/1.Tokenization/superTokenizers_BPE/chinese_origin_subset100k_superbpe_16000/tokenizer.json"
chinese_text = "中国是根据2020年美国人口普查的数据中发现的！"
test_tokenizer("Chinese Origin", chinese_tk, chinese_text)

