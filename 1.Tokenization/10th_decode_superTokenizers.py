import os
import json
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "superTokenizers_BPE")
OUTPUT_DIR = os.path.join(BASE_DIR, "decoded_superTokenizers")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_tokenizers():
    print(f"扫描目录: {INPUT_DIR} ...\n")
    
    processed_count = 0
    for root, dirs, files in os.walk(INPUT_DIR):
        if "tokenizer.json" in files:
            tk_path = os.path.join(root, "tokenizer.json")
            tk_name = os.path.basename(root)
            
            # 仅处理最终的 superbpe 分词器（跳过过程中的 stage1）
            if "superbpe" not in tk_name:
                continue
            
            print(f"正在解码: {tk_name} ...")
            try:
                tk = Tokenizer.from_file(tk_path)
                tk.decoder = ByteLevel()
                
                vocab = tk.get_vocab()
                
                # 解码并构建新的词表 { "解码后的文本": id }
                decoded_vocab = {}
                for raw_token, tid in vocab.items():
                    decoded_text = tk.decode([tid])
                    # 注意：如你所求，这里绝对不使用 .strip() 剥离空格
                    decoded_vocab[decoded_text] = tid
                
                # 按照 ID 顺序从小到大排序，方便人类阅读
                sorted_decoded_vocab = {k: v for k, v in sorted(decoded_vocab.items(), key=lambda item: item[1])}
                
                out_path = os.path.join(OUTPUT_DIR, f"{tk_name}_decoded.json")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(sorted_decoded_vocab, f, ensure_ascii=False, indent=2)
                
                print(f"  ✅ 已保存至 {os.path.basename(out_path)} (有效词汇量: {len(sorted_decoded_vocab)})\n")
                processed_count += 1
            except Exception as e:
                print(f"  ❌ 解析失败: {e}\n")
                
    print(f"🎯 处理完毕！共生成了 {processed_count} 个解码词表文件，存放在 {OUTPUT_DIR} 目录下。")

if __name__ == "__main__":
    process_tokenizers()
