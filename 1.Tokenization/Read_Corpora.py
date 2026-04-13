import os
import random

CORPUS_DIR = "corpora"
SAMPLE_LENGTH = 200

def sample_from_file(filepath, sample_length=200):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        
        text = text.strip()
        if len(text) < sample_length:
            return text  # 太短就直接返回
        
        start = random.randint(0, len(text) - sample_length)
        return text[start:start + sample_length]
    
    except Exception as e:
        return f"[ERROR reading {filepath}: {e}]"


def main():
    files = [f for f in os.listdir(CORPUS_DIR) if (f.endswith(".txt") or f.endswith(".jsonl")) and "train90" in f]
    
    if not files:
        print("No files found in corpus directory.")
        return
    
    for filename in files:
        filepath = os.path.join(CORPUS_DIR, filename)
        
        print("=" * 60)
        print(f"FILE: {filename}")
        print("-" * 60)
        
        sample = sample_from_file(filepath, SAMPLE_LENGTH)
        print(sample)
        print("\n")


if __name__ == "__main__":
    main()