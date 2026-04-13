import os
import random

CORPUS_DIR = "./corpora"
TRAIN_RATIO = 0.9
SEED = 42

random.seed(SEED)


def split_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    
    if len(lines) < 10:
        print(f"Skip too small file: {filepath}")
        return
    
    random.shuffle(lines)
    
    split_idx = int(len(lines) * TRAIN_RATIO)
    
    train_lines = lines[:split_idx]
    test_lines = lines[split_idx:]
    
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    
    train_path = os.path.join(CORPUS_DIR, f"{base_name}_train90.txt")
    test_path = os.path.join(CORPUS_DIR, f"{base_name}_test10.txt")
    
    with open(train_path, "w", encoding="utf-8") as f:
        for l in train_lines:
            f.write(l + "\n")
    
    with open(test_path, "w", encoding="utf-8") as f:
        for l in test_lines:
            f.write(l + "\n")
    
    print(f"\nProcessed: {base_name}")
    print(f"Train: {len(train_lines)} → {train_path}")
    print(f"Test : {len(test_lines)} → {test_path}")


def main():
    files = [f for f in os.listdir(CORPUS_DIR) if f.endswith(".txt")]
    
    for file in files:
        filepath = os.path.join(CORPUS_DIR, file)
        
        # 跳过已经 split 的文件
        if "_train90" in file or "_test10" in file:
            continue
        
        split_file(filepath)


if __name__ == "__main__":
    main()