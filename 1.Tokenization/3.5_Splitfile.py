import os
import random

CORPUS_DIR = "./corpora"
TRAIN_RATIO = 0.9
SEED = 42

"""
⚠️  重要：该脚本使用统一的行索引分割来确保所有corpus文件分割到相同的行
    即使是不同格式的tokenizer（中文、无声调拼音、带号码拼音、带符号拼音等）
    
工作原理：
1. 先读第一个完整的corpus文件，确定总行数
2. 生成随机索引顺序 [line_num1, line_num2, ...]
3. 对所有其他corpus文件使用同一个索引列表进行分割
4. 这样保证了4个train90%文件包含的是完全对应的行
"""

def split_files_with_shared_indices(corpus_dir, train_ratio=0.9, seed=42):
    """使用共享的行索引来分割所有corpus文件"""
    
    random.seed(seed)
    
    # 第1步：明确指定要分割的4个文件
    complete_files = [
        "chinese_origin_中国.txt",
        "pinyin_toned_spaced.txt",
        "pinyin_toneless_spaced.txt",
        "pinyin_diacritic_spaced.txt"
    ]
    
    # 检查文件是否存在
    complete_files = [f for f in complete_files if os.path.exists(os.path.join(corpus_dir, f))]
    
    if not complete_files:
        print("❌ 没有找到未分割的corpus文件")
        return
    
    print(f"发现 {len(complete_files)} 个完整的corpus文件")
    print()
    
    # 第2步：读第一个文件来确定总行数
    first_file_path = os.path.join(corpus_dir, complete_files[0])
    with open(first_file_path, "r", encoding="utf-8") as f:
        total_lines = sum(1 for line in f if line.strip())
    
    print(f"总行数: {total_lines}")
    print()
    
    if total_lines < 10:
        print("❌ 文件行数太少，无法分割")
        return
    
    # 第3步：生成随机索引顺序（这是关键！）
    indices = list(range(total_lines))
    random.shuffle(indices)  # 使用seed=42打乱索引
    
    # 第4步：根据TRAIN_RATIO分割索引
    split_idx = int(total_lines * train_ratio)
    train_indices = set(indices[:split_idx])
    test_indices = set(indices[split_idx:])
    
    print(f"Train索引数: {len(train_indices)}")
    print(f"Test索引数:  {len(test_indices)}")
    print()
    
    # 第5步：对每个完整的corpus文件使用同一个索引进行分割
    for file in complete_files:
        filepath = os.path.join(corpus_dir, file)
        base_name = os.path.splitext(file)[0]
        
        # 读取所有行
        with open(filepath, "r", encoding="utf-8") as f:
            all_lines = [l.strip() for l in f if l.strip()]
        
        if len(all_lines) != total_lines:
            print(f"⚠️  警告: {file} 的行数 ({len(all_lines)}) 与第一个文件不同 ({total_lines})")
            continue
        
        # 使用同一个索引进行分割
        train_lines = [all_lines[i] for i in sorted(train_indices)]
        test_lines = [all_lines[i] for i in sorted(test_indices)]
        
        # 保存分割后的文件
        train_path = os.path.join(corpus_dir, f"{base_name}_train90.txt")
        test_path = os.path.join(corpus_dir, f"{base_name}_test10.txt")
        
        with open(train_path, "w", encoding="utf-8") as f:
            for line in train_lines:
                f.write(line + "\n")
        
        with open(test_path, "w", encoding="utf-8") as f:
            for line in test_lines:
                f.write(line + "\n")
        
        print(f"✓ {base_name}")
        print(f"  ├─ Train: {len(train_lines)} 行 → {train_path}")
        print(f"  └─ Test:  {len(test_lines)} 行 → {test_path}")


def main():
    split_files_with_shared_indices(CORPUS_DIR, train_ratio=TRAIN_RATIO, seed=SEED)


if __name__ == "__main__":
    main()