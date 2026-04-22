import os
import shutil
import subprocess
import random

# ===== Paths Configuration =====
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
CORPUS_DIR = os.path.join(BASE_DIR, "corpora")
OUTPUT_DIR = os.path.join(BASE_DIR, "superTokenizers_BPE")
SUPERBPE_VENV_PYTHON = os.path.abspath(os.path.join(BASE_DIR, "..", "superbpe", "superbpe_venv", "bin", "python"))
TRAIN_TOKENIZER_SCRIPT = os.path.abspath(os.path.join(BASE_DIR, "..", "superbpe", "train_tokenizer.py"))

os.makedirs(OUTPUT_DIR, exist_ok=True)

STAGE1_REGEX = r"\S+|\s+"
STAGE2_REGEX = r"[^\p{L}\p{N}\s]+|[\r\n]+"

def build_subset_corpus(original_path, subset_path, num_lines=100000, seed=42):
    if os.path.exists(subset_path):
        print(f"Subset already exists at {subset_path}. Using existing subset.")
        return

    print(f"Generating {num_lines}-line subset from {original_path}...")
    with open(original_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    random.seed(seed)
    subset_lines = random.sample(lines, min(num_lines, len(lines)))
    
    with open(subset_path, "w", encoding="utf-8") as f:
        f.writelines(subset_lines)
    print(f"✅ Subset generated at {subset_path}")

def ensure_corpus_dir(txt_file_path):
    parent_dir = os.path.dirname(txt_file_path)
    base_name = os.path.basename(txt_file_path)
    stem = os.path.splitext(base_name)[0]
    
    tmp_dir = os.path.join(parent_dir, f"{stem}_dir")
    os.makedirs(tmp_dir, exist_ok=True)
    
    symlink_path = os.path.join(tmp_dir, base_name)
    if not os.path.exists(symlink_path):
        os.symlink(os.path.abspath(txt_file_path), symlink_path)
        
    return tmp_dir

def run_superbpe_pipeline(corpus_path, total_vocab_size, name):
    t_point = int(total_vocab_size * 0.10)
    print(f"\n{'='*50}\nStarting SuperBPE Pipeline for {name} ({total_vocab_size//1000}k)\n{'='*50}")
    
    corpus_dir = ensure_corpus_dir(corpus_path)
    stage1_out = os.path.join(OUTPUT_DIR, f"{name}_stage1_{total_vocab_size}")
    
    print(f"\n[Stage 1] Training base BPE to {total_vocab_size} (will truncate later)...")
    if not os.path.exists(os.path.join(stage1_out, "tokenizer.json")):
        cmd1 = [
            SUPERBPE_VENV_PYTHON, TRAIN_TOKENIZER_SCRIPT,
            "--output_dir", stage1_out,
            "--corpus_dir", corpus_dir,
            "--vocab_size", str(total_vocab_size),
            "--regex_string", STAGE1_REGEX
        ]
        subprocess.run(cmd1, check=True)
    else:
        print("Stage 1 completed previously. Skipping.")

    stage2_out = os.path.join(OUTPUT_DIR, f"{name}_superbpe_{total_vocab_size}")
    os.makedirs(stage2_out, exist_ok=True)
    
    merges_txt = os.path.join(stage1_out, "merges.txt")
    meta_json = os.path.join(stage1_out, "meta.json")
    
    stage2_merges = os.path.join(stage2_out, "merges.txt")
    stage2_meta = os.path.join(stage2_out, "meta.json")
    
    print(f"\n[Stage 2 Prep] Inheriting top {t_point} merges from Stage 1...")
    with open(merges_txt, "r", encoding="utf-8") as f_in:
        lines = f_in.readlines()
        
    with open(stage2_merges, "w", encoding="utf-8") as f_out:
        f_out.writelines(lines[:t_point + 1])
        
    shutil.copy(meta_json, stage2_meta)
    
    print(f"\n[Stage 2] Extending tokenizer to {total_vocab_size} with SuperBPE regex...")
    cmd2 = [
        SUPERBPE_VENV_PYTHON, TRAIN_TOKENIZER_SCRIPT,
        "--output_dir", stage2_out,
        "--vocab_size", str(total_vocab_size),
        "--regex_string", STAGE2_REGEX
    ]
    subprocess.run(cmd2, check=True)
    
    print(f"\n✅ Finished Pipeline for {name} ({total_vocab_size//1000}k). Saved to {stage2_out}")

if __name__ == "__main__":
    corpora = {
        "chinese_origin": os.path.join(CORPUS_DIR, "chinese_origin_中国_train90.txt"),
        "pinyin_toned": os.path.join(CORPUS_DIR, "pinyin_toned_spaced_train90.txt"),
        "pinyin_toneless": os.path.join(CORPUS_DIR, "pinyin_toneless_spaced_train90.txt"),
        "pinyin_diacritic": os.path.join(CORPUS_DIR, "pinyin_diacritic_spaced_train90.txt")
    }

    vocab_sizes = [8000, 16000, 32000, 64000]

    for name, orig_path in corpora.items():
        if not os.path.exists(orig_path):
            print(f"⚠️ Missing original file: {orig_path}")
            continue
            
        subset_path = os.path.join(CORPUS_DIR, f"{name}_subset100k.txt")
        build_subset_corpus(orig_path, subset_path, num_lines=100000, seed=42)

        for v_size in vocab_sizes:
            run_superbpe_pipeline(subset_path, v_size, f"{name}_subset100k")