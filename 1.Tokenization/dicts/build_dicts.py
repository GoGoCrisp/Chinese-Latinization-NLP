import re
from pypinyin import pinyin, Style

INPUT_FILE = "./dicts/cedict_ts.u8"
OUTPUT_DIR = "./dicts"

toned_set = set()
toneless_set = set()
diacritic_set = set()

def hanzi_to_diacritic(hanzi):
    # zhōngguó 这种形式
    return "".join([item[0] for item in pinyin(hanzi, style=Style.TONE)])

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if line.startswith("#"):
            continue

        try:
            # 拆结构
            parts = re.split(r"\[|\]", line)
            hanzi_part = parts[0].strip().split()

            if len(hanzi_part) < 2:
                continue

            simplified = hanzi_part[1]     # 中国
            pinyin_raw = parts[1]          # Zhong1 guo2

            # ===== toned =====
            toned = pinyin_raw.lower().replace(" ", "")
            toned_set.add(toned)

            # ===== toneless =====
            toneless = re.sub(r"[0-9]", "", toned)
            toneless_set.add(toneless)

            # ===== diacritic =====
            diacritic = hanzi_to_diacritic(simplified)
            diacritic_set.add(diacritic)

        except Exception:
            continue

# ===== 保存 =====
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save(name, data):
    path = os.path.join(OUTPUT_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        for w in sorted(data):
            f.write(w + "\n")

save("dict_toned.txt", toned_set)
save("dict_toneless.txt", toneless_set)
save("dict_diacritic.txt", diacritic_set)

print("✅ Done!")
print(f"toned: {len(toned_set)}")
print(f"toneless: {len(toneless_set)}")
print(f"diacritic: {len(diacritic_set)}")