"""
从Unihan_Readings.txt解析拼音，创建全面的字符到拼音映射
"""
import re
import os

print("=" * 100)
print("从Unihan数据库解析拼音信息")
print("=" * 100)

# 从Unihan_Readings.txt解析拼音
char_to_unihan_pinyin = {}
unihan_file = 'dicts/Unihan_Readings.txt'

if os.path.exists(unihan_file):
    print(f"\n正在解析 {unihan_file}...")
    
    with open(unihan_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            # 跳过注释和空行
            if not line or line.startswith('#'):
                continue
            
            # Unihan格式: U+XXXX	kMandarin	pin3 yin1
            if 'kMandarin' in line:
                parts = line.split('\t')
                if len(parts) >= 3:
                    try:
                        unicode_str = parts[0].replace('U+', '')
                        char_code = int(unicode_str, 16)
                        char = chr(char_code)
                        
                        # 获取拼音（可能有多个）
                        pinyins = parts[2].split()  # "pin3 yin1" -> ["pin3", "yin1"]
                        
                        if char not in char_to_unihan_pinyin:
                            char_to_unihan_pinyin[char] = pinyins
                    except:
                        pass
            
            if line_num % 100000 == 0:
                print(f"  已处理 {line_num:,} 行...", end='\r')

print(f"\n✓ 从Unihan_Readings.txt解析出 {len(char_to_unihan_pinyin):,} 个字符的拼音")

# 现在对比CEDICT的覆盖
char_to_pinyin_cedict = {}
cedict_file = 'dicts/cedict_ts.u8'

if os.path.exists(cedict_file):
    with open(cedict_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            try:
                parts = line.split()
                if len(parts) >= 3:
                    simp = parts[1]
                    pinyin_str = parts[2].strip('[]')
                    pinyins = pinyin_str.split('/')
                    for i, char in enumerate(simp):
                        if char not in char_to_pinyin_cedict:
                            char_to_pinyin_cedict[char] = []
                        if i < len(pinyins):
                            char_to_pinyin_cedict[char].append(pinyins[i])
            except:
                pass

print(f"✓ CEDICT覆盖 {len(char_to_pinyin_cedict):,} 个字符的拼音")

# 分析Unihan相比CEDICT增加了什么
unihan_only = set(char_to_unihan_pinyin.keys()) - set(char_to_pinyin_cedict.keys())
print(f"\n✓ Unihan相比CEDICT新增的字符: {len(unihan_only):,} 个")

# 创建合并后的字典（优先CEDICT，然后Unihan）
char_to_pinyin_merged = {}
char_to_pinyin_merged.update(char_to_pinyin_cedict)
char_to_pinyin_merged.update(char_to_unihan_pinyin)

print(f"✓ 合并后的总字符数: {len(char_to_pinyin_merged):,} 个")

# 验证：看看那些生僻字现在有没有
test_chars = ['両', '乁', '乗', '乚', '乫', '亜']
print(f"\n【验证：生僻字是否在Unihan中】")
for char in test_chars:
    if char in char_to_unihan_pinyin:
        print(f"  ✓ {char}: {char_to_unihan_pinyin[char]}")
    else:
        print(f"  ✗ {char}: 未找到")

# 保存合并后的字典（JSON格式）
import json

# 转换为可序列化的格式
data_dict = {}
for char, pinyins in char_to_pinyin_merged.items():
    if isinstance(pinyins, list) and len(pinyins) > 0:
        data_dict[char] = pinyins[0]
    elif isinstance(pinyins, str):
        data_dict[char] = pinyins

merged_dict = {
    'total_chars': len(data_dict),
    'cedict_chars': len(char_to_pinyin_cedict),
    'unihan_chars': len(char_to_unihan_pinyin),
    'unihan_only': len(unihan_only),
    'data': data_dict
}

output_file = 'dicts/merged_pinyin_dict.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(merged_dict, f, ensure_ascii=False, indent=2)

print(f"\n✓ 已保存合并字典到 {output_file}")
print(f"  (包含 {len(merged_dict['data']):,} 个字符)")

print("\n" + "=" * 100)
