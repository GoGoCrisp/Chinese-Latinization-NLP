with open("9th_compare_tokenizers_overlap_superBPE.py", "r", encoding="utf-8") as f:
    content = f.read()

old_str = """            elif name1 == "B" and name2 == "C":
                # B(无声调) → C(带数字)
                # 禁用前向精确匹配，防止B中的残留脏词（如带数字拼音）错误匹配C
                # 仅使用后段的安全反向推导（去调匹配）
                pass
                
            elif name1 == "B" and name2 == "D":
                # B(无声调) → D(带声调符号)
                # 同理禁用前向精确匹配
                pass"""

new_str = """            elif name1 == "B" and name2 == "C":
                # B(无声调) → C(带数字)
                # 直接保留token以备反向查找
                candidates_in_vocab2.add(token1_clean)
                
            elif name1 == "B" and name2 == "D":
                # B(无声调) → D(带声调符号)
                # 直接保留token以备反向查找
                candidates_in_vocab2.add(token1_clean)"""

content = content.replace(old_str, new_str)

with open("9th_compare_tokenizers_overlap_superBPE.py", "w", encoding="utf-8") as f:
    f.write(content)
