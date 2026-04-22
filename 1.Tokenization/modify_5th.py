import re

with open('/Users/crisp/Desktop/code_field/python/Chinese_Latinization_NLP/1.Tokenization/5th_Analyzation for 16 tokenization.py', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Update TOKENIZER_DIR
text = text.replace('TOKENIZER_DIR = os.path.join(BASE_DIR, "tokenizers")', 'TOKENIZER_DIR = os.path.join(BASE_DIR, "superTokenizers_BPE")')

# 2. Comment out 4c english_vocab sections
text = re.sub(
    r'(# ===== ENGLISH VOCAB（baseline）=====.*?)(?=# ===== MAIN =====)',
    r"'''\n\g<1>\n'''\n",
    text,
    flags=re.DOTALL
)

# 3. Add VALID_PINYIN and pinyin checker
pinyin_code = """
import unicodedata
import re

VALID_PINYIN = set(['a', 'ai', 'an', 'ang', 'ao', 'ba', 'bai', 'ban', 'bang', 'bao', 'bei', 'ben', 'beng', 'bi', 'bian', 'biao', 'bie', 'bin', 'bing', 'bo', 'bu', 'ca', 'cai', 'can', 'cang', 'cao', 'ce', 'cen', 'ceng', 'cha', 'chai', 'chan', 'chang', 'chao', 'che', 'chen', 'cheng', 'chi', 'chong', 'chou', 'chu', 'chua', 'chuai', 'chuan', 'chuang', 'chui', 'chun', 'chuo', 'ci', 'cong', 'cou', 'cu', 'cuan', 'cui', 'cun', 'cuo', 'da', 'dai', 'dan', 'dang', 'dao', 'de', 'dei', 'deng', 'di', 'dian', 'diao', 'die', 'ding', 'diu', 'dong', 'dou', 'du', 'duan', 'dui', 'dun', 'duo', 'e', 'ei', 'en', 'eng', 'er', 'fa', 'fan', 'fang', 'fei', 'fen', 'feng', 'fo', 'fou', 'fu', 'ga', 'gai', 'gan', 'gang', 'gao', 'ge', 'gei', 'gen', 'geng', 'gong', 'gou', 'gu', 'gua', 'guai', 'guan', 'guang', 'gui', 'gun', 'guo', 'ha', 'hai', 'han', 'hang', 'hao', 'he', 'hei', 'hen', 'heng', 'hm', 'hng', 'hong', 'hou', 'hu', 'hua', 'huai', 'huan', 'huang', 'hui', 'hun', 'huo', 'ji', 'jia', 'jian', 'jiang', 'jiao', 'jie', 'jin', 'jing', 'jiong', 'jiu', 'ju', 'juan', 'jue', 'jun', 'ka', 'kai', 'kan', 'kang', 'kao', 'ke', 'kei', 'ken', 'keng', 'kong', 'kou', 'ku', 'kua', 'kuai', 'kuan', 'kuang', 'kui', 'kun', 'kuo', 'la', 'lai', 'lan', 'lang', 'lao', 'le', 'lei', 'leng', 'li', 'lia', 'lian', 'liang', 'liao', 'lie', 'lin', 'ling', 'liu', 'lo', 'long', 'lou', 'lu', 'lv', 'luan', 'lue', 'lve', 'lun', 'luo', 'm', 'ma', 'mai', 'man', 'mang', 'mao', 'me', 'mei', 'men', 'meng', 'mi', 'mian', 'miao', 'mie', 'min', 'ming', 'miu', 'mo', 'mou', 'mu', 'na', 'nai', 'nan', 'nang', 'nao', 'ne', 'nei', 'nen', 'neng', 'ng', 'ni', 'nian', 'niang', 'niao', 'nie', 'nin', 'ning', 'niu', 'nong', 'nou', 'nu', 'nv', 'nuan', 'nue', 'nve', 'nun', 'nuo', 'o', 'ou', 'pa', 'pai', 'pan', 'pang', 'pao', 'pei', 'pen', 'peng', 'pi', 'pian', 'piao', 'pie', 'pin', 'ping', 'po', 'pou', 'pu', 'qi', 'qia', 'qian', 'qiang', 'qiao', 'qie', 'qin', 'qing', 'qiong', 'qiu', 'qu', 'quan', 'que', 'qun', 'ran', 'rang', 'rao', 're', 'ren', 'reng', 'ri', 'rong', 'rou', 'ru', 'ruan', 'rui', 'run', 'ruo', 'sa', 'sai', 'san', 'sang', 'sao', 'se', 'sen', 'seng', 'sha', 'shai', 'shan', 'shang', 'shao', 'she', 'shei', 'shen', 'sheng', 'shi', 'shou', 'shu', 'shua', 'shuai', 'shuan', 'shuang', 'shui', 'shun', 'shuo', 'si', 'song', 'sou', 'su', 'suan', 'sui', 'sun', 'suo', 'ta', 'tai', 'tan', 'tang', 'tao', 'te', 'teng', 'ti', 'tian', 'tiao', 'tie', 'ting', 'tong', 'tou', 'tu', 'tuan', 'tui', 'tun', 'tuo', 'wa', 'wai', 'wan', 'wang', 'wei', 'wen', 'weng', 'wo', 'wu', 'xi', 'xia', 'xian', 'xiang', 'xiao', 'xie', 'xin', 'xing', 'xiong', 'xiu', 'xu', 'xuan', 'xue', 'xun', 'ya', 'yan', 'yang', 'yao', 'ye', 'yi', 'yin', 'ying', 'yo', 'yong', 'you', 'yu', 'yuan', 'yue', 'yun', 'za', 'zai', 'zan', 'zang', 'zao', 'ze', 'zei', 'zen', 'zeng', 'zha', 'zhai', 'zhan', 'zhang', 'zhao', 'zhe', 'zhei', 'zhen', 'zheng', 'zhi', 'zhong', 'zhou', 'zhu', 'zhua', 'zhuai', 'zhuan', 'zhuang', 'zhui', 'zhun', 'zhuo', 'zi', 'zong', 'zou', 'zu', 'zuan', 'zui', 'zun', 'zuo'])

def is_valid_pinyin_syllable(s: str) -> bool:
    if any('A' <= c <= 'Z' for c in s):
        return False
    s = s.replace('ü', 'v').replace('u:', 'v')
    s = re.sub(r'[1-5]', '', s)
    s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    return s in VALID_PINYIN

def is_chinese_chars(s: str) -> bool:
    if not s: return False
    return all('\u4e00' <= c <= '\u9fff' for c in s)

# ===== MAIN =====
"""

text = text.replace('# ===== MAIN =====', pinyin_code)

# 4. Modify loop to read superTokenizers
old_loop = '''for file in os.listdir(TOKENIZER_DIR):
    if not file.endswith(".json"):
        continue

    t_type = detect_type(file)'''

new_loop = '''for folder in os.listdir(TOKENIZER_DIR):
    if "_superbpe_" not in folder:
        continue

    t_type = detect_type(folder)
    
    tokenizer_path = os.path.join(TOKENIZER_DIR, folder, "tokenizer.json")
    if not os.path.exists(tokenizer_path):
        continue
        
    file = folder # Using folder name as model identifier'''

text = text.replace(old_loop, new_loop)
text = text.replace('Tokenizer.from_file(os.path.join(TOKENIZER_DIR, file))', 'Tokenizer.from_file(tokenizer_path)')

# 5. Modify 4b check logic
old_4b = '''        # ===== 4b =====
        if dictionary:
            for tok in tokens:
                if len(tok) >= 2:
                    checked_tokens += 1
                    if tok in dictionary:
                        valid_tokens += 1'''

new_4b = '''        # ===== 4b =====
        for tok in tokens:
            clean_tok = tok.replace("##", "").replace("Ġ", "").strip()
            if not clean_tok:
                continue
                
            # Skip purely punctuation or non-alphanumeric/non-CJK
            if re.match(r'^[^\w\u4e00-\u9fff]+$', clean_tok):
                continue
                
            # If length >= 2, we consider it for morph checking
            if len(clean_tok) >= 2:
                checked_tokens += 1
                is_valid = False
                
                # Check 1: Dictionary Match
                if dictionary and clean_tok in dictionary:
                    is_valid = True
                
                # Check 2: Chinese Origin pure CJK
                elif t_type == "origin" and is_chinese_chars(clean_tok):
                    is_valid = True
                
                # Check 3: Valid Pinyin Combinations (with optional numbers)
                elif t_type != "origin":
                    pinyin_parts = [p.strip() for p in clean_tok.split(" ") if p.strip()]
                    if pinyin_parts:
                        all_parts_valid = True
                        for p in pinyin_parts:
                            if not (p.isdigit() or is_valid_pinyin_syllable(p)):
                                all_parts_valid = False
                                break
                        if all_parts_valid:
                            is_valid = True
                
                if is_valid:
                    valid_tokens += 1'''

text = text.replace(old_4b, new_4b)

# 6. Comment out 4c execution
text = text.replace('''    # ===== 4c =====
    vocab = set(tokenizer.get_vocab().keys())
    overlap = len(vocab & english_vocab) / len(vocab)''', '''    # ===== 4c (Commented out) =====
    # vocab = set(tokenizer.get_vocab().keys())
    # overlap = len(vocab & english_vocab) / len(vocab)
    overlap = 0.0 # Placeholder value''')

# 7. Remove overlap from output header
# text = text.replace('"cross_lingual_overlap",\n', '')

with open('/Users/crisp/Desktop/code_field/python/Chinese_Latinization_NLP/1.Tokenization/5th_Analyzation for 16 tokenization.py', 'w', encoding='utf-8') as f:
    f.write(text)
