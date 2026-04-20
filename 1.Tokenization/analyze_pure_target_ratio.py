import json
import os
import re

DIR = "/Users/crisp/Desktop/code_field/python/Chinese_Latinization_NLP/1.Tokenization/decoded_superTokenizers"

pinyin_syllables = set(["a", "ai", "an", "ang", "ao", "ba", "bai", "ban", "bang", "bao", "bei", "ben", "beng", "bi", "bian", "biao", "bie", "bin", "bing", "bo", "bu", "ca", "cai", "can", "cang", "cao", "ce", "cen", "ceng", "cha", "chai", "chan", "chang", "chao", "che", "chen", "cheng", "chi", "chong", "chou", "chu", "chua", "chuai", "chuan", "chuang", "chui", "chun", "chuo", "ci", "cong", "cou", "cu", "cuan", "cui", "cun", "cuo", "da", "dai", "dan", "dang", "dao", "de", "deng", "di", "dian", "diao", "die", "ding", "diu", "dong", "dou", "du", "duan", "dui", "dun", "duo", "e", "ei", "en", "eng", "er", "fa", "fan", "fang", "fei", "fen", "feng", "fo", "fou", "fu", "ga", "gai", "gan", "gang", "gao", "ge", "gei", "gen", "geng", "gong", "gou", "gu", "gua", "guai", "guan", "guang", "gui", "gun", "guo", "ha", "hai", "han", "hang", "hao", "he", "hei", "hen", "heng", "hong", "hou", "hu", "hua", "huai", "huan", "huang", "hui", "hun", "huo", "ji", "jia", "jian", "jiang", "jiao", "jie", "jin", "jing", "jiong", "jiu", "ju", "juan", "jue", "jun", "ka", "kai", "kan", "kang", "kao", "ke", "ken", "keng", "kong", "kou", "ku", "kua", "kuai", "kuan", "kuang", "kui", "kun", "kuo", "la", "lai", "lan", "lang", "lao", "le", "lei", "leng", "li", "lia", "lian", "liang", "liao", "lie", "lin", "ling", "liu", "long", "lou", "lu", "luan", "lun", "luo", "lv", "lve", "ma", "mai", "man", "mang", "mao", "me", "mei", "men", "meng", "mi", "mian", "miao", "mie", "min", "ming", "miu", "mo", "mou", "mu", "na", "nai", "nan", "nang", "nao", "ne", "nei", "nen", "neng", "ni", "nian", "niang", "niao", "nie", "nin", "ning", "niu", "nong", "nou", "nu", "nuan", "nuo", "nv", "nve", "o", "ou", "pa", "pai", "pan", "pang", "pao", "pei", "pen", "peng", "pi", "pian", "piao", "pie", "pin", "ping", "po", "pou", "pu", "qi", "qia", "qian", "qiang", "qiao", "qie", "qin", "qing", "qiong", "qiu", "qu", "quan", "que", "qun", "ran", "rang", "rao", "re", "ren", "reng", "ri", "rong", "rou", "ru", "ruan", "rui", "run", "ruo", "sa", "sai", "san", "sang", "sao", "se", "sen", "seng", "sha", "shai", "shan", "shang", "shao", "she", "shen", "sheng", "shi", "shou", "shu", "shua", "shuai", "shuan", "shuang", "shui", "shun", "shuo", "si", "song", "sou", "su", "suan", "sui", "sun", "suo", "ta", "tai", "tan", "tang", "tao", "te", "teng", "ti", "tian", "tiao", "tie", "ting", "tong", "tou", "tu", "tuan", "tui", "tun", "tuo", "wa", "wai", "wan", "wang", "wei", "wen", "weng", "wo", "wu", "xi", "xia", "xian", "xiang", "xiao", "xie", "xin", "xing", "xiong", "xiu", "xu", "xuan", "xue", "xun", "ya", "yan", "yang", "yao", "ye", "yi", "yin", "ying", "yo", "yong", "you", "yu", "yuan", "yue", "yun", "za", "zai", "zan", "zang", "zao", "ze", "zei", "zen", "zeng", "zha", "zhai", "zhan", "zhang", "zhao", "zhe", "zhen", "zheng", "zhi", "zhong", "zhou", "zhu", "zhua", "zhuai", "zhuan", "zhuang", "zhui", "zhun", "zhuo", "zi", "zong", "zou", "zu", "zuan", "zui", "zun", "zuo"])

def remove_tones(text):
    text = re.sub(r'[0-9]', '', text)
    tone_map = {
        'ā':'a','á':'a','ǎ':'a','à':'a',
        'ē':'e','é':'e','ě':'e','è':'e',
        'ī':'i','í':'i','ǐ':'i','ì':'i',
        'ō':'o','ó':'o','ǒ':'o','ò':'o',
        'ū':'u','ú':'u','ǔ':'u','ù':'u',
        'ǖ':'v','ǘ':'v','ǚ':'v','ǜ':'v','ü':'v'
    }
    return "".join(tone_map.get(c, c) for c in text)

def is_pure_chinese(token):
    clean = token.strip()
    if not clean: return False
    return bool(re.fullmatch(r'[\u4e00-\u9fa5]+', clean))

def is_pure_pinyin(token):
    # This is a strict verification of Pinyin
    clean = remove_tones(token).strip()
    if not clean: return False
    if any(c.isupper() for c in clean): return False
    if not bool(re.fullmatch(r'[a-z \-]+', clean)): return False
    
    parts = clean.split()
    for p in parts:
        if not p: continue
        if p not in pinyin_syllables:
            is_sub = False
            for sy in pinyin_syllables:
                if p in sy:
                    is_sub = True
                    break
            if not is_sub:
                return False
    return True

print(f"{'Tokenizer Model':<40} | {'Vocab':<6} | {'Total':<6} | {'Target Pure Count':<18} | {'Ratio':<6}")
print("-" * 88)

for t in ["chinese_origin", "pinyin_toneless", "pinyin_toned", "pinyin_diacritic"]:
    for size in [8000, 16000, 32000, 64000]:
        fname = f"{t}_subset100k_superbpe_{size}_decoded.json"
        fpath = os.path.join(DIR, fname)
        if not os.path.exists(fpath): continue
        with open(fpath, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        
        pure_count = 0
        total = len(vocab)
        
        for token in vocab.keys():
            if t == "chinese_origin":
                if is_pure_chinese(token): pure_count += 1
            else:
                if is_pure_pinyin(token): pure_count += 1
                
        pct = (pure_count / total) * 100 if total > 0 else 0
        label = "Pure Chinese" if t == "chinese_origin" else "Pure Pinyin"
        print(f"{t:<40} | {size:<6} | {total:<6} | {pure_count:<5} ({label:<11}) | {pct:>5.1f}%")
    print("-" * 88)
