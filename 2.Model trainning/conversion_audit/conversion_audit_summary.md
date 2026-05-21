# Conversion Audit Summary

## Pipeline Settings Used

- Jieba segmentation: `jieba.cut(text, cut_all=False)` after the existing light punctuation normalization.
- Pinyin-Toneless: `pypinyin.pinyin(word, style=Style.NORMAL, strict=False)` per jieba token.
- Pinyin-Toned: `pypinyin.pinyin(word, style=Style.TONE3, strict=False)` per jieba token.
- Pinyin-Diacritic: `pypinyin.pinyin(word, style=Style.TONE, strict=False)` per jieba token.
- Main split files: `data/raw/train.zh.txt`, `valid.zh.txt`, `test.zh.txt`, aligned to `train.diacritic.txt`, `valid.diacritic.txt`, `test.diacritic.txt`.
- Eval2/Eval4 stored diacritic fields are preserved when present; missing fields use the existing eval builder convention with `lazy_pinyin(..., style=Style.TONE, neutral_tone_with_five=False, errors=lambda chunk: list(chunk))`.

## Generated Samples

- zhwiki: `/Users/crisp/Desktop/code_field/python/Chinese_Latinization_NLP/2.Model trainning/conversion_audit/zhwiki_conversion_audit_sample.csv`; sampled/items = 500
- eval2: `/Users/crisp/Desktop/code_field/python/Chinese_Latinization_NLP/2.Model trainning/conversion_audit/eval2_conversion_audit_sample.csv`; sampled/items = 100
- eval4: `/Users/crisp/Desktop/code_field/python/Chinese_Latinization_NLP/2.Model trainning/conversion_audit/eval4_conversion_audit_sample.csv`; sampled/items = 150
- collision_sanity: `/Users/crisp/Desktop/code_field/python/Chinese_Latinization_NLP/2.Model trainning/conversion_audit/diacritic_collision_sanity_check.csv`; sampled/items = 50

## Manual Labels

Manual review is pending. Fill the manual columns in copied labeled CSVs, then rerun with `--manual_labels`.
