# 4.4 Morphological Coherence

This folder stores the standalone 4.4 vocabulary-level morphological coherence outputs.

Method memory:
- We moved from test-set dictionary hit rate to vocabulary-level token type quality.
- The research question is now: how many tokens in the tokenizer vocabulary are linguistically reasonable units?
- Labels:
  - SV = strong_valid: complete and natural lexical/morphemic token.
  - WV = weak_valid: understandable and linguistically plausible, but not an ideal standalone lexical token.
  - IV = invalid: arbitrary fragment, broken syllable, incomplete numeric/time phrase, or uninterpretable unit.
  - EX = excluded: punctuation, empty/control/symbol tokens, or out-of-scope foreign inventory items.
- Denominator for rates is SV + WV + IV, not vocab_size. Coverage reports (SV+WV+IV)/vocab_size.
- Main metrics:
  - strict_valid_rate = SV / (SV + WV + IV)
  - inclusive_valid_rate = (SV + WV) / (SV + WV + IV)
  - invalid_rate = IV / (SV + WV + IV)
  - coverage_over_vocab = (SV + WV + IV) / vocab_size

Chinese memory:
- CEDICT hits, single CJK morphemes, and complete numeric/time expressions are SV.
- Content unit + 的/了/着/过/地/得, locative/range suffixes, and natural multi-unit phrases are WV.
- Proper substrings of longer dictionary words are not automatically IV: if they can be naturally decomposed,
  they are WV; otherwise they are IV.
- Current large-dictionary trial uses CEDICT as the base lexicon, then adds tokenizer-vocab compounds
  that can be conservatively segmented into dictionary words plus productive bound morphemes. Examples:
  战俘营 = 战俘 + 营, 系教授 = 系 + 教授, 区道 = 区 + 道, 本篇 = 本 + 篇.
- The same enhanced Chinese compound source is converted with pypinyin into toned/toneless/diacritic
  pinyin entries, so Chinese and pinyin systems receive the same lexical-source expansion.
- Example decisions:
  - 中华人民共和国 -> SV
  - 中华人民共和国的 -> WV
  - 华人民共和国 -> IV, because it is a residual substring of 中华人民共和国.
  - 人民共和国 -> WV, because it can be decomposed as 人民 + 共和国 and appears naturally in XX人民共和国.
  - 的一个 / 是在 / 面积为 -> WV, because they are function/predicate combinations with a known unit.

Pinyin memory:
- We decided not to force pinyin tokens to uniquely map back to Chinese characters.
- Pinyin tokens are judged by complete syllable integrity plus plausible linguistic evidence.
- Ambiguity is acceptable if at least one reasonable Chinese reading exists.
- Single Latin inventory letters such as m/p/q/t/w are EX even if they appear in the pinyin dictionary,
  because they are better treated as base inventory symbols than morphological pinyin units.
- Example decisions from manual review:
  - zui hou yi likely 最后一; the label can be WV even if a naive splitter proposes the wrong split.
  - dui huo de can be 对获得.
  - huai yi shi is 怀疑是.
  - wang zi de is 王子的.
  - jiang zi ji is 将自己.
  - tui te shang is 推特上.
  - jin zhong jiang may be 仅中奖 or 金钟奖, so ambiguity alone is not invalid.
  - di 1 can be 第1 and is acceptable.
  - 1 ri ping jun shang xia che ren ci wei, zai 20 shi ji, and
    50 ping fang qian mi are WV because the number is followed by a complete
    time/measure unit such as 日、世纪、平方千米.
  - yu 2013 is incomplete because it lacks 年 or another time unit, so IV.
  - nian 6 yue 2 ri is incomplete because it lacks the leading year number, so IV.

This is a reproducible heuristic, not a claim that morphology has a single universal tokenizer-validity standard.
The TSV files in this folder are meant for manual inspection and future rule tightening.
The metrics_table.txt file is the human-readable table version of metrics.csv.

Manual audit memory:
- manual_audit_sample.csv and manual_audit_sample.html sample up to 100 tokens from
  each automatic SV/WV/IV group for each tokenizer.
- With four custom tokenizers, the expected maximum is 100 * 3 * 4 = 1200 rows.
- The HTML table gives a single-select manual label control: SV, WV, IV, or UNK.
- UNK is for cases that are plausible but genuinely uncertain and should not be
  forced into a hard SV/WV/IV decision during a quick pass.
- We sample all three automatic groups because the heuristic can miss valid tokens
  in IV, but it can also overestimate SV/WV. This audit is meant to estimate both
  false negatives and false positives.
- The HTML page stores edits in browser localStorage; use Download CSV to export
  the reviewed labels.
