# 4.3 Fertility

This folder stores the standalone 4.3 fertility outputs in addition to the full comparison report.

Method memory:
- Fertility is an occurrence-level tokenization efficiency metric.
- The script tokenizes each tokenizer-specific test file and reports tokens per sample, tokens per surface character, total tokens, total characters, and chars/token.
- Earlier discussion noted that pinyin and Chinese surface strings have different character lengths, so 4.3 is useful for within-representation efficiency and should be read carefully for cross-representation claims.
- In the current fast 4B iteration run, 4.3 is intentionally disabled in run_full_analysis(); this folder may therefore contain only this README unless 4A is re-enabled.
