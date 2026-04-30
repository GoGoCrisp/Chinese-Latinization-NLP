# 4.5 Compression Efficiency

This folder stores the standalone 4.5 compression efficiency outputs in addition to the full comparison report.

Method memory:
- The current script calls this function compression_efficiency; in the paper outline it corresponds to 4.5.
- It reports vocab-size-derived bits/token, bits/character, bytes/token, and average token id over each tokenizer-specific test file.
- Earlier discussion separated this from fertility: fertility counts tokenizer output length, while compression efficiency translates that length through an assumed fixed-width token id cost.
- In the current fast 4B iteration run, 4.5 is intentionally disabled in run_full_analysis(); this folder may therefore contain only this README unless 4D is re-enabled.
