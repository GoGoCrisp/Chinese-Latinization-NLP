# Conversion Quality Manual Audit

This directory contains CSV samples for checking whether Chinese-to-Pinyin conversion quality could confound the reported model comparisons. The automatic columns are review aids only; do not treat them as gold error labels.

## How to label zhwiki_conversion_audit_sample.csv

- `manual_polyphone_error_count`: enter the number of characters or words whose Pinyin pronunciation is wrong in context. A polyphone error means the Pinyin pronunciation is wrong in context.
- `manual_segmentation_error_count`: enter the number of jieba segmentation boundary errors that create an unnatural or misleading word boundary affecting Pinyin grouping or a downstream contrast.
- `manual_affects_meaning_or_eval`: use `1`/`true` when the conversion error changes meaning or would plausibly affect an evaluation contrast; otherwise use `0`/`false`.
- `manual_corrected_pinyin_optional`: optionally enter a corrected Pinyin string.
- `manual_comments`: add short evidence or uncertainty notes.

The `potential_polyphone_chars` column is only an automatic review flag produced with pypinyin heteronym mode. Do not mark it as an error unless the pipeline pronunciation is wrong in context.

## How to label Eval2 and Eval4 samples

- `manual_conversion_error_good`: use `1`/`true` if the good/gold side has a conversion error.
- `manual_conversion_error_bad`: use `1`/`true` if the bad/distractor side has a conversion error.
- `manual_error_changes_contrast`: use `1`/`true` if the conversion error changes the intended good/bad contrast, or creates/removes a Pinyin collapse not licensed by true pronunciation.
- `manual_comments`: add the corrected pronunciation or the reason for the label.

Do not mark genuine homophones as conversion errors. Genuine same-pronunciation pairs are expected in homophone probes and collision groups.

## After labeling

Save a labeled copy, for example:

```bash
python scripts/audit_conversion_quality.py \
  --manual_labels conversion_audit/zhwiki_conversion_audit_sample_labeled.csv \
  --manual_labels conversion_audit/eval2_conversion_audit_sample_labeled.csv \
  --manual_labels conversion_audit/eval4_conversion_audit_sample_labeled.csv
```
