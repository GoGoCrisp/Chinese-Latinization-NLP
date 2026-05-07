# Appendix: 64k Max Deviation from K=2048

This table uses K=2048 as the baseline and compares K=1000, K=3200 (5%), and K=6400 (10%). Each cell is selected by the largest absolute percentage deviation from K=2048, but the displayed percentage is signed. K=0 is intentionally excluded from this robustness table.

For overlap total shared, higher is marked as better and lower as worse. For overlap independent and fertility, higher is marked as worse and lower as better. Vocabulary-total rows are directional only, because higher/lower vocabulary totals are not treated as intrinsically better or worse here.

Overlap is computed for the Chinese-Origin to Pinyin-Diacritic pair. For Chinese, total shared is the number of Chinese vocabulary entries mapped to a Diacritic token; for Diacritic, total shared is the number of Diacritic vocabulary entries reached by at least one Chinese entry.

| Metric | Chinese | Diacritic |
|---|---:|---:|
| Vocabulary total, outside parentheses | +0.05% (K=1000, higher) | +1.46% (K=1000, higher) |
| Vocabulary total, inside parentheses | +0.00% (K=1000, higher) | +0.01% (K=6400, higher) |
| Overlap total shared | +1.69% (K=1000, better) | +1.94% (K=1000, better) |
| Overlap independent | -3.14% (K=1000, better) | +6.84% (K=6400, worse) |
| Fertility tokens/original char | -0.02% (K=6400, better) | +0.25% (K=6400, worse) |

## Baseline Values

| Metric | Chinese K=2048 | Diacritic K=2048 |
|---|---:|---:|
| Vocabulary total, outside parentheses | 61,143 | 42,653 |
| Vocabulary total, inside parentheses | 63,499 | 63,754 |
| Overlap total shared | 40,374 | 34,733 |
| Overlap independent | 20,768 | 7,919 |
| Fertility tokens/original char | 0.5448 | 0.5974 |
