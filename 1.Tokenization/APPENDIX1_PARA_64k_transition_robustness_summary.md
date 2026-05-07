# Appendix: 64k SuperBPE Transition Robustness

This appendix compares the 64k SuperBPE tokenizers trained with five transition settings: fixed K=0, fixed K=1000, 5%, fixed K=2048, and 10%. For 64k vocabularies, 5% and 10% correspond to K=3200 and K=6400. The goal is not to claim that K=2048 is best on every individual metric, but to show how sensitive the main paper results are to the transition point, including the boundary case where SuperBPE starts immediately at K=0.

## Table A. Fertility and Vocabulary Size

| Parameter | Tokenizer | Tokens/original char | Tokens/sample | Total test tokens | Vocabulary total |
|---|---|---:|---:|---:|---:|
| K=0 | Chinese Origin | 0.5448 | 312.8350 | 41,841,991 | 61,174 (63,500) |
| K=0 | Pinyin-Toned | 0.5753 | 330.3163 | 44,180,132 | 50,638 (63,743) |
| K=0 | Pinyin-Toneless | 0.5574 | 320.0353 | 42,805,044 | 49,794 (63,743) |
| K=0 | Pinyin-Diacritic | 0.5980 | 343.3605 | 45,924,808 | 38,336 (63,763) |
| K=1000 | Chinese Origin | 0.5448 | 312.8352 | 41,842,027 | 61,174 (63,500) |
| K=1000 | Pinyin-Toned | 0.5972 | 342.9137 | 45,865,050 | 43,248 (63,756) |
| K=1000 | Pinyin-Toneless | 0.5704 | 327.5111 | 43,804,942 | 46,828 (63,742) |
| K=1000 | Pinyin-Diacritic | 0.5971 | 342.8773 | 45,860,181 | 43,274 (63,760) |
| 5% | Chinese Origin | 0.5448 | 312.7935 | 41,836,443 | 61,167 (63,499) |
| 5% | Pinyin-Toned | 0.5974 | 343.0208 | 45,879,372 | 42,448 (63,756) |
| 5% | Pinyin-Toneless | 0.5705 | 327.5714 | 43,813,002 | 46,394 (63,747) |
| 5% | Pinyin-Diacritic | 0.5974 | 343.0076 | 45,877,605 | 42,439 (63,756) |
| K=2048 | Chinese Origin | 0.5448 | 312.8213 | 41,840,163 | 61,143 (63,499) |
| K=2048 | Pinyin-Toned | 0.5974 | 343.0240 | 45,879,799 | 42,663 (63,755) |
| K=2048 | Pinyin-Toneless | 0.5703 | 327.4896 | 43,802,065 | 46,602 (63,741) |
| K=2048 | Pinyin-Diacritic | 0.5974 | 343.0136 | 45,878,409 | 42,653 (63,754) |
| 10% | Chinese Origin | 0.5447 | 312.7378 | 41,828,988 | 61,135 (63,499) |
| 10% | Pinyin-Toned | 0.5989 | 343.9132 | 45,998,734 | 42,934 (63,763) |
| 10% | Pinyin-Toneless | 0.5726 | 328.7561 | 43,971,452 | 46,739 (63,756) |
| 10% | Pinyin-Diacritic | 0.5989 | 343.9062 | 45,997,795 | 42,927 (63,763) |

Vocabulary total is formatted as unique whitespace-insensitive content count followed by raw vocabulary-entry count in parentheses, matching the convention used in the vocabulary-composition table.

## Table B. Chinese-Origin to Pinyin-Toned Mapping

| Parameter | Mapping type | Chinese source tokens | Pinyin target tokens |
|---|---|---:|---:|
| K=0 | Shared 1:1 | 41,649 | 41,649 |
| K=0 | Shared 2:1 | 3,800 | 1,900 |
| K=0 | Shared 3:1 | 1,023 | 341 |
| K=0 | Shared 4:1 | 584 | 146 |
| K=0 | Shared >4:1 | 3,025 | 362 |
| K=0 | Unique | 11,092 | 6,239 |
| K=1000 | Shared 1:1 | 32,628 | 32,628 |
| K=1000 | Shared 2:1 | 3,728 | 1,864 |
| K=1000 | Shared 3:1 | 1,023 | 341 |
| K=1000 | Shared 4:1 | 584 | 146 |
| K=1000 | Shared >4:1 | 3,025 | 362 |
| K=1000 | Unique | 20,185 | 7,906 |
| 5% | Shared 1:1 | 31,864 | 31,864 |
| 5% | Shared 2:1 | 3,732 | 1,866 |
| 5% | Shared 3:1 | 1,017 | 339 |
| 5% | Shared 4:1 | 584 | 146 |
| 5% | Shared >4:1 | 3,025 | 362 |
| 5% | Unique | 20,944 | 7,870 |
| K=2048 | Shared 1:1 | 32,006 | 32,006 |
| K=2048 | Shared 2:1 | 3,720 | 1,860 |
| K=2048 | Shared 3:1 | 1,017 | 339 |
| K=2048 | Shared 4:1 | 584 | 146 |
| K=2048 | Shared >4:1 | 3,025 | 362 |
| K=2048 | Unique | 20,790 | 7,949 |
| 10% | Shared 1:1 | 31,748 | 31,748 |
| 10% | Shared 2:1 | 3,716 | 1,858 |
| 10% | Shared 3:1 | 1,023 | 341 |
| 10% | Shared 4:1 | 580 | 145 |
| 10% | Shared >4:1 | 3,025 | 362 |
| 10% | Unique | 21,042 | 8,479 |

## Interpretation

K=0 is qualitatively different from the nonzero transition settings. Chinese-Origin remains unchanged at about 0.5448 tokens/original character, and Pinyin-Diacritic remains close to the nonzero settings at about 0.5980. However, Pinyin-Toned drops to 0.5753 under K=0, compared with 0.5972-0.5989 for K=1000 through 10%; Pinyin-Toneless drops to 0.5574, compared with 0.5703-0.5726 for the nonzero settings. Thus K=0 does not simply confirm robustness: it changes tokenization efficiency for the numbered-pinyin settings.

Vocabulary composition also shifts under K=0. For Pinyin-Toned, the unique whitespace-insensitive vocabulary total rises to 50,638, far above the 42,448-43,248 range of the nonzero settings. For Pinyin-Toneless, K=0 rises to 49,794, above the 46,394-46,828 nonzero range. In contrast, Pinyin-Diacritic falls to 38,336, below the 42,439-43,274 nonzero range. This indicates that immediately enabling SuperBPE changes how the 64k vocabulary budget is allocated, rather than only making a small perturbation.

The Chinese-Origin to Pinyin-Toned mapping table shows the same discontinuity. K=0 has 41,649 Shared 1:1 tokens, much higher than the 31,748-32,628 range for K=1000 through 10%, and the Chinese-Origin unique count drops to 11,092 rather than about 20k-21k. The high-order many-to-one structure is still stable: Shared >4:1 remains exactly 3,025 Chinese source tokens to 362 Pinyin target tokens in every setting, and Shared 3:1/4:1 barely move. So K=0 does not remove the structural many-to-one effect of romanization, but it substantially changes the balance between 1:1 shared tokens and unique tokens.

Overall, K=0 should be treated as a boundary-case ablation, not as an interchangeable replacement for K=2048. The nonzero settings from K=1000 to 10% are stable, but K=0 changes fertility, vocabulary composition, and 1:1 overlap enough that it is less suitable as the main experimental setting. K=2048 remains more defensible because it preserves an initial ordinary-BPE phase, is fixed rather than percentage-dependent, and sits in the empirically stable nonzero range.
