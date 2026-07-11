# Korean KHDB Dev Fertility Report

Fertility is computed on the held-out 10% dev split only.
The denominator for cross-representation comparison is the non-space
character count of the original mixed-script dev source.

## Inputs

- mixed tokenizer: `4.Korean/korean_khdb_magazine_audit/data/tokenizers/korean_mixed_bpe_32k.json`
- Hangulized tokenizer: `4.Korean/korean_khdb_magazine_audit/data/tokenizers/korean_hangulized_bpe_32k.json`
- mixed dev: `4.Korean/korean_khdb_magazine_audit/data/corpus/final_aligned/splits/seed42_90_10/dev.mixed_chunks_nospace.txt`
- Hangulized dev: `4.Korean/korean_khdb_magazine_audit/data/corpus/final_aligned/splits/seed42_90_10/dev.hangulized_chunks_nospace.txt`

## Results

| corpus | vocab | lines | tokens/sample | tokens/surface char | tokens/original source char | total tokens | original source chars | UNK/10k original chars |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mixed | 32000 | 1381 | 329.372918 | 0.575685 | 0.575685 | 454864 | 790127 | 3.138736 |
| hangulized | 32000 | 1381 | 293.918899 | 0.513974 | 0.513717 | 405902 | 790127 | 0.999839 |

## Comparison

- absolute fertility reduction: `0.061967` tokens/original source char
- relative fertility reduction: `10.76%`
- total dev token reduction: `48962` tokens

## Notes

- `tokens_per_surface_char` uses each representation's own non-space character count.
- `tokens_per_original_source_char` uses the mixed-script dev source denominator for both tokenizers.
- No special tokens are added during encoding.
