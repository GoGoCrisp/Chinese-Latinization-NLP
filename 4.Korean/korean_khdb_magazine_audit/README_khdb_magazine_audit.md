# KHDB Modern Magazine Mixed-Script Audit

This audit prepares a Korean Hanja-Hangul mixed-script source for the EMNLP
Chinese Latinization diagnostic work. The source is KHDB / 한국사데이터베이스
근현대잡지자료, limited to the 19 magazines officially marked as 원문제공잡지.

## Source and Scope

- Source root: https://db.history.go.kr/modern/level.do?itemId=ma
- Source introduction: https://db.history.go.kr/introduction/modern/intro_ma.html
- Included material: public KHDB HTML pages for the 19 원문제공잡지 magazines.
- Excluded material: newspapers, images, OCR, PDFs, login-only pages, hidden APIs,
  and any access-restricted content.

The target magazine list is embedded in `scripts/khdb_common.py`.

## Crawling Policy

The crawler is deliberately conservative:

- configurable delay, default `1.0` second;
- local HTML cache under `data/raw_html/`;
- resumable runs by reusing cached HTML;
- no authentication, captcha, robots, rate-limit, or access-control bypassing;
- small debug runs before any full crawl.

## Current Status

- `01_discover_khdb_magazines.py` finds all 19/19 target 원문제공잡지.
- Seed article extraction works for the three known public article URLs.
- Full article discovery is now partially solved through the public child-tree
  endpoint:
  `https://db.history.go.kr/modern/getChildItemLevelListAjax.do?parentId=...`
- Full 19-magazine dynamic discovery has been run in `--index-only` mode:
  15,747 public candidate article pages, zero duplicate article IDs, and zero
  unresolved dynamic-tree issues after one retry.
- The bounded diagnostic download reached the small tokenizer target:
  5,908 article HTML pages downloaded, 3,472 selected loose-pass articles, and
  8,000,021 non-space source chars. The target was `8,000,000`, minimum
  `5,000,000`, maximum `12,000,000`, at most `8,000` downloaded articles, and
  at most `1,600,000` selected chars from any one magazine.
- Step 2 no-space and Hangulized corpus preparation has been run with
  Gukhanmun `0.2.0`: 3,472 aligned article lines and 13,813 aligned chunk
  lines. No-space and line-alignment checks pass. After KHDB page-marker
  removal, mixed no-space source chars are 7,919,621. Gukhanmun output is
  post-cleaned to remove Hanja parenthetical annotations and any remaining
  unconverted Hanja; independent scans show zero Hanja characters in the
  Hangulized article-level and chunk-level files.
- Step 3 seed-42 split has been run on aligned chunks: 12,432 train lines and
  1,381 dev lines. Mixed and Hangulized split files remain line-aligned.
- Step 4 has trained two standard HuggingFace BPE tokenizers on the seed-42
  90% training split only. Both mixed-script and Hangulized tokenizers have
  exactly 32,000 vocabulary entries including `[UNK]`, `[PAD]`, `[BOS]`, and
  `[EOS]`. Byte fallback is disabled. Dev-split UNK sanity rates are low:
  3.1387 UNK tokens per 10k source chars for mixed-script and 1.0003 for
  Hangulized; all warning flags are `ok`.
- Step 5 dev fertility has been computed on the held-out 10% split using the
  original mixed-script non-space chars as the cross-representation denominator:
  mixed-script BPE is `0.575685` tokens/original source char and Hangulized BPE
  is `0.513717`, a `10.76%` relative reduction.
- Step 6 vocabulary N:1 collision analysis has been run with Gukhanmun `0.2.0`.
  The mixed tokenizer has `19,999` Hanja-containing vocab tokens; `18,068`
  exactly match a Hangulized vocab surface after conversion. The Hanja-token
  exact overlap rate is `0.903445`. There are `1,413` N:1 collision groups,
  maximum collision size is `89`, and `61.53%` of dev Hanja-token occurrences
  belong to N:1 collision groups. As an additional robustness slice within the
  same result, `1,019` collision groups have converted Hangul length at least
  `2`; these account for `11.36%` of dev Hanja-token occurrences.

## Extraction Caveat

KHDB article text is curated/transcribed public HTML. It may be normalized by
KHDB and should not be treated as raw scanned historical print. The extraction
scripts preserve Hanja and Hangul, normalize Unicode with NFKC, and avoid
modernizing the text.

## Filtering Criteria

The filtering stage records every article and assigns several tiers:

- `basic_pass`: length and minimum Hanja/Hangul counts pass.
- `balanced_mixed`: `0.05 <= hanja_ratio <= 0.70` and no explicit Japanese
  omission marker failure.
- `hanja_heavy_mixed`: `0.70 < hanja_ratio <= 0.90`, enough Hangul context,
  and enough Hanja+Korean particle contexts.
- `loose`: balanced or Hanja-heavy mixed.
- `strict`: loose plus the heuristic content-mixed filter.
- `near_classical_or_mostly_hanja`: mostly Hanja or too little Hangul context.

The strict filter is intentionally heuristic. Name/office/title markers are
diagnostics, not standalone rejection reasons. Rejected articles are not
deleted; all filter decisions and metrics are saved.

## Outputs

```text
4.Korean/korean_khdb_magazine_audit/
  scripts/
    00_inspect_khdb_html.py
    01_discover_khdb_magazines.py
    02_download_khdb_articles.py
    03_extract_khdb_article_text.py
    04_filter_mixed_script_articles.py
    05_investigate_khdb_dynamic_tree.py
    khdb_common.py
    06_download_bounded_article_html.py
    07_select_bounded_corpus_index.py
    08_export_selected_corpus_text.py
    09_prepare_no_space_and_hangulized_corpus.py
    10_split_aligned_corpus.py
    11_train_korean_bpe_tokenizers.py
    12_compute_korean_tokenizer_fertility.py
    13_korean_vocab_n_to_1_analysis.py
  data/
    raw_html/
    index/
      magazines.jsonl
      articles_index.jsonl
      articles_index_full19.jsonl
      articles_downloaded_bounded.jsonl
      nodes_index.jsonl
      nodes_index_full19.jsonl
      crawl_graph.jsonl
      crawl_graph_full19.jsonl
      download_summary.json
      download_summary_full19.json
      bounded_download_summary.json
      dynamic_endpoint_candidates.json
    extracted/
      articles_extracted.jsonl
      articles_extracted_bounded.jsonl
    filtered/
      all_articles_with_filter_flags.jsonl
      all_articles_with_filter_flags_bounded.jsonl
      balanced_mixed_article_index.jsonl
      balanced_mixed_article_index_bounded.jsonl
      hanja_heavy_mixed_article_index.jsonl
      hanja_heavy_mixed_article_index_bounded.jsonl
      strict_pass_article_index.jsonl
      strict_pass_article_index_bounded.jsonl
      loose_pass_article_index.jsonl
      loose_pass_article_index_bounded.jsonl
      near_classical_or_mostly_hanja_index.jsonl
      near_classical_or_mostly_hanja_index_bounded.jsonl
      rejected_article_index.jsonl
      rejected_article_index_bounded.jsonl
      filter_summary_by_magazine.csv
      filter_summary_by_magazine_bounded.csv
      filter_summary_overall.json
      filter_summary_overall_bounded.json
      selected_diagnostic_article_index.jsonl
      selected_diagnostic_corpus_summary.json
    corpus/
      article_level/
        selected_diagnostic_mixed_articles.jsonl
        selected_diagnostic_mixed_article_texts.txt
        selected_diagnostic_mixed_articles_nospace.jsonl
        selected_diagnostic_mixed_nospace.txt
        selected_diagnostic_hangulized_nospace.txt
      final_aligned/
        README.md
        selected_diagnostic_mixed_chunks_nospace.txt
        selected_diagnostic_hangulized_chunks_nospace.txt
        selected_diagnostic_chunk_index.jsonl
        splits/
          seed42_90_10/
            train.mixed_chunks_nospace.txt
            train.hangulized_chunks_nospace.txt
            dev.mixed_chunks_nospace.txt
            dev.hangulized_chunks_nospace.txt
            train.chunk_index.jsonl
            dev.chunk_index.jsonl
            split_summary.json
      summaries/
        selected_diagnostic_mixed_corpus_summary.json
        step2_no_space_hangulized_summary.json
      debug/
        debug20/
    tokenizers/
      korean_mixed_bpe_32k.json
      korean_hangulized_bpe_32k.json
      korean_mixed_bpe_32k_vocab.txt
      korean_hangulized_bpe_32k_vocab.txt
      tokenizer_training_summary.json
      tokenizer_training_report.md
      fertility_dev_summary.json
      fertility_dev_results.csv
  results/
    samples/
      extraction_samples.md
      extraction_samples_bounded.md
    reports/
      html_structure_inspection.md
      khdb_dynamic_tree_investigation.md
      article_discovery_report.md
      article_discovery_full19_report.md
      bounded_download_report.md
      magazine_discovery_report.md
      mixed_script_filter_report.md
      mixed_script_filter_report_bounded.md
      selected_diagnostic_corpus_report.md
      step2_no_space_hangulized_report.md
      tokenizer_dev_fertility_report.md
    korean_n_to_1/
      korean_vocab_n_to_1_summary.json
      mixed_vocab_converted_to_hangul.jsonl
      n_to_1_groups.jsonl
      n_to_1_groups_collisions_only.jsonl
      top_n_to_1_groups_by_size.csv
      top_n_to_1_groups_by_dev_frequency.csv
      n_to_1_groups_collisions_converted_len_ge_2.jsonl
      top_n_to_1_groups_len_ge_2_by_size.csv
      top_n_to_1_groups_len_ge_2_by_dev_frequency.csv
      korean_n_to_1_report.md
      korean_n_to_1_robustness_len_ge_2_report.md
```

## Reproducible Commands

Run from the repository workspace root.

Debug inspection:

```bash
python 4.Korean/korean_khdb_magazine_audit/scripts/00_inspect_khdb_html.py
```

Discover the 19 magazine roots:

```bash
python 4.Korean/korean_khdb_magazine_audit/scripts/01_discover_khdb_magazines.py \
  --root-url https://db.history.go.kr/modern/level.do?itemId=ma \
  --output-dir 4.Korean/korean_khdb_magazine_audit/data/index \
  --cache-dir 4.Korean/korean_khdb_magazine_audit/data/raw_html \
  --delay 1.0 \
  --debug
```

Dynamic tree investigation:

```bash
python 4.Korean/korean_khdb_magazine_audit/scripts/05_investigate_khdb_dynamic_tree.py \
  --max-magazines 2 \
  --delay 1.0
```

Small dynamic article discovery test:

```bash
python 4.Korean/korean_khdb_magazine_audit/scripts/02_download_khdb_articles.py \
  --magazines-index 4.Korean/korean_khdb_magazine_audit/data/index/magazines.jsonl \
  --output-index 4.Korean/korean_khdb_magazine_audit/data/index/articles_index.jsonl \
  --cache-dir 4.Korean/korean_khdb_magazine_audit/data/raw_html \
  --discovery-mode dynamic \
  --delay 1.0 \
  --magazine-title 삼천리 \
  --max-articles 10 \
  --max-depth 4 \
  --debug
```

Full 19-magazine dynamic discovery, index only:

```bash
python 4.Korean/korean_khdb_magazine_audit/scripts/02_download_khdb_articles.py \
  --magazines-index 4.Korean/korean_khdb_magazine_audit/data/index/magazines.jsonl \
  --output-index 4.Korean/korean_khdb_magazine_audit/data/index/articles_index_full19.jsonl \
  --nodes-index 4.Korean/korean_khdb_magazine_audit/data/index/nodes_index_full19.jsonl \
  --graph-index 4.Korean/korean_khdb_magazine_audit/data/index/crawl_graph_full19.jsonl \
  --summary 4.Korean/korean_khdb_magazine_audit/data/index/download_summary_full19.json \
  --cache-dir 4.Korean/korean_khdb_magazine_audit/data/raw_html \
  --discovery-mode dynamic \
  --index-only \
  --delay 1.0 \
  --max-depth 10 \
  --resume \
  --report 4.Korean/korean_khdb_magazine_audit/results/reports/article_discovery_full19_report.md
```

Bounded article HTML download toward the diagnostic target:

```bash
python 4.Korean/korean_khdb_magazine_audit/scripts/06_download_bounded_article_html.py \
  --articles-index 4.Korean/korean_khdb_magazine_audit/data/index/articles_index_full19.jsonl \
  --cache-dir 4.Korean/korean_khdb_magazine_audit/data/raw_html \
  --output-index 4.Korean/korean_khdb_magazine_audit/data/index/articles_downloaded_bounded.jsonl \
  --target-source-chars 8000000 \
  --minimum-source-chars 5000000 \
  --max-source-chars 12000000 \
  --max-articles 8000 \
  --per-magazine-char-cap 1600000 \
  --delay 1.0 \
  --seed 42 \
  --resume
```

Known article-page debug using `삼천리`:

```bash
python 4.Korean/korean_khdb_magazine_audit/scripts/02_download_khdb_articles.py \
  --magazines-index 4.Korean/korean_khdb_magazine_audit/data/index/magazines.jsonl \
  --output-index 4.Korean/korean_khdb_magazine_audit/data/index/articles_index.jsonl \
  --cache-dir 4.Korean/korean_khdb_magazine_audit/data/raw_html \
  --delay 1.0 \
  --magazine-title 삼천리 \
  --max-pages-per-magazine 50 \
  --debug
```

Known direct article seed debug:

```bash
python 4.Korean/korean_khdb_magazine_audit/scripts/02_download_khdb_articles.py \
  --magazines-index 4.Korean/korean_khdb_magazine_audit/data/index/magazines.jsonl \
  --output-index 4.Korean/korean_khdb_magazine_audit/data/index/articles_index.jsonl \
  --cache-dir 4.Korean/korean_khdb_magazine_audit/data/raw_html \
  --discovery-mode dynamic \
  --delay 1.0 \
  --max-magazines 0 \
  --max-articles 3 \
  --seed-article-url https://db.history.go.kr/id/ma_002_0050_0330 \
  --seed-article-url https://db.history.go.kr/id/ma_016_0020_0220 \
  --seed-article-url https://db.history.go.kr/id/ma_016_0840_0480 \
  --debug
```

Extract small test:

```bash
python 4.Korean/korean_khdb_magazine_audit/scripts/03_extract_khdb_article_text.py \
  --articles-index 4.Korean/korean_khdb_magazine_audit/data/index/articles_index.jsonl \
  --output-jsonl 4.Korean/korean_khdb_magazine_audit/data/extracted/articles_extracted.jsonl \
  --output-samples 4.Korean/korean_khdb_magazine_audit/results/samples/extraction_samples.md \
  --max-articles 50 \
  --debug
```

Filter small test:

```bash
python 4.Korean/korean_khdb_magazine_audit/scripts/04_filter_mixed_script_articles.py \
  --input-jsonl 4.Korean/korean_khdb_magazine_audit/data/extracted/articles_extracted.jsonl \
  --output-dir 4.Korean/korean_khdb_magazine_audit/data/filtered \
  --report 4.Korean/korean_khdb_magazine_audit/results/reports/mixed_script_filter_report.md
```

Extract bounded articles:

```bash
python 4.Korean/korean_khdb_magazine_audit/scripts/03_extract_khdb_article_text.py \
  --articles-index 4.Korean/korean_khdb_magazine_audit/data/index/articles_downloaded_bounded.jsonl \
  --output-jsonl 4.Korean/korean_khdb_magazine_audit/data/extracted/articles_extracted_bounded.jsonl \
  --output-samples 4.Korean/korean_khdb_magazine_audit/results/samples/extraction_samples_bounded.md
```

Filter bounded articles without overwriting debug outputs:

```bash
python 4.Korean/korean_khdb_magazine_audit/scripts/04_filter_mixed_script_articles.py \
  --input-jsonl 4.Korean/korean_khdb_magazine_audit/data/extracted/articles_extracted_bounded.jsonl \
  --output-dir 4.Korean/korean_khdb_magazine_audit/data/filtered \
  --output-suffix bounded \
  --report 4.Korean/korean_khdb_magazine_audit/results/reports/mixed_script_filter_report_bounded.md
```

Select final diagnostic article index:

```bash
python 4.Korean/korean_khdb_magazine_audit/scripts/07_select_bounded_corpus_index.py \
  --filtered-jsonl 4.Korean/korean_khdb_magazine_audit/data/filtered/all_articles_with_filter_flags_bounded.jsonl \
  --output-selected-index 4.Korean/korean_khdb_magazine_audit/data/filtered/selected_diagnostic_article_index.jsonl \
  --output-summary 4.Korean/korean_khdb_magazine_audit/data/filtered/selected_diagnostic_corpus_summary.json \
  --target-source-chars 8000000 \
  --minimum-source-chars 5000000 \
  --max-source-chars 12000000 \
  --per-magazine-char-cap 1600000 \
  --seed 42
```

Export selected mixed-script corpus text:

```bash
python 4.Korean/korean_khdb_magazine_audit/scripts/08_export_selected_corpus_text.py \
  --selected-index 4.Korean/korean_khdb_magazine_audit/data/filtered/selected_diagnostic_article_index.jsonl \
  --filtered-jsonl 4.Korean/korean_khdb_magazine_audit/data/filtered/all_articles_with_filter_flags_bounded.jsonl \
  --output-jsonl 4.Korean/korean_khdb_magazine_audit/data/corpus/article_level/selected_diagnostic_mixed_articles.jsonl \
  --output-text 4.Korean/korean_khdb_magazine_audit/data/corpus/article_level/selected_diagnostic_mixed_article_texts.txt \
  --output-summary 4.Korean/korean_khdb_magazine_audit/data/corpus/summaries/selected_diagnostic_mixed_corpus_summary.json
```

## Step 2: No-Space and Hangulized Corpora

Step 2 prepares tokenizer-diagnostic inputs. It removes internal whitespace and
KHDB page markers from selected article text, then creates aligned mixed-script
and Hangulized files. This does not train tokenizers, compute fertility, or
compute N:1 pairs.

Install the primary external converter, Gukhanmun:

```bash
cargo install gukhanmun-cli gukhanmun-mkdict
```

Alternatively, download a prebuilt `gukhanmun` executable from:
https://github.com/dahlia/gukhanmun/releases

Optional fallback, for sanity comparison or if Gukhanmun is unavailable:

```bash
pip install hanja
```

Gukhanmun is GPL-3.0. Its source is not vendored in this repository; the local
pipeline only calls the installed CLI as an external preprocessing tool and
records converter name/version in the summary.

Run a 20-article debug test first:

```bash
python 4.Korean/korean_khdb_magazine_audit/scripts/09_prepare_no_space_and_hangulized_corpus.py \
  --input-jsonl 4.Korean/korean_khdb_magazine_audit/data/corpus/article_level/selected_diagnostic_mixed_articles.jsonl \
  --output-dir 4.Korean/korean_khdb_magazine_audit/data/corpus \
  --converter gukhanmun \
  --fallback-converter hanja \
  --chunk-size 600 \
  --min-chunk-size 300 \
  --max-chunk-size 800 \
  --max-articles 20 \
  --debug
```

Run full Step 2:

```bash
python 4.Korean/korean_khdb_magazine_audit/scripts/09_prepare_no_space_and_hangulized_corpus.py \
  --input-jsonl 4.Korean/korean_khdb_magazine_audit/data/corpus/article_level/selected_diagnostic_mixed_articles.jsonl \
  --output-dir 4.Korean/korean_khdb_magazine_audit/data/corpus \
  --converter gukhanmun \
  --fallback-converter hanja \
  --chunk-size 600 \
  --min-chunk-size 300 \
  --max-chunk-size 800
```

Step 2 outputs:

- `selected_diagnostic_mixed_articles_nospace.jsonl`: article metadata plus
  original body, no-space mixed text, Hangulized no-space text, counts, and
  conversion notes under `data/corpus/article_level/`.
- `selected_diagnostic_mixed_nospace.txt` and
  `selected_diagnostic_hangulized_nospace.txt`: article-level aligned text
  under `data/corpus/article_level/`.
- `data/corpus/final_aligned/selected_diagnostic_mixed_chunks_nospace.txt` and
  `data/corpus/final_aligned/selected_diagnostic_hangulized_chunks_nospace.txt`:
  the two primary corpora for downstream tokenizer diagnostics. They are
  approximately 600-source-char chunks, preserve article boundaries, and line
  `i` is aligned across both files.
- `data/corpus/final_aligned/selected_diagnostic_chunk_index.jsonl`: chunk-to-article metadata and
  conversion diagnostics.
- `step2_no_space_hangulized_summary.json` and
  `step2_no_space_hangulized_report.md`: validation summary under
  `data/corpus/summaries/` and human-readable audit report under
  `results/reports/`.

The Hangulized corpus is automatic Hanja-to-Hangul conversion, not gold
annotation. Historical spellings, names, rare Hanja, and Classical
Chinese-heavy passages may be converted imperfectly. The final Hangulized files
are post-cleaned so they contain zero Hanja characters.

## Step 3: Seed-42 9:1 Split

Step 3 splits the aligned chunk-level corpora with the same shuffled line
indices for mixed and Hangulized text.

```bash
python 4.Korean/korean_khdb_magazine_audit/scripts/10_split_aligned_corpus.py \
  --mixed 4.Korean/korean_khdb_magazine_audit/data/corpus/final_aligned/selected_diagnostic_mixed_chunks_nospace.txt \
  --hangulized 4.Korean/korean_khdb_magazine_audit/data/corpus/final_aligned/selected_diagnostic_hangulized_chunks_nospace.txt \
  --chunk-index 4.Korean/korean_khdb_magazine_audit/data/corpus/final_aligned/selected_diagnostic_chunk_index.jsonl \
  --output-dir 4.Korean/korean_khdb_magazine_audit/data/corpus/final_aligned/splits/seed42_90_10 \
  --seed 42 \
  --train-ratio 0.9
```

Step 3 outputs:

- `data/corpus/final_aligned/splits/seed42_90_10/train.mixed_chunks_nospace.txt`
- `data/corpus/final_aligned/splits/seed42_90_10/train.hangulized_chunks_nospace.txt`
- `data/corpus/final_aligned/splits/seed42_90_10/dev.mixed_chunks_nospace.txt`
- `data/corpus/final_aligned/splits/seed42_90_10/dev.hangulized_chunks_nospace.txt`
- `data/corpus/final_aligned/splits/seed42_90_10/train.chunk_index.jsonl`
- `data/corpus/final_aligned/splits/seed42_90_10/dev.chunk_index.jsonl`
- `data/corpus/final_aligned/splits/seed42_90_10/split_summary.json`

## Step 4: 32K BPE Tokenizer Training

Step 4 trains two comparable 32K standard BPE tokenizers, one on the
mixed-script train split and one on the Hangulized train split. It uses only the
90% train split for training; the 10% dev split is used only for loading and
UNK sanity checks. This step does not compute final fertility or N:1 pairs.

The tokenizer configuration is:

- HuggingFace `tokenizers` BPE;
- vocab size `32000`, including `[UNK]`, `[PAD]`, `[BOS]`, and `[EOS]`;
- NFKC normalizer;
- no whitespace pre-tokenizer;
- byte fallback disabled by default.

Run tokenizer training:

```bash
python 4.Korean/korean_khdb_magazine_audit/scripts/11_train_korean_bpe_tokenizers.py \
  --train-mixed 4.Korean/korean_khdb_magazine_audit/data/corpus/final_aligned/splits/seed42_90_10/train.mixed_chunks_nospace.txt \
  --train-hangulized 4.Korean/korean_khdb_magazine_audit/data/corpus/final_aligned/splits/seed42_90_10/train.hangulized_chunks_nospace.txt \
  --dev-mixed 4.Korean/korean_khdb_magazine_audit/data/corpus/final_aligned/splits/seed42_90_10/dev.mixed_chunks_nospace.txt \
  --dev-hangulized 4.Korean/korean_khdb_magazine_audit/data/corpus/final_aligned/splits/seed42_90_10/dev.hangulized_chunks_nospace.txt \
  --output-dir 4.Korean/korean_khdb_magazine_audit/data/tokenizers \
  --vocab-size 32000 \
  --seed 42
```

Step 4 outputs:

- `data/tokenizers/korean_mixed_bpe_32k.json`
- `data/tokenizers/korean_hangulized_bpe_32k.json`
- `data/tokenizers/korean_mixed_bpe_32k_vocab.txt`
- `data/tokenizers/korean_hangulized_bpe_32k_vocab.txt`
- `data/tokenizers/tokenizer_training_summary.json`
- `data/tokenizers/tokenizer_training_report.md`

Current Step 4 sanity results:

- mixed-script vocab size: `32000`
- Hangulized vocab size: `32000`
- train lines: `12432`; dev lines: `1381`
- line alignment, no-space, and Hangulized no-Hanja checks: pass
- mixed-script dev UNK rate: `3.1387` per 10k source chars
- Hangulized dev UNK rate: `1.0003` per 10k source chars
- all UNK warning flags: `ok`

The token-per-source-char numbers in the training summary are preliminary
tokenizer sanity diagnostics only. Step 5 computes the dev fertility explicitly.

## Step 5: Dev Fertility

Step 5 follows the earlier `1.Tokenization` fertility convention:

- `tokens_per_sample = total_tokens / dev_line_count`;
- `tokens_per_surface_char = total_tokens / non-space chars in the evaluated
  representation`;
- `tokens_per_original_source_char = total_tokens / non-space chars in the
  original mixed-script dev source`.

Run dev fertility:

```bash
python 4.Korean/korean_khdb_magazine_audit/scripts/12_compute_korean_tokenizer_fertility.py \
  --mixed-tokenizer 4.Korean/korean_khdb_magazine_audit/data/tokenizers/korean_mixed_bpe_32k.json \
  --hangulized-tokenizer 4.Korean/korean_khdb_magazine_audit/data/tokenizers/korean_hangulized_bpe_32k.json \
  --dev-mixed 4.Korean/korean_khdb_magazine_audit/data/corpus/final_aligned/splits/seed42_90_10/dev.mixed_chunks_nospace.txt \
  --dev-hangulized 4.Korean/korean_khdb_magazine_audit/data/corpus/final_aligned/splits/seed42_90_10/dev.hangulized_chunks_nospace.txt
```

Step 5 outputs:

- `data/tokenizers/fertility_dev_summary.json`
- `data/tokenizers/fertility_dev_results.csv`
- `results/reports/tokenizer_dev_fertility_report.md`

Current Step 5 dev results:

| corpus | tokens/sample | tokens/surface char | tokens/original source char | total tokens |
|---|---:|---:|---:|---:|
| mixed | 329.372918 | 0.575685 | 0.575685 | 454864 |
| hangulized | 293.918899 | 0.513974 | 0.513717 | 405902 |

The Hangulized tokenizer reduces dev fertility by `0.061967` tokens/original
source char, or `10.76%` relative to the mixed-script tokenizer.

## Step 6: Vocabulary N:1 Collision Analysis

Step 6 analyzes the recoverability-cost side of the tokenizer trade-off. It
takes mixed-script BPE vocabulary tokens, converts them to Hangul with the same
Gukhanmun settings and post-cleaning rules used in Step 2, then groups
Hanja-containing source tokens by their converted Hangul surface.

Run N:1 analysis:

```bash
python 4.Korean/korean_khdb_magazine_audit/scripts/13_korean_vocab_n_to_1_analysis.py \
  --mixed-tokenizer 4.Korean/korean_khdb_magazine_audit/data/tokenizers/korean_mixed_bpe_32k.json \
  --hangulized-tokenizer 4.Korean/korean_khdb_magazine_audit/data/tokenizers/korean_hangulized_bpe_32k.json \
  --train-mixed 4.Korean/korean_khdb_magazine_audit/data/corpus/final_aligned/splits/seed42_90_10/train.mixed_chunks_nospace.txt \
  --dev-mixed 4.Korean/korean_khdb_magazine_audit/data/corpus/final_aligned/splits/seed42_90_10/dev.mixed_chunks_nospace.txt \
  --output-dir 4.Korean/korean_khdb_magazine_audit/results/korean_n_to_1 \
  --converter gukhanmun \
  --seed 42
```

Step 6 outputs:

- `results/korean_n_to_1/korean_vocab_n_to_1_summary.json`
- `results/korean_n_to_1/mixed_vocab_converted_to_hangul.jsonl`
- `results/korean_n_to_1/n_to_1_groups.jsonl`
- `results/korean_n_to_1/n_to_1_groups_collisions_only.jsonl`
- `results/korean_n_to_1/top_n_to_1_groups_by_size.csv`
- `results/korean_n_to_1/top_n_to_1_groups_by_dev_frequency.csv`
- `results/korean_n_to_1/korean_n_to_1_report.md`
- `results/korean_n_to_1/n_to_1_groups_collisions_converted_len_ge_2.jsonl`
- `results/korean_n_to_1/top_n_to_1_groups_len_ge_2_by_size.csv`
- `results/korean_n_to_1/top_n_to_1_groups_len_ge_2_by_dev_frequency.csv`
- `results/korean_n_to_1/korean_n_to_1_robustness_len_ge_2_report.md`

Current Step 6 N:1 distribution:

| collision size | mixed source tokens | Hangulized surfaces |
|---|---:|---:|
| 1:1 | 10159 | 10159 |
| 2:1 | 1590 | 795 |
| 3:1 | 567 | 189 |
| 4:1 | 320 | 80 |
| >4:1 | 6139 | 349 |

Key Step 6 results:

- mixed Hanja-containing vocab tokens: `19999`
- Hangulized pure Hangul strict tokens: `25461`
- Hanja-token exact overlap: `18068 / 19999 = 0.903445`
- N:1 collision groups: `1413`
- max collision size: `89`
- dev Hanja-token occurrences in N:1 groups: `149129 / 242352 = 61.53%`

Length>=2 subset within the same N:1 collision result:

| collision size | mixed source tokens | Hangulized surfaces |
|---|---:|---:|
| 2:1 | 1552 | 776 |
| 3:1 | 486 | 162 |
| 4:1 | 208 | 52 |
| >4:1 | 170 | 29 |

- length>=2 collision groups: `1019 / 1413`
- length>=2 max collision size: `11`
- length>=2 dev Hanja-token occurrences: `27534 / 242352 = 11.36%`

Interpretation: the Hangulized tokenizer is more compact on dev fertility, but
many distinct mixed-script Hanja vocabulary tokens collapse to the same Hangul
surface. This is a tokenizer-vocabulary diagnostic, not a gold lexical
ambiguity dataset.
