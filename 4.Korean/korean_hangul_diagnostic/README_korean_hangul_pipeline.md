# Korean Hanja-Hangul Diagnostic

This folder is the downstream Korean diagnostic area parallel to the existing
Chinese/Pinyin pipeline and the Japanese kana diagnostic. The corpus source has
changed for this diagnostic.

The new source preparation work starts in:

```text
4.Korean/korean_khdb_magazine_audit/
```

That experiment audits KHDB / 한국사데이터베이스 근현대잡지자료, limited to the
19 magazines officially marked as 원문제공잡지. It downloads only public HTML
pages, extracts article text, and records mixed-script quality filters.

Do not use the previous encyclopedia-dump extraction or Hanja-to-Hangul
conversion path for the current corpus audit.

## Current Scope

For the KHDB source change, the immediate work is:

1. inspect KHDB HTML structure;
2. discover the 19 target magazine indexes;
3. cache candidate issue/article HTML pages with a polite delay;
4. extract metadata and body text while preserving Hanja and Hangul;
5. count mixed-script metrics and save strict/loose article indexes.

This stage does not train tokenizers, does not convert Hanja to Hangul, does
not run fertility or N:1 collision analysis, and does not process newspapers,
images, OCR, hidden APIs, or restricted pages.

## Numbered Outputs

Legacy numbered tokenizer outputs in this folder are not the active first step
for the KHDB corpus source. The active first-step outputs are:

```text
4.Korean/korean_khdb_magazine_audit/
  scripts/
    00_inspect_khdb_html.py
    01_discover_khdb_magazines.py
    02_download_khdb_articles.py
    03_extract_khdb_article_text.py
    04_filter_mixed_script_articles.py
  data/index/
    magazines.jsonl
    articles_index.jsonl
  data/extracted/
    articles_extracted.jsonl
  data/filtered/
    all_articles_with_filter_flags.jsonl
    strict_pass_article_index.jsonl
    loose_pass_article_index.jsonl
    rejected_article_index.jsonl
    filter_summary_by_magazine.csv
    filter_summary_overall.json
  results/reports/
    html_structure_inspection.md
    magazine_discovery_report.md
    mixed_script_filter_report.md
```

## Reproducible Commands

Run these from the repository workspace root:

```bash
python 4.Korean/korean_khdb_magazine_audit/scripts/00_inspect_khdb_html.py
```

```bash
python 4.Korean/korean_khdb_magazine_audit/scripts/01_discover_khdb_magazines.py \
  --root-url https://db.history.go.kr/modern/level.do?itemId=ma \
  --output-dir 4.Korean/korean_khdb_magazine_audit/data/index \
  --cache-dir 4.Korean/korean_khdb_magazine_audit/data/raw_html \
  --delay 1.0 \
  --debug
```

```bash
python 4.Korean/korean_khdb_magazine_audit/scripts/02_download_khdb_articles.py \
  --magazines-index 4.Korean/korean_khdb_magazine_audit/data/index/magazines.jsonl \
  --output-index 4.Korean/korean_khdb_magazine_audit/data/index/articles_index.jsonl \
  --cache-dir 4.Korean/korean_khdb_magazine_audit/data/raw_html \
  --delay 1.0 \
  --max-magazines 1 \
  --max-pages-per-magazine 50 \
  --debug
```

For extraction debugging with known public article pages:

```bash
python 4.Korean/korean_khdb_magazine_audit/scripts/02_download_khdb_articles.py \
  --magazines-index 4.Korean/korean_khdb_magazine_audit/data/index/magazines.jsonl \
  --output-index 4.Korean/korean_khdb_magazine_audit/data/index/articles_index.jsonl \
  --cache-dir 4.Korean/korean_khdb_magazine_audit/data/raw_html \
  --delay 1.0 \
  --max-magazines 0 \
  --seed-article-url https://db.history.go.kr/id/ma_002_0050_0330 \
  --seed-article-url https://db.history.go.kr/id/ma_016_0020_0220 \
  --seed-article-url https://db.history.go.kr/id/ma_016_0840_0480 \
  --debug
```

```bash
python 4.Korean/korean_khdb_magazine_audit/scripts/03_extract_khdb_article_text.py \
  --articles-index 4.Korean/korean_khdb_magazine_audit/data/index/articles_index.jsonl \
  --output-jsonl 4.Korean/korean_khdb_magazine_audit/data/extracted/articles_extracted.jsonl \
  --output-samples 4.Korean/korean_khdb_magazine_audit/results/samples/extraction_samples.md \
  --max-articles 50 \
  --debug
```

```bash
python 4.Korean/korean_khdb_magazine_audit/scripts/04_filter_mixed_script_articles.py \
  --input-jsonl 4.Korean/korean_khdb_magazine_audit/data/extracted/articles_extracted.jsonl \
  --output-dir 4.Korean/korean_khdb_magazine_audit/data/filtered \
  --report 4.Korean/korean_khdb_magazine_audit/results/reports/mixed_script_filter_report.md
```

## Dependencies

The KHDB audit scripts use Python standard-library HTML parsing and network
utilities. No tokenizer or Hanja conversion dependency is required for this
first corpus audit.

## Notes

The target is a mixed Hanja-Hangul historical Korean corpus. The filtering stage
keeps both strict and loose pass indexes and never permanently deletes rejected
articles. The name/office/list filter is heuristic and needs manual inspection
before any downstream diagnostic uses the selected article set.
