# Final Aligned Korean Diagnostic Corpora

Use these two files as the primary paired corpus inputs for downstream tokenizer
diagnostics:

- `selected_diagnostic_mixed_chunks_nospace.txt`
- `selected_diagnostic_hangulized_chunks_nospace.txt`

Line `i` in the mixed-script file corresponds to line `i` in the Hangulized
file. Both files are no-space corpora with KHDB page markers removed. Article
boundaries are preserved during chunking; chunks are approximately 600 source
characters with hard bounds of 300-800 except final article remainders.

The Hangulized side is automatically produced by Gukhanmun and is not gold
annotation. Check `../summaries/step2_no_space_hangulized_summary.json` and
`../../../results/reports/step2_no_space_hangulized_report.md` before using it
for analysis.

The final Hangulized files are post-cleaned after conversion. Current validation
shows zero Hanja characters in the Hangulized article-level and chunk-level
outputs.

Seed-42 9:1 split outputs are under `splits/seed42_90_10/`:

- `train.mixed_chunks_nospace.txt`
- `train.hangulized_chunks_nospace.txt`
- `dev.mixed_chunks_nospace.txt`
- `dev.hangulized_chunks_nospace.txt`
- `train.chunk_index.jsonl`
- `dev.chunk_index.jsonl`
- `split_summary.json`
