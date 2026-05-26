# Chinese Latinization NLP

This repository contains the code and compact experiment artifacts for:

**What If Chinese Were Latinized? A Counterfactual Study of Script, Tokenization, and Language Modeling**

The project asks a controlled counterfactual question: if Chinese text were written in Latinized Pinyin rather than characters, how would tokenizer behavior and language-model learning change? The experiments keep the underlying Chinese Wikipedia content fixed, convert it into several Pinyin representations, train SuperBPE tokenizers, and compare small Llama-style language models trained on Chinese characters versus tone-marked Pinyin.

## Main Idea

The main comparison is between:

- `Chinese-Origin`: ordinary Chinese character text.
- `Pinyin-Diacritic`: syllable-separated Pinyin with tone marks, for example `zhōng guó rén mín`.

Tokenizer analyses also include:

- `Pinyin-Toneless`: Pinyin without tones.
- `Pinyin-Toned`: Pinyin with numeric tones.
- SuperBPE vocabularies at 8K, 16K, 32K, and 64K.

The central finding is that Latinization is not a neutral re-encoding. Even with tone marks, Pinyin collapses many character-distinguished lexical items and gives the model a harder learning problem under matched-token compute.

## Key Results

| Metric | Chinese-Origin | Pinyin-Diacritic |
| --- | ---: | ---: |
| 64K tokenizer fertility, tokens/line | 312.82 | 343.01 |
| Per-character PPL, matched-token pretraining | 9.14 | 10.53 |
| ZhoBLiMP overall accuracy | 68.40% | 61.19% |

Additional headline results:

- 64K Chinese-to-Pinyin tokenizer alignment contains widespread N:1 homophone collisions; the largest observed Pinyin-Diacritic collision is 41:1.
- In the homophone-interference probe, Chinese models outperform Pinyin models even after removing examples that collapse to the exact same Pinyin surface form.
- Pinyin-Diacritic remains above chance on ZhoBLiMP, but drops most on phenomena where character-distinguished lexical or functional items carry the contrast.

## Repository Layout

```text
Chinese_Latinization_NLP/
├── 1.Tokenization/
│   ├── 1st_Clean_wiki.py
│   ├── 2nd_Segment&token.py
│   ├── 3rd_v2_Pinyin_4corpus.py
│   ├── 4th_v2_superBPE.py
│   ├── 7th_Tokenizer_Comparison_with_AI.py
│   ├── 9th_compare_tokenizers_overlap_superBPE.py
│   ├── 10th_decode_superTokenizers.py
│   ├── 11th_semantic_dispersion_collision_embeddings.py
│   ├── corpora/
│   ├── decoded_superTokenizers*/
│   └── dicts/
├── 2.Model trainning/
│   ├── configs/
│   ├── scripts/
│   ├── tokenizers/
│   ├── eval_data/
│   ├── eval_results/
│   ├── figures/
│   └── robustness_training_plan/
├── superbpe/
└── README.md
```

The directory name `2.Model trainning` preserves the original local spelling.

## Code Map

- `1.Tokenization/3rd_v2_Pinyin_4corpus.py` converts segmented Chinese into toneless, numeric-tone, and diacritic Pinyin corpora using `pypinyin`.
- `1.Tokenization/4th_v2_superBPE.py` and appendix variants train SuperBPE tokenizers under different vocabulary sizes and transition settings.
- `1.Tokenization/9th_compare_tokenizers_overlap_superBPE.py` analyzes cross-script token overlap and homophone collisions.
- `1.Tokenization/11th_semantic_dispersion_collision_embeddings.py` measures semantic dispersion inside collapsed Pinyin groups.
- `2.Model trainning/step5_train_lm_formal.py` trains the final 12-layer Llama-style decoder models.
- `2.Model trainning/scripts/` contains data checks, tokenization, model packaging, and evaluation entry points.
- `2.Model trainning/scripts/robustness/` contains multi-seed robustness evaluation and aggregation utilities.

## Reproduction Notes

Create an environment from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r "2.Model trainning/requirements.txt"
```

Some tokenizer and analysis scripts additionally require packages such as `pypinyin`, `jieba`, `sentence-transformers`, `scikit-learn`, and `pandas`, depending on which stage is being rerun.

Example tokenizer analysis:

```bash
cd 1.Tokenization
python 9th_compare_tokenizers_overlap_superBPE.py
python 7th_Tokenizer_Comparison_with_AI.py
```

Example model evaluation:

```bash
cd "2.Model trainning"
bash scripts/run_eval_normalized_ppl_linelevel.sh
bash scripts/run_homophone_probe_v2.sh
bash scripts/run_nonhomophone_control_v2.sh
bash scripts/run_eval4_chinese_blimp_style.sh
bash scripts/run_eval5_chid_idiom_cloze.sh
```

Robustness utilities:

```bash
cd "2.Model trainning"
python scripts/robustness/validate_robustness_configs.py
bash scripts/robustness/run_robustness_evaluation.sh
python scripts/robustness/summarize_robustness_eval.py
```

These commands assume that the corresponding corpora, tokenized datasets, checkpoints, or server outputs have been restored locally. Large data and model artifacts are intentionally not tracked in Git.

## Tracked vs. Excluded

Tracked:

- Source code, shell scripts, model configs, tokenizer files, and small metadata.
- Summary tables, reports, diagnostics, plots, and evaluation data needed to inspect the final experiments.
- Server handoff notes and robustness plans.

Excluded:

- Raw Wikipedia dumps, large converted corpora, tokenized Arrow datasets, checkpoints, model weights, optimizer states, logs, and generated server bundles.
- Large per-line or per-item score files that can be regenerated from the scripts.

## Data and License Notes

The experiments use public Chinese Wikipedia text, CC-CEDICT/Unihan-derived dictionary resources, and public evaluation datasets or derived probes. Follow the original licenses of each data source when rerunning or redistributing artifacts. Code in this repository is intended for research use; third-party data, models, and tokenizer implementations remain under their own licenses.
