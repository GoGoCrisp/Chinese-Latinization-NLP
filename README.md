# Chinese Latinization NLP

中文拉丁化与拼音表示对语言模型的影响研究。This repository studies how Chinese surface writing and Latinized Pinyin representations change tokenizer behavior, corpus ambiguity, and downstream language-model evaluation.

## 中文版

### 项目概述

本项目围绕一个核心问题展开：当中文文本被转换成拼音，尤其是带声调符号的拼音时，分词器、训练语料、语言模型困惑度、同音词判断和语法最小对评测会发生什么变化？

仓库包含三类工作：

- Tokenization：从中文维基语料出发，构造中文原文、无声调拼音、数字声调拼音、声调符号拼音等表示，训练并分析 BPE/SuperBPE tokenizer。
- Model training：构建中文原文模型和拼音声调符号模型的训练配置、数据处理脚本、服务器运行脚本和复现实验配置。
- Evaluation：评估归一化 PPL、同音词 probe、非同音 control、CEVAL/CMMLU 子集、C3 对话选项、ZhoBLiMP 风格中文最小对、CHID 成语填空和多 seed robustness。

### 研究对象

主要比较两种模型输入表示：

- `Chinese-Origin`：原始中文字符序列。
- `Pinyin-Diacritic`：带声调符号的拼音序列，例如 `zhōng guó`。

Tokenizer 层面还分析：

- `pinyin_toneless`：无声调拼音。
- `pinyin_toned`：数字声调拼音。
- `pinyin_diacritic`：声调符号拼音。
- SuperBPE 不同参数、不同 vocabulary size 和不同 K 值设置。

### 仓库结构

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
│   ├── decoded_superTokenizers*/
│   ├── superTokenizers_BPE*/
│   └── dicts/
├── 2.Model trainning/
│   ├── configs/
│   ├── configs/robustness/
│   ├── eval_data/
│   ├── eval_results/
│   ├── figures/
│   ├── scripts/
│   ├── scripts/robustness/
│   ├── tokenizers/
│   └── robustness_training_plan/
├── superbpe
├── .gitignore
└── README.md
```

说明：目录名 `2.Model trainning` 沿用了原始拼写。大语料、训练 checkpoint、模型权重、tokenized Arrow 数据、服务器输出包、缓存和逐行/逐题大明细不纳入 Git。

### 主要流程

1. 清理和切分维基语料。
2. 将中文语料转换为多种拼音表示。
3. 训练和解码 BPE/SuperBPE tokenizer。
4. 分析 tokenizer fertility、词表重叠、同音 collapse、形态一致性和语义分散。
5. 训练中文原文模型与拼音声调符号模型。
6. 在 PPL、homophone probe、control probe、多项选择、语法最小对、成语填空等任务上比较两类表示。
7. 通过 seed 43/44 等 robustness run 检查结果稳定性。

### 评估任务

| 任务 | 目的 | 主要输出 |
| --- | --- | --- |
| Eval1 normalized PPL | 比较字符归一化困惑度 | `eval_results/**/summary.csv` |
| Eval2 homophone probe | 测试同音/近音候选区分能力 | `eval_results/eval2/eval2_integrated_report.md` |
| Eval2 controls | 排除非同音随机或困难候选造成的假差异 | `homophone_vs_control_gap.csv` |
| Eval3 CEVAL/CMMLU subset | 多项选择知识/推理子集 | `eval_results/eval3/` |
| Eval3b C3 dialogue | 对话选项打分 | `eval_results/eval3b_c3_dialogue_option_text/` |
| Eval4 ZhoBLiMP-style | 中文语法最小对 | `eval_results/eval4_chinese_blimp_style/` |
| Eval5 CHID | 成语完形填空 | `eval_results/eval5_chid_idiom_cloze/` |
| Robustness | 多 seed、多训练 regime 汇总 | `eval_results/robustness_134m_eval/summary/` |

### 当前结果摘要

Eval2 4-epoch 主分析使用 `candidate_plus_suffix`。在 noncollapsed homophone subset 上，中文模型为 95.98%，拼音声调符号模型为 88.20%，差距为 +7.78 pp。Hard control 差距为 +6.00 pp，Easy control 差距为 +5.40 pp，说明 homophone 场景中的差距高于普通 control。

Eval4 ZhoBLiMP 风格任务显示，不同语法现象对拼音化的敏感度不同。`fci_licensing`、`npi_licensing`、`verb_phrase`、`BA` 等现象中中文模型优势较大；`question`、`classifier`、`quantifiers` 等现象中拼音声调符号模型在当前设置下并不总是落后。

Robustness 汇总中，matched-token seed 43/44 的整体趋势保持一致：中文原文模型在 Eval1 字符级 PPL、Eval2 controls/homophone 和 Eval4 overall 上通常优于拼音声调符号模型；拼音模型在部分语法现象和特定 scoring 条件下仍有可分析的局部优势。

### 如何复现

建议使用 Python 3.10+，并在项目根目录创建虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r "2.Model trainning/requirements.txt"
```

运行 tokenizer 分析示例：

```bash
cd 1.Tokenization
python 9th_compare_tokenizers_overlap_superBPE.py
python 7th_Tokenizer_Comparison_with_AI.py
```

运行评估示例：

```bash
cd "2.Model trainning"
bash scripts/run_eval_normalized_ppl_linelevel.sh
bash scripts/run_homophone_probe_v2.sh
bash scripts/run_nonhomophone_control_v2.sh
bash scripts/run_eval4_chinese_blimp_style.sh
bash scripts/run_eval5_chid_idiom_cloze.sh
```

Robustness 相关入口：

```bash
cd "2.Model trainning"
python scripts/robustness/validate_robustness_configs.py
bash scripts/robustness/run_robustness_evaluation.sh
python scripts/robustness/summarize_robustness_eval.py
```

### Git 中包含与排除的内容

包含：

- 源代码、运行脚本、训练配置、评估配置。
- 小型 tokenizer 文件、字典文件和元数据。
- 评估 summary、report、diagnostics、表格和图。
- 服务器运行说明与 robustness 计划文档。

排除：

- 维基原始 dump、清理后大语料、拼音大语料。
- tokenized Arrow 数据、checkpoint、模型权重、optimizer state。
- `.tar.gz` 训练包和服务器输出包。
- `.matplotlib-cache`、虚拟环境、日志、系统文件。
- `per_line_scores.csv`、`item_scores.csv` 等可再生成的大型逐行/逐题明细。

### 数据与许可

本项目使用公开中文维基语料、CC-CEDICT/Unihan 相关字典资源，以及公开评测数据的派生构造。请在复现实验时遵守各数据源原始许可。代码默认按 MIT License 使用；如后续添加第三方数据或模型权重，请以其原始许可证为准。

## English

### Overview

This project investigates how Latinizing Chinese into Pinyin, especially tone-marked Pinyin, changes tokenization, lexical ambiguity, language-model training, and downstream evaluation.

The repository covers:

- Tokenization: building Chinese, toneless Pinyin, numbered-tone Pinyin, and diacritic-tone Pinyin corpora and BPE/SuperBPE tokenizers.
- Model training: data preparation, tokenizer packaging, model configs, server scripts, and robustness training plans for Chinese-origin and Pinyin-diacritic language models.
- Evaluation: normalized PPL, homophone probes, non-homophone controls, CEVAL/CMMLU subsets, C3 dialogue option scoring, ZhoBLiMP-style Chinese minimal pairs, CHID idiom cloze, and multi-seed robustness analysis.

### Representations

The main model-level comparison is between:

- `Chinese-Origin`: the original Chinese character surface form.
- `Pinyin-Diacritic`: tone-marked Pinyin, such as `zhōng guó`.

Tokenizer-level analyses also include toneless Pinyin, numbered-tone Pinyin, multiple SuperBPE vocabulary sizes, and several SuperBPE parameter settings.

### Pipeline

1. Clean and split Chinese Wikipedia text.
2. Convert Chinese text into multiple Pinyin representations.
3. Train and decode BPE/SuperBPE tokenizers.
4. Measure fertility, vocabulary overlap, homophone collapse, morphological coherence, and semantic dispersion.
5. Train Chinese-origin and Pinyin-diacritic language models.
6. Evaluate the models on PPL, homophone probes, controls, multiple-choice tasks, grammatical minimal pairs, and idiom cloze.
7. Aggregate robustness runs across random seeds and training regimes.

### Key Results

In the 4-epoch Eval2 `candidate_plus_suffix` setting, the Chinese model reaches 95.98% on the noncollapsed homophone subset, while the Pinyin-diacritic model reaches 88.20%, a +7.78 pp Chinese advantage. The hard-control gap is +6.00 pp and the easy-control gap is +5.40 pp, so the homophone setting shows an additional degradation beyond ordinary controls.

In Eval4 ZhoBLiMP-style minimal pairs, the effect is phenomenon-specific. Chinese is substantially stronger on phenomena such as `fci_licensing`, `npi_licensing`, `verb_phrase`, and `BA`, while the Pinyin-diacritic model is competitive or better on some categories such as `question`, `classifier`, and `quantifiers` under the current setup.

The robustness summaries across matched-token seeds 43/44 preserve the same broad pattern: Chinese-origin models usually outperform Pinyin-diacritic models on character-level PPL, Eval2 probes/controls, and Eval4 overall accuracy, while localized Pinyin advantages remain worth analyzing by phenomenon and scoring mode.

### Reproduction

Create an environment from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r "2.Model trainning/requirements.txt"
```

Tokenizer analysis:

```bash
cd 1.Tokenization
python 9th_compare_tokenizers_overlap_superBPE.py
python 7th_Tokenizer_Comparison_with_AI.py
```

Evaluation:

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

### What Is Tracked

Tracked:

- Source code, shell scripts, configs, tokenizer metadata, small tokenizer assets.
- Evaluation summaries, reports, diagnostics, tables, and figures.
- Server handoff notes and robustness training plans.

Excluded:

- Raw Wikipedia dumps, cleaned large corpora, generated Pinyin corpora.
- Tokenized Arrow datasets, checkpoints, model weights, optimizer states.
- Server output bundles and `.tar.gz` training packages.
- Local caches, virtual environments, logs, and OS metadata.
- Regenerable large detail files such as `per_line_scores.csv` and `item_scores.csv`.

### Citation and License Notes

The project uses public Chinese Wikipedia text, CC-CEDICT/Unihan-style lexical resources, and derived public evaluation data. Follow the licenses of the upstream datasets when reproducing experiments. Project code is intended for MIT-style use unless a file or upstream asset states otherwise.
