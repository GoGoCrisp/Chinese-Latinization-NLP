# Chinese Latinization NLP

一个专注于中文拉丁化（拼音转换）的自然语言处理项目，包含多种拼音表示形式的tokenizer比较分析。

## 项目概概述

本项目用于建立和比较中文原始文本与各种拼音表示形式（无声调拼音、带数字声调拼音、带声调符号拼音）之间的tokenizer，并分析它们的重叠率、多音字现象等。

## 目录结构

```
Chinese_Latinization_NLP/
├── 1.Tokenization/              # 主要的tokenization代码和分析
│   ├── 1st_Clean_wiki.py       # 第一步：清理Wikipedia数据
│   ├── 2nd_Segment&token.py    # 第二步：分词和tokenization
│   ├── 3rd_Pinyin_4corpus.py   # 第三步：生成拼音语料
│   ├── 4th_Tokenization_Trainning.py  # 第四步：训练tokenizer
│   ├── 5th_Analyzation.py      # 第五步：分析
│   ├── 6th_Compare with AI.py   # 第六步：与AI tokenizer对比
│   ├── 7th_Tokenizer_Comparison_with_AI.py  # 深度对比分析
│   ├── 9th_compare_tokenizers_overlap.py    # **新增**：多对1映射和多音字检测
│   ├── cleaned_wiki.jsonl       # 清理后的Wikipedia数据
│   ├── corpora/                 # 各种拼音格式的语料库
│   ├── dicts/                   # CC-CEDICT字典文件
│   ├── tokenizers/              # 训练好的tokenizer文件（64K vocab）
│   └── extracted/               # 提取的数据
└── README.md
```

## Tokenizer类型

项目中包含4种主要的tokenizer（64K vocabulary）：

| 类型 | 文件名 | 示例 | 说明 |
|------|--------|------|------|
| **A** | chinese_origin_64k_train90.json | 中文原始 | 原始中文字符 |
| **B** | pinyin_toneless_64k_train90.json | "zhongguo" | 无声调拼音 |
| **C** | pinyin_toned_64k_train90.json | "zhong1guo2" | 带数字声调 |
| **D** | pinyin_diacritic_64k_train90.json | "zhōngguó" | 带声调符号 |

## 核心功能

### 1. Tokenizer对比分析（9th_compare_tokenizers_overlap.py）

主要功能：
- **正向映射（1对N）**：检测同一个词汇对应多个目标词汇的情况
- **反向映射（N对1）**：检测多个不同词汇映射到同一个目标词汇的情况（多音字现象）
- **进度条显示**：实时显示比较进度
- **详细统计**：生成完整的统计报告

### 2. 主要发现

#### A vs B（中文 ↔ 无声调拼音）
- **1对1映射**：43,147个 (67.4%)
- **多对1映射**：3,733个 (5.8%)
  - 2对1: 2,300个
  - 3对1: 672个
  - 最多169对1

**例子**：多个中文词映射到同一个拼音
```
"bujin" ← ["不尽", "不仅"]
"jinxing" ← ["进行", "金星"]
"mushi" ← ["墓室", "牧师", "模式"]
```

#### B vs C（无声调 ↔ 带数字声调）
- **1对1**：100个
- **1对2**：93个 - 两个声调的词
- **1对3**：149个 - 三个声调的词
- **1对4**：313个 - 四个声调的词

**例子**：多音字检测
```
"xie" → ["xie1", "xie4", "xie2", "xie3"]
"sheng" → ["sheng1", "sheng4", "sheng2", "sheng3"]
```

## 使用方法

### 环境设置

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 运行分析

```bash
cd 1.Tokenization

# 比较tokenizer重叠率和多音字现象
python 9th_compare_tokenizers_overlap.py

# 输出：
# - 进度条显示6对比较的进度
# - 生成 tokenizers/tokenizer_overlap_analysis.txt 报告
# - 包含正向和反向映射的详细分析和示例
```

### 输出结果

```
SUMMARY TABLE
Pair            | 1对1      | 1对多        | 多对1        | 独立(1)    | 总覆盖
A_vs_B          | 43147    | 0          | 3733       | 20848    | 46880
A_vs_C          | 39050    | 0          | 2174       | 24945    | 41224
...
```

## 关键脚本说明

### `9th_compare_tokenizers_overlap.py`（主要分析脚本）

**功能**：
- 加载4个64K的tokenizer
- 使用CC-CEDICT字典建立汉字→拼音映射
- 比较所有6对tokenizer的重叠情况
- 检测多音字和同音词现象

**比较规则**：
- **A→B**: 汉字查字典转无声调拼音
- **A→C**: 汉字查字典转带数字拼音
- **A→D**: 汉字查字典转带声调符号拼音
- **B→C**: 拼音添加1-4声调
- **B→D**: 无声调拼音对应带声调
- **C→D**: 去掉数字转声调符号

**输出**：
- `tokenizers/tokenizer_overlap_analysis.txt` - 完整的分析报告

## 数据来源

- **Wikipedia**：中文Wikipedia语料库
- **CC-CEDICT**：CC-CEDICT汉字-拼音字典（121,106个字符）

## 依赖

- Python 3.7+
- tokenizers（HuggingFace）
- transformers
- tqdm
- regex
- pypinyin（某些脚本需要）
- jieba（分词用）

## 文献

- [HuggingFace Tokenizers](https://github.com/huggingface/tokenizers)
- [CC-CEDICT](https://cedict.org/)

## 许可证

MIT License

## 联系方式

如有问题或建议，欢迎提issue。
