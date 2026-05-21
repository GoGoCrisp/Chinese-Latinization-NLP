# Eval2 整合报告

生成时间：2026-05-08 10:41:19

## 范围

Eval2 包括 Homophone Probe v2、Hard Non-Homophone Control v2、Easy Random Non-Homophone Control v2。主分析使用 `candidate_plus_suffix`；`candidate_only` 作为辅助诊断。

模型：

- `chinese_4epoch`：Chinese 4epoch 模型。
- `diacritic_matched_token_4epoch`：Diacritic matched-token 4epoch 模型。

## 数据集构建概览

| 数据集 | 条目数 | Collapsed | 无效 control collision | 有距离信息条目 | 未使用 train 数据 |
| --- | --- | --- | --- | --- | --- |
| Homophone Probe v2 | 1000 | 229 |  | 1000 | True |
| Hard Non-Homophone Control v2 | 1000 |  | 0 | 1000 | True |
| Easy Random Non-Homophone Control v2 | 1000 |  | 0 | 1000 | True |

## 结果文件检查

每个 1000-item 结果应有 4000 行得分：1000 items x 2 models x 2 scoring modes。

| 结果 | 得分行数 |
| --- | --- |
| Homophone Probe v2 | 4000 |
| Hard Non-Homophone Control v2 | 4000 |
| Easy Random Non-Homophone Control v2 | 4000 |

## 主结果：candidate_plus_suffix

| Probe | Chinese accuracy | Diacritic accuracy | Chinese-Diacritic gap | Homophone 减 probe gap |
| --- | --- | --- | --- | --- |
| Homophone Probe v2 noncollapsed subset | 95.98% | 88.20% | +7.78 pp |  |
| Hard Non-Homophone Control v2 | 97.70% | 91.70% | +6.00 pp | +1.78 pp |
| Easy Random Non-Homophone Control v2 | 98.90% | 93.50% | +5.40 pp | +2.38 pp |

Homophone matched-subset 细节：

| 模型 | All-item accuracy | Chance-adjusted all accuracy | Noncollapsed accuracy | Collapsed accuracy | Mean margin all scored |
| --- | --- | --- | --- | --- | --- |
| Chinese | 95.40% | 95.40% | 95.98% | 93.45% | 0.3913 |
| Diacritic matched-token | 79.45% | 79.45% | 88.20% | 50.00% | 0.1615 |

## 辅助诊断：candidate_only

| Probe | Chinese accuracy | Diacritic accuracy | Chinese-Diacritic gap | Homophone 减 probe gap |
| --- | --- | --- | --- | --- |
| Homophone Probe v2 noncollapsed subset | 88.72% | 69.52% | +19.20 pp |  |
| Hard Non-Homophone Control v2 | 93.20% | 68.80% | +24.40 pp | -5.20 pp |
| Easy Random Non-Homophone Control v2 | 87.70% | 64.90% | +22.80 pp | -3.60 pp |

## 解读

- 主分析中 Homophone noncollapsed gap 为 +7.78 pp：Chinese 95.98% vs Diacritic 88.20%.
- Hard-control gap 为 +6.00 pp；Homophone 减 hard-control gap 为 +1.78 pp。
- Easy-control gap 为 +5.40 pp；Homophone 减 easy-control gap 为 +2.38 pp。
- Homophone 比较使用两个模型共同的 noncollapsed matched subset，避免把 Chinese all-item accuracy 和 Diacritic 中不可分辨的 collapsed items 错位比较。

## 来源文件

- `eval_results/eval2/homophone_probe_v2/summary_matched_subsets.csv`
- `eval_results/eval2/nonhomophone_control_v2/homophone_vs_control_gap.csv`
- `eval_results/eval2/easy_random_control_v2/three_probe_gap_comparison.csv`

