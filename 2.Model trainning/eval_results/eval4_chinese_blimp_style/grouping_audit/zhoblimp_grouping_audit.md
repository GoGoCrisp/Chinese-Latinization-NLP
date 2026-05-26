# ZhoBLiMP 分组审计报告

## 数据来源与方法

本报告只读取已有 Eval4/ZhoBLiMP 输出，没有重新跑模型推理，也没有改动原始评测结果。使用的文件包括：

- `eval_results/eval4_chinese_blimp_style/summary_by_phenomenon.csv`：seed42 的 Chinese 与 Pinyin-token。
- `eval_results/robustness_134m_eval/eval4_zhoblimp/summary_by_phenomenon.csv`：Chinese seed43/44、Pinyin-token seed43/44、Pinyin-data seed42。
- `eval_results/robustness_matched_data_diacritic_seed43_44_eval/eval4_zhoblimp/summary_by_phenomenon.csv`：Pinyin-data seed43/44。
- `eval_data/eval4_chinese_blimp_style/eval4_chinese_blimp_style.jsonl` 与 `item_scores.csv` 用于核对标签字段和 collapsed/noncollapsed 定义。

聚合方式：每个 analysis category 先在每个 seed 内对所含官方现象做 macro average，然后报告三个 seed 的 mean +/- sample SD。`Anaphor*` 按你的当前主表设定使用 `noncollapsed_accuracy`；其他组使用 all-item `accuracy`。CSV 中另附 `item_weighted_accuracy` 作为检查列，但正文建议明确使用 macro-over-phenomena。

## 官方现象标签与分组映射

这些标签均直接来自现有结果文件，未手工假设。

| Official phenomenon | Analysis category    | Metric                | n/item | Note                                                              |
| ------------------- | -------------------- | --------------------- | ------ | ----------------------------------------------------------------- |
| anaphor             | Anaphor*             | noncollapsed_accuracy | 1800   | Anaphor; main table uses non-collapsed subset                     |
| argument_structure  | Argument structure   | accuracy              | 2100   | Argument structure                                                |
| BA                  | Constructional       | accuracy              | 3900   | BA construction                                                   |
| control_raising     | Constructional       | accuracy              | 1200   | Control/raising                                                   |
| verb_phrase         | Constructional       | accuracy              | 4200   | Verb phrase                                                       |
| fci_licensing       | Functional licensing | accuracy              | 1500   | FCI licensing                                                     |
| npi_licensing       | Functional licensing | accuracy              | 2700   | NPI licensing                                                     |
| passive             | Order-sensitive      | accuracy              | 3600   | Passive                                                           |
| relativization      | Order-sensitive      | accuracy              | 1200   | Relativization                                                    |
| topicalization      | Order-sensitive      | accuracy              | 1200   | Topicalization                                                    |
| classifier          | Unstable categories  | accuracy              | 900    | Unstable: not otherwise grouped; near/below chance                |
| ellipsis            | Unstable categories  | accuracy              | 900    | Unstable: not otherwise grouped; near/below chance                |
| nominal_expression  | Unstable categories  | accuracy              | 3300   | Unstable: not otherwise grouped; near/below chance                |
| quantifiers         | Unstable categories  | accuracy              | 600    | Unstable: not otherwise grouped; below chance in at least one run |
| question            | Unstable categories  | accuracy              | 6300   | Unstable: not otherwise grouped; near chance in at least one run  |

## 重算聚合结果

数值为 accuracy 百分比，gap 为 Chinese minus Pinyin，单位同为百分点。

| Category             | n | Chinese      | Pinyin-token | Pinyin-data  | Zh-Token gap  | Zh-Data gap   |
| -------------------- | - | ------------ | ------------ | ------------ | ------------- | ------------- |
| Functional licensing | 2 | 80.0 +/- 3.1 | 50.7 +/- 3.0 | 52.0 +/- 5.2 | +29.4 +/- 4.3 | +28.0 +/- 4.0 |
| Constructional       | 3 | 81.6 +/- 1.2 | 71.4 +/- 1.8 | 71.3 +/- 1.5 | +10.2 +/- 3.0 | +10.3 +/- 2.7 |
| Anaphor*             | 1 | 57.3 +/- 1.8 | 49.9 +/- 2.0 | 45.6 +/- 2.3 | +7.5 +/- 2.0  | +11.7 +/- 1.1 |
| Argument structure   | 1 | 68.9 +/- 1.0 | 63.9 +/- 0.8 | 64.3 +/- 1.6 | +5.0 +/- 0.2  | +4.6 +/- 2.6  |
| Order-sensitive      | 3 | 77.4 +/- 2.2 | 72.5 +/- 1.0 | 71.9 +/- 2.3 | +4.9 +/- 1.6  | +5.4 +/- 0.3  |
| Unstable categories  | 5 | 45.1 +/- 4.4 | 48.5 +/- 1.0 | 48.2 +/- 1.8 | -3.3 +/- 3.6  | -3.1 +/- 2.7  |

## 官方现象 gap 排名

Chinese - Pinyin-token，按三 seed 平均 gap 排序，前 8 个：

| phenomenon         | gap           |
| ------------------ | ------------- |
| fci_licensing      | +39.7 +/- 6.2 |
| anaphor            | +21.8 +/- 0.9 |
| npi_licensing      | +19.0 +/- 7.6 |
| verb_phrase        | +12.1 +/- 2.0 |
| control_raising    | +9.4 +/- 4.4  |
| BA                 | +9.0 +/- 3.2  |
| passive            | +7.0 +/- 1.6  |
| argument_structure | +5.0 +/- 0.2  |

Chinese - Pinyin-data，按三 seed 平均 gap 排序，前 8 个：

| phenomenon         | gap            |
| ------------------ | -------------- |
| fci_licensing      | +36.6 +/- 10.0 |
| anaphor            | +24.6 +/- 1.5  |
| npi_licensing      | +19.5 +/- 4.5  |
| control_raising    | +14.4 +/- 5.3  |
| verb_phrase        | +10.3 +/- 1.6  |
| passive            | +6.9 +/- 1.4   |
| BA                 | +6.1 +/- 4.5   |
| nominal_expression | +5.9 +/- 7.9   |

`anaphor` 的 all-item gap 很大，但主表使用 noncollapsed 子集后明显变小；因此若正文讨论 lexical/functional-marker gap，不应把 all-item anaphor 当作主要证据。

## 分组判断

| Group                | Judgment              | Reason                                                                                                                                        |
| -------------------- | --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| Functional licensing | reasonable            | FCI/NPI 都围绕 renhe/任何 等许可条件；也是最大且最稳定的中文优势。                                                                                                     |
| Constructional       | reasonable but rename | BA、control/raising、verb phrase 都是 construction/syntax-sensitive，但内部机制偏宽；建议叫 construction-level syntax 或 constructions and predicate syntax。   |
| Anaphor*             | reasonable but rename | 必须明确是 non-collapsed anaphor subset；all-item anaphor gap 会被拼音表面坍缩夸大。                                                                           |
| Argument structure   | reasonable            | 单一官方现象，直接保留最清楚。                                                                                                                               |
| Order-sensitive      | reasonable but rename | Topicalization 和 relativization 很合适；passive 也涉及被/受、agent deletion、constituent position，表现接近但不纯粹是 word order；Order-sensitive 比 Word order 更准确。 |
| Unstable categories  | reasonable but rename | n=5 可重现，但需要写明规则；建议 caption 用 near-chance/unstable residual categories，不要暗示同一语言学机制。                                                            |

## 对主 claim 的审计

你的核心 claim 基本受支持，但建议表述为“largest and most consistent gaps”，不要说所有 lexical/marker 对比都大。重算后 Functional licensing 是最强组：相对 Pinyin-token 为 +29.4 +/- 4.3 pp，相对 Pinyin-data 为 +28.0 +/- 4.0 pp。官方现象层面，`fci_licensing` 和 `npi_licensing` 也是 gap 最大的两个稳定现象之一，这与“依赖 lexical identity 或 character-distinguishable functional markers”一致。

Constructional 组有中等 gap：相对 Pinyin-token +10.2 +/- 3.0 pp，相对 Pinyin-data +10.3 +/- 2.7 pp，其中 `verb_phrase`、`BA`、`control_raising` 都有正 gap，但机制不完全相同。Order-sensitive 组 gap 较小：相对 Pinyin-token +4.9 +/- 1.6 pp，相对 Pinyin-data +5.4 +/- 0.3 pp；这支持“topicalization、relativization、passive 等主要顺序/配置敏感现象上的 gap 较小”。

## Order-sensitive 组专项审计

| phenomenon     | Chinese      | Pinyin-token | Pinyin-data  | Zh-Token gap | Zh-Data gap  |
| -------------- | ------------ | ------------ | ------------ | ------------ | ------------ |
| topicalization | 80.5 +/- 2.2 | 76.5 +/- 1.0 | 74.6 +/- 2.5 | +4.0 +/- 3.1 | +5.8 +/- 1.8 |
| relativization | 72.4 +/- 5.8 | 68.7 +/- 3.4 | 68.8 +/- 4.8 | +3.7 +/- 2.5 | +3.6 +/- 1.5 |
| passive        | 79.2 +/- 2.4 | 72.3 +/- 1.1 | 72.4 +/- 0.9 | +7.0 +/- 1.6 | +6.9 +/- 1.4 |

`Order-sensitive` 比 `Word order` 更合适。`Word order` 会把 Passive 说得太窄，因为 passive 不只是线性词序，还涉及 `被/受/所`、agent deletion、argument realization 与 constructional licensing。Passive 的 gap 与 topicalization/relativization 同量级，且三者在 Chinese 与两个 Pinyin 设置中都明显高于 chance；把它们合为“order/configuration-sensitive phenomena”是可辩护的。若你想更保守，建议标签改成 `Order/configuration-sensitive`，正文写“including topicalization and relativization, with passive treated as a configuration-sensitive construction rather than a pure word-order test.”

## Constructional 组专项审计

BA + control/raising + verb phrase 可以作为宽口径 construction-level syntax 分组，但 `Constructional` 单词略泛。更清晰的替代名是：

- `Construction-level syntax`
- `Constructions and predicate syntax`
- `Construction/predicate configuration`

如果主表空间允许，更干净的拆法是把 `BA construction` 单列，把 `control_raising + verb_phrase` 合为 `Predicate/configuration syntax`。这样可以避免读者质疑 BA 与 VP 内部副词/体貌、control/raising 是否同一机制。

## Unstable categories 审计

当前 n=5 对应以下官方现象：

| phenomenon         | min_accuracy_any_run | max_accuracy_any_run | mean_accuracy_all_runs |
| ------------------ | -------------------- | -------------------- | ---------------------- |
| classifier         | 39.0                 | 53.3                 | 45.6                   |
| ellipsis           | 33.3                 | 38.0                 | 36.0                   |
| nominal_expression | 41.9                 | 55.7                 | 46.9                   |
| quantifiers        | 26.5                 | 54.7                 | 43.8                   |
| question           | 57.0                 | 72.7                 | 64.0                   |

可重现的规则应写成：在未被其他解释性分组收纳的官方现象中，只要任一 setting/seed 的 all-item accuracy <= 60%，就归入 residual near-chance/unstable categories。这个规则能复原 `classifier`、`ellipsis`、`nominal_expression`、`question`、`quantifiers` 五项。

警告：如果规则写成“严格低于 chance 50%”，`question` 不应进入；如果规则全局应用而不先排除已单列的组，`anaphor` 的 all-item 或 noncollapsed Pinyin-data 也会触发。因此 caption/正文必须说明这是 residual/near-chance criterion，并给出阈值，避免显得事后挑选。

## 建议 caption

Table X. ZhoBLiMP results by analysis category. Values are macro-averages over official ZhoBLiMP phenomena within each category, reported as mean +/- SD across seeds 42/43/44. `Anaphor*` uses the non-collapsed subset because one anaphor paradigm collapses under Pinyin conversion. The residual near-chance/unstable category contains official phenomena not otherwise grouped for which at least one setting/seed has accuracy <= 60%.

## 建议正文表述

The largest and most consistent Chinese-over-Pinyin gaps occur in licensing phenomena, especially FCI and NPI licensing, where the contrast depends on lexical identity and functional markers such as `renhe`/`任何` and licensing environments. Construction-level phenomena show smaller but still positive gaps. By contrast, order/configuration-sensitive phenomena, including topicalization, relativization, and passive, have substantially smaller gaps and remain above chance for both Chinese and Pinyin settings. We therefore interpret the ZhoBLiMP degradation as selective rather than uniform: Pinyin conversion most strongly affects contrasts tied to lexical identity and character-distinguishable functional material, while primarily configurational contrasts are less affected.

## 过度声称风险

- 不要把 `Unstable categories` 解释成统一语言学机制；它是 residual diagnostic bucket。
- 不要用 all-item `anaphor` gap 支撑主要论点；主表应坚持 `Anaphor*` noncollapsed。
- `Passive` 可以和 topicalization/relativization 同组，但应称为 configuration/order-sensitive，不宜称为纯 `Word order`。
- `near chance` 需要阈值；建议在 caption 写 `<= 60% in at least one setting/seed`。

## 输出文件

- `zhoblimp_grouping_mapping.csv`
- `zhoblimp_group_aggregates_by_seed.csv`
- `zhoblimp_group_aggregate_stats.csv`
- `zhoblimp_group_gaps_by_seed.csv`
- `zhoblimp_group_gap_stats.csv`
- `zhoblimp_official_phenomenon_by_seed.csv`
- `zhoblimp_official_phenomenon_setting_stats.csv`
- `zhoblimp_official_phenomenon_gaps_by_seed.csv`
- `zhoblimp_official_phenomenon_gap_stats.csv`
- `zhoblimp_anaphor_noncollapsed_gaps_by_seed.csv`
- `zhoblimp_unstable_criterion_audit.csv`
