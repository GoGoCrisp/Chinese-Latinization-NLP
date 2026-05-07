# BAAI/bge-small-zh-v1.5

Token filter: CJK character count >= 2. Collision groups kept only when at least 2 source tokens remain.

Overlap pair: AC. Details CSV: /Users/crisp/Desktop/code_field/python/Chinese_Latinization_NLP/1.Tokenization/decoded_superTokenizers_2048_subset100k/table2/table2_ac_overlap_superBPE_outputs/table2_ac_overlap_superBPE_details.csv

## Baselines

| baseline | pair_count | mean_dist | median_dist | p90_dist | max_dist | min_dist |
| --- | --- | --- | --- | --- | --- | --- |
| collision_source_random_pairs | 10000 | 0.6286 | 0.6385 | 0.7143 | 0.8373 | 0.2021 |
| chinese_vocab_random_pairs | 10000 | 0.65 | 0.6592 | 0.7405 | 0.8901 | 0.1881 |

## Target collision groups

| pinyin_token | N | mean_dist | median_dist | max_dist | max_pair | min_dist |
| --- | --- | --- | --- | --- | --- | --- |
| ALL | ALL | 0.4026 | 0.379 | 0.8333 | shi4 she4 (N=2) | 0.0425 |
| ta1 de | 4 | 0.3154 | 0.3133 | 0.431 | 她的\|牠的 | 0.2254 |
| an4 jian4 | 2 | 0.6486 | 0.6486 | 0.6486 | 按键\|案件 | 0.6486 |
| dian4 shi4 | 2 | 0.6639 | 0.6639 | 0.6639 | 殿试\|电视 | 0.6639 |
| fu2 shi4 | 2 | 0.3179 | 0.3179 | 0.3179 | 服侍\|服饰 | 0.3179 |
| lie4 shi4 | 2 | 0.6796 | 0.6796 | 0.6796 | 劣势\|烈士 | 0.6796 |
| yu4 shi4 | 2 | 0.6634 | 0.6634 | 0.6634 | 浴室\|预示 | 0.6634 |
| gong1 shi4 | 6 | 0.4934 | 0.5622 | 0.6835 | 公视\|攻势 | 0.3018 |
| xing2 shi4 | 5 | 0.4682 | 0.5147 | 0.6705 | 刑事\|型式 | 0.1815 |
| qi2 shi4 | 4 | 0.6436 | 0.6807 | 0.7388 | 其事\|骑士 | 0.4063 |
| jing1 li4 | 2 | 0.5071 | 0.5071 | 0.5071 | 精力\|经历 | 0.5071 |

## Aggregate collision group means

| N | pinyin_token | pair_count | mean_dist | median_dist | p90_dist | max_dist | max_pair | min_dist | min_pair | source_tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | N=2 | 1656 | 0.3981 | 0.3687 | 0.6457 | 0.8333 | shi4 she4 (N=2) | 0.0425 | zhang4 hu4 (N=2) | equal-weight mean over filtered N=2 collision groups |
| 3 | N=3 | 198 | 0.4297 | 0.4264 | 0.5889 | 0.703 | he2 li3 (N=3) | 0.1752 | wu2 xu1 (N=3) | equal-weight mean over filtered N=3 collision groups |
| 4 | N=4 | 30 | 0.443 | 0.4748 | 0.574 | 0.6436 | qi2 shi4 (N=4) | 0.2085 | ta1 men de (N=4) | equal-weight mean over filtered N=4 collision groups |
| 5 | N=5 | 11 | 0.4551 | 0.4601 | 0.5478 | 0.5586 | shi4 ji4 (N=5) | 0.3416 | zhong1 shi4 (N=5) | equal-weight mean over filtered N=5 collision groups |
| 6 | N=6 | 3 | 0.4853 | 0.4934 | 0.5091 | 0.513 | zheng4 shi4 (N=6) | 0.4496 | shi4 wei4 (N=6) | equal-weight mean over filtered N=6 collision groups |
| 8 | N=8 | 1 | 0.4528 | 0.4528 | 0.4528 | 0.4528 | shi4 de (N=8) | 0.4528 | shi4 de (N=8) | equal-weight mean over filtered N=8 collision groups |
| ALL | ALL | 1899 | 0.4026 | 0.379 | 0.6414 | 0.8333 | shi4 she4 (N=2) | 0.0425 | zhang4 hu4 (N=2) | equal-weight mean over filtered N>1 collision groups |

## Sampled collision groups by filtered N

| N | pinyin_token | pair_count | mean_dist | median_dist | max_dist | max_pair | min_dist | source_tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | jin3 jin3 | 1 | 0.585 | 0.585 | 0.585 | 仅仅\|紧紧 | 0.585 | 仅仅 / 紧紧 |
| 2 | ren2 he2 | 1 | 0.4174 | 0.4174 | 0.4174 | 人和\|仁和 | 0.4174 | 人和 / 仁和 |
| 2 | sheng4 gong1 | 1 | 0.3169 | 0.3169 | 0.3169 | 圣公\|圣宫 | 0.3169 | 圣公 / 圣宫 |
| 2 | wu1 si1 | 1 | 0.3005 | 0.3005 | 0.3005 | 乌思\|乌斯 | 0.3005 | 乌思 / 乌斯 |
| 2 | xian4 nei4 | 1 | 0.4123 | 0.4123 | 0.4123 | 县内\|线内 | 0.4123 | 县内 / 线内 |
| 3 | he2 li3 | 3 | 0.703 | 0.7014 | 0.7087 | 合理\|和李 | 0.6988 | 合理 / 和李 / 荷里 |
| 3 | ke1 ke4 | 3 | 0.5886 | 0.7409 | 0.7535 | 科克\|苛刻 | 0.2713 | 柯克 / 科克 / 苛刻 |
| 3 | quan2 li4 | 3 | 0.3939 | 0.4123 | 0.5864 | 全力\|权利 | 0.1829 | 全力 / 权利 / 权力 |
| 3 | shi2 cai2 | 3 | 0.5939 | 0.6542 | 0.7172 | 时才\|石材 | 0.4103 | 时才 / 石材 / 食材 |
| 3 | zuo4 de | 3 | 0.4155 | 0.5171 | 0.5176 | 做的\|座的 | 0.2117 | 作的 / 做的 / 座的 |
| 4 | lian2 jie2 | 6 | 0.509 | 0.5704 | 0.6849 | 廉洁\|连结 | 0.1878 | 廉洁 / 联捷 / 联结 / 连结 |
| 4 | liu2 shi4 | 6 | 0.5113 | 0.5374 | 0.6328 | 刘氏\|流逝 | 0.3333 | 刘氏 / 流士 / 流式 / 流逝 |
| 4 | shi4 li4 | 6 | 0.6273 | 0.6321 | 0.7438 | 市立\|视力 | 0.469 | 势力 / 市立 / 示例 / 视力 |
| 4 | ta1 zai4 | 6 | 0.3205 | 0.2653 | 0.5049 | 他再\|她在 | 0.2055 | 他再 / 他在 / 她在 / 它在 |
| 4 | zi4 xing2 | 6 | 0.3622 | 0.3138 | 0.5887 | 字形\|自行 | 0.1487 | 字型 / 字形 / 字行 / 自行 |
| 5 | ma3 li4 | 10 | 0.3858 | 0.3822 | 0.6061 | 玛丽\|马力 | 0.1784 | 玛丽 / 玛利 / 玛莉 / 马利 / 马力 |
| 5 | shi4 zhong1 | 10 | 0.3789 | 0.3965 | 0.4566 | 侍中\|是中 | 0.254 | 事中 / 侍中 / 室中 / 式中 / 是中 |
| 5 | xing2 shi4 | 10 | 0.4682 | 0.5147 | 0.6705 | 刑事\|型式 | 0.1815 | 刑事 / 型式 / 形势 / 形式 / 行事 |
| 5 | yi1 shi4 | 10 | 0.5366 | 0.6389 | 0.6935 | 一世\|伊士 | 0.2476 | 一世 / 一事 / 一是 / 伊势 / 伊士 |
| 5 | yi4 wei4 | 10 | 0.4016 | 0.3994 | 0.5833 | 亦未\|意味 | 0.2393 | 亦为 / 亦未 / 意为 / 意味 / 译为 |
| 6 | gong1 shi4 | 15 | 0.4934 | 0.5622 | 0.6835 | 公视\|攻势 | 0.3018 | 公事 / 公式 / 公示 / 公视 / 工事 / 攻势 |
| 6 | shi4 wei4 | 15 | 0.4496 | 0.4044 | 0.7142 | 侍卫\|是为 | 0.2649 | 侍卫 / 式为 / 是为 / 是位 / 氏为 / 视为 |
| 6 | zheng4 shi4 | 15 | 0.513 | 0.5928 | 0.6645 | 正视\|郑氏 | 0.2579 | 政事 / 正室 / 正式 / 正是 / 正视 / 郑氏 |
| 8 | shi4 de | 28 | 0.4528 | 0.4606 | 0.569 | 士的\|市的 | 0.3238 | 世的 / 事的 / 士的 / 室的 / 市的 / 式的 / 是的 / 氏的 |

# shibing624/text2vec-base-chinese

Token filter: CJK character count >= 2. Collision groups kept only when at least 2 source tokens remain.

Overlap pair: AC. Details CSV: /Users/crisp/Desktop/code_field/python/Chinese_Latinization_NLP/1.Tokenization/decoded_superTokenizers_2048_subset100k/table2/table2_ac_overlap_superBPE_outputs/table2_ac_overlap_superBPE_details.csv

## Baselines

| baseline | pair_count | mean_dist | median_dist | p90_dist | max_dist | min_dist |
| --- | --- | --- | --- | --- | --- | --- |
| collision_source_random_pairs | 10000 | 0.5908 | 0.5982 | 0.7231 | 0.9189 | 0.1807 |
| chinese_vocab_random_pairs | 10000 | 0.6231 | 0.63 | 0.7519 | 0.9768 | 0.2143 |

## Target collision groups

| pinyin_token | N | mean_dist | median_dist | max_dist | max_pair | min_dist |
| --- | --- | --- | --- | --- | --- | --- |
| ALL | ALL | 0.4343 | 0.4497 | 0.9099 | de2 yi3 (N=2) | 0.0164 |
| ta1 de | 4 | 0.3219 | 0.2968 | 0.5017 | 他的\|牠的 | 0.1447 |
| an4 jian4 | 2 | 0.5867 | 0.5867 | 0.5867 | 按键\|案件 | 0.5867 |
| dian4 shi4 | 2 | 0.7092 | 0.7092 | 0.7092 | 殿试\|电视 | 0.7092 |
| fu2 shi4 | 2 | 0.3558 | 0.3558 | 0.3558 | 服侍\|服饰 | 0.3558 |
| lie4 shi4 | 2 | 0.6137 | 0.6137 | 0.6137 | 劣势\|烈士 | 0.6137 |
| yu4 shi4 | 2 | 0.7626 | 0.7626 | 0.7626 | 浴室\|预示 | 0.7626 |
| gong1 shi4 | 6 | 0.5556 | 0.5279 | 0.7292 | 公示\|攻势 | 0.4421 |
| xing2 shi4 | 5 | 0.5103 | 0.5181 | 0.6478 | 刑事\|形式 | 0.2554 |
| qi2 shi4 | 4 | 0.6659 | 0.6735 | 0.7412 | 棋士\|歧视 | 0.5551 |
| jing1 li4 | 2 | 0.5491 | 0.5491 | 0.5491 | 精力\|经历 | 0.5491 |

## Aggregate collision group means

| N | pinyin_token | pair_count | mean_dist | median_dist | p90_dist | max_dist | max_pair | min_dist | min_pair | source_tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | N=2 | 1656 | 0.431 | 0.4449 | 0.6391 | 0.9099 | de2 yi3 (N=2) | 0.0164 | huo4 qi2 ta1 (N=2) | equal-weight mean over filtered N=2 collision groups |
| 3 | N=3 | 198 | 0.4554 | 0.4642 | 0.6038 | 0.7103 | yi1 fu2 (N=3) | 0.0904 | dan4 ta1 (N=3) | equal-weight mean over filtered N=3 collision groups |
| 4 | N=4 | 30 | 0.4565 | 0.5032 | 0.5888 | 0.6659 | qi2 shi4 (N=4) | 0.1837 | ai4 li4 (N=4) | equal-weight mean over filtered N=4 collision groups |
| 5 | N=5 | 11 | 0.4729 | 0.4652 | 0.5329 | 0.5487 | xian4 zhi4 (N=5) | 0.4027 | shi4 he2 (N=5) | equal-weight mean over filtered N=5 collision groups |
| 6 | N=6 | 3 | 0.53 | 0.5556 | 0.5608 | 0.5621 | zheng4 shi4 (N=6) | 0.4724 | shi4 wei4 (N=6) | equal-weight mean over filtered N=6 collision groups |
| 8 | N=8 | 1 | 0.4323 | 0.4323 | 0.4323 | 0.4323 | shi4 de (N=8) | 0.4323 | shi4 de (N=8) | equal-weight mean over filtered N=8 collision groups |
| ALL | ALL | 1899 | 0.4343 | 0.4497 | 0.6358 | 0.9099 | de2 yi3 (N=2) | 0.0164 | huo4 qi2 ta1 (N=2) | equal-weight mean over filtered N>1 collision groups |

## Sampled collision groups by filtered N

| N | pinyin_token | pair_count | mean_dist | median_dist | max_dist | max_pair | min_dist | source_tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | jin3 jin3 | 1 | 0.5056 | 0.5056 | 0.5056 | 仅仅\|紧紧 | 0.5056 | 仅仅 / 紧紧 |
| 2 | ren2 he2 | 1 | 0.4247 | 0.4247 | 0.4247 | 人和\|仁和 | 0.4247 | 人和 / 仁和 |
| 2 | sheng4 gong1 | 1 | 0.3018 | 0.3018 | 0.3018 | 圣公\|圣宫 | 0.3018 | 圣公 / 圣宫 |
| 2 | wu1 si1 | 1 | 0.2087 | 0.2087 | 0.2087 | 乌思\|乌斯 | 0.2087 | 乌思 / 乌斯 |
| 2 | xian4 nei4 | 1 | 0.3763 | 0.3763 | 0.3763 | 县内\|线内 | 0.3763 | 县内 / 线内 |
| 3 | he2 li3 | 3 | 0.5504 | 0.5574 | 0.5753 | 和李\|荷里 | 0.5186 | 合理 / 和李 / 荷里 |
| 3 | ke1 ke4 | 3 | 0.4783 | 0.6011 | 0.6246 | 柯克\|苛刻 | 0.2092 | 柯克 / 科克 / 苛刻 |
| 3 | quan2 li4 | 3 | 0.4938 | 0.576 | 0.6146 | 全力\|权利 | 0.2907 | 全力 / 权利 / 权力 |
| 3 | shi2 cai2 | 3 | 0.5754 | 0.5724 | 0.6259 | 时才\|石材 | 0.528 | 时才 / 石材 / 食材 |
| 3 | zuo4 de | 3 | 0.4611 | 0.4858 | 0.5607 | 做的\|座的 | 0.3368 | 作的 / 做的 / 座的 |
| 4 | lian2 jie2 | 6 | 0.5448 | 0.5901 | 0.7891 | 廉洁\|连结 | 0.1754 | 廉洁 / 联捷 / 联结 / 连结 |
| 4 | liu2 shi4 | 6 | 0.5746 | 0.5578 | 0.7772 | 刘氏\|流式 | 0.3893 | 刘氏 / 流士 / 流式 / 流逝 |
| 4 | shi4 li4 | 6 | 0.6012 | 0.5857 | 0.6965 | 市立\|视力 | 0.5181 | 势力 / 市立 / 示例 / 视力 |
| 4 | ta1 zai4 | 6 | 0.2627 | 0.2739 | 0.3294 | 他再\|她在 | 0.1776 | 他再 / 他在 / 她在 / 它在 |
| 4 | zi4 xing2 | 6 | 0.4355 | 0.4507 | 0.6884 | 字行\|自行 | 0.082 | 字型 / 字形 / 字行 / 自行 |
| 5 | ma3 li4 | 10 | 0.4589 | 0.5305 | 0.716 | 玛莉\|马力 | 0.0879 | 玛丽 / 玛利 / 玛莉 / 马利 / 马力 |
| 5 | shi4 zhong1 | 10 | 0.5165 | 0.5082 | 0.6472 | 侍中\|式中 | 0.3528 | 事中 / 侍中 / 室中 / 式中 / 是中 |
| 5 | xing2 shi4 | 10 | 0.5103 | 0.5181 | 0.6478 | 刑事\|形式 | 0.2554 | 刑事 / 型式 / 形势 / 形式 / 行事 |
| 5 | yi1 shi4 | 10 | 0.4849 | 0.4827 | 0.6473 | 一世\|伊士 | 0.3033 | 一世 / 一事 / 一是 / 伊势 / 伊士 |
| 5 | yi4 wei4 | 10 | 0.4074 | 0.3803 | 0.618 | 亦未\|意为 | 0.2848 | 亦为 / 亦未 / 意为 / 意味 / 译为 |
| 6 | gong1 shi4 | 15 | 0.5556 | 0.5279 | 0.7292 | 公示\|攻势 | 0.4421 | 公事 / 公式 / 公示 / 公视 / 工事 / 攻势 |
| 6 | shi4 wei4 | 15 | 0.4724 | 0.4886 | 0.6288 | 氏为\|视为 | 0.2436 | 侍卫 / 式为 / 是为 / 是位 / 氏为 / 视为 |
| 6 | zheng4 shi4 | 15 | 0.5621 | 0.59 | 0.7096 | 正视\|郑氏 | 0.3474 | 政事 / 正室 / 正式 / 正是 / 正视 / 郑氏 |
| 8 | shi4 de | 28 | 0.4323 | 0.4173 | 0.5656 | 是的\|氏的 | 0.343 | 世的 / 事的 / 士的 / 室的 / 市的 / 式的 / 是的 / 氏的 |
