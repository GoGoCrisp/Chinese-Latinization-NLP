# BAAI/bge-small-zh-v1.5

Token filter: CJK character count >= 2. Collision groups kept only when at least 2 source tokens remain.

Overlap pair: AD. Details CSV: /Users/crisp/Desktop/code_field/python/Chinese_Latinization_NLP/1.Tokenization/decoded_superTokenizers_2048_subset100k/table2/table2_ad_overlap_superBPE_outputs/table2_ad_overlap_superBPE_details.csv

## Baselines

| baseline | pair_count | mean_dist | median_dist | p90_dist | max_dist | min_dist |
| --- | --- | --- | --- | --- | --- | --- |
| collision_source_random_pairs | 10000 | 0.6303 | 0.6397 | 0.7154 | 0.8442 | 0.2021 |
| chinese_vocab_random_pairs | 10000 | 0.6503 | 0.659 | 0.7386 | 0.8998 | 0.1077 |

## Target collision groups

| pinyin_token | N | mean_dist | median_dist | max_dist | max_pair | min_dist |
| --- | --- | --- | --- | --- | --- | --- |
| ALL | ALL | 0.4028 | 0.3791 | 0.8333 | shì shè (N=2) | 0.0425 |
| tā de | 4 | 0.3154 | 0.3133 | 0.431 | 她的\|牠的 | 0.2254 |
| àn jiàn | 2 | 0.6486 | 0.6486 | 0.6486 | 按键\|案件 | 0.6486 |
| diàn shì | 2 | 0.6639 | 0.6639 | 0.6639 | 殿试\|电视 | 0.6639 |
| fú shì | 2 | 0.3179 | 0.3179 | 0.3179 | 服侍\|服饰 | 0.3179 |
| liè shì | 2 | 0.6796 | 0.6796 | 0.6796 | 劣势\|烈士 | 0.6796 |
| yù shì | 2 | 0.6634 | 0.6634 | 0.6634 | 浴室\|预示 | 0.6634 |
| gōng shì | 6 | 0.4934 | 0.5622 | 0.6835 | 公视\|攻势 | 0.3018 |
| xíng shì | 5 | 0.4682 | 0.5147 | 0.6705 | 刑事\|型式 | 0.1815 |
| qí shì | 4 | 0.6436 | 0.6807 | 0.7388 | 其事\|骑士 | 0.4063 |
| jīng lì | 2 | 0.5071 | 0.5071 | 0.5071 | 精力\|经历 | 0.5071 |

## Aggregate collision group means

| N | pinyin_token | pair_count | mean_dist | median_dist | p90_dist | max_dist | max_pair | min_dist | min_pair | source_tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | N=2 | 1657 | 0.3983 | 0.3687 | 0.6461 | 0.8333 | shì shè (N=2) | 0.0425 | zhàng hù (N=2) | equal-weight mean over filtered N=2 collision groups |
| 3 | N=3 | 198 | 0.4297 | 0.4264 | 0.5889 | 0.703 | hé lǐ (N=3) | 0.1752 | wú xū (N=3) | equal-weight mean over filtered N=3 collision groups |
| 4 | N=4 | 30 | 0.443 | 0.4748 | 0.574 | 0.6436 | qí shì (N=4) | 0.2085 | tā men de (N=4) | equal-weight mean over filtered N=4 collision groups |
| 5 | N=5 | 11 | 0.4551 | 0.4601 | 0.5478 | 0.5586 | shì jì (N=5) | 0.3416 | zhōng shì (N=5) | equal-weight mean over filtered N=5 collision groups |
| 6 | N=6 | 3 | 0.4853 | 0.4934 | 0.5091 | 0.513 | zhèng shì (N=6) | 0.4496 | shì wèi (N=6) | equal-weight mean over filtered N=6 collision groups |
| 8 | N=8 | 1 | 0.4528 | 0.4528 | 0.4528 | 0.4528 | shì de (N=8) | 0.4528 | shì de (N=8) | equal-weight mean over filtered N=8 collision groups |
| ALL | ALL | 1900 | 0.4028 | 0.3791 | 0.6419 | 0.8333 | shì shè (N=2) | 0.0425 | zhàng hù (N=2) | equal-weight mean over filtered N>1 collision groups |

## Sampled collision groups by filtered N

| N | pinyin_token | pair_count | mean_dist | median_dist | max_dist | max_pair | min_dist | source_tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | jūn de | 1 | 0.5387 | 0.5387 | 0.5387 | 军的\|菌的 | 0.5387 | 军的 / 菌的 |
| 2 | shàng yī | 1 | 0.3611 | 0.3611 | 0.3611 | 上一\|上衣 | 0.3611 | 上一 / 上衣 |
| 2 | shí sì | 1 | 0.6433 | 0.6433 | 0.6433 | 十四\|食肆 | 0.6433 | 十四 / 食肆 |
| 2 | xiàn shí de | 1 | 0.244 | 0.244 | 0.244 | 现实的\|现时的 | 0.244 | 现实的 / 现时的 |
| 2 | xī kǒu | 1 | 0.3838 | 0.3838 | 0.3838 | 溪口\|西口 | 0.3838 | 溪口 / 西口 |
| 3 | hé lǐ | 3 | 0.703 | 0.7014 | 0.7087 | 合理\|和李 | 0.6988 | 合理 / 和李 / 荷里 |
| 3 | liǎng jí | 3 | 0.402 | 0.404 | 0.4363 | 两极\|两集 | 0.3657 | 两极 / 两级 / 两集 |
| 3 | quán lì | 3 | 0.3939 | 0.4123 | 0.5864 | 全力\|权利 | 0.1829 | 全力 / 权利 / 权力 |
| 3 | shì wēi | 3 | 0.5756 | 0.6191 | 0.6812 | 士威\|式微 | 0.4265 | 士威 / 式微 / 示威 |
| 3 | ān xī | 3 | 0.4266 | 0.4293 | 0.4365 | 安息\|安溪 | 0.4141 | 安息 / 安溪 / 安西 |
| 4 | lǐ shì | 6 | 0.569 | 0.5964 | 0.6578 | 李世\|里士 | 0.3912 | 李世 / 李氏 / 理事 / 里士 |
| 4 | quán shì | 6 | 0.5544 | 0.583 | 0.667 | 全市\|权势 | 0.3728 | 全市 / 全是 / 权势 / 诠释 |
| 4 | tā de | 6 | 0.3154 | 0.3133 | 0.431 | 她的\|牠的 | 0.2254 | 他的 / 她的 / 它的 / 牠的 |
| 4 | xī lín | 6 | 0.4762 | 0.4054 | 0.7312 | 西临\|锡林 | 0.3013 | 西临 / 西林 / 西邻 / 锡林 |
| 4 | ài lì | 6 | 0.2543 | 0.2554 | 0.3198 | 艾力\|艾莉 | 0.1716 | 艾丽 / 艾利 / 艾力 / 艾莉 |
| 5 | mǎ lì | 10 | 0.3858 | 0.3822 | 0.6061 | 玛丽\|马力 | 0.1784 | 玛丽 / 玛利 / 玛莉 / 马利 / 马力 |
| 5 | shì zhōng | 10 | 0.3789 | 0.3965 | 0.4566 | 侍中\|是中 | 0.254 | 事中 / 侍中 / 室中 / 式中 / 是中 |
| 5 | xíng shì | 10 | 0.4682 | 0.5147 | 0.6705 | 刑事\|型式 | 0.1815 | 刑事 / 型式 / 形势 / 形式 / 行事 |
| 5 | yì wèi | 10 | 0.4016 | 0.3994 | 0.5833 | 亦未\|意味 | 0.2393 | 亦为 / 亦未 / 意为 / 意味 / 译为 |
| 5 | yī shì | 10 | 0.5366 | 0.6389 | 0.6935 | 一世\|伊士 | 0.2476 | 一世 / 一事 / 一是 / 伊势 / 伊士 |
| 6 | gōng shì | 15 | 0.4934 | 0.5622 | 0.6835 | 公视\|攻势 | 0.3018 | 公事 / 公式 / 公示 / 公视 / 工事 / 攻势 |
| 6 | shì wèi | 15 | 0.4496 | 0.4044 | 0.7142 | 侍卫\|是为 | 0.2649 | 侍卫 / 式为 / 是为 / 是位 / 氏为 / 视为 |
| 6 | zhèng shì | 15 | 0.513 | 0.5928 | 0.6645 | 正视\|郑氏 | 0.2579 | 政事 / 正室 / 正式 / 正是 / 正视 / 郑氏 |
| 8 | shì de | 28 | 0.4528 | 0.4606 | 0.569 | 士的\|市的 | 0.3238 | 世的 / 事的 / 士的 / 室的 / 市的 / 式的 / 是的 / 氏的 |

# shibing624/text2vec-base-chinese

Token filter: CJK character count >= 2. Collision groups kept only when at least 2 source tokens remain.

Overlap pair: AD. Details CSV: /Users/crisp/Desktop/code_field/python/Chinese_Latinization_NLP/1.Tokenization/decoded_superTokenizers_2048_subset100k/table2/table2_ad_overlap_superBPE_outputs/table2_ad_overlap_superBPE_details.csv

## Baselines

| baseline | pair_count | mean_dist | median_dist | p90_dist | max_dist | min_dist |
| --- | --- | --- | --- | --- | --- | --- |
| collision_source_random_pairs | 10000 | 0.5913 | 0.5978 | 0.7232 | 0.9281 | 0.1807 |
| chinese_vocab_random_pairs | 10000 | 0.6237 | 0.6317 | 0.7497 | 0.9975 | 0.2016 |

## Target collision groups

| pinyin_token | N | mean_dist | median_dist | max_dist | max_pair | min_dist |
| --- | --- | --- | --- | --- | --- | --- |
| ALL | ALL | 0.4345 | 0.45 | 0.9099 | dé yǐ (N=2) | 0.0164 |
| tā de | 4 | 0.3219 | 0.2968 | 0.5017 | 他的\|牠的 | 0.1447 |
| àn jiàn | 2 | 0.5867 | 0.5867 | 0.5867 | 按键\|案件 | 0.5867 |
| diàn shì | 2 | 0.7092 | 0.7092 | 0.7092 | 殿试\|电视 | 0.7092 |
| fú shì | 2 | 0.3558 | 0.3558 | 0.3558 | 服侍\|服饰 | 0.3558 |
| liè shì | 2 | 0.6137 | 0.6137 | 0.6137 | 劣势\|烈士 | 0.6137 |
| yù shì | 2 | 0.7626 | 0.7626 | 0.7626 | 浴室\|预示 | 0.7626 |
| gōng shì | 6 | 0.5556 | 0.5279 | 0.7292 | 公示\|攻势 | 0.4421 |
| xíng shì | 5 | 0.5103 | 0.5181 | 0.6478 | 刑事\|形式 | 0.2554 |
| qí shì | 4 | 0.6659 | 0.6735 | 0.7412 | 棋士\|歧视 | 0.5551 |
| jīng lì | 2 | 0.5491 | 0.5491 | 0.5491 | 精力\|经历 | 0.5491 |

## Aggregate collision group means

| N | pinyin_token | pair_count | mean_dist | median_dist | p90_dist | max_dist | max_pair | min_dist | min_pair | source_tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | N=2 | 1657 | 0.4312 | 0.4451 | 0.6396 | 0.9099 | dé yǐ (N=2) | 0.0164 | huò qí tā (N=2) | equal-weight mean over filtered N=2 collision groups |
| 3 | N=3 | 198 | 0.4554 | 0.4642 | 0.6038 | 0.7103 | yī fú (N=3) | 0.0904 | dàn tā (N=3) | equal-weight mean over filtered N=3 collision groups |
| 4 | N=4 | 30 | 0.4565 | 0.5032 | 0.5888 | 0.6659 | qí shì (N=4) | 0.1837 | ài lì (N=4) | equal-weight mean over filtered N=4 collision groups |
| 5 | N=5 | 11 | 0.4729 | 0.4652 | 0.5329 | 0.5487 | xiàn zhì (N=5) | 0.4027 | shì hé (N=5) | equal-weight mean over filtered N=5 collision groups |
| 6 | N=6 | 3 | 0.53 | 0.5556 | 0.5608 | 0.5621 | zhèng shì (N=6) | 0.4724 | shì wèi (N=6) | equal-weight mean over filtered N=6 collision groups |
| 8 | N=8 | 1 | 0.4323 | 0.4323 | 0.4323 | 0.4323 | shì de (N=8) | 0.4323 | shì de (N=8) | equal-weight mean over filtered N=8 collision groups |
| ALL | ALL | 1900 | 0.4345 | 0.45 | 0.6358 | 0.9099 | dé yǐ (N=2) | 0.0164 | huò qí tā (N=2) | equal-weight mean over filtered N>1 collision groups |

## Sampled collision groups by filtered N

| N | pinyin_token | pair_count | mean_dist | median_dist | max_dist | max_pair | min_dist | source_tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | jūn de | 1 | 0.5481 | 0.5481 | 0.5481 | 军的\|菌的 | 0.5481 | 军的 / 菌的 |
| 2 | shàng yī | 1 | 0.5326 | 0.5326 | 0.5326 | 上一\|上衣 | 0.5326 | 上一 / 上衣 |
| 2 | shí sì | 1 | 0.6947 | 0.6947 | 0.6947 | 十四\|食肆 | 0.6947 | 十四 / 食肆 |
| 2 | xiàn shí de | 1 | 0.3873 | 0.3873 | 0.3873 | 现实的\|现时的 | 0.3873 | 现实的 / 现时的 |
| 2 | xī kǒu | 1 | 0.5934 | 0.5934 | 0.5934 | 溪口\|西口 | 0.5934 | 溪口 / 西口 |
| 3 | hé lǐ | 3 | 0.5504 | 0.5574 | 0.5753 | 和李\|荷里 | 0.5186 | 合理 / 和李 / 荷里 |
| 3 | liǎng jí | 3 | 0.4218 | 0.4466 | 0.4739 | 两极\|两集 | 0.3449 | 两极 / 两级 / 两集 |
| 3 | quán lì | 3 | 0.4938 | 0.576 | 0.6146 | 全力\|权利 | 0.2907 | 全力 / 权利 / 权力 |
| 3 | shì wēi | 3 | 0.5181 | 0.5198 | 0.5288 | 式微\|示威 | 0.5057 | 士威 / 式微 / 示威 |
| 3 | ān xī | 3 | 0.454 | 0.456 | 0.4989 | 安息\|安溪 | 0.4071 | 安息 / 安溪 / 安西 |
| 4 | lǐ shì | 6 | 0.5293 | 0.5658 | 0.6473 | 李氏\|理事 | 0.2728 | 李世 / 李氏 / 理事 / 里士 |
| 4 | quán shì | 6 | 0.5472 | 0.5701 | 0.6332 | 全是\|权势 | 0.4041 | 全市 / 全是 / 权势 / 诠释 |
| 4 | tā de | 6 | 0.3219 | 0.2968 | 0.5017 | 他的\|牠的 | 0.1447 | 他的 / 她的 / 它的 / 牠的 |
| 4 | xī lín | 6 | 0.5195 | 0.5007 | 0.7126 | 西邻\|锡林 | 0.2624 | 西临 / 西林 / 西邻 / 锡林 |
| 4 | ài lì | 6 | 0.1837 | 0.178 | 0.2822 | 艾力\|艾莉 | 0.0912 | 艾丽 / 艾利 / 艾力 / 艾莉 |
| 5 | mǎ lì | 10 | 0.4589 | 0.5305 | 0.716 | 玛莉\|马力 | 0.0879 | 玛丽 / 玛利 / 玛莉 / 马利 / 马力 |
| 5 | shì zhōng | 10 | 0.5165 | 0.5082 | 0.6472 | 侍中\|式中 | 0.3528 | 事中 / 侍中 / 室中 / 式中 / 是中 |
| 5 | xíng shì | 10 | 0.5103 | 0.5181 | 0.6478 | 刑事\|形式 | 0.2554 | 刑事 / 型式 / 形势 / 形式 / 行事 |
| 5 | yì wèi | 10 | 0.4074 | 0.3803 | 0.618 | 亦未\|意为 | 0.2848 | 亦为 / 亦未 / 意为 / 意味 / 译为 |
| 5 | yī shì | 10 | 0.4849 | 0.4827 | 0.6473 | 一世\|伊士 | 0.3033 | 一世 / 一事 / 一是 / 伊势 / 伊士 |
| 6 | gōng shì | 15 | 0.5556 | 0.5279 | 0.7292 | 公示\|攻势 | 0.4421 | 公事 / 公式 / 公示 / 公视 / 工事 / 攻势 |
| 6 | shì wèi | 15 | 0.4724 | 0.4886 | 0.6288 | 氏为\|视为 | 0.2436 | 侍卫 / 式为 / 是为 / 是位 / 氏为 / 视为 |
| 6 | zhèng shì | 15 | 0.5621 | 0.59 | 0.7096 | 正视\|郑氏 | 0.3474 | 政事 / 正室 / 正式 / 正是 / 正视 / 郑氏 |
| 8 | shì de | 28 | 0.4323 | 0.4173 | 0.5656 | 是的\|氏的 | 0.343 | 世的 / 事的 / 士的 / 室的 / 市的 / 式的 / 是的 / 氏的 |
