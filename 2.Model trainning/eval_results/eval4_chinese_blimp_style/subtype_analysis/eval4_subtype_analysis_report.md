# Eval 4 ZhoBLiMP Subtype Analysis

## Loaded Data

- items: 35400
- score rows: 70800
- models: chinese_4epoch, diacritic_matched_token_4epoch
- unique phenomena: 15
- unique subtypes: 3792
- main-table subtypes with n >= 10: 448

## Top 20 Subtypes By n_items

| phenomenon | subtype | n | ch_acc | di_acc | gap | collapse |
| --- | --- | --- | --- | --- | --- | --- |
| nominal_expression | bad inserts 们 | 900 | 0.0789 | 0.0900 | -0.0111 | 0.0000 |
| question | 到底 → 难道 | 900 | 0.0444 | 0.9367 | -0.8922 | 0.0000 |
| question | multiple edits: 难道->\|->难道 | 688 | 0.6032 | 0.7267 | -0.1235 | 0.0000 |
| BA | bad inserts 所 | 600 | 0.5683 | 0.3550 | 0.2133 | 0.0000 |
| anaphor | bad deletes 们 | 600 | 0.5617 | 0.3483 | 0.2133 | 0.0000 |
| npi_licensing | multiple edits: 没有->\|->没有 | 600 | 0.6233 | 0.3267 | 0.2967 | 0.0000 |
| question | multiple edits: ->是\|是-> | 600 | 0.8050 | 0.8650 | -0.0600 | 0.0000 |
| question | 呢 → 吗 | 600 | 0.9100 | 0.1733 | 0.7367 | 0.0000 |
| question | 难道 → 到底 | 600 | 0.9950 | 0.3883 | 0.6067 | 0.0000 |
| question | multiple edits: ->不\|不-> | 399 | 0.8421 | 0.8571 | -0.0150 | 0.0000 |
| verb_phrase | multiple edits: 没有->\|->没有 | 381 | 0.8425 | 0.6667 | 0.1759 | 0.0000 |
| passive | bad deletes 我 | 362 | 0.9006 | 0.8260 | 0.0746 | 0.0000 |
| anaphor | 她 → 他 | 308 | 0.9091 | 0.0000 | 0.9091 | 1.0000 |
| BA | bad deletes 了 | 300 | 0.9833 | 0.9100 | 0.0733 | 0.0000 |
| BA | bad inserts 了 | 300 | 0.5900 | 0.7200 | -0.1300 | 0.0000 |
| BA | bad inserts 把没 | 300 | 0.5467 | 0.4500 | 0.0967 | 0.0000 |
| BA | 恨 → 怕 | 300 | 0.8633 | 0.4900 | 0.3733 | 0.0000 |
| anaphor | bad inserts 们 | 300 | 0.4800 | 0.8567 | -0.3767 | 0.0000 |
| fci_licensing | bad deletes 都 | 300 | 0.7667 | 0.8867 | -0.1200 | 0.0000 |
| fci_licensing | bad inserts 都 | 300 | 0.9967 | 0.0033 | 0.9933 | 0.0000 |

## Top 20 Chinese-Advantage Subtypes

| phenomenon | subtype | n | ch_acc | di_acc | gap | collapse |
| --- | --- | --- | --- | --- | --- | --- |
| fci_licensing | bad inserts 都 | 300 | 0.9967 | 0.0033 | 0.9933 | 0.0000 |
| npi_licensing | multiple edits: ->你有\|你的-> | 300 | 0.9833 | 0.0467 | 0.9367 | 0.0000 |
| anaphor | 她 → 他 | 308 | 0.9091 | 0.0000 | 0.9091 | 1.0000 |
| fci_licensing | 任何 → 有些 | 300 | 0.7867 | 0.0200 | 0.7667 | 0.0000 |
| passive | bad deletes 小明 | 24 | 1.0000 | 0.2500 | 0.7500 | 0.0000 |
| question | 呢 → 吗 | 600 | 0.9100 | 0.1733 | 0.7367 | 0.0000 |
| argument_structure | bad deletes 她 | 33 | 1.0000 | 0.3333 | 0.6667 | 0.0000 |
| question | 难道 → 到底 | 600 | 0.9950 | 0.3883 | 0.6067 | 0.0000 |
| verb_phrase | 以为 → 告知 | 39 | 0.8718 | 0.2821 | 0.5897 | 0.0000 |
| control_raising | multiple edits: ->马上会\|马上会-> | 23 | 0.8696 | 0.3043 | 0.5652 | 0.0000 |
| passive | multiple edits: ->被他\|被他-> | 25 | 0.8800 | 0.3200 | 0.5600 | 0.0000 |
| npi_licensing | 这位 → 任何 | 43 | 0.7209 | 0.2093 | 0.5116 | 0.0000 |
| nominal_expression | bad inserts 有 | 300 | 0.8967 | 0.3933 | 0.5033 | 0.0000 |
| passive | bad deletes 李先生 | 20 | 1.0000 | 0.5000 | 0.5000 | 0.0000 |
| argument_structure | 有点 → 专心 | 80 | 1.0000 | 0.5125 | 0.4875 | 0.0000 |
| argument_structure | bad deletes 它 | 27 | 1.0000 | 0.5185 | 0.4815 | 0.0000 |
| npi_licensing | multiple edits: 任何->\|她->任何人 | 99 | 0.9899 | 0.5354 | 0.4545 | 0.0000 |
| npi_licensing | 那位 → 任何 | 57 | 0.7544 | 0.3158 | 0.4386 | 0.0000 |
| npi_licensing | 觉得 → 知道 | 55 | 1.0000 | 0.5636 | 0.4364 | 0.0000 |
| control_raising | multiple edits: ->就要\|就要-> | 30 | 0.8333 | 0.4000 | 0.4333 | 0.0000 |

## Top 20 Diacritic-Advantage Subtypes

| phenomenon | subtype | n | ch_acc | di_acc | gap | collapse |
| --- | --- | --- | --- | --- | --- | --- |
| classifier | multiple edits: ->张桌子是\|张桌子->的 | 20 | 0.0000 | 0.9000 | -0.9000 | 0.0000 |
| question | 到底 → 难道 | 900 | 0.0444 | 0.9367 | -0.8922 | 0.0000 |
| passive | 鼻 → 袜 | 23 | 0.0000 | 0.8696 | -0.8696 | 0.0000 |
| passive | multiple edits: 飞机上被->\|->飞机上 | 29 | 0.3103 | 0.9655 | -0.6552 | 0.0000 |
| BA | multiple edits: 他把->\|->被他 | 23 | 0.3913 | 1.0000 | -0.6087 | 0.0000 |
| question | multiple edits: ->不\|不愿意->想 | 30 | 0.1667 | 0.7667 | -0.6000 | 0.0000 |
| passive | multiple edits: 货箱上被->\|->货箱上 | 126 | 0.3413 | 0.7937 | -0.4524 | 0.0000 |
| verb_phrase | multiple edits: 鱼->\|->鱼 | 38 | 0.4211 | 0.8684 | -0.4474 | 0.0000 |
| nominal_expression | multiple edits: ->热\|热-> | 24 | 0.4583 | 0.8750 | -0.4167 | 0.0000 |
| BA | multiple edits: 我把->\|->被我 | 29 | 0.5172 | 0.9310 | -0.4138 | 0.0000 |
| question | multiple edits: 不从->\|->不从 | 242 | 0.5124 | 0.9050 | -0.3926 | 0.0000 |
| anaphor | bad inserts 们 | 300 | 0.4800 | 0.8567 | -0.3767 | 0.0000 |
| nominal_expression | multiple edits: ->十个\|十个-> | 38 | 0.6316 | 1.0000 | -0.3684 | 0.0000 |
| quantifiers | multiple edits: ->没\|的-> | 300 | 0.5133 | 0.8800 | -0.3667 | 0.0000 |
| question | multiple edits: ->不\|不愿意->希望 | 40 | 0.5500 | 0.9000 | -0.3500 | 0.0000 |
| nominal_expression | multiple edits: ->两个\|两个-> | 34 | 0.5294 | 0.8235 | -0.2941 | 0.0000 |
| question | multiple edits: ->不\|不想->愿意 | 25 | 0.7200 | 1.0000 | -0.2800 | 0.0000 |
| question | multiple edits: 从不->\|->不从 | 58 | 0.6552 | 0.9310 | -0.2759 | 0.0000 |
| npi_licensing | 她的 → 任何 | 89 | 0.4719 | 0.7191 | -0.2472 | 0.0000 |
| BA | multiple edits: 她把->\|->被她 | 29 | 0.7586 | 1.0000 | -0.2414 | 0.0000 |

## Highest Diacritic Collapse Rate

| phenomenon | subtype | n | ch_acc | di_acc | gap | collapse |
| --- | --- | --- | --- | --- | --- | --- |
| anaphor | 她 → 他 | 308 | 0.9091 | 0.0000 | 0.9091 | 1.0000 |
| anaphor | 他 → 她 | 292 | 0.1027 | 0.0000 | 0.1027 | 1.0000 |

## Anaphor Collapse Summary

- n_items: 1800
- collapsed_count: 600
- collapsed_rate: 0.3333
- tie_count: 600
- tie_rate: 0.3333

Top anaphor subtypes:
| subtype | n | collapse_rate | ch_all | di_all | ch_noncollapsed | di_noncollapsed |
| --- | --- | --- | --- | --- | --- | --- |
| bad deletes 们 | 600 | 0.0000 | 0.5617 | 0.3483 | 0.5617 | 0.3483 |
| 她 → 他 | 308 | 1.0000 | 0.9091 | 0.0000 |  |  |
| bad inserts 们 | 300 | 0.0000 | 0.4800 | 0.8567 | 0.4800 | 0.8567 |
| 他 → 她 | 292 | 1.0000 | 0.1027 | 0.0000 |  |  |
| 王大娘 → 刘先生 | 6 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 宋女士 → 刘先生 | 5 | 0.0000 | 1.0000 | 0.4000 | 1.0000 | 0.4000 |
| 徐小姐 → 赵大爷 | 5 | 0.0000 | 0.0000 | 0.8000 | 0.0000 | 0.8000 |
| 胡大爷 → 张夫人 | 5 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 赵大爷 → 李太太 | 5 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 |
| multiple edits: 周->杨\|妈->哥 | 4 | 0.0000 | 0.7500 | 0.0000 | 0.7500 | 0.0000 |
| multiple edits: 郑->杨\|妈->哥 | 4 | 0.0000 | 0.2500 | 0.5000 | 0.2500 | 0.5000 |
| multiple edits: 郑->胡\|妈->爷 | 4 | 0.0000 | 0.7500 | 1.0000 | 0.7500 | 1.0000 |
| 刘先生 → 王小姐 | 4 | 0.0000 | 1.0000 | 0.7500 | 1.0000 | 0.7500 |
| 吴太太 → 李先生 | 4 | 0.0000 | 0.2500 | 1.0000 | 0.2500 | 1.0000 |
| 大娘 → 先生 | 4 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 宋女士 → 张先生 | 4 | 0.0000 | 0.5000 | 0.0000 | 0.5000 | 0.0000 |
| 张夫人 → 王先生 | 4 | 0.0000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| 李先生 → 王小姐 | 4 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 李太太 → 胡大爷 | 4 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 杨大哥 → 徐小姐 | 4 | 0.0000 | 1.0000 | 0.2500 | 1.0000 | 0.2500 |

## Baseline-Aware Interpretation

- Chinese above baseline and Diacritic near chance: 24 subtypes
- Both models solve / close: 105 subtypes
- Both models below chance or unstable: 43 subtypes
- Diacritic-favoring: 104 subtypes

### Chinese Above, Diacritic Near Chance

| phenomenon | subtype | n | ch_acc | di_acc | gap | collapse |
| --- | --- | --- | --- | --- | --- | --- |
| BA | 恨 → 怕 | 300 | 0.8633 | 0.4900 | 0.3733 | 0.0000 |
| argument_structure | bad deletes 它 | 27 | 1.0000 | 0.5185 | 0.4815 | 0.0000 |
| argument_structure | 有点 → 专心 | 80 | 1.0000 | 0.5125 | 0.4875 | 0.0000 |
| classifier | 个 → 串 | 28 | 0.6429 | 0.5000 | 0.1429 | 0.0000 |
| classifier | 位 → 桶 | 12 | 0.8333 | 0.5000 | 0.3333 | 0.0000 |
| classifier | 位 → 瓶 | 14 | 0.6429 | 0.5000 | 0.1429 | 0.0000 |
| control_raising | multiple edits: 那片面包->\|->那片面包 | 11 | 0.8182 | 0.5455 | 0.2727 | 0.0000 |
| ellipsis | 部 → 本 | 25 | 0.6400 | 0.4800 | 0.1600 | 0.0000 |
| nominal_expression | multiple edits: ->五个\|五个-> | 34 | 0.7353 | 0.5000 | 0.2353 | 0.0000 |
| nominal_expression | multiple edits: ->四个\|四个-> | 35 | 0.7714 | 0.5143 | 0.2571 | 0.0000 |
| npi_licensing | multiple edits: 任何->\|她->任何人 | 99 | 0.9899 | 0.5354 | 0.4545 | 0.0000 |
| passive | bad deletes 张三 | 22 | 0.7273 | 0.5455 | 0.1818 | 0.0000 |
| passive | bad deletes 李先生 | 20 | 1.0000 | 0.5000 | 0.5000 | 0.0000 |
| passive | bad deletes 李四 | 15 | 0.9333 | 0.5333 | 0.4000 | 0.0000 |
| passive | multiple edits: ->被你们\|被你们-> | 25 | 0.8800 | 0.5200 | 0.3600 | 0.0000 |
| passive | multiple edits: ->被她\|被她-> | 24 | 0.7500 | 0.5417 | 0.2083 | 0.0000 |
| passive | 耳朵 → 手套 | 15 | 1.0000 | 0.4667 | 0.5333 | 0.0000 |
| topicalization | multiple edits: ->什么鱼\|什么鱼-> | 16 | 0.8125 | 0.5000 | 0.3125 | 0.0000 |
| verb_phrase | multiple edits: ->创作过\|创作过-> | 12 | 0.6667 | 0.5000 | 0.1667 | 0.0000 |
| verb_phrase | multiple edits: ->屠宰着\|屠宰着-> | 15 | 1.0000 | 0.5333 | 0.4667 | 0.0000 |

### Both Models Solve

| phenomenon | subtype | n | ch_acc | di_acc | gap | collapse |
| --- | --- | --- | --- | --- | --- | --- |
| BA | bad inserts 他 | 133 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| BA | bad inserts 她 | 167 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| BA | multiple edits: ->把鱼\|把鱼-> | 24 | 0.9583 | 0.9167 | 0.0417 | 0.0000 |
| BA | multiple edits: 你->\|->你 | 17 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| BA | multiple edits: 你们->\|->你们 | 10 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| BA | multiple edits: 把轮船->\|->轮船 | 37 | 0.9459 | 0.9459 | 0.0000 | 0.0000 |
| BA | multiple edits: 把飞机->\|->飞机 | 30 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| argument_structure | bad deletes 他们 | 42 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| argument_structure | bad deletes 你 | 31 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| argument_structure | bad deletes 你们 | 33 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| argument_structure | bad deletes 她们 | 37 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| argument_structure | bad deletes 我 | 40 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| argument_structure | bad deletes 我们 | 33 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| argument_structure | 可能 → 努力 | 81 | 0.9506 | 1.0000 | -0.0494 | 0.0000 |
| classifier | 头 → 个 | 61 | 0.6557 | 0.6230 | 0.0328 | 0.0000 |
| control_raising | multiple edits: ->会\|会-> | 68 | 1.0000 | 0.9706 | 0.0294 | 0.0000 |
| control_raising | 可以 → 想要 | 51 | 0.9804 | 1.0000 | -0.0196 | 0.0000 |
| control_raising | 可以 → 愿意 | 35 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| control_raising | 可以 → 期待 | 39 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| control_raising | 应该 → 想要 | 45 | 0.9556 | 0.9778 | -0.0222 | 0.0000 |

### Both Models Fail

| phenomenon | subtype | n | ch_acc | di_acc | gap | collapse |
| --- | --- | --- | --- | --- | --- | --- |
| BA | multiple edits: 在包扎->把\|->包扎 | 12 | 0.1667 | 0.0000 | 0.1667 | 0.0000 |
| BA | multiple edits: 在屠宰->把\|->屠宰 | 12 | 0.3333 | 0.3333 | 0.0000 | 0.0000 |
| BA | multiple edits: 在打断->把\|->打断 | 14 | 0.0000 | 0.2857 | -0.2857 | 0.0000 |
| BA | multiple edits: 在炖->把\|->炖 | 15 | 0.1333 | 0.0000 | 0.1333 | 0.0000 |
| BA | multiple edits: 在烧->把\|->烧 | 14 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| BA | multiple edits: 在爆炒->把\|->爆炒 | 10 | 0.2000 | 0.0000 | 0.2000 | 0.0000 |
| BA | multiple edits: 在麻醉->把\|->麻醉 | 14 | 0.1429 | 0.0000 | 0.1429 | 0.0000 |
| BA | 使 → 把 | 145 | 0.4966 | 0.3517 | 0.1448 | 0.0000 |
| anaphor | 他 → 她 | 292 | 0.1027 | 0.0000 | 0.1027 | 1.0000 |
| argument_structure | bad deletes 他 | 24 | 0.4583 | 0.2917 | 0.1667 | 0.0000 |
| classifier | multiple edits: ->串香蕉是\|串香蕉->的 | 14 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| classifier | multiple edits: ->只手套是\|只手套->的 | 11 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| classifier | multiple edits: ->只袜子是\|只袜子->的 | 12 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| classifier | multiple edits: ->本教材是\|本教材->的 | 12 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| classifier | multiple edits: ->桶啤酒是\|桶啤酒->的 | 23 | 0.0000 | 0.0870 | -0.0870 | 0.0000 |
| classifier | multiple edits: ->桶方便面是\|桶方便面->的 | 11 | 0.0000 | 0.3636 | -0.3636 | 0.0000 |
| classifier | multiple edits: ->片面包是\|片面包->的 | 19 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| classifier | 个 → 条 | 18 | 0.4444 | 0.3333 | 0.1111 | 0.0000 |
| classifier | 条 → 个 | 49 | 0.4082 | 0.4898 | -0.0816 | 0.0000 |
| classifier | 条 → 位 | 51 | 0.3725 | 0.3529 | 0.0196 | 0.0000 |

### Diacritic-Favoring

| phenomenon | subtype | n | ch_acc | di_acc | gap | collapse |
| --- | --- | --- | --- | --- | --- | --- |
| BA | bad inserts 了 | 300 | 0.5900 | 0.7200 | -0.1300 | 0.0000 |
| BA | multiple edits: ->鱼被\|把鱼-> | 20 | 0.8000 | 1.0000 | -0.2000 | 0.0000 |
| BA | multiple edits: ->鸡被\|把鸡-> | 17 | 0.7647 | 0.8824 | -0.1176 | 0.0000 |
| BA | multiple edits: ->鸭被\|把鸭-> | 16 | 0.6250 | 0.8750 | -0.2500 | 0.0000 |
| BA | multiple edits: 他把->\|->被他 | 23 | 0.3913 | 1.0000 | -0.6087 | 0.0000 |
| BA | multiple edits: 你们把->\|->被你们 | 10 | 0.3000 | 0.9000 | -0.6000 | 0.0000 |
| BA | multiple edits: 你把->\|->被你 | 21 | 0.8095 | 0.9048 | -0.0952 | 0.0000 |
| BA | multiple edits: 在打断->把\|->打断 | 14 | 0.0000 | 0.2857 | -0.2857 | 0.0000 |
| BA | multiple edits: 在清洗->把\|->清洗 | 15 | 0.3333 | 0.6667 | -0.3333 | 0.0000 |
| BA | multiple edits: 她把->\|->被她 | 29 | 0.7586 | 1.0000 | -0.2414 | 0.0000 |
| BA | multiple edits: 我把->\|->被我 | 29 | 0.5172 | 0.9310 | -0.4138 | 0.0000 |
| BA | multiple edits: 把货车->\|->货车 | 29 | 0.8966 | 0.9655 | -0.0690 | 0.0000 |
| anaphor | bad inserts 们 | 300 | 0.4800 | 0.8567 | -0.3767 | 0.0000 |
| classifier | multiple edits: ->张桌子是\|张桌子->的 | 20 | 0.0000 | 0.9000 | -0.9000 | 0.0000 |
| classifier | multiple edits: ->把椅子是\|把椅子->的 | 12 | 0.0000 | 0.6667 | -0.6667 | 0.0000 |
| classifier | multiple edits: ->桶啤酒是\|桶啤酒->的 | 23 | 0.0000 | 0.0870 | -0.0870 | 0.0000 |
| classifier | multiple edits: ->桶方便面是\|桶方便面->的 | 11 | 0.0000 | 0.3636 | -0.3636 | 0.0000 |
| classifier | 个 → 只 | 14 | 0.4286 | 0.5000 | -0.0714 | 0.0000 |
| classifier | 头 → 位 | 47 | 0.4043 | 0.5957 | -0.1915 | 0.0000 |
| classifier | 条 → 个 | 49 | 0.4082 | 0.4898 | -0.0816 | 0.0000 |
