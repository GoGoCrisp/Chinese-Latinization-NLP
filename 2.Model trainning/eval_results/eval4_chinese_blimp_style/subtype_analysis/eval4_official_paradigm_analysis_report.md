# Eval 4 Official ZhoBLiMP Paradigm Analysis

## Loaded Data

- dataset: /Users/crisp/Desktop/code_field/python/Chinese_Latinization_NLP/2.Model trainning/eval_data/eval4_chinese_blimp_style/eval4_chinese_blimp_style.jsonl
- scores: /Users/crisp/Desktop/code_field/python/Chinese_Latinization_NLP/2.Model trainning/eval_results/eval4_chinese_blimp_style/item_scores.csv
- items: 35400
- score rows: 70800
- official template JSON files loaded: 131
- official unique template UIDs loaded: 129
- official paradigms represented in dataset: 120
- unique phenomenon labels: 15
- unique paradigm + observed surface diff rows: 3955

## Phenomenon Level

| phenomenon | n | ch_acc | di_acc | gap | ch_margin | di_margin | collapse | tie |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BA | 3900 | 0.7546 | 0.6305 | 0.1241 | 0.2655 | 0.1111 | 0.0000 | 0.0000 |
| anaphor | 1800 | 0.5539 | 0.3461 | 0.2078 | 0.1034 | -0.0018 | 0.3333 | 0.3333 |
| argument_structure | 2100 | 0.6971 | 0.6443 | 0.0529 | 0.2853 | 0.2186 | 0.0000 | 0.0000 |
| classifier | 900 | 0.3900 | 0.4400 | -0.0500 | -0.2932 | -0.0871 | 0.0000 | 0.0000 |
| control_raising | 1200 | 0.8767 | 0.7333 | 0.1433 | 0.4308 | 0.2739 | 0.0000 | 0.0000 |
| ellipsis | 900 | 0.3644 | 0.3556 | 0.0089 | -0.4458 | -0.4194 | 0.0000 | 0.0000 |
| fci_licensing | 1500 | 0.8767 | 0.5193 | 0.3573 | 0.2224 | 0.0062 | 0.0000 | 0.0000 |
| nominal_expression | 3300 | 0.4900 | 0.4400 | 0.0500 | 0.0507 | 0.0219 | 0.0000 | 0.0000 |
| npi_licensing | 2700 | 0.7915 | 0.5185 | 0.2730 | 0.2290 | 0.0329 | 0.0000 | 0.0000 |
| passive | 3600 | 0.7878 | 0.7283 | 0.0594 | 0.2912 | 0.2662 | 0.0000 | 0.0000 |
| quantifiers | 600 | 0.2650 | 0.5033 | -0.2383 | -0.0901 | -0.0187 | 0.0000 | 0.0000 |
| question | 6300 | 0.5968 | 0.6330 | -0.0362 | 0.1294 | 0.2387 | 0.0000 | 0.0000 |
| relativization | 1200 | 0.6633 | 0.6492 | 0.0142 | 0.1417 | 0.1472 | 0.0000 | 0.0000 |
| topicalization | 1200 | 0.7983 | 0.7642 | 0.0342 | 0.4970 | 0.4691 | 0.0000 | 0.0000 |
| verb_phrase | 4200 | 0.8595 | 0.7190 | 0.1405 | 0.3677 | 0.2331 | 0.0000 | 0.0000 |

## All Official Paradigms

This is the compact subtype report: each row is the official ZhoBLiMP UID, the intended template rule replacement, and both models' accuracy and mean margin.

| phenomenon | uid | n | rule_diff | ch_acc | di_acc | gap | ch_margin | di_margin | collapse | tie |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BA | BA_BEI_subj_drop | 300 | INSERT_BAD:pos:NN + 被 \| DELETE_BAD:把 + pos:NN | 0.6133 | 0.8400 | -0.2267 | 0.0850 | 0.3680 | 0.0000 | 0.0000 |
| BA | BA_deletion | 300 | DELETE_BAD:把 + pos:NN subcat:vehicle \| pos:NN subcat:container \| INSERT_BAD:pos:NN subcat:vehicle \| pos:NN subcat:container +  | 0.9600 | 0.8333 | 0.1267 | 0.4874 | 0.3329 | 0.0000 | 0.0000 |
| BA | BA_duplicate_argument | 300 |  -> pos:PN mPos:0 mPro:gender/number | 1.0000 | 1.0000 | 0.0000 | 0.5011 | 0.4069 | 0.0000 | 0.0000 |
| BA | BA_inversion | 300 | DELETE_BAD:phrase:Person \| phrase:ComplexPerson \| INSERT_BAD:phrase:Person \| phrase:ComplexPerson | 0.9467 | 0.7233 | 0.2233 | 0.5291 | 0.2160 | 0.0000 | 0.0000 |
| BA | BA_meiba | 300 | 把 -> 把没把 | 0.5467 | 0.4500 | 0.0967 | 0.0155 | -0.0164 | 0.0000 | 0.0000 |
| BA | BA_negation | 300 | DELETE_BAD:没有 \| INSERT_BAD:没有 | 0.9300 | 0.6733 | 0.2567 | 0.4575 | 0.0966 | 0.0000 | 0.0000 |
| BA | BA_no_progressive | 300 | 正在 + pos:VV transitivity:tran mPos:5 mPro:tran_verb -> 正把 \| INSERT_BAD:pos:VV transitivity:tran mPos:5 mPro:tran_verb | 0.5133 | 0.3033 | 0.2100 | 0.0478 | -0.2545 | 0.0000 | 0.0000 |
| BA | BA_no_stative_verb | 300 | 恨 -> 怕 | 0.8633 | 0.4900 | 0.3733 | 0.2073 | -0.0046 | 0.0000 | 0.0000 |
| BA | BA_suo_adverbial_a | 300 |  -> 所 | 0.2733 | 0.2600 | 0.0133 | -0.1011 | -0.1187 | 0.0000 | 0.0000 |
| BA | BA_suo_adverbial_b | 300 |  -> 所 | 0.8633 | 0.4500 | 0.4133 | 0.1793 | -0.0423 | 0.0000 | 0.0000 |
| BA | BA_verb_le_a | 300 | 把 -> 把了 | 0.5900 | 0.7200 | -0.1300 | 0.0306 | 0.0643 | 0.0000 | 0.0000 |
| BA | BA_verb_le_b | 300 | 了。 -> 。 | 0.9833 | 0.9100 | 0.0733 | 0.8191 | 0.3947 | 0.0000 | 0.0000 |
| BA | causative_shi_ba | 300 | 使 让 -> 把 | 0.7267 | 0.5433 | 0.1833 | 0.1926 | 0.0011 | 0.0000 | 0.0000 |
| anaphor | anaphor_gender_agreement | 300 | pos:NR subcat:person matchedPosition:2 matchedProperties:gender -> pos:NR subcat:person mismatchedPosition:2 mismatchedProperties:gender | 0.6867 | 0.5233 | 0.1633 | 0.2396 | 0.0125 | 0.0000 | 0.0000 |
| anaphor | anaphor_number_agreement | 300 | 们自己 -> 自己 | 0.5267 | 0.3533 | 0.1733 | 0.2451 | -0.0590 | 0.0000 | 0.0000 |
| anaphor | principle_A_c_command | 300 | pos:PN number:singular mPos:2 mPro:gender -> pos:PN number:singular mmPos:2 mmPro:gender | 0.5133 | 0.0000 | 0.5133 | 0.0361 | 0.0000 | 1.0000 | 1.0000 |
| anaphor | principle_A_c_command_number | 300 | 们自己 -> 自己 | 0.5967 | 0.3433 | 0.2533 | 0.1885 | -0.0577 | 0.0000 | 0.0000 |
| anaphor | principle_A_domain | 300 | pos:PN number:singular matchedPosition:2 matchedProperties:gender -> pos:PN number:singular mismatchedPosition:2 mismatchedProperties:gender | 0.5200 | 0.0000 | 0.5200 | 0.0388 | 0.0000 | 1.0000 | 1.0000 |
| anaphor | principle_A_domain_number | 300 | 自己 -> 们自己 | 0.4800 | 0.8567 | -0.3767 | -0.1275 | 0.0936 | 0.0000 | 0.0000 |
| argument_structure | agent_animacy_adv | 300 | 有点 可能 -> pos:AD animate:1 | 0.9600 | 0.7533 | 0.2067 | 0.5311 | 0.2558 | 0.0000 | 0.0000 |
| argument_structure | agent_animacy_passive | 300 | phrase:Person -> pos:NN animate:0 | 0.5933 | 0.4900 | 0.1033 | 0.1403 | 0.0419 | 0.0000 | 0.0000 |
| argument_structure | agent_animacy_subj | 300 | pos:NN subcat:person -> pos:NN animate:0 | 0.3633 | 0.4067 | -0.0433 | -0.1856 | -0.1111 | 0.0000 | 0.0000 |
| argument_structure | agent_causative | 300 | transitivity:tran matchedPosition:4 matchedProperties:tran_verb -> pos:VE | 0.7500 | 0.8200 | -0.0700 | 0.4381 | 0.5757 | 0.0000 | 0.0000 |
| argument_structure | agent_deletion | 300 | pos:PN ->  | 0.9567 | 0.8267 | 0.1300 | 0.6011 | 0.3309 | 0.0000 | 0.0000 |
| argument_structure | intransitive_double_obj | 300 | INSERT_BAD:拿 + pos:NN animate:1 + 的 \| DELETE_BAD:pos:CD + pos:M mPro:classifier mPos:7 + pos:NN animate:1 | 0.7867 | 0.7067 | 0.0800 | 0.4889 | 0.3990 | 0.0000 | 0.0000 |
| argument_structure | intransitive_no_obj | 300 | transitivity:intran animate:1 -> transitivity:tran | 0.4700 | 0.5067 | -0.0367 | -0.0170 | 0.0385 | 0.0000 | 0.0000 |
| classifier | classifier_noun_agreement | 300 | pos:M subcat:person -> pos:M subcat:food \| pos:M subcat:beverage \| pos:M subcat:animal | 0.6433 | 0.4700 | 0.1733 | 0.2380 | 0.0148 | 0.0000 | 0.0000 |
| classifier | classifier_noun_agreement_no_gap | 300 | pos:M mPos:7 mPro:classifier -> pos:M mPos:5 mPro:classifier | 0.5200 | 0.5367 | -0.0167 | 0.0959 | 0.0241 | 0.0000 | 0.0000 |
| classifier | classifier_noun_subj | 300 | INSERT_BAD:pos:M mPos:3 mPro:classifier + pos:NN subcat:food \| pos:NN subcat:obj \| pos:NN subcat:beverage + 是 \| pos:M mPos:3 mPro:classifier + pos:NN subcat:food \| pos:NN subcat:obj \| pos:NN subcat:beverage +  + 。 -> 的。 | 0.0067 | 0.3133 | -0.3067 | -1.2137 | -0.3003 | 0.0000 | 0.0000 |
| control_raising | control_modal_vs_raising_modal | 300 | pos:VV subcat:modal attitude:pos -> pos:VV subcat:control attitude:pos | 0.9833 | 0.9633 | 0.0200 | 0.5149 | 0.4521 | 0.0000 | 0.0000 |
| control_raising | existential_there_subject_raising | 300 | 有 ->  | 0.8300 | 0.7600 | 0.0700 | 0.4267 | 0.3105 | 0.0000 | 0.0000 |
| control_raising | modal_raising_hui | 300 | INSERT_BAD:马上 可能 就 \|  + 会 要 \| DELETE_BAD:马上 可能 就 \|  + 会 要 | 0.9367 | 0.7033 | 0.2333 | 0.5261 | 0.3004 | 0.0000 | 0.0000 |
| control_raising | modal_raising_topicalization | 300 | INSERT_BAD:phrase:Person + 应该 + 能 可以 会 \| DELETE_BAD:phrase:Person + 应该 + 能 可以 会 | 0.7567 | 0.5067 | 0.2500 | 0.2553 | 0.0323 | 0.0000 | 0.0000 |
| ellipsis | ellipsis_adj | 300 | pos:VV transitivity:tran -> pos:VV transitivity:intran animate:1 \| pos:NN mPos:1 mPro:tran_verb -> 一小时 一会儿 一分钟 一天 很久 | 0.4067 | 0.3633 | 0.0433 | -0.1905 | -0.3099 | 0.0000 | 0.0000 |
| ellipsis | ellipsis_double_object | 300 | 是 +  +  -> pos:VV transitivity:ditran attitude:pos + 给了 + pos:NR subcat:person | 0.0000 | 0.0000 | 0.0000 | -1.2610 | -1.0709 | 0.0000 | 0.0000 |
| ellipsis | ellipsis_n_bar_class | 300 | pos:M mPos:4 mPro:expression -> pos:M mmPos:5 mmPro:classifier | 0.6867 | 0.7033 | -0.0167 | 0.1140 | 0.1224 | 0.0000 | 0.0000 |
| fci_licensing | fci_renhe_dou | 300 | 都 ->  | 0.7667 | 0.8867 | -0.1200 | 0.0861 | 0.1415 | 0.0000 | 0.0000 |
| fci_licensing | fci_renhe_prepP | 300 | 可以跟任何人一起 -> 可以跟没有人一起 | 1.0000 | 1.0000 | 0.0000 | 0.4290 | 0.5873 | 0.0000 | 0.0000 |
| fci_licensing | fci_renhe_ruguo | 300 |  -> 都 | 0.9967 | 0.0033 | 0.9933 | 0.3399 | -0.2616 | 0.0000 | 0.0000 |
| fci_licensing | fci_renhe_subj | 300 | 任何人都 -> 有些人都 | 0.7867 | 0.0200 | 0.7667 | 0.1661 | -0.5019 | 0.0000 | 0.0000 |
| fci_licensing | fci_renhe_suoyou | 300 | 所有 -> 任何 | 0.8333 | 0.6867 | 0.1467 | 0.0911 | 0.0659 | 0.0000 | 0.0000 |
| nominal_expression | PN_numP_a | 300 | pos:PN subcat:person number:singular -> pos:NN subcat:person | 0.5367 | 0.5967 | -0.0600 | 0.0330 | 0.0449 | 0.0000 | 0.0000 |
| nominal_expression | PN_numP_b | 300 | DELETE_BAD:pos:PN subcat:person number:plural \| INSERT_BAD:pos:PN subcat:person number:plural | 0.6333 | 0.4933 | 0.1400 | -0.0689 | 0.0587 | 0.0000 | 0.0000 |
| nominal_expression | nominal_definite_men | 300 |  -> 们 | 0.1100 | 0.0933 | 0.0167 | -0.3289 | -0.3627 | 0.0000 | 0.0000 |
| nominal_expression | nominal_modal_insertion | 300 | INSERT_BAD:pos:VV subcat:modal \| DELETE_BAD:pos:VV subcat:modal | 0.6767 | 0.6233 | 0.0533 | 0.0912 | 0.1127 | 0.0000 | 0.0000 |
| nominal_expression | noun_adjective_shi | 300 | 是 ->  | 0.8433 | 0.8367 | 0.0067 | 0.7052 | 0.7611 | 0.0000 | 0.0000 |
| nominal_expression | noun_phrase_conjunction_jian | 300 | 和\|跟 -> 兼 | 0.6100 | 0.6433 | -0.0333 | 0.0776 | 0.0731 | 0.0000 | 0.0000 |
| nominal_expression | plural_cardinal_men_a | 300 | DELETE_BAD: \| INSERT_BAD:们 | 0.0867 | 0.0900 | -0.0033 | -0.2510 | -0.3064 | 0.0000 | 0.0000 |
| nominal_expression | plural_cardinal_men_b | 300 | DELETE_BAD: \| INSERT_BAD:们 | 0.0400 | 0.0867 | -0.0467 | -0.2495 | -0.2677 | 0.0000 | 0.0000 |
| nominal_expression | singular_PN_but_plural_pron | 300 | 们 ->  | 0.0333 | 0.0967 | -0.0633 | -0.2384 | -0.1444 | 0.0000 | 0.0000 |
| nominal_expression | you_quantifier_adj | 300 | INSERT_BAD:pos:VA mPos:9 mPro:subcat \| DELETE_BAD:pos:VA mPos:9 mPro:subcat | 0.9233 | 0.8867 | 0.0367 | 0.5547 | 0.4698 | 0.0000 | 0.0000 |
| nominal_expression | you_yige | 300 |  -> 有 | 0.8967 | 0.3933 | 0.5033 | 0.2322 | -0.1978 | 0.0000 | 0.0000 |
| npi_licensing | npi_renhe_A_not_A_question | 300 | 有 + 没有 ->  +  \| 任何 -> 了任何 \| ？ -> 。 | 0.8667 | 0.6433 | 0.2233 | 0.3159 | 0.1934 | 0.0000 | 0.0000 |
| npi_licensing | npi_renhe_conditional | 300 | 如果有任何人 -> 如果有人 \| 我 他 她 -> 任何人 | 0.8267 | 0.6500 | 0.1767 | 0.1880 | 0.1253 | 0.0000 | 0.0000 |
| npi_licensing | npi_renhe_neg_scope_locP | 300 | INSERT_BAD:phrase:PPRH \| DELETE_BAD:phrase:PPRH | 0.8933 | 0.2567 | 0.6367 | 0.2525 | -0.2263 | 0.0000 | 0.0000 |
| npi_licensing | npi_renhe_neg_scope_subj | 300 | DELETE_BAD:没有 \| INSERT_BAD:没有 | 0.3533 | 0.3967 | -0.0433 | -0.1236 | -0.1146 | 0.0000 | 0.0000 |
| npi_licensing | npi_renhe_wh_question_obj | 300 | phrase:Possessive -> 任何 | 0.5500 | 0.6700 | -0.1200 | 0.0905 | 0.1486 | 0.0000 | 0.0000 |
| npi_licensing | npi_renhe_wh_question_subj | 300 | pos:CD \| pos:DT subcat:woCD + pos:M matchedPosition:2 matchedProperties:classifier -> 任何 +  | 0.7000 | 0.4033 | 0.2967 | 0.1548 | -0.0729 | 0.0000 | 0.0000 |
| npi_licensing | renhe_no_episodic_sentences | 300 | DELETE_BAD:没 \| INSERT_BAD:了 | 0.9767 | 0.7700 | 0.2067 | 0.5100 | 0.2910 | 0.0000 | 0.0000 |
| npi_licensing | renhe_no_superordinate_negation | 300 | INSERT_BAD:你 + 有 \| DELETE_BAD:你 + 的 | 0.9833 | 0.0467 | 0.9367 | 0.3357 | -0.2643 | 0.0000 | 0.0000 |
| npi_licensing | renhe_non_factive_verb | 300 | 觉得 相信 希望 -> 知道 主张 | 0.9733 | 0.8300 | 0.1433 | 0.3370 | 0.2159 | 0.0000 | 0.0000 |
| passive | BEI_construction_a | 300 | INSERT_BAD:被 + phrase:Person \| DELETE_BAD:被 + phrase:Person | 0.9067 | 0.7333 | 0.1733 | 0.3375 | 0.2973 | 0.0000 | 0.0000 |
| passive | BEI_construction_b | 300 | DELETE_BAD:phrase:Person \| INSERT_BAD:phrase:Person | 0.8700 | 0.7900 | 0.0800 | 0.3435 | 0.4208 | 0.0000 | 0.0000 |
| passive | BEI_deletion | 300 | DELETE_BAD:pos:NN subcat:vehicle \| pos:NN subcat:container + 上被 \| INSERT_BAD:pos:NN subcat:vehicle \| pos:NN subcat:container + 上 | 0.5467 | 0.7267 | -0.1800 | 0.0712 | 0.1735 | 0.0000 | 0.0000 |
| passive | BEI_preposition | 300 | DELETE_BAD:pos:PN subcat:person \| INSERT_BAD:pos:PN subcat:person | 0.9033 | 0.8400 | 0.0633 | 0.5621 | 0.6028 | 0.0000 | 0.0000 |
| passive | passive_agent_deletion_long_left | 300 | pos:NR subcat:anyone ->  | 0.9967 | 0.9933 | 0.0033 | 0.3965 | 0.4827 | 0.0000 | 0.0000 |
| passive | passive_agent_deletion_long_right_a | 300 | 被我叫 -> 被叫 | 0.9200 | 0.8100 | 0.1100 | 0.1091 | 0.1007 | 0.0000 | 0.0000 |
| passive | passive_agent_deletion_long_right_b | 300 | pos:PN subcat:person \| pos:NR subcat:person mmPos:0 mmPro:expression ->  | 0.7467 | 0.6967 | 0.0500 | 0.2236 | 0.1937 | 0.0000 | 0.0000 |
| passive | passive_agent_deletion_short | 300 | pos:PN subcat:person \| pos:NR subcat:person mmPos:0 mmPro:expression ->  | 0.6800 | 0.5433 | 0.1367 | 0.1830 | 0.0428 | 0.0000 | 0.0000 |
| passive | passive_body_part | 300 | pos:NN subcat:body -> pos:NN subcat:obj | 0.5067 | 0.7000 | -0.1933 | 0.0181 | 0.1796 | 0.0000 | 0.0000 |
| passive | passive_intransitive | 300 | pos:VV transitivity:tran subcat2:stative \| pos:VV transitivity:tran subcat2:action -> pos:VV transitivity:intran animate:1 | 0.6467 | 0.6000 | 0.0467 | 0.2886 | 0.1443 | 0.0000 | 0.0000 |
| passive | passive_no_adj | 300 | pos:VV subcat:person subcat2:action -> pos:VA | 0.8367 | 0.6000 | 0.2367 | 0.5464 | 0.1540 | 0.0000 | 0.0000 |
| passive | passive_suo | 300 | pos:PN subcat:person \| pos:NR subcat:person ->  | 0.8933 | 0.7067 | 0.1867 | 0.4152 | 0.4017 | 0.0000 | 0.0000 |
| quantifiers | superlative_quantifiers_1 | 300 | 超过 -> 至少 | 0.0167 | 0.1267 | -0.1100 | -0.2328 | -0.3837 | 0.0000 | 0.0000 |
| quantifiers | superlative_quantifiers_2 | 300 | 有的 -> 没有 | 0.5133 | 0.8800 | -0.3667 | 0.0525 | 0.3463 | 0.0000 | 0.0000 |
| question | question_A_not_A | 300 |  -> phrase:Person | 0.1733 | 0.2233 | -0.0500 | -0.4240 | -0.5619 | 0.0000 | 0.0000 |
| question | question_A_not_A_daodi_a | 300 | INSERT_BAD:不 \| DELETE_BAD:不 | 0.7600 | 0.7833 | -0.0233 | 0.2531 | 0.4314 | 0.0000 | 0.0000 |
| question | question_A_not_A_daodi_b | 300 | INSERT_BAD:不 \| DELETE_BAD:不 | 0.7500 | 0.7500 | 0.0000 | 0.2479 | 0.3474 | 0.0000 | 0.0000 |
| question | question_A_not_A_indirect | 300 |  + 。 -> 呢 + ？ | 0.0000 | 0.1067 | -0.1067 | -0.4042 | -0.1236 | 0.0000 | 0.0000 |
| question | question_V_not_VP_1 | 300 | INSERT_BAD:pos:NN subcat:location + pos:VV subcat2:leave \| DELETE_BAD:pos:NN subcat:location + pos:VV subcat2:leave | 0.5400 | 0.9100 | -0.3700 | 0.0360 | 0.3471 | 0.0000 | 0.0000 |
| question | question_V_not_VP_2 | 300 | DELETE_BAD:把 + 不 \| INSERT_BAD:不 + 把 | 0.8233 | 0.6067 | 0.2167 | 0.1321 | 0.0781 | 0.0000 | 0.0000 |
| question | question_daodi_nandao_1 | 300 | 难道 -> 到底 | 1.0000 | 0.7733 | 0.2267 | 0.8618 | 0.2832 | 0.0000 | 0.0000 |
| question | question_daodi_nandao_2 | 300 | 到底 -> 难道 | 0.0300 | 0.9433 | -0.9133 | -0.3867 | 0.8371 | 0.0000 | 0.0000 |
| question | question_daodi_nandao_A_not_A_intran | 300 | 到底 -> 难道 | 0.0300 | 0.9533 | -0.9233 | -0.5260 | 0.8167 | 0.0000 | 0.0000 |
| question | question_daodi_nandao_A_not_A_tran | 300 | 到底 -> 难道 | 0.0733 | 0.9133 | -0.8400 | -0.3902 | 0.6829 | 0.0000 | 0.0000 |
| question | question_daodi_negation | 300 | INSERT_BAD:不 \| DELETE_BAD:不 | 0.9967 | 0.8467 | 0.1500 | 1.1162 | 0.3187 | 0.0000 | 0.0000 |
| question | question_nandao_negation | 300 | INSERT_BAD:不 \| 不 + 愿意 \| 想 \| 希望  -> 愿意 \| 想 \| 希望 | 0.5633 | 0.8400 | -0.2767 | 0.0435 | 0.4678 | 0.0000 | 0.0000 |
| question | question_nandao_raising_1_a | 300 | INSERT_BAD:是 \| DELETE_BAD:是 | 0.9767 | 0.9733 | 0.0033 | 0.6686 | 0.6385 | 0.0000 | 0.0000 |
| question | question_nandao_raising_1_b | 300 | DELETE_BAD:难道 \| INSERT_BAD:难道 | 0.9933 | 0.9433 | 0.0500 | 0.7323 | 0.4571 | 0.0000 | 0.0000 |
| question | question_nandao_raising_2 | 300 | DELETE_BAD:难道 \| INSERT_BAD:难道 | 1.0000 | 0.8500 | 0.1500 | 0.6088 | 0.5360 | 0.0000 | 0.0000 |
| question | question_nandao_raising_3 | 300 | INSERT_BAD:是 \| DELETE_BAD:是 | 0.6333 | 0.7567 | -0.1233 | 0.0738 | 0.2341 | 0.0000 | 0.0000 |
| question | question_nandao_scope_1 | 300 | DELETE_BAD:难道 \| INSERT_BAD:难道 | 0.0900 | 0.5500 | -0.4600 | -0.2749 | -0.0229 | 0.0000 | 0.0000 |
| question | question_nandao_scope_2 | 300 | INSERT_BAD:认为 \| 确定 \| 相信 \| DELETE_BAD:认为 \| 确定 \| 相信 | 0.2900 | 0.2200 | 0.0700 | -0.1226 | -0.1684 | 0.0000 | 0.0000 |
| question | question_particle_daodi_choice_intran | 300 | 呢 -> 吗 | 0.9567 | 0.1300 | 0.8267 | 0.1168 | -0.0792 | 0.0000 | 0.0000 |
| question | question_particle_daodi_choice_tran | 300 | 呢 -> 吗 | 0.8633 | 0.2167 | 0.6467 | 0.0601 | -0.0484 | 0.0000 | 0.0000 |
| question | question_particle_nandao | 300 | 难道 -> 到底 | 0.9900 | 0.0033 | 0.9867 | 0.2943 | -0.4593 | 0.0000 | 0.0000 |
| relativization | relative_operator_intepretation | 300 | 原因 ->  | 0.7967 | 0.8233 | -0.0267 | 0.2137 | 0.2492 | 0.0000 | 0.0000 |
| relativization | relative_operator_who | 300 | pos:PN subcat:person number:singular -> 谁 \| 。 -> ？ | 0.7700 | 0.8633 | -0.0933 | 0.1431 | 0.2703 | 0.0000 | 0.0000 |
| relativization | relativization_movement_no_gap | 300 |  -> pos:PN subcat:person subcat2:person3 number:singular matchedPosition:7 matchedProperties:gender | 0.7733 | 0.5967 | 0.1767 | 0.2669 | 0.1322 | 0.0000 | 0.0000 |
| relativization | relativization_movement_when_where | 300 |  -> 所 | 0.3133 | 0.3133 | 0.0000 | -0.0570 | -0.0627 | 0.0000 | 0.0000 |
| topicalization | topicalization_OSV | 300 | INSERT_BAD:一 + pos:M matchedPosition:5 matchedProperties:classifier + pos:NN matchedPosition:2 matchedProperties:tran_verb \| DELETE_BAD:一 + pos:M matchedPosition:5 matchedProperties:classifier + pos:NN matchedPosition:2 matchedProperties:tran_verb | 0.8833 | 0.7733 | 0.1100 | 0.6393 | 0.6525 | 0.0000 | 0.0000 |
| topicalization | topicalization_OSV_mei | 300 | INSERT_BAD:任何 什么 + pos:NN matchedPosition:2 matchedProperties:tran_verb \| DELETE_BAD:任何 什么 + pos:NN matchedPosition:2 matchedProperties:tran_verb | 0.8833 | 0.7400 | 0.1433 | 0.6057 | 0.3882 | 0.0000 | 0.0000 |
| topicalization | topicalization_SOV | 300 | DELETE_BAD:在 + pos:VV transitivity:tran \| INSERT_BAD:在 + pos:VV transitivity:tran | 0.4833 | 0.6333 | -0.1500 | -0.0136 | 0.1731 | 0.0000 | 0.0000 |
| topicalization | topicalization_SOV_mei | 300 | INSERT_BAD:任何 什么 + pos:NN matchedPosition:2 matchedProperties:tran_verb \| DELETE_BAD:任何 什么 + pos:NN matchedPosition:2 matchedProperties:tran_verb | 0.9433 | 0.9100 | 0.0333 | 0.7567 | 0.6624 | 0.0000 | 0.0000 |
| verb_phrase | adjective_transitive_dui | 300 | 对 -> 有点 比较 很 非常 + pos:VA subcat:emotion +  \| DELETE_BAD:有点 比较 很 非常 + pos:VA subcat:emotion | 0.9100 | 0.8300 | 0.0800 | 0.5446 | 0.4687 | 0.0000 | 0.0000 |
| verb_phrase | left_adverbial_b | 300 | DELETE_BAD:pos:AD matchedPosition:2 matchedProperties:verb \| INSERT_BAD:pos:AD matchedPosition:2 matchedProperties:verb | 0.8933 | 0.7133 | 0.1800 | 0.4190 | 0.1925 | 0.0000 | 0.0000 |
| verb_phrase | left_adverbial_d | 300 | DELETE_BAD:pos:AD matchedPosition:2 matchedProperties:verb \| INSERT_BAD:pos:AD matchedPosition:2 matchedProperties:verb | 0.7933 | 0.5533 | 0.2400 | 0.2470 | 0.0993 | 0.0000 | 0.0000 |
| verb_phrase | left_adverbial_e | 300 | DELETE_BAD:pos:AD subcat:transfer \| INSERT_BAD:pos:AD subcat:transfer | 0.8767 | 0.7367 | 0.1400 | 0.2518 | 0.1721 | 0.0000 | 0.0000 |
| verb_phrase | left_adverbial_negation | 300 | INSERT_BAD:pos:AD mPos:3 mPro:verb \| DELETE_BAD:pos:AD mPos:3 mPro:verb | 0.8300 | 0.6700 | 0.1600 | 0.2424 | 0.1505 | 0.0000 | 0.0000 |
| verb_phrase | left_dou | 300 | DELETE_BAD:都 \| INSERT_BAD:都 | 0.9733 | 0.8033 | 0.1700 | 0.3451 | 0.1970 | 0.0000 | 0.0000 |
| verb_phrase | preposition_deletion | 300 | 在 ->  | 0.9367 | 0.9000 | 0.0367 | 0.4904 | 0.4641 | 0.0000 | 0.0000 |
| verb_phrase | preposition_insertion | 300 | INSERT_BAD:pos:VV transitivity:tran \| pos:VV transitivity:tran + 了 ->  | 0.9867 | 0.9133 | 0.0733 | 0.8458 | 0.5429 | 0.0000 | 0.0000 |
| verb_phrase | right_yijing_a | 300 | 给 -> 过 \| DELETE_BAD:pos:NN subcat:animal \| pos:NN subcat3:movable animate:0 \| 了 -> pos:NN subcat:animal \| pos:NN subcat3:movable animate:0 +  | 0.9700 | 0.8633 | 0.1067 | 0.7331 | 0.4857 | 0.0000 | 0.0000 |
| verb_phrase | right_yijing_b | 300 | DELETE_BAD:pos:NN subcat:animal \| pos:NN subcat3:movable animate:0 \| INSERT_BAD:pos:NN subcat:animal \| pos:NN subcat3:movable animate:0 | 0.7000 | 0.6767 | 0.0233 | 0.0874 | 0.1794 | 0.0000 | 0.0000 |
| verb_phrase | verb_negation_particle | 300 | 过 -> 了 | 0.8667 | 0.6900 | 0.1767 | 0.3587 | 0.1635 | 0.0000 | 0.0000 |
| verb_phrase | verb_phrase_left_adverbial | 300 | INSERT_BAD:被 \| DELETE_BAD:被 | 0.8667 | 0.5100 | 0.3567 | 0.2222 | -0.0065 | 0.0000 | 0.0000 |
| verb_phrase | verb_phrase_left_negation | 300 | INSERT_BAD:把 +  + pos:PN subcat:person mismatchedPosition:0 mismatchedProperties:expression \| pos:NR subcat:person \| DELETE_BAD:把 + pos:PN subcat:person mismatchedPosition:0 mismatchedProperties:expression \| pos:NR subcat:person +  | 0.9233 | 0.7700 | 0.1533 | 0.3568 | 0.1836 | 0.0000 | 0.0000 |
| verb_phrase | ya_insertion | 300 | 说 认为 觉得 以为 -> 告诉 告知 劝 | 0.5067 | 0.4367 | 0.0700 | 0.0031 | -0.0291 | 0.0000 | 0.0000 |

## Top 20 Official Paradigms By n_items

| phenomenon | uid | n | rule_diff | ch_acc | di_acc | gap | ch_margin | di_margin | collapse | tie |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BA | BA_BEI_subj_drop | 300 | INSERT_BAD:pos:NN + 被 \| DELETE_BAD:把 + pos:NN | 0.6133 | 0.8400 | -0.2267 | 0.0850 | 0.3680 | 0.0000 | 0.0000 |
| BA | BA_deletion | 300 | DELETE_BAD:把 + pos:NN subcat:vehicle \| pos:NN subcat:container \| INSERT_BAD:pos:NN subcat:vehicle \| pos:NN subcat:container +  | 0.9600 | 0.8333 | 0.1267 | 0.4874 | 0.3329 | 0.0000 | 0.0000 |
| BA | BA_duplicate_argument | 300 |  -> pos:PN mPos:0 mPro:gender/number | 1.0000 | 1.0000 | 0.0000 | 0.5011 | 0.4069 | 0.0000 | 0.0000 |
| BA | BA_inversion | 300 | DELETE_BAD:phrase:Person \| phrase:ComplexPerson \| INSERT_BAD:phrase:Person \| phrase:ComplexPerson | 0.9467 | 0.7233 | 0.2233 | 0.5291 | 0.2160 | 0.0000 | 0.0000 |
| BA | BA_meiba | 300 | 把 -> 把没把 | 0.5467 | 0.4500 | 0.0967 | 0.0155 | -0.0164 | 0.0000 | 0.0000 |
| BA | BA_negation | 300 | DELETE_BAD:没有 \| INSERT_BAD:没有 | 0.9300 | 0.6733 | 0.2567 | 0.4575 | 0.0966 | 0.0000 | 0.0000 |
| BA | BA_no_progressive | 300 | 正在 + pos:VV transitivity:tran mPos:5 mPro:tran_verb -> 正把 \| INSERT_BAD:pos:VV transitivity:tran mPos:5 mPro:tran_verb | 0.5133 | 0.3033 | 0.2100 | 0.0478 | -0.2545 | 0.0000 | 0.0000 |
| BA | BA_no_stative_verb | 300 | 恨 -> 怕 | 0.8633 | 0.4900 | 0.3733 | 0.2073 | -0.0046 | 0.0000 | 0.0000 |
| BA | BA_suo_adverbial_a | 300 |  -> 所 | 0.2733 | 0.2600 | 0.0133 | -0.1011 | -0.1187 | 0.0000 | 0.0000 |
| BA | BA_suo_adverbial_b | 300 |  -> 所 | 0.8633 | 0.4500 | 0.4133 | 0.1793 | -0.0423 | 0.0000 | 0.0000 |
| BA | BA_verb_le_a | 300 | 把 -> 把了 | 0.5900 | 0.7200 | -0.1300 | 0.0306 | 0.0643 | 0.0000 | 0.0000 |
| BA | BA_verb_le_b | 300 | 了。 -> 。 | 0.9833 | 0.9100 | 0.0733 | 0.8191 | 0.3947 | 0.0000 | 0.0000 |
| passive | BEI_construction_a | 300 | INSERT_BAD:被 + phrase:Person \| DELETE_BAD:被 + phrase:Person | 0.9067 | 0.7333 | 0.1733 | 0.3375 | 0.2973 | 0.0000 | 0.0000 |
| passive | BEI_construction_b | 300 | DELETE_BAD:phrase:Person \| INSERT_BAD:phrase:Person | 0.8700 | 0.7900 | 0.0800 | 0.3435 | 0.4208 | 0.0000 | 0.0000 |
| passive | BEI_deletion | 300 | DELETE_BAD:pos:NN subcat:vehicle \| pos:NN subcat:container + 上被 \| INSERT_BAD:pos:NN subcat:vehicle \| pos:NN subcat:container + 上 | 0.5467 | 0.7267 | -0.1800 | 0.0712 | 0.1735 | 0.0000 | 0.0000 |
| passive | BEI_preposition | 300 | DELETE_BAD:pos:PN subcat:person \| INSERT_BAD:pos:PN subcat:person | 0.9033 | 0.8400 | 0.0633 | 0.5621 | 0.6028 | 0.0000 | 0.0000 |
| nominal_expression | PN_numP_a | 300 | pos:PN subcat:person number:singular -> pos:NN subcat:person | 0.5367 | 0.5967 | -0.0600 | 0.0330 | 0.0449 | 0.0000 | 0.0000 |
| nominal_expression | PN_numP_b | 300 | DELETE_BAD:pos:PN subcat:person number:plural \| INSERT_BAD:pos:PN subcat:person number:plural | 0.6333 | 0.4933 | 0.1400 | -0.0689 | 0.0587 | 0.0000 | 0.0000 |
| verb_phrase | adjective_transitive_dui | 300 | 对 -> 有点 比较 很 非常 + pos:VA subcat:emotion +  \| DELETE_BAD:有点 比较 很 非常 + pos:VA subcat:emotion | 0.9100 | 0.8300 | 0.0800 | 0.5446 | 0.4687 | 0.0000 | 0.0000 |
| argument_structure | agent_animacy_adv | 300 | 有点 可能 -> pos:AD animate:1 | 0.9600 | 0.7533 | 0.2067 | 0.5311 | 0.2558 | 0.0000 | 0.0000 |

## Top 20 Chinese-Advantage Official Paradigms

| phenomenon | uid | n | rule_diff | ch_acc | di_acc | gap | ch_margin | di_margin | collapse | tie |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fci_licensing | fci_renhe_ruguo | 300 |  -> 都 | 0.9967 | 0.0033 | 0.9933 | 0.3399 | -0.2616 | 0.0000 | 0.0000 |
| question | question_particle_nandao | 300 | 难道 -> 到底 | 0.9900 | 0.0033 | 0.9867 | 0.2943 | -0.4593 | 0.0000 | 0.0000 |
| npi_licensing | renhe_no_superordinate_negation | 300 | INSERT_BAD:你 + 有 \| DELETE_BAD:你 + 的 | 0.9833 | 0.0467 | 0.9367 | 0.3357 | -0.2643 | 0.0000 | 0.0000 |
| question | question_particle_daodi_choice_intran | 300 | 呢 -> 吗 | 0.9567 | 0.1300 | 0.8267 | 0.1168 | -0.0792 | 0.0000 | 0.0000 |
| fci_licensing | fci_renhe_subj | 300 | 任何人都 -> 有些人都 | 0.7867 | 0.0200 | 0.7667 | 0.1661 | -0.5019 | 0.0000 | 0.0000 |
| question | question_particle_daodi_choice_tran | 300 | 呢 -> 吗 | 0.8633 | 0.2167 | 0.6467 | 0.0601 | -0.0484 | 0.0000 | 0.0000 |
| npi_licensing | npi_renhe_neg_scope_locP | 300 | INSERT_BAD:phrase:PPRH \| DELETE_BAD:phrase:PPRH | 0.8933 | 0.2567 | 0.6367 | 0.2525 | -0.2263 | 0.0000 | 0.0000 |
| anaphor | principle_A_domain | 300 | pos:PN number:singular matchedPosition:2 matchedProperties:gender -> pos:PN number:singular mismatchedPosition:2 mismatchedProperties:gender | 0.5200 | 0.0000 | 0.5200 | 0.0388 | 0.0000 | 1.0000 | 1.0000 |
| anaphor | principle_A_c_command | 300 | pos:PN number:singular mPos:2 mPro:gender -> pos:PN number:singular mmPos:2 mmPro:gender | 0.5133 | 0.0000 | 0.5133 | 0.0361 | 0.0000 | 1.0000 | 1.0000 |
| nominal_expression | you_yige | 300 |  -> 有 | 0.8967 | 0.3933 | 0.5033 | 0.2322 | -0.1978 | 0.0000 | 0.0000 |
| BA | BA_suo_adverbial_b | 300 |  -> 所 | 0.8633 | 0.4500 | 0.4133 | 0.1793 | -0.0423 | 0.0000 | 0.0000 |
| BA | BA_no_stative_verb | 300 | 恨 -> 怕 | 0.8633 | 0.4900 | 0.3733 | 0.2073 | -0.0046 | 0.0000 | 0.0000 |
| verb_phrase | verb_phrase_left_adverbial | 300 | INSERT_BAD:被 \| DELETE_BAD:被 | 0.8667 | 0.5100 | 0.3567 | 0.2222 | -0.0065 | 0.0000 | 0.0000 |
| npi_licensing | npi_renhe_wh_question_subj | 300 | pos:CD \| pos:DT subcat:woCD + pos:M matchedPosition:2 matchedProperties:classifier -> 任何 +  | 0.7000 | 0.4033 | 0.2967 | 0.1548 | -0.0729 | 0.0000 | 0.0000 |
| BA | BA_negation | 300 | DELETE_BAD:没有 \| INSERT_BAD:没有 | 0.9300 | 0.6733 | 0.2567 | 0.4575 | 0.0966 | 0.0000 | 0.0000 |
| anaphor | principle_A_c_command_number | 300 | 们自己 -> 自己 | 0.5967 | 0.3433 | 0.2533 | 0.1885 | -0.0577 | 0.0000 | 0.0000 |
| control_raising | modal_raising_topicalization | 300 | INSERT_BAD:phrase:Person + 应该 + 能 可以 会 \| DELETE_BAD:phrase:Person + 应该 + 能 可以 会 | 0.7567 | 0.5067 | 0.2500 | 0.2553 | 0.0323 | 0.0000 | 0.0000 |
| verb_phrase | left_adverbial_d | 300 | DELETE_BAD:pos:AD matchedPosition:2 matchedProperties:verb \| INSERT_BAD:pos:AD matchedPosition:2 matchedProperties:verb | 0.7933 | 0.5533 | 0.2400 | 0.2470 | 0.0993 | 0.0000 | 0.0000 |
| passive | passive_no_adj | 300 | pos:VV subcat:person subcat2:action -> pos:VA | 0.8367 | 0.6000 | 0.2367 | 0.5464 | 0.1540 | 0.0000 | 0.0000 |
| control_raising | modal_raising_hui | 300 | INSERT_BAD:马上 可能 就 \|  + 会 要 \| DELETE_BAD:马上 可能 就 \|  + 会 要 | 0.9367 | 0.7033 | 0.2333 | 0.5261 | 0.3004 | 0.0000 | 0.0000 |

## Top 20 Diacritic-Advantage Official Paradigms

| phenomenon | uid | n | rule_diff | ch_acc | di_acc | gap | ch_margin | di_margin | collapse | tie |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| question | question_daodi_nandao_A_not_A_intran | 300 | 到底 -> 难道 | 0.0300 | 0.9533 | -0.9233 | -0.5260 | 0.8167 | 0.0000 | 0.0000 |
| question | question_daodi_nandao_2 | 300 | 到底 -> 难道 | 0.0300 | 0.9433 | -0.9133 | -0.3867 | 0.8371 | 0.0000 | 0.0000 |
| question | question_daodi_nandao_A_not_A_tran | 300 | 到底 -> 难道 | 0.0733 | 0.9133 | -0.8400 | -0.3902 | 0.6829 | 0.0000 | 0.0000 |
| question | question_nandao_scope_1 | 300 | DELETE_BAD:难道 \| INSERT_BAD:难道 | 0.0900 | 0.5500 | -0.4600 | -0.2749 | -0.0229 | 0.0000 | 0.0000 |
| anaphor | principle_A_domain_number | 300 | 自己 -> 们自己 | 0.4800 | 0.8567 | -0.3767 | -0.1275 | 0.0936 | 0.0000 | 0.0000 |
| question | question_V_not_VP_1 | 300 | INSERT_BAD:pos:NN subcat:location + pos:VV subcat2:leave \| DELETE_BAD:pos:NN subcat:location + pos:VV subcat2:leave | 0.5400 | 0.9100 | -0.3700 | 0.0360 | 0.3471 | 0.0000 | 0.0000 |
| quantifiers | superlative_quantifiers_2 | 300 | 有的 -> 没有 | 0.5133 | 0.8800 | -0.3667 | 0.0525 | 0.3463 | 0.0000 | 0.0000 |
| classifier | classifier_noun_subj | 300 | INSERT_BAD:pos:M mPos:3 mPro:classifier + pos:NN subcat:food \| pos:NN subcat:obj \| pos:NN subcat:beverage + 是 \| pos:M mPos:3 mPro:classifier + pos:NN subcat:food \| pos:NN subcat:obj \| pos:NN subcat:beverage +  + 。 -> 的。 | 0.0067 | 0.3133 | -0.3067 | -1.2137 | -0.3003 | 0.0000 | 0.0000 |
| question | question_nandao_negation | 300 | INSERT_BAD:不 \| 不 + 愿意 \| 想 \| 希望  -> 愿意 \| 想 \| 希望 | 0.5633 | 0.8400 | -0.2767 | 0.0435 | 0.4678 | 0.0000 | 0.0000 |
| BA | BA_BEI_subj_drop | 300 | INSERT_BAD:pos:NN + 被 \| DELETE_BAD:把 + pos:NN | 0.6133 | 0.8400 | -0.2267 | 0.0850 | 0.3680 | 0.0000 | 0.0000 |
| passive | passive_body_part | 300 | pos:NN subcat:body -> pos:NN subcat:obj | 0.5067 | 0.7000 | -0.1933 | 0.0181 | 0.1796 | 0.0000 | 0.0000 |
| passive | BEI_deletion | 300 | DELETE_BAD:pos:NN subcat:vehicle \| pos:NN subcat:container + 上被 \| INSERT_BAD:pos:NN subcat:vehicle \| pos:NN subcat:container + 上 | 0.5467 | 0.7267 | -0.1800 | 0.0712 | 0.1735 | 0.0000 | 0.0000 |
| topicalization | topicalization_SOV | 300 | DELETE_BAD:在 + pos:VV transitivity:tran \| INSERT_BAD:在 + pos:VV transitivity:tran | 0.4833 | 0.6333 | -0.1500 | -0.0136 | 0.1731 | 0.0000 | 0.0000 |
| BA | BA_verb_le_a | 300 | 把 -> 把了 | 0.5900 | 0.7200 | -0.1300 | 0.0306 | 0.0643 | 0.0000 | 0.0000 |
| question | question_nandao_raising_3 | 300 | INSERT_BAD:是 \| DELETE_BAD:是 | 0.6333 | 0.7567 | -0.1233 | 0.0738 | 0.2341 | 0.0000 | 0.0000 |
| fci_licensing | fci_renhe_dou | 300 | 都 ->  | 0.7667 | 0.8867 | -0.1200 | 0.0861 | 0.1415 | 0.0000 | 0.0000 |
| npi_licensing | npi_renhe_wh_question_obj | 300 | phrase:Possessive -> 任何 | 0.5500 | 0.6700 | -0.1200 | 0.0905 | 0.1486 | 0.0000 | 0.0000 |
| quantifiers | superlative_quantifiers_1 | 300 | 超过 -> 至少 | 0.0167 | 0.1267 | -0.1100 | -0.2328 | -0.3837 | 0.0000 | 0.0000 |
| question | question_A_not_A_indirect | 300 |  + 。 -> 呢 + ？ | 0.0000 | 0.1067 | -0.1067 | -0.4042 | -0.1236 | 0.0000 | 0.0000 |
| relativization | relative_operator_who | 300 | pos:PN subcat:person number:singular -> 谁 \| 。 -> ？ | 0.7700 | 0.8633 | -0.0933 | 0.1431 | 0.2703 | 0.0000 | 0.0000 |

## Top 20 Collapse-Affected Official Paradigms

| phenomenon | uid | n | rule_diff | ch_acc | di_acc | gap | ch_margin | di_margin | collapse | tie |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anaphor | principle_A_c_command | 300 | pos:PN number:singular mPos:2 mPro:gender -> pos:PN number:singular mmPos:2 mmPro:gender | 0.5133 | 0.0000 | 0.5133 | 0.0361 | 0.0000 | 1.0000 | 1.0000 |
| anaphor | principle_A_domain | 300 | pos:PN number:singular matchedPosition:2 matchedProperties:gender -> pos:PN number:singular mismatchedPosition:2 mismatchedProperties:gender | 0.5200 | 0.0000 | 0.5200 | 0.0388 | 0.0000 | 1.0000 | 1.0000 |

## Top 30 Paradigm + Observed Surface Diff Rows By n_items

| phenomenon | uid | surface_diff | n | ch_acc | di_acc | gap | ch_margin | di_margin | collapse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BA | BA_meiba | bad inserts 把没 | 300 | 0.5467 | 0.4500 | 0.0967 | 0.0155 | -0.0164 | 0.0000 |
| BA | BA_no_stative_verb | 恨 → 怕 | 300 | 0.8633 | 0.4900 | 0.3733 | 0.2073 | -0.0046 | 0.0000 |
| BA | BA_suo_adverbial_a | bad inserts 所 | 300 | 0.2733 | 0.2600 | 0.0133 | -0.1011 | -0.1187 | 0.0000 |
| BA | BA_suo_adverbial_b | bad inserts 所 | 300 | 0.8633 | 0.4500 | 0.4133 | 0.1793 | -0.0423 | 0.0000 |
| BA | BA_verb_le_a | bad inserts 了 | 300 | 0.5900 | 0.7200 | -0.1300 | 0.0306 | 0.0643 | 0.0000 |
| BA | BA_verb_le_b | bad deletes 了 | 300 | 0.9833 | 0.9100 | 0.0733 | 0.8191 | 0.3947 | 0.0000 |
| anaphor | anaphor_number_agreement | bad deletes 们 | 300 | 0.5267 | 0.3533 | 0.1733 | 0.2451 | -0.0590 | 0.0000 |
| fci_licensing | fci_renhe_dou | bad deletes 都 | 300 | 0.7667 | 0.8867 | -0.1200 | 0.0861 | 0.1415 | 0.0000 |
| fci_licensing | fci_renhe_prepP | 任何 → 没有 | 300 | 1.0000 | 1.0000 | 0.0000 | 0.4290 | 0.5873 | 0.0000 |
| fci_licensing | fci_renhe_ruguo | bad inserts 都 | 300 | 0.9967 | 0.0033 | 0.9933 | 0.3399 | -0.2616 | 0.0000 |
| fci_licensing | fci_renhe_subj | 任何 → 有些 | 300 | 0.7867 | 0.0200 | 0.7667 | 0.1661 | -0.5019 | 0.0000 |
| fci_licensing | fci_renhe_suoyou | 所有 → 任何 | 300 | 0.8333 | 0.6867 | 0.1467 | 0.0911 | 0.0659 | 0.0000 |
| verb_phrase | left_dou | multiple edits: 都->\|->都 | 300 | 0.9733 | 0.8033 | 0.1700 | 0.3451 | 0.1970 | 0.0000 |
| nominal_expression | nominal_definite_men | bad inserts 们 | 300 | 0.1100 | 0.0933 | 0.0167 | -0.3289 | -0.3627 | 0.0000 |
| nominal_expression | noun_adjective_shi | bad deletes 是 | 300 | 0.8433 | 0.8367 | 0.0067 | 0.7052 | 0.7611 | 0.0000 |
| npi_licensing | npi_renhe_A_not_A_question | multiple edits: 有没有->\|->了\|？->。 | 300 | 0.8667 | 0.6433 | 0.2233 | 0.3159 | 0.1934 | 0.0000 |
| npi_licensing | npi_renhe_neg_scope_locP | multiple edits: 没有->\|->没有 | 300 | 0.8933 | 0.2567 | 0.6367 | 0.2525 | -0.2263 | 0.0000 |
| npi_licensing | npi_renhe_neg_scope_subj | multiple edits: 没有->\|->没有 | 300 | 0.3533 | 0.3967 | -0.0433 | -0.1236 | -0.1146 | 0.0000 |
| passive | passive_agent_deletion_long_right_a | bad deletes 我 | 300 | 0.9200 | 0.8100 | 0.1100 | 0.1091 | 0.1007 | 0.0000 |
| nominal_expression | plural_cardinal_men_a | bad inserts 们 | 300 | 0.0867 | 0.0900 | -0.0033 | -0.2510 | -0.3064 | 0.0000 |
| nominal_expression | plural_cardinal_men_b | bad inserts 们 | 300 | 0.0400 | 0.0867 | -0.0467 | -0.2495 | -0.2677 | 0.0000 |
| verb_phrase | preposition_deletion | bad deletes 在 | 300 | 0.9367 | 0.9000 | 0.0367 | 0.4904 | 0.4641 | 0.0000 |
| anaphor | principle_A_c_command_number | bad deletes 们 | 300 | 0.5967 | 0.3433 | 0.2533 | 0.1885 | -0.0577 | 0.0000 |
| anaphor | principle_A_domain_number | bad inserts 们 | 300 | 0.4800 | 0.8567 | -0.3767 | -0.1275 | 0.0936 | 0.0000 |
| question | question_A_not_A_indirect | 。 → 呢？ | 300 | 0.0000 | 0.1067 | -0.1067 | -0.4042 | -0.1236 | 0.0000 |
| question | question_V_not_VP_2 | multiple edits: 把不->\|->不把 | 300 | 0.8233 | 0.6067 | 0.2167 | 0.1321 | 0.0781 | 0.0000 |
| question | question_daodi_nandao_1 | 难道 → 到底 | 300 | 1.0000 | 0.7733 | 0.2267 | 0.8618 | 0.2832 | 0.0000 |
| question | question_daodi_nandao_2 | 到底 → 难道 | 300 | 0.0300 | 0.9433 | -0.9133 | -0.3867 | 0.8371 | 0.0000 |
| question | question_daodi_nandao_A_not_A_intran | 到底 → 难道 | 300 | 0.0300 | 0.9533 | -0.9233 | -0.5260 | 0.8167 | 0.0000 |
| question | question_daodi_nandao_A_not_A_tran | 到底 → 难道 | 300 | 0.0733 | 0.9133 | -0.8400 | -0.3902 | 0.6829 | 0.0000 |

## Anaphor Collapse Paradigms

| phenomenon | uid | n | rule_diff | ch_acc | di_acc | gap | ch_margin | di_margin | collapse | tie |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anaphor | principle_A_c_command | 300 | pos:PN number:singular mPos:2 mPro:gender -> pos:PN number:singular mmPos:2 mmPro:gender | 0.5133 | 0.0000 | 0.5133 | 0.0361 | 0.0000 | 1.0000 | 1.0000 |
| anaphor | principle_A_domain | 300 | pos:PN number:singular matchedPosition:2 matchedProperties:gender -> pos:PN number:singular mismatchedPosition:2 mismatchedProperties:gender | 0.5200 | 0.0000 | 0.5200 | 0.0388 | 0.0000 | 1.0000 | 1.0000 |

## Question Particle Paradigms

| phenomenon | uid | n | rule_diff | ch_acc | di_acc | gap | ch_margin | di_margin | collapse | tie |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| question | question_particle_nandao | 300 | 难道 -> 到底 | 0.9900 | 0.0033 | 0.9867 | 0.2943 | -0.4593 | 0.0000 | 0.0000 |
| question | question_daodi_nandao_A_not_A_intran | 300 | 到底 -> 难道 | 0.0300 | 0.9533 | -0.9233 | -0.5260 | 0.8167 | 0.0000 | 0.0000 |
| question | question_daodi_nandao_2 | 300 | 到底 -> 难道 | 0.0300 | 0.9433 | -0.9133 | -0.3867 | 0.8371 | 0.0000 | 0.0000 |
| question | question_daodi_nandao_A_not_A_tran | 300 | 到底 -> 难道 | 0.0733 | 0.9133 | -0.8400 | -0.3902 | 0.6829 | 0.0000 | 0.0000 |
| question | question_particle_daodi_choice_intran | 300 | 呢 -> 吗 | 0.9567 | 0.1300 | 0.8267 | 0.1168 | -0.0792 | 0.0000 | 0.0000 |
| question | question_particle_daodi_choice_tran | 300 | 呢 -> 吗 | 0.8633 | 0.2167 | 0.6467 | 0.0601 | -0.0484 | 0.0000 | 0.0000 |
| question | question_nandao_scope_1 | 300 | DELETE_BAD:难道 \| INSERT_BAD:难道 | 0.0900 | 0.5500 | -0.4600 | -0.2749 | -0.0229 | 0.0000 | 0.0000 |
| question | question_V_not_VP_1 | 300 | INSERT_BAD:pos:NN subcat:location + pos:VV subcat2:leave \| DELETE_BAD:pos:NN subcat:location + pos:VV subcat2:leave | 0.5400 | 0.9100 | -0.3700 | 0.0360 | 0.3471 | 0.0000 | 0.0000 |
| question | question_nandao_negation | 300 | INSERT_BAD:不 \| 不 + 愿意 \| 想 \| 希望  -> 愿意 \| 想 \| 希望 | 0.5633 | 0.8400 | -0.2767 | 0.0435 | 0.4678 | 0.0000 | 0.0000 |
| question | question_daodi_nandao_1 | 300 | 难道 -> 到底 | 1.0000 | 0.7733 | 0.2267 | 0.8618 | 0.2832 | 0.0000 | 0.0000 |
| question | question_V_not_VP_2 | 300 | DELETE_BAD:把 + 不 \| INSERT_BAD:不 + 把 | 0.8233 | 0.6067 | 0.2167 | 0.1321 | 0.0781 | 0.0000 | 0.0000 |
| question | question_daodi_negation | 300 | INSERT_BAD:不 \| DELETE_BAD:不 | 0.9967 | 0.8467 | 0.1500 | 1.1162 | 0.3187 | 0.0000 | 0.0000 |
| question | question_nandao_raising_2 | 300 | DELETE_BAD:难道 \| INSERT_BAD:难道 | 1.0000 | 0.8500 | 0.1500 | 0.6088 | 0.5360 | 0.0000 | 0.0000 |
| question | question_nandao_raising_3 | 300 | INSERT_BAD:是 \| DELETE_BAD:是 | 0.6333 | 0.7567 | -0.1233 | 0.0738 | 0.2341 | 0.0000 | 0.0000 |
| question | question_A_not_A_indirect | 300 |  + 。 -> 呢 + ？ | 0.0000 | 0.1067 | -0.1067 | -0.4042 | -0.1236 | 0.0000 | 0.0000 |
| question | question_nandao_scope_2 | 300 | INSERT_BAD:认为 \| 确定 \| 相信 \| DELETE_BAD:认为 \| 确定 \| 相信 | 0.2900 | 0.2200 | 0.0700 | -0.1226 | -0.1684 | 0.0000 | 0.0000 |
| question | question_A_not_A | 300 |  -> phrase:Person | 0.1733 | 0.2233 | -0.0500 | -0.4240 | -0.5619 | 0.0000 | 0.0000 |
| question | question_nandao_raising_1_b | 300 | DELETE_BAD:难道 \| INSERT_BAD:难道 | 0.9933 | 0.9433 | 0.0500 | 0.7323 | 0.4571 | 0.0000 | 0.0000 |
| question | question_A_not_A_daodi_a | 300 | INSERT_BAD:不 \| DELETE_BAD:不 | 0.7600 | 0.7833 | -0.0233 | 0.2531 | 0.4314 | 0.0000 | 0.0000 |
| question | question_nandao_raising_1_a | 300 | INSERT_BAD:是 \| DELETE_BAD:是 | 0.9767 | 0.9733 | 0.0033 | 0.6686 | 0.6385 | 0.0000 | 0.0000 |
| question | question_A_not_A_daodi_b | 300 | INSERT_BAD:不 \| DELETE_BAD:不 | 0.7500 | 0.7500 | 0.0000 | 0.2479 | 0.3474 | 0.0000 | 0.0000 |

## BA/BEI Paradigms

| phenomenon | uid | n | rule_diff | ch_acc | di_acc | gap | ch_margin | di_margin | collapse | tie |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BA | BA_suo_adverbial_b | 300 |  -> 所 | 0.8633 | 0.4500 | 0.4133 | 0.1793 | -0.0423 | 0.0000 | 0.0000 |
| BA | BA_no_stative_verb | 300 | 恨 -> 怕 | 0.8633 | 0.4900 | 0.3733 | 0.2073 | -0.0046 | 0.0000 | 0.0000 |
| verb_phrase | verb_phrase_left_adverbial | 300 | INSERT_BAD:被 \| DELETE_BAD:被 | 0.8667 | 0.5100 | 0.3567 | 0.2222 | -0.0065 | 0.0000 | 0.0000 |
| BA | BA_negation | 300 | DELETE_BAD:没有 \| INSERT_BAD:没有 | 0.9300 | 0.6733 | 0.2567 | 0.4575 | 0.0966 | 0.0000 | 0.0000 |
| passive | passive_no_adj | 300 | pos:VV subcat:person subcat2:action -> pos:VA | 0.8367 | 0.6000 | 0.2367 | 0.5464 | 0.1540 | 0.0000 | 0.0000 |
| BA | BA_BEI_subj_drop | 300 | INSERT_BAD:pos:NN + 被 \| DELETE_BAD:把 + pos:NN | 0.6133 | 0.8400 | -0.2267 | 0.0850 | 0.3680 | 0.0000 | 0.0000 |
| BA | BA_inversion | 300 | DELETE_BAD:phrase:Person \| phrase:ComplexPerson \| INSERT_BAD:phrase:Person \| phrase:ComplexPerson | 0.9467 | 0.7233 | 0.2233 | 0.5291 | 0.2160 | 0.0000 | 0.0000 |
| question | question_V_not_VP_2 | 300 | DELETE_BAD:把 + 不 \| INSERT_BAD:不 + 把 | 0.8233 | 0.6067 | 0.2167 | 0.1321 | 0.0781 | 0.0000 | 0.0000 |
| BA | BA_no_progressive | 300 | 正在 + pos:VV transitivity:tran mPos:5 mPro:tran_verb -> 正把 \| INSERT_BAD:pos:VV transitivity:tran mPos:5 mPro:tran_verb | 0.5133 | 0.3033 | 0.2100 | 0.0478 | -0.2545 | 0.0000 | 0.0000 |
| passive | passive_body_part | 300 | pos:NN subcat:body -> pos:NN subcat:obj | 0.5067 | 0.7000 | -0.1933 | 0.0181 | 0.1796 | 0.0000 | 0.0000 |
| passive | passive_suo | 300 | pos:PN subcat:person \| pos:NR subcat:person ->  | 0.8933 | 0.7067 | 0.1867 | 0.4152 | 0.4017 | 0.0000 | 0.0000 |
| BA | causative_shi_ba | 300 | 使 让 -> 把 | 0.7267 | 0.5433 | 0.1833 | 0.1926 | 0.0011 | 0.0000 | 0.0000 |
| passive | BEI_deletion | 300 | DELETE_BAD:pos:NN subcat:vehicle \| pos:NN subcat:container + 上被 \| INSERT_BAD:pos:NN subcat:vehicle \| pos:NN subcat:container + 上 | 0.5467 | 0.7267 | -0.1800 | 0.0712 | 0.1735 | 0.0000 | 0.0000 |
| passive | BEI_construction_a | 300 | INSERT_BAD:被 + phrase:Person \| DELETE_BAD:被 + phrase:Person | 0.9067 | 0.7333 | 0.1733 | 0.3375 | 0.2973 | 0.0000 | 0.0000 |
| verb_phrase | verb_phrase_left_negation | 300 | INSERT_BAD:把 +  + pos:PN subcat:person mismatchedPosition:0 mismatchedProperties:expression \| pos:NR subcat:person \| DELETE_BAD:把 + pos:PN subcat:person mismatchedPosition:0 mismatchedProperties:expression \| pos:NR subcat:person +  | 0.9233 | 0.7700 | 0.1533 | 0.3568 | 0.1836 | 0.0000 | 0.0000 |
| passive | passive_agent_deletion_short | 300 | pos:PN subcat:person \| pos:NR subcat:person mmPos:0 mmPro:expression ->  | 0.6800 | 0.5433 | 0.1367 | 0.1830 | 0.0428 | 0.0000 | 0.0000 |
| BA | BA_verb_le_a | 300 | 把 -> 把了 | 0.5900 | 0.7200 | -0.1300 | 0.0306 | 0.0643 | 0.0000 | 0.0000 |
| BA | BA_deletion | 300 | DELETE_BAD:把 + pos:NN subcat:vehicle \| pos:NN subcat:container \| INSERT_BAD:pos:NN subcat:vehicle \| pos:NN subcat:container +  | 0.9600 | 0.8333 | 0.1267 | 0.4874 | 0.3329 | 0.0000 | 0.0000 |
| passive | passive_agent_deletion_long_right_a | 300 | 被我叫 -> 被叫 | 0.9200 | 0.8100 | 0.1100 | 0.1091 | 0.1007 | 0.0000 | 0.0000 |
| BA | BA_meiba | 300 | 把 -> 把没把 | 0.5467 | 0.4500 | 0.0967 | 0.0155 | -0.0164 | 0.0000 | 0.0000 |
| passive | BEI_construction_b | 300 | DELETE_BAD:phrase:Person \| INSERT_BAD:phrase:Person | 0.8700 | 0.7900 | 0.0800 | 0.3435 | 0.4208 | 0.0000 | 0.0000 |
| BA | BA_verb_le_b | 300 | 了。 -> 。 | 0.9833 | 0.9100 | 0.0733 | 0.8191 | 0.3947 | 0.0000 | 0.0000 |
| passive | BEI_preposition | 300 | DELETE_BAD:pos:PN subcat:person \| INSERT_BAD:pos:PN subcat:person | 0.9033 | 0.8400 | 0.0633 | 0.5621 | 0.6028 | 0.0000 | 0.0000 |
| passive | passive_agent_deletion_long_right_b | 300 | pos:PN subcat:person \| pos:NR subcat:person mmPos:0 mmPro:expression ->  | 0.7467 | 0.6967 | 0.0500 | 0.2236 | 0.1937 | 0.0000 | 0.0000 |
| passive | passive_intransitive | 300 | pos:VV transitivity:tran subcat2:stative \| pos:VV transitivity:tran subcat2:action -> pos:VV transitivity:intran animate:1 | 0.6467 | 0.6000 | 0.0467 | 0.2886 | 0.1443 | 0.0000 | 0.0000 |
| BA | BA_suo_adverbial_a | 300 |  -> 所 | 0.2733 | 0.2600 | 0.0133 | -0.1011 | -0.1187 | 0.0000 | 0.0000 |
| passive | passive_agent_deletion_long_left | 300 | pos:NR subcat:anyone ->  | 0.9967 | 0.9933 | 0.0033 | 0.3965 | 0.4827 | 0.0000 | 0.0000 |
| BA | BA_duplicate_argument | 300 |  -> pos:PN mPos:0 mPro:gender/number | 1.0000 | 1.0000 | 0.0000 | 0.5011 | 0.4069 | 0.0000 | 0.0000 |

## 的/地/得 Paradigms

| phenomenon | uid | n | rule_diff | ch_acc | di_acc | gap | ch_margin | di_margin | collapse | tie |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| npi_licensing | renhe_no_superordinate_negation | 300 | INSERT_BAD:你 + 有 \| DELETE_BAD:你 + 的 | 0.9833 | 0.0467 | 0.9367 | 0.3357 | -0.2643 | 0.0000 | 0.0000 |
| quantifiers | superlative_quantifiers_2 | 300 | 有的 -> 没有 | 0.5133 | 0.8800 | -0.3667 | 0.0525 | 0.3463 | 0.0000 | 0.0000 |
| classifier | classifier_noun_subj | 300 | INSERT_BAD:pos:M mPos:3 mPro:classifier + pos:NN subcat:food \| pos:NN subcat:obj \| pos:NN subcat:beverage + 是 \| pos:M mPos:3 mPro:classifier + pos:NN subcat:food \| pos:NN subcat:obj \| pos:NN subcat:beverage +  + 。 -> 的。 | 0.0067 | 0.3133 | -0.3067 | -1.2137 | -0.3003 | 0.0000 | 0.0000 |
| npi_licensing | renhe_non_factive_verb | 300 | 觉得 相信 希望 -> 知道 主张 | 0.9733 | 0.8300 | 0.1433 | 0.3370 | 0.2159 | 0.0000 | 0.0000 |
| argument_structure | intransitive_double_obj | 300 | INSERT_BAD:拿 + pos:NN animate:1 + 的 \| DELETE_BAD:pos:CD + pos:M mPro:classifier mPos:7 + pos:NN animate:1 | 0.7867 | 0.7067 | 0.0800 | 0.4889 | 0.3990 | 0.0000 | 0.0000 |
| verb_phrase | ya_insertion | 300 | 说 认为 觉得 以为 -> 告诉 告知 劝 | 0.5067 | 0.4367 | 0.0700 | 0.0031 | -0.0291 | 0.0000 | 0.0000 |
