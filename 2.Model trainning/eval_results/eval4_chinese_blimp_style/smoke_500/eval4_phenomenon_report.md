# Eval 4 ZhoBLiMP Phenomenon Report

Tie definition: `abs(mean_logprob_margin) <= 1e-5`, or identical model-input good/bad strings. Tie items are not counted as correct.

`collapsed` means `good_sentence_diacritic == bad_sentence_diacritic`. `noncollapsed_*` accuracy excludes collapsed items from that phenomenon.

## All Phenomena Summary

| phenomenon | n_items | Chinese acc | Diacritic acc | gap | collapsed count | collapsed rate | tie count | tie rate | noncollapsed n | Chinese noncollapsed acc | Diacritic noncollapsed acc | noncollapsed gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| npi_licensing | 33 | 0.8182 | 0.3939 | +0.4242 | 0 | 0.0000 | 0 | 0.0000 | 33 | 0.8182 | 0.3939 | +0.4242 |
| fci_licensing | 33 | 0.9394 | 0.5455 | +0.3939 | 0 | 0.0000 | 0 | 0.0000 | 33 | 0.9394 | 0.5455 | +0.3939 |
| control_raising | 34 | 0.9412 | 0.5882 | +0.3529 | 0 | 0.0000 | 0 | 0.0000 | 34 | 0.9412 | 0.5882 | +0.3529 |
| anaphor | 34 | 0.5294 | 0.2647 | +0.2647 | 13 | 0.3824 | 13 | 0.3824 | 21 | 0.5714 | 0.4286 | +0.1429 |
| verb_phrase | 33 | 0.9091 | 0.7273 | +0.1818 | 0 | 0.0000 | 0 | 0.0000 | 33 | 0.9091 | 0.7273 | +0.1818 |
| BA | 34 | 0.8824 | 0.7647 | +0.1176 | 0 | 0.0000 | 0 | 0.0000 | 34 | 0.8824 | 0.7647 | +0.1176 |
| ellipsis | 33 | 0.5152 | 0.4242 | +0.0909 | 0 | 0.0000 | 0 | 0.0000 | 33 | 0.5152 | 0.4242 | +0.0909 |
| passive | 33 | 0.9091 | 0.8182 | +0.0909 | 0 | 0.0000 | 0 | 0.0000 | 33 | 0.9091 | 0.8182 | +0.0909 |
| nominal_expression | 33 | 0.5152 | 0.4545 | +0.0606 | 0 | 0.0000 | 0 | 0.0000 | 33 | 0.5152 | 0.4545 | +0.0606 |
| topicalization | 33 | 0.7576 | 0.6970 | +0.0606 | 0 | 0.0000 | 0 | 0.0000 | 33 | 0.7576 | 0.6970 | +0.0606 |
| argument_structure | 34 | 0.7059 | 0.6471 | +0.0588 | 0 | 0.0000 | 0 | 0.0000 | 34 | 0.7059 | 0.6471 | +0.0588 |
| question | 33 | 0.6364 | 0.6061 | +0.0303 | 0 | 0.0000 | 0 | 0.0000 | 33 | 0.6364 | 0.6061 | +0.0303 |
| relativization | 33 | 0.5455 | 0.5758 | -0.0303 | 0 | 0.0000 | 0 | 0.0000 | 33 | 0.5455 | 0.5758 | -0.0303 |
| classifier | 34 | 0.2059 | 0.3824 | -0.1765 | 0 | 0.0000 | 0 | 0.0000 | 34 | 0.2059 | 0.3824 | -0.1765 |
| quantifiers | 33 | 0.2424 | 0.4242 | -0.1818 | 0 | 0.0000 | 0 | 0.0000 | 33 | 0.2424 | 0.4242 | -0.1818 |

## Per-Phenomenon Tables

### BA

| metric | value |
|---|---:|
| n_items | 34 |
| Chinese accuracy | 0.882353 |
| Diacritic accuracy | 0.764706 |
| gap | +0.117647 |
| collapsed count | 0 |
| collapsed rate | 0.000000 |
| tie count | 0 |
| tie rate | 0.000000 |
| noncollapsed n_items | 34 |
| Chinese noncollapsed accuracy | 0.882353 |
| Diacritic noncollapsed accuracy | 0.764706 |
| noncollapsed gap | +0.117647 |

### anaphor

| metric | value |
|---|---:|
| n_items | 34 |
| Chinese accuracy | 0.529412 |
| Diacritic accuracy | 0.264706 |
| gap | +0.264706 |
| collapsed count | 13 |
| collapsed rate | 0.382353 |
| tie count | 13 |
| tie rate | 0.382353 |
| noncollapsed n_items | 21 |
| Chinese noncollapsed accuracy | 0.571429 |
| Diacritic noncollapsed accuracy | 0.428571 |
| noncollapsed gap | +0.142857 |

### argument_structure

| metric | value |
|---|---:|
| n_items | 34 |
| Chinese accuracy | 0.705882 |
| Diacritic accuracy | 0.647059 |
| gap | +0.058824 |
| collapsed count | 0 |
| collapsed rate | 0.000000 |
| tie count | 0 |
| tie rate | 0.000000 |
| noncollapsed n_items | 34 |
| Chinese noncollapsed accuracy | 0.705882 |
| Diacritic noncollapsed accuracy | 0.647059 |
| noncollapsed gap | +0.058824 |

### classifier

| metric | value |
|---|---:|
| n_items | 34 |
| Chinese accuracy | 0.205882 |
| Diacritic accuracy | 0.382353 |
| gap | -0.176471 |
| collapsed count | 0 |
| collapsed rate | 0.000000 |
| tie count | 0 |
| tie rate | 0.000000 |
| noncollapsed n_items | 34 |
| Chinese noncollapsed accuracy | 0.205882 |
| Diacritic noncollapsed accuracy | 0.382353 |
| noncollapsed gap | -0.176471 |

### control_raising

| metric | value |
|---|---:|
| n_items | 34 |
| Chinese accuracy | 0.941176 |
| Diacritic accuracy | 0.588235 |
| gap | +0.352941 |
| collapsed count | 0 |
| collapsed rate | 0.000000 |
| tie count | 0 |
| tie rate | 0.000000 |
| noncollapsed n_items | 34 |
| Chinese noncollapsed accuracy | 0.941176 |
| Diacritic noncollapsed accuracy | 0.588235 |
| noncollapsed gap | +0.352941 |

### ellipsis

| metric | value |
|---|---:|
| n_items | 33 |
| Chinese accuracy | 0.515152 |
| Diacritic accuracy | 0.424242 |
| gap | +0.090909 |
| collapsed count | 0 |
| collapsed rate | 0.000000 |
| tie count | 0 |
| tie rate | 0.000000 |
| noncollapsed n_items | 33 |
| Chinese noncollapsed accuracy | 0.515152 |
| Diacritic noncollapsed accuracy | 0.424242 |
| noncollapsed gap | +0.090909 |

### fci_licensing

| metric | value |
|---|---:|
| n_items | 33 |
| Chinese accuracy | 0.939394 |
| Diacritic accuracy | 0.545455 |
| gap | +0.393939 |
| collapsed count | 0 |
| collapsed rate | 0.000000 |
| tie count | 0 |
| tie rate | 0.000000 |
| noncollapsed n_items | 33 |
| Chinese noncollapsed accuracy | 0.939394 |
| Diacritic noncollapsed accuracy | 0.545455 |
| noncollapsed gap | +0.393939 |

### nominal_expression

| metric | value |
|---|---:|
| n_items | 33 |
| Chinese accuracy | 0.515152 |
| Diacritic accuracy | 0.454545 |
| gap | +0.060606 |
| collapsed count | 0 |
| collapsed rate | 0.000000 |
| tie count | 0 |
| tie rate | 0.000000 |
| noncollapsed n_items | 33 |
| Chinese noncollapsed accuracy | 0.515152 |
| Diacritic noncollapsed accuracy | 0.454545 |
| noncollapsed gap | +0.060606 |

### npi_licensing

| metric | value |
|---|---:|
| n_items | 33 |
| Chinese accuracy | 0.818182 |
| Diacritic accuracy | 0.393939 |
| gap | +0.424242 |
| collapsed count | 0 |
| collapsed rate | 0.000000 |
| tie count | 0 |
| tie rate | 0.000000 |
| noncollapsed n_items | 33 |
| Chinese noncollapsed accuracy | 0.818182 |
| Diacritic noncollapsed accuracy | 0.393939 |
| noncollapsed gap | +0.424242 |

### passive

| metric | value |
|---|---:|
| n_items | 33 |
| Chinese accuracy | 0.909091 |
| Diacritic accuracy | 0.818182 |
| gap | +0.090909 |
| collapsed count | 0 |
| collapsed rate | 0.000000 |
| tie count | 0 |
| tie rate | 0.000000 |
| noncollapsed n_items | 33 |
| Chinese noncollapsed accuracy | 0.909091 |
| Diacritic noncollapsed accuracy | 0.818182 |
| noncollapsed gap | +0.090909 |

### quantifiers

| metric | value |
|---|---:|
| n_items | 33 |
| Chinese accuracy | 0.242424 |
| Diacritic accuracy | 0.424242 |
| gap | -0.181818 |
| collapsed count | 0 |
| collapsed rate | 0.000000 |
| tie count | 0 |
| tie rate | 0.000000 |
| noncollapsed n_items | 33 |
| Chinese noncollapsed accuracy | 0.242424 |
| Diacritic noncollapsed accuracy | 0.424242 |
| noncollapsed gap | -0.181818 |

### question

| metric | value |
|---|---:|
| n_items | 33 |
| Chinese accuracy | 0.636364 |
| Diacritic accuracy | 0.606061 |
| gap | +0.030303 |
| collapsed count | 0 |
| collapsed rate | 0.000000 |
| tie count | 0 |
| tie rate | 0.000000 |
| noncollapsed n_items | 33 |
| Chinese noncollapsed accuracy | 0.636364 |
| Diacritic noncollapsed accuracy | 0.606061 |
| noncollapsed gap | +0.030303 |

### relativization

| metric | value |
|---|---:|
| n_items | 33 |
| Chinese accuracy | 0.545455 |
| Diacritic accuracy | 0.575758 |
| gap | -0.030303 |
| collapsed count | 0 |
| collapsed rate | 0.000000 |
| tie count | 0 |
| tie rate | 0.000000 |
| noncollapsed n_items | 33 |
| Chinese noncollapsed accuracy | 0.545455 |
| Diacritic noncollapsed accuracy | 0.575758 |
| noncollapsed gap | -0.030303 |

### topicalization

| metric | value |
|---|---:|
| n_items | 33 |
| Chinese accuracy | 0.757576 |
| Diacritic accuracy | 0.696970 |
| gap | +0.060606 |
| collapsed count | 0 |
| collapsed rate | 0.000000 |
| tie count | 0 |
| tie rate | 0.000000 |
| noncollapsed n_items | 33 |
| Chinese noncollapsed accuracy | 0.757576 |
| Diacritic noncollapsed accuracy | 0.696970 |
| noncollapsed gap | +0.060606 |

### verb_phrase

| metric | value |
|---|---:|
| n_items | 33 |
| Chinese accuracy | 0.909091 |
| Diacritic accuracy | 0.727273 |
| gap | +0.181818 |
| collapsed count | 0 |
| collapsed rate | 0.000000 |
| tie count | 0 |
| tie rate | 0.000000 |
| noncollapsed n_items | 33 |
| Chinese noncollapsed accuracy | 0.909091 |
| Diacritic noncollapsed accuracy | 0.727273 |
| noncollapsed gap | +0.181818 |
