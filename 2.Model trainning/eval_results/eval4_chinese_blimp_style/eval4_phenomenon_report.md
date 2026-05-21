# Eval 4 ZhoBLiMP Phenomenon Report

Tie definition: `abs(mean_logprob_margin) <= 1e-5`, or identical model-input good/bad strings. Tie items are not counted as correct.

`collapsed` means `good_sentence_diacritic == bad_sentence_diacritic`. `noncollapsed_*` accuracy excludes collapsed items from that phenomenon.

## All Phenomena Summary

| phenomenon | n_items | Chinese acc | Diacritic acc | gap | collapsed count | collapsed rate | tie count | tie rate | noncollapsed n | Chinese noncollapsed acc | Diacritic noncollapsed acc | noncollapsed gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fci_licensing | 1500 | 0.8767 | 0.5193 | +0.3573 | 0 | 0.0000 | 0 | 0.0000 | 1500 | 0.8767 | 0.5193 | +0.3573 |
| npi_licensing | 2700 | 0.7915 | 0.5185 | +0.2730 | 0 | 0.0000 | 0 | 0.0000 | 2700 | 0.7915 | 0.5185 | +0.2730 |
| anaphor | 1800 | 0.5539 | 0.3461 | +0.2078 | 600 | 0.3333 | 600 | 0.3333 | 1200 | 0.5725 | 0.5192 | +0.0533 |
| control_raising | 1200 | 0.8767 | 0.7333 | +0.1433 | 0 | 0.0000 | 0 | 0.0000 | 1200 | 0.8767 | 0.7333 | +0.1433 |
| verb_phrase | 4200 | 0.8595 | 0.7190 | +0.1405 | 0 | 0.0000 | 0 | 0.0000 | 4200 | 0.8595 | 0.7190 | +0.1405 |
| BA | 3900 | 0.7546 | 0.6305 | +0.1241 | 0 | 0.0000 | 0 | 0.0000 | 3900 | 0.7546 | 0.6305 | +0.1241 |
| passive | 3600 | 0.7878 | 0.7283 | +0.0594 | 0 | 0.0000 | 0 | 0.0000 | 3600 | 0.7878 | 0.7283 | +0.0594 |
| argument_structure | 2100 | 0.6971 | 0.6443 | +0.0529 | 0 | 0.0000 | 0 | 0.0000 | 2100 | 0.6971 | 0.6443 | +0.0529 |
| nominal_expression | 3300 | 0.4900 | 0.4400 | +0.0500 | 0 | 0.0000 | 0 | 0.0000 | 3300 | 0.4900 | 0.4400 | +0.0500 |
| topicalization | 1200 | 0.7983 | 0.7642 | +0.0342 | 0 | 0.0000 | 0 | 0.0000 | 1200 | 0.7983 | 0.7642 | +0.0342 |
| relativization | 1200 | 0.6633 | 0.6492 | +0.0142 | 0 | 0.0000 | 0 | 0.0000 | 1200 | 0.6633 | 0.6492 | +0.0142 |
| ellipsis | 900 | 0.3644 | 0.3556 | +0.0089 | 0 | 0.0000 | 0 | 0.0000 | 900 | 0.3644 | 0.3556 | +0.0089 |
| question | 6300 | 0.5968 | 0.6330 | -0.0362 | 0 | 0.0000 | 0 | 0.0000 | 6300 | 0.5968 | 0.6330 | -0.0362 |
| classifier | 900 | 0.3900 | 0.4400 | -0.0500 | 0 | 0.0000 | 0 | 0.0000 | 900 | 0.3900 | 0.4400 | -0.0500 |
| quantifiers | 600 | 0.2650 | 0.5033 | -0.2383 | 0 | 0.0000 | 0 | 0.0000 | 600 | 0.2650 | 0.5033 | -0.2383 |

## Per-Phenomenon Tables

### BA

| metric | value |
|---|---:|
| n_items | 3900 |
| Chinese accuracy | 0.754615 |
| Diacritic accuracy | 0.630513 |
| gap | +0.124103 |
| collapsed count | 0 |
| collapsed rate | 0.000000 |
| tie count | 0 |
| tie rate | 0.000000 |
| noncollapsed n_items | 3900 |
| Chinese noncollapsed accuracy | 0.754615 |
| Diacritic noncollapsed accuracy | 0.630513 |
| noncollapsed gap | +0.124103 |

### anaphor

| metric | value |
|---|---:|
| n_items | 1800 |
| Chinese accuracy | 0.553889 |
| Diacritic accuracy | 0.346111 |
| gap | +0.207778 |
| collapsed count | 600 |
| collapsed rate | 0.333333 |
| tie count | 600 |
| tie rate | 0.333333 |
| noncollapsed n_items | 1200 |
| Chinese noncollapsed accuracy | 0.572500 |
| Diacritic noncollapsed accuracy | 0.519167 |
| noncollapsed gap | +0.053333 |

### argument_structure

| metric | value |
|---|---:|
| n_items | 2100 |
| Chinese accuracy | 0.697143 |
| Diacritic accuracy | 0.644286 |
| gap | +0.052857 |
| collapsed count | 0 |
| collapsed rate | 0.000000 |
| tie count | 0 |
| tie rate | 0.000000 |
| noncollapsed n_items | 2100 |
| Chinese noncollapsed accuracy | 0.697143 |
| Diacritic noncollapsed accuracy | 0.644286 |
| noncollapsed gap | +0.052857 |

### classifier

| metric | value |
|---|---:|
| n_items | 900 |
| Chinese accuracy | 0.390000 |
| Diacritic accuracy | 0.440000 |
| gap | -0.050000 |
| collapsed count | 0 |
| collapsed rate | 0.000000 |
| tie count | 0 |
| tie rate | 0.000000 |
| noncollapsed n_items | 900 |
| Chinese noncollapsed accuracy | 0.390000 |
| Diacritic noncollapsed accuracy | 0.440000 |
| noncollapsed gap | -0.050000 |

### control_raising

| metric | value |
|---|---:|
| n_items | 1200 |
| Chinese accuracy | 0.876667 |
| Diacritic accuracy | 0.733333 |
| gap | +0.143333 |
| collapsed count | 0 |
| collapsed rate | 0.000000 |
| tie count | 0 |
| tie rate | 0.000000 |
| noncollapsed n_items | 1200 |
| Chinese noncollapsed accuracy | 0.876667 |
| Diacritic noncollapsed accuracy | 0.733333 |
| noncollapsed gap | +0.143333 |

### ellipsis

| metric | value |
|---|---:|
| n_items | 900 |
| Chinese accuracy | 0.364444 |
| Diacritic accuracy | 0.355556 |
| gap | +0.008889 |
| collapsed count | 0 |
| collapsed rate | 0.000000 |
| tie count | 0 |
| tie rate | 0.000000 |
| noncollapsed n_items | 900 |
| Chinese noncollapsed accuracy | 0.364444 |
| Diacritic noncollapsed accuracy | 0.355556 |
| noncollapsed gap | +0.008889 |

### fci_licensing

| metric | value |
|---|---:|
| n_items | 1500 |
| Chinese accuracy | 0.876667 |
| Diacritic accuracy | 0.519333 |
| gap | +0.357333 |
| collapsed count | 0 |
| collapsed rate | 0.000000 |
| tie count | 0 |
| tie rate | 0.000000 |
| noncollapsed n_items | 1500 |
| Chinese noncollapsed accuracy | 0.876667 |
| Diacritic noncollapsed accuracy | 0.519333 |
| noncollapsed gap | +0.357333 |

### nominal_expression

| metric | value |
|---|---:|
| n_items | 3300 |
| Chinese accuracy | 0.490000 |
| Diacritic accuracy | 0.440000 |
| gap | +0.050000 |
| collapsed count | 0 |
| collapsed rate | 0.000000 |
| tie count | 0 |
| tie rate | 0.000000 |
| noncollapsed n_items | 3300 |
| Chinese noncollapsed accuracy | 0.490000 |
| Diacritic noncollapsed accuracy | 0.440000 |
| noncollapsed gap | +0.050000 |

### npi_licensing

| metric | value |
|---|---:|
| n_items | 2700 |
| Chinese accuracy | 0.791481 |
| Diacritic accuracy | 0.518519 |
| gap | +0.272963 |
| collapsed count | 0 |
| collapsed rate | 0.000000 |
| tie count | 0 |
| tie rate | 0.000000 |
| noncollapsed n_items | 2700 |
| Chinese noncollapsed accuracy | 0.791481 |
| Diacritic noncollapsed accuracy | 0.518519 |
| noncollapsed gap | +0.272963 |

### passive

| metric | value |
|---|---:|
| n_items | 3600 |
| Chinese accuracy | 0.787778 |
| Diacritic accuracy | 0.728333 |
| gap | +0.059444 |
| collapsed count | 0 |
| collapsed rate | 0.000000 |
| tie count | 0 |
| tie rate | 0.000000 |
| noncollapsed n_items | 3600 |
| Chinese noncollapsed accuracy | 0.787778 |
| Diacritic noncollapsed accuracy | 0.728333 |
| noncollapsed gap | +0.059444 |

### quantifiers

| metric | value |
|---|---:|
| n_items | 600 |
| Chinese accuracy | 0.265000 |
| Diacritic accuracy | 0.503333 |
| gap | -0.238333 |
| collapsed count | 0 |
| collapsed rate | 0.000000 |
| tie count | 0 |
| tie rate | 0.000000 |
| noncollapsed n_items | 600 |
| Chinese noncollapsed accuracy | 0.265000 |
| Diacritic noncollapsed accuracy | 0.503333 |
| noncollapsed gap | -0.238333 |

### question

| metric | value |
|---|---:|
| n_items | 6300 |
| Chinese accuracy | 0.596825 |
| Diacritic accuracy | 0.633016 |
| gap | -0.036190 |
| collapsed count | 0 |
| collapsed rate | 0.000000 |
| tie count | 0 |
| tie rate | 0.000000 |
| noncollapsed n_items | 6300 |
| Chinese noncollapsed accuracy | 0.596825 |
| Diacritic noncollapsed accuracy | 0.633016 |
| noncollapsed gap | -0.036190 |

### relativization

| metric | value |
|---|---:|
| n_items | 1200 |
| Chinese accuracy | 0.663333 |
| Diacritic accuracy | 0.649167 |
| gap | +0.014167 |
| collapsed count | 0 |
| collapsed rate | 0.000000 |
| tie count | 0 |
| tie rate | 0.000000 |
| noncollapsed n_items | 1200 |
| Chinese noncollapsed accuracy | 0.663333 |
| Diacritic noncollapsed accuracy | 0.649167 |
| noncollapsed gap | +0.014167 |

### topicalization

| metric | value |
|---|---:|
| n_items | 1200 |
| Chinese accuracy | 0.798333 |
| Diacritic accuracy | 0.764167 |
| gap | +0.034167 |
| collapsed count | 0 |
| collapsed rate | 0.000000 |
| tie count | 0 |
| tie rate | 0.000000 |
| noncollapsed n_items | 1200 |
| Chinese noncollapsed accuracy | 0.798333 |
| Diacritic noncollapsed accuracy | 0.764167 |
| noncollapsed gap | +0.034167 |

### verb_phrase

| metric | value |
|---|---:|
| n_items | 4200 |
| Chinese accuracy | 0.859524 |
| Diacritic accuracy | 0.719048 |
| gap | +0.140476 |
| collapsed count | 0 |
| collapsed rate | 0.000000 |
| tie count | 0 |
| tie rate | 0.000000 |
| noncollapsed n_items | 4200 |
| Chinese noncollapsed accuracy | 0.859524 |
| Diacritic noncollapsed accuracy | 0.719048 |
| noncollapsed gap | +0.140476 |
