# Matched-Data Extra Seeds Setup

Prepared locally in `Chinese_Latinization_NLP/2.Model trainning`.

## Runs

This bundle is only for two Pinyin-Diacritic matched-data runs:

- `diacritic_125m_b1024_matched_data_4epoch_seed43`
- `diacritic_125m_b1024_matched_data_4epoch_seed44`

## Shared Setup

- tokenizer: `tokenizers/pinyin_diacritic_32k_eos`
- train data: `data/tokenized/diacritic_train_full_eos_1024`
- valid data: `data/tokenized/diacritic_valid_full_eos_1024`
- expected tokenizer vocab size: `32001`
- expected EOS token id: `32000`
- expected PAD token id: `32000`
- expected parameter count: `134107392`

## Schedule

- `updates_per_train_pass = 7441`
- `train_passes = 4`
- `max_steps = 29764`
- `tokens_per_update = 65536`
- `expected_tokens_seen = 1950613504`
- expected source-data passes: `4.0`

## Architecture And Optimization

Both configs preserve the same architecture and hyperparameters as the previous Pinyin-Diacritic matched-data seed42 control, changing only `seed`, `output_dir`, and run identity.
