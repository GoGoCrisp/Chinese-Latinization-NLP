# Confirmed Existing Setup

Source checked locally in `Chinese_Latinization_NLP/2.Model trainning` on 2026-05-12.

## Existing Seed42 Matched-Token Runs

- Chinese-Origin config: `configs/step5_chinese_125m_b1024_4epoch.yaml`
- Chinese-Origin output: `server_outputs/4epoch/outputs/chinese_125m_b1024_4epoch_seed42/checkpoint-27176`
- Chinese-Origin tokenizer: `tokenizers/chinese_origin_32k_eos`
- Pinyin-Diacritic config: `configs/step5_diacritic_125m_b1024_matched_token_4epoch.yaml`
- Pinyin-Diacritic output: `server_outputs/4epoch/diacritic_125m_b1024_matched_token_4epoch_seed42_outputs/outputs/diacritic_125m_b1024_matched_token_4epoch_seed42/checkpoint-27176`
- Pinyin-Diacritic tokenizer: `tokenizers/pinyin_diacritic_32k_eos`

Both existing matched-token configs and saved `training_args.json` files confirm `seed: 42`.

## Model And Tokenizer

- Saved `run_config_resolved.json` and `run_summary.json` for both main runs report `parameter_count: 134107392`.
- Both tokenizer EOS update reports confirm:
  - `vocab_size` / `new_vocab_size`: 32001
  - `eos_token_id`: 32000
  - `pad_token_id`: 32000
- Both existing configs use the same architecture:
  - `hidden_size: 768`
  - `intermediate_size: 2048`
  - `num_hidden_layers: 12`
  - `num_attention_heads: 12`
  - `max_position_embeddings: 1024`
  - `rms_norm_eps: 1.0e-5`
  - `hidden_act: silu`
  - `rope_theta: 10000.0`
  - `tie_word_embeddings: false`

## Data Paths

Confirmed current train/valid paths:

- Chinese train: `data/tokenized/chinese_train_full_eos_1024`
- Chinese valid: `data/tokenized/chinese_valid_full_eos_1024`
- Pinyin-Diacritic train: `data/tokenized/diacritic_train_full_eos_1024`
- Pinyin-Diacritic valid: `data/tokenized/diacritic_valid_full_eos_1024`

Saved metadata reports:

- Chinese train rows: 434846
- Chinese valid rows: 4374
- Pinyin-Diacritic train rows: 476236
- Pinyin-Diacritic valid rows: 4798

## Matched-Token Schedule

The training script computes:

`tokens_per_update = per_device_train_batch_size * gradient_accumulation_steps * max_position_embeddings`

With `16 * 4 * 1024`, the confirmed `tokens_per_update` is 65536.

Saved run metadata confirms:

- Chinese-Origin `updates_per_train_pass`: 6794
- Pinyin-Diacritic `updates_per_train_pass`: 7441
- Current matched-token `max_steps`: 27176
- Current matched-token `tokens_seen`: 1781006336

Pass exposure:

- Chinese-Origin: `27176 / 6794 = 4.0` full source-data passes.
- Pinyin-Diacritic: `27176 / 7441 = 3.6521972853` source-data passes, about 3.65.

## New Robustness Schedule

For the fixed-data Pinyin-Diacritic control:

- `updates_per_train_pass = 7441`
- `train_passes = 4`
- `max_steps = 29764`
- `tokens_per_update = 65536`
- `expected_tokens_seen = 1950613504`

This gives Pinyin-Diacritic approximately 4 full source-data passes while preserving all other seed42 Pinyin-Diacritic hyperparameters.
