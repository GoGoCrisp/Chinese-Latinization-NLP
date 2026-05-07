# Step 5 Server Run: Experiment 2 Pinyin-Diacritic 125M b1024

This bundle is for the A100 Pinyin-Diacritic runs:

- Variant: Pinyin-Diacritic
- Tokenizer: EOS tokenizer, vocab_size 32001
- eos_token_id: 32000
- pad_token_id: 32000
- block_size: 1024
- Model: Llama-style causal LM from scratch
- Formal 4epoch architecture: 12 transformer layers
- matched-token run: 6794 steps
- matched-token strict 4epoch run: 27176 steps, 1,781,006,336 tokens_seen
- matched-content run: 7441 steps

Do not put API keys, private keys, passwords, billing information, real IPs, or real ports in this repository or bundle.

## 1. Upload Archive From Local Machine

Option A, scp:

```bash
scp -P <PORT> server_bundle_exp2_diacritic_125m_b1024.tar.gz \
  root@<IP>:/workspace/
```

Option B, rsync:

```bash
rsync -avP -e "ssh -p <PORT>" \
  server_bundle_exp2_diacritic_125m_b1024.tar.gz \
  root@<IP>:/workspace/
```

## 2. SSH Into Server

```bash
ssh -p <PORT> root@<IP>
```

## 3. Extract Archive On Server

```bash
cd /workspace
mkdir -p Chinese_Latinization_NLP
tar -xzf server_bundle_exp2_diacritic_125m_b1024.tar.gz -C Chinese_Latinization_NLP
cd Chinese_Latinization_NLP/server_bundle_exp2_diacritic_125m_b1024
```

## 4. Create Python Environment

Use the server image's recommended CUDA PyTorch environment if it already has one. Otherwise:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

If the server image does not include CUDA-enabled PyTorch, install the CUDA wheel following the official PyTorch instructions for that server image before installing or running training.

## 5. Verify Bundle

```bash
bash scripts/step5_verify_diacritic_server_bundle.sh
```

This checks tokenizer loading, vocab/EOS/PAD ids, dataset loading, row length 1024, metadata, configs, training script, and disk usage.

## 6. Run Diacritic Smoke Test

```bash
bash scripts/step5_run_diacritic_125m_smoke.sh 2>&1 | tee outputs/diacritic_125m_b1024_smoke.log
```

Expected output directory:

```text
outputs/diacritic_125m_b1024_smoke
```

## 7. Run Matched-Token Strict 4epoch Training In tmux

```bash
tmux new -s diacritic_matched_token_4epoch
```

Run:

```bash
bash scripts/step5_run_diacritic_125m_matched_token_4epoch.sh 2>&1 | tee outputs/diacritic_125m_b1024_matched_token_4epoch_seed42.log
```

Detach tmux with `Ctrl-b`, then `d`.

Expected output directory:

```text
outputs/diacritic_125m_b1024_matched_token_4epoch_seed42
```

The matched-token strict 4epoch run uses:

- per_device_train_batch_size: 16
- gradient_accumulation_steps: 4
- block_size: 1024
- tokens_per_update: 65,536
- max_steps: 27,176
- num_hidden_layers: 12
- eval_steps/save_steps: 1,000
- logging_steps: 20
- save_total_limit: 3
- total tokens_seen at final step: 1,781,006,336

## 8. Optional Old Matched-Content Training In tmux

```bash
tmux new -s diacritic_matched_content
```

Run:

```bash
bash scripts/step5_run_diacritic_125m_matched_content.sh 2>&1 | tee outputs/diacritic_125m_b1024_matched_content_seed42.log
```

Detach tmux with `Ctrl-b`, then `d`.

Expected output directory:

```text
outputs/diacritic_125m_b1024_matched_content_seed42
```

The matched-content run uses:

- per_device_train_batch_size: 16
- gradient_accumulation_steps: 4
- block_size: 1024
- tokens_per_update: 65,536
- max_steps: 7,441
- total tokens_seen at final step: 487,653,376

## 9. Stop At Next Checkpoint Safely

This script only waits and prints a safe-to-interrupt message. It does not kill training and does not stop or destroy the Vast instance.

```bash
bash scripts/step5_wait_for_next_checkpoint_then_exit.sh \
  outputs/diacritic_125m_b1024_matched_token_4epoch_seed42
```

## 10. Monitor

GPU:

```bash
nvidia-smi
```

Logs:

```bash
tail -f outputs/diacritic_125m_b1024_matched_token_4epoch_seed42.log
tail -f outputs/diacritic_125m_b1024_matched_token_4epoch_seed42/train_log.jsonl
tail -f outputs/diacritic_125m_b1024_matched_content_seed42.log
tail -f outputs/diacritic_125m_b1024_matched_content_seed42/train_log.jsonl
```

Disk:

```bash
df -h
du -sh outputs/diacritic_125m_b1024_matched_token_4epoch_seed42
du -sh outputs/diacritic_125m_b1024_matched_content_seed42
```

## 11. Download Outputs From Local Machine

The safest workflow is to pull from the server using your local Mac terminal.

Full matched-token output:

```bash
rsync -avP -e "ssh -p <PORT>" \
  root@<IP>:<REMOTE_PROJECT_DIR>/outputs/diacritic_125m_b1024_matched_token_4epoch_seed42/ \
  "./server_outputs/diacritic_125m_b1024_matched_token_4epoch_seed42/"
```

Full matched-content output:

```bash
rsync -avP -e "ssh -p <PORT>" \
  root@<IP>:<REMOTE_PROJECT_DIR>/outputs/diacritic_125m_b1024_matched_content_seed42/ \
  "./server_outputs/diacritic_125m_b1024_matched_content_seed42/"
```

## 12. Stop Or Destroy Vast Instance

Stop or destroy the Vast instance manually only after:

1. The training logs and checkpoints have been downloaded.
2. The downloaded checkpoint directory contains model weights, optimizer, scheduler, trainer state, args/config, and logs.
3. You have verified that the local files are readable.

Do not run Vast stop or destroy commands from any helper script unless you explicitly decide to do so.
