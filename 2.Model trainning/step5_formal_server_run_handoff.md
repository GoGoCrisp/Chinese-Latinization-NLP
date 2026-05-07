# Step 5 Formal Server Run Handoff

Last updated: 2026-05-05

This file records the Experiment 2 formal A100 run workflow and run metadata so a new chat window can continue without relying on long conversation history.

## Current Status

The first full A100 formal runs for Experiment 2 are complete and downloaded locally.

Completed models:

| Variant | Run type | Output dir in archive | Final step | Tokens seen | Final eval loss | Wall clock |
|---|---:|---|---:|---:|---:|---:|
| Chinese-Origin | one epoch budget | `outputs/chinese_125m_b1024_oneepoch_seed42` | 6794 | 445,251,584 | 4.370833 | 3528.35 s |
| Pinyin-Diacritic | matched-token budget | `outputs/diacritic_125m_b1024_matched_token_seed42` | 6794 | 445,251,584 | 4.264621 | 4538.56 s |
| Pinyin-Diacritic | matched-content budget | `outputs/diacritic_125m_b1024_matched_content_seed42` | 7441 | 487,653,376 | 4.210126 | 4966.71 s |

Local downloaded archives:

```bash
Chinese_Latinization_NLP/2.Model trainning/server_outputs/chinese_125m_b1024_oneepoch_seed42_outputs.tar.gz
Chinese_Latinization_NLP/2.Model trainning/server_outputs/diacritic_125m_b1024_matched_token_seed42_outputs.tar.gz
Chinese_Latinization_NLP/2.Model trainning/server_outputs/diacritic_125m_b1024_matched_content_seed42_outputs.tar.gz
```

Each archive was locally checked with `tar -tzf`. Each contains the final checkpoint plus logs and summaries:

- `model.safetensors`
- `optimizer.pt`
- `scheduler.pt`
- `trainer_state.json`
- `training_args.json`
- `rng_state.pth`
- `config.json`
- `generation_config.json`
- `train_log.jsonl`
- `run_summary.json`
- outer `.log`

The last retained checkpoints follow `save_total_limit=3`:

- Chinese: `checkpoint-6000`, `checkpoint-6500`, `checkpoint-6794`
- Diacritic matched-token: `checkpoint-6000`, `checkpoint-6500`, `checkpoint-6794`
- Diacritic matched-content: `checkpoint-6500`, `checkpoint-7000`, `checkpoint-7441`

## Main Local Project Paths

Project root:

```bash
Chinese_Latinization_NLP/2.Model trainning
```

Important scripts:

```bash
Chinese_Latinization_NLP/2.Model trainning/step5_train_lm_formal.py
Chinese_Latinization_NLP/2.Model trainning/scripts/step5_package_chinese_125m_b1024_bundle.sh
Chinese_Latinization_NLP/2.Model trainning/scripts/step5_package_diacritic_125m_b1024_bundle.sh
Chinese_Latinization_NLP/2.Model trainning/scripts/step5_verify_server_bundle.sh
Chinese_Latinization_NLP/2.Model trainning/scripts/step5_verify_diacritic_server_bundle.sh
Chinese_Latinization_NLP/2.Model trainning/scripts/step5_run_chinese_125m_smoke.sh
Chinese_Latinization_NLP/2.Model trainning/scripts/step5_run_chinese_125m_oneepoch.sh
Chinese_Latinization_NLP/2.Model trainning/scripts/step5_run_diacritic_125m_smoke.sh
Chinese_Latinization_NLP/2.Model trainning/scripts/step5_run_diacritic_125m_matched_token.sh
Chinese_Latinization_NLP/2.Model trainning/scripts/step5_run_diacritic_125m_matched_content.sh
Chinese_Latinization_NLP/2.Model trainning/scripts/wait_for_next_checkpoint_then_exit.sh
```

Important configs:

```bash
Chinese_Latinization_NLP/2.Model trainning/configs/step5_chinese_125m_b1024.yaml
Chinese_Latinization_NLP/2.Model trainning/configs/step5_diacritic_125m_b1024_matched_token.yaml
Chinese_Latinization_NLP/2.Model trainning/configs/step5_diacritic_125m_b1024_matched_content.yaml
```

Server bundle archives created locally:

```bash
Chinese_Latinization_NLP/2.Model trainning/server_bundle_exp2_chinese_125m_b1024.tar.gz
Chinese_Latinization_NLP/2.Model trainning/server_bundle_exp2_diacritic_125m_b1024.tar.gz
```

Extracted local bundle directories:

```bash
Chinese_Latinization_NLP/2.Model trainning/server_bundle_exp2_chinese_125m_b1024
Chinese_Latinization_NLP/2.Model trainning/server_bundle_exp2_diacritic_125m_b1024
```

## SSH And Server Connection

The A100 server was a Vast.ai instance with 1 x A100 40GB.

Direct SSH used during this run:

```bash
ssh -i ~/.ssh/vast_ai_nopass -o IdentitiesOnly=yes -p 23844 root@209.146.116.50
```

Original Vast direct SSH form shown by Vast:

```bash
ssh -p 23844 root@209.146.116.50 -L 8080:localhost:8080
```

Original Vast proxy SSH form:

```bash
ssh -p 12119 root@ssh1.vast.ai -L 8080:localhost:8080
```

The no-pass local private key path used by Codex:

```bash
~/.ssh/vast_ai_nopass
```

Do not put private key contents, API keys, passwords, or billing information in scripts or notes.

Useful server preflight commands:

```bash
nvidia-smi
df -h
```

Torch/CUDA check:

```bash
python3 - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("torch cuda:", torch.version.cuda)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("bf16:", torch.cuda.is_bf16_supported())
PY
```

Observed environment:

- GPU: `NVIDIA A100-PCIE-40GB`
- CUDA available: yes
- bf16 supported: yes
- torch: `2.11.0+cu128`
- transformers: `5.7.0`
- datasets: `4.8.5`

## Local To Server Bundle Workflow

### Package Chinese Bundle Locally

From local project root:

```bash
cd "Chinese_Latinization_NLP/2.Model trainning"
bash scripts/step5_package_chinese_125m_b1024_bundle.sh
```

This creates:

```bash
server_bundle_exp2_chinese_125m_b1024.tar.gz
```

Included content:

- `tokenizers/chinese_origin_32k_eos`
- `data/tokenized/chinese_train_full_eos_1024`
- `data/tokenized/chinese_valid_full_eos_1024`
- `data/tokenized/chinese_test_full_eos_1024`
- `step5_train_lm_formal.py`
- configs
- run scripts
- verification scripts
- `requirements.txt`
- `README_server_run.md`
- `MANIFEST.txt`

Excluded content:

- no-EOS tokenizers
- 512-block debug datasets
- old debug outputs
- old checkpoints
- raw data unless explicitly needed
- decoded vocabulary analysis files

### Package Diacritic Bundle Locally

```bash
cd "Chinese_Latinization_NLP/2.Model trainning"
bash scripts/step5_package_diacritic_125m_b1024_bundle.sh
```

This creates:

```bash
server_bundle_exp2_diacritic_125m_b1024.tar.gz
```

Included content:

- `tokenizers/pinyin_diacritic_32k_eos`
- `data/tokenized/diacritic_train_full_eos_1024`
- `data/tokenized/diacritic_valid_full_eos_1024`
- `data/tokenized/diacritic_test_full_eos_1024`
- `step5_train_lm_formal.py`
- configs
- run scripts
- verification scripts
- `requirements.txt`
- `README_server_run_diacritic.md`
- `MANIFEST.txt`

### Upload Bundle To Server

Chinese upload:

```bash
scp -i ~/.ssh/vast_ai_nopass -o IdentitiesOnly=yes -P 23844 \
  "Chinese_Latinization_NLP/2.Model trainning/server_bundle_exp2_chinese_125m_b1024.tar.gz" \
  root@209.146.116.50:/workspace/
```

Diacritic upload:

```bash
scp -i ~/.ssh/vast_ai_nopass -o IdentitiesOnly=yes -P 23844 \
  "Chinese_Latinization_NLP/2.Model trainning/server_bundle_exp2_diacritic_125m_b1024.tar.gz" \
  root@209.146.116.50:/workspace/
```

Generic template:

```bash
scp -P <PORT> server_bundle_exp2_chinese_125m_b1024.tar.gz \
  root@<IP>:/workspace/

rsync -avP -e "ssh -p <PORT>" \
  server_bundle_exp2_chinese_125m_b1024.tar.gz \
  root@<IP>:/workspace/
```

### Extract Bundle On Server

Chinese:

```bash
ssh -i ~/.ssh/vast_ai_nopass -o IdentitiesOnly=yes -p 23844 root@209.146.116.50
cd /workspace
mkdir -p Chinese_Latinization_NLP
tar -xzf server_bundle_exp2_chinese_125m_b1024.tar.gz -C Chinese_Latinization_NLP
cd /workspace/Chinese_Latinization_NLP/server_bundle_exp2_chinese_125m_b1024
```

Diacritic:

```bash
ssh -i ~/.ssh/vast_ai_nopass -o IdentitiesOnly=yes -p 23844 root@209.146.116.50
cd /workspace
mkdir -p Chinese_Latinization_NLP
tar -xzf server_bundle_exp2_diacritic_125m_b1024.tar.gz -C Chinese_Latinization_NLP
cd /workspace/Chinese_Latinization_NLP/server_bundle_exp2_diacritic_125m_b1024
```

The extraction printed harmless macOS extended attribute warnings such as `LIBARCHIVE.xattr.com.apple.provenance`.

## Server Python Environment

Chinese bundle environment was created under:

```bash
/workspace/Chinese_Latinization_NLP/server_bundle_exp2_chinese_125m_b1024/.venv
```

Typical setup:

```bash
cd /workspace/Chinese_Latinization_NLP/server_bundle_exp2_chinese_125m_b1024
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Diacritic bundle reused the same venv by symlink:

```bash
cd /workspace/Chinese_Latinization_NLP/server_bundle_exp2_diacritic_125m_b1024
ln -s ../server_bundle_exp2_chinese_125m_b1024/.venv .venv
. .venv/bin/activate
```

Verify after activation:

```bash
python3 - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("torch cuda:", torch.version.cuda)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("bf16:", torch.cuda.is_bf16_supported())
PY
```

## Server Bundle Verification

Chinese:

```bash
cd /workspace/Chinese_Latinization_NLP/server_bundle_exp2_chinese_125m_b1024
. .venv/bin/activate
bash scripts/step5_verify_server_bundle.sh
```

Diacritic:

```bash
cd /workspace/Chinese_Latinization_NLP/server_bundle_exp2_diacritic_125m_b1024
. .venv/bin/activate
bash scripts/step5_verify_diacritic_server_bundle.sh
```

Verification checks:

- tokenizer directory exists
- tokenizer loads
- vocab size is `32001`
- `eos_token_id == 32000`
- `pad_token_id == 32000`
- tokenized datasets load with `datasets.load_from_disk`
- first row length is `1024`
- metadata exists and says `block_size=1024`
- training script exists
- config exists
- requirements exist
- disk usage printed

## Formal Training Commands

All formal runs use:

- script: `step5_train_lm_formal.py`
- model: Llama-style decoder-only causal LM from scratch
- precision: bf16 on CUDA/A100
- fallback: fp32 on CPU
- scheduler: cosine
- optimizer: AdamW
- `save_total_limit=3`
- `dataloader_drop_last=true`
- `tokens_per_update = 16 * 4 * 1024 = 65,536`

Run inside `tmux` so training survives SSH disconnect.

### Chinese Smoke Test

```bash
cd /workspace/Chinese_Latinization_NLP/server_bundle_exp2_chinese_125m_b1024
. .venv/bin/activate
bash scripts/step5_run_chinese_125m_smoke.sh
```

Smoke output:

```bash
outputs/chinese_125m_b1024_smoke
```

### Chinese One-Epoch Run

Run:

```bash
cd /workspace/Chinese_Latinization_NLP/server_bundle_exp2_chinese_125m_b1024
tmux new -d -s chinese_oneepoch \
  "cd /workspace/Chinese_Latinization_NLP/server_bundle_exp2_chinese_125m_b1024 && . .venv/bin/activate && bash scripts/step5_run_chinese_125m_oneepoch.sh 2>&1 | tee outputs/chinese_125m_b1024_oneepoch_seed42.log"
```

Output:

```bash
outputs/chinese_125m_b1024_oneepoch_seed42
```

### Diacritic Smoke Test

```bash
cd /workspace/Chinese_Latinization_NLP/server_bundle_exp2_diacritic_125m_b1024
. .venv/bin/activate
bash scripts/step5_run_diacritic_125m_smoke.sh
```

Smoke output:

```bash
outputs/diacritic_125m_b1024_smoke
```

### Diacritic Matched-Token Run

Run:

```bash
cd /workspace/Chinese_Latinization_NLP/server_bundle_exp2_diacritic_125m_b1024
tmux new -d -s diacritic_matched_token \
  "cd /workspace/Chinese_Latinization_NLP/server_bundle_exp2_diacritic_125m_b1024 && . .venv/bin/activate && bash scripts/step5_run_diacritic_125m_matched_token.sh 2>&1 | tee outputs/diacritic_125m_b1024_matched_token_seed42.log"
```

Output:

```bash
outputs/diacritic_125m_b1024_matched_token_seed42
```

### Diacritic Matched-Content Run

Run:

```bash
cd /workspace/Chinese_Latinization_NLP/server_bundle_exp2_diacritic_125m_b1024
tmux new -d -s diacritic_matched_content \
  "cd /workspace/Chinese_Latinization_NLP/server_bundle_exp2_diacritic_125m_b1024 && . .venv/bin/activate && bash scripts/step5_run_diacritic_125m_matched_content.sh 2>&1 | tee outputs/diacritic_125m_b1024_matched_content_seed42.log"
```

Output:

```bash
outputs/diacritic_125m_b1024_matched_content_seed42
```

## Monitoring Commands

Check training session:

```bash
tmux ls
tmux attach -t chinese_oneepoch
tmux attach -t diacritic_matched_token
tmux attach -t diacritic_matched_content
```

Detach from tmux:

```text
Ctrl-b then d
```

Tail logs:

```bash
tail -f outputs/chinese_125m_b1024_oneepoch_seed42.log
tail -f outputs/diacritic_125m_b1024_matched_token_seed42.log
tail -f outputs/diacritic_125m_b1024_matched_content_seed42.log
```

GPU and disk:

```bash
nvidia-smi
df -h
```

Checkpoint directories:

```bash
find outputs/chinese_125m_b1024_oneepoch_seed42 -maxdepth 1 -type d -name "checkpoint-*" | sort -V
find outputs/diacritic_125m_b1024_matched_token_seed42 -maxdepth 1 -type d -name "checkpoint-*" | sort -V
find outputs/diacritic_125m_b1024_matched_content_seed42 -maxdepth 1 -type d -name "checkpoint-*" | sort -V
```

Checkpoint completeness expected:

```bash
model.safetensors
optimizer.pt
scheduler.pt
trainer_state.json
training_args.json
rng_state.pth
config.json
generation_config.json
```

## Stop Safely At Checkpoint

Helper script:

```bash
scripts/wait_for_next_checkpoint_then_exit.sh
```

Purpose:

- monitor an output directory
- wait until a new checkpoint appears complete
- print that it is safe to interrupt
- never kill training automatically unless explicitly requested
- never stop or destroy a Vast instance automatically

## Compress Server Outputs

Chinese:

```bash
cd /workspace/Chinese_Latinization_NLP/server_bundle_exp2_chinese_125m_b1024
tar -czf /workspace/chinese_125m_b1024_oneepoch_seed42_outputs.tar.gz \
  outputs/chinese_125m_b1024_oneepoch_seed42 \
  outputs/chinese_125m_b1024_oneepoch_seed42.log
du -sh /workspace/chinese_125m_b1024_oneepoch_seed42_outputs.tar.gz
```

Diacritic matched-token:

```bash
cd /workspace/Chinese_Latinization_NLP/server_bundle_exp2_diacritic_125m_b1024
tar -czf /workspace/diacritic_125m_b1024_matched_token_seed42_outputs.tar.gz \
  outputs/diacritic_125m_b1024_matched_token_seed42 \
  outputs/diacritic_125m_b1024_matched_token_seed42.log
du -sh /workspace/diacritic_125m_b1024_matched_token_seed42_outputs.tar.gz
```

Diacritic matched-content:

```bash
cd /workspace/Chinese_Latinization_NLP/server_bundle_exp2_diacritic_125m_b1024
tar -czf /workspace/diacritic_125m_b1024_matched_content_seed42_outputs.tar.gz \
  outputs/diacritic_125m_b1024_matched_content_seed42 \
  outputs/diacritic_125m_b1024_matched_content_seed42.log
du -sh /workspace/diacritic_125m_b1024_matched_content_seed42_outputs.tar.gz
```

Observed server archive sizes were about `1.6G` each. Local downloaded sizes show as about `1.5G` each.

## Download Server Outputs To Local

Chinese:

```bash
scp -i ~/.ssh/vast_ai_nopass -o IdentitiesOnly=yes -P 23844 \
  root@209.146.116.50:/workspace/chinese_125m_b1024_oneepoch_seed42_outputs.tar.gz \
  "Chinese_Latinization_NLP/2.Model trainning/server_outputs/"
```

Diacritic matched-token:

```bash
scp -i ~/.ssh/vast_ai_nopass -o IdentitiesOnly=yes -P 23844 \
  root@209.146.116.50:/workspace/diacritic_125m_b1024_matched_token_seed42_outputs.tar.gz \
  "Chinese_Latinization_NLP/2.Model trainning/server_outputs/"
```

Diacritic matched-content:

```bash
scp -i ~/.ssh/vast_ai_nopass -o IdentitiesOnly=yes -P 23844 \
  root@209.146.116.50:/workspace/diacritic_125m_b1024_matched_content_seed42_outputs.tar.gz \
  "Chinese_Latinization_NLP/2.Model trainning/server_outputs/"
```

Generic pull template:

```bash
rsync -avP -e "ssh -p <PORT>" \
  root@<IP>:/workspace/Chinese_Latinization_NLP/<BUNDLE_DIR>/outputs/<OUTPUT_DIR>/ \
  "./server_outputs/<OUTPUT_DIR>/"
```

## Local Archive Verification

Run this before destroying a server instance:

```bash
cd /Users/crisp/Desktop/code_field/python
ls -lh "Chinese_Latinization_NLP/2.Model trainning/server_outputs"
tar -tzf "Chinese_Latinization_NLP/2.Model trainning/server_outputs/chinese_125m_b1024_oneepoch_seed42_outputs.tar.gz" >/dev/null
tar -tzf "Chinese_Latinization_NLP/2.Model trainning/server_outputs/diacritic_125m_b1024_matched_token_seed42_outputs.tar.gz" >/dev/null
tar -tzf "Chinese_Latinization_NLP/2.Model trainning/server_outputs/diacritic_125m_b1024_matched_content_seed42_outputs.tar.gz" >/dev/null
```

Final checkpoint checks:

```bash
tar -tzf "Chinese_Latinization_NLP/2.Model trainning/server_outputs/chinese_125m_b1024_oneepoch_seed42_outputs.tar.gz" | grep "checkpoint-6794/"
tar -tzf "Chinese_Latinization_NLP/2.Model trainning/server_outputs/diacritic_125m_b1024_matched_token_seed42_outputs.tar.gz" | grep "checkpoint-6794/"
tar -tzf "Chinese_Latinization_NLP/2.Model trainning/server_outputs/diacritic_125m_b1024_matched_content_seed42_outputs.tar.gz" | grep "checkpoint-7441/"
```

Summary checks:

```bash
tar -xOzf "Chinese_Latinization_NLP/2.Model trainning/server_outputs/chinese_125m_b1024_oneepoch_seed42_outputs.tar.gz" \
  outputs/chinese_125m_b1024_oneepoch_seed42/run_summary.json

tar -xOzf "Chinese_Latinization_NLP/2.Model trainning/server_outputs/diacritic_125m_b1024_matched_token_seed42_outputs.tar.gz" \
  outputs/diacritic_125m_b1024_matched_token_seed42/run_summary.json

tar -xOzf "Chinese_Latinization_NLP/2.Model trainning/server_outputs/diacritic_125m_b1024_matched_content_seed42_outputs.tar.gz" \
  outputs/diacritic_125m_b1024_matched_content_seed42/run_summary.json
```

## Dataset And Text Sources

Experiment 2 compares:

1. Chinese-Origin
2. Pinyin-Diacritic

The Pinyin-Diacritic version uses the spaced pinyin-diacritic corpus variant prepared before Step 2. The final model training uses copied local artifacts inside `2.Model trainning` rather than reading from Experiment 1 folders.

Raw split files:

```bash
data/raw/train.zh.txt
data/raw/valid.zh.txt
data/raw/test.zh.txt
data/raw/train.diacritic.txt
data/raw/valid.diacritic.txt
data/raw/test.diacritic.txt
```

Split ratio:

```text
train:valid:test = 98:1:1
```

Line counts:

| Split | Chinese-Origin lines | Pinyin-Diacritic lines |
|---|---:|---:|
| train | 1,310,755 | 1,310,755 |
| valid | 13,375 | 13,375 |
| test | 13,375 | 13,375 |

The paired corpora were checked to be aligned line by line. Empty line count in final 1024-tokenized metadata is `0`.

## Tokenizers

EOS tokenizer directories used for formal training:

```bash
tokenizers/chinese_origin_32k_eos
tokenizers/pinyin_diacritic_32k_eos
```

Tokenizer invariants:

```text
vocab_size = 32001
eos_token_id = 32000
pad_token_id = 32000
eos_token = <|endoftext|>
```

Original no-EOS tokenizers had vocab size `32000`. The EOS token was added in Step 3 with:

```bash
step3_add_eos_to_tokenizers.py
```

Formal training always uses the `_eos` tokenizers.

## Tokenization Settings

Formal block size:

```text
block_size = 1024
```

Tokenization logic:

- load local HuggingFace-compatible tokenizer
- hard-check `vocab_size=32001`
- hard-check `eos_token_id=32000`
- hard-check `pad_token_id=32000`
- read input line by line
- strip only trailing newline
- preserve internal spaces
- encode with `add_special_tokens=False`
- append `eos_token_id` after every line
- concatenate token ids into one stream
- chunk into fixed-length blocks of 1024
- drop final incomplete block
- save HuggingFace Dataset with one column: `input_ids`
- save `metadata.json`

Formal tokenized dataset paths:

```bash
data/tokenized/chinese_train_full_eos_1024
data/tokenized/chinese_valid_full_eos_1024
data/tokenized/chinese_test_full_eos_1024
data/tokenized/diacritic_train_full_eos_1024
data/tokenized/diacritic_valid_full_eos_1024
data/tokenized/diacritic_test_full_eos_1024
```

Metadata:

| Dataset | Lines | Tokens | Blocks/rows | Dropped tokens | Empty lines |
|---|---:|---:|---:|---:|---:|
| Chinese train | 1,310,755 | 445,283,267 | 434,846 | 963 | 0 |
| Chinese valid | 13,375 | 4,479,760 | 4,374 | 784 | 0 |
| Chinese test | 13,375 | 4,454,628 | 4,350 | 228 | 0 |
| Diacritic train | 1,310,755 | 487,665,889 | 476,236 | 225 | 0 |
| Diacritic valid | 13,375 | 4,913,279 | 4,798 | 127 | 0 |
| Diacritic test | 13,375 | 4,884,201 | 4,769 | 745 | 0 |

Diacritic/Chinese train token ratio:

```text
487,665,889 / 445,283,267 = approximately 1.0952
```

## Model Configuration

All formal runs use the same 10-layer 125M-ish Llama-style architecture.

```yaml
vocab_size: 32001
hidden_size: 768
intermediate_size: 2048
num_hidden_layers: 10
num_attention_heads: 12
max_position_embeddings: 1024
rms_norm_eps: 1.0e-5
hidden_act: silu
rope_theta: 10000.0
tie_word_embeddings: false
eos_token_id: 32000
pad_token_id: 32000
```

Parameter count printed by the training script:

```text
119,948,544
```

This is the 125M-class pilot/formal configuration used for Experiment 2 server runs. It is not the earlier 12M local debug model.

## Training Hyperparameters

Common training hyperparameters:

```yaml
per_device_train_batch_size: 16
per_device_eval_batch_size: 16
gradient_accumulation_steps: 4
sequence_length: 1024
tokens_per_update: 65536
learning_rate: 3.0e-4
weight_decay: 0.1
adam_beta1: 0.9
adam_beta2: 0.95
adam_epsilon: 1.0e-8
scheduler: cosine
warmup_ratio: 0.03
max_grad_norm: 1.0
eval_steps: 500
save_steps: 500
logging_steps: 10
save_total_limit: 3
dataloader_drop_last: true
seed: 42
precision: bf16 on CUDA/A100
tf32: true when CUDA is available
```

Budget choices:

| Run | max_steps | Token budget | Rationale |
|---|---:|---:|---|
| Chinese oneepoch | 6794 | 445,251,584 | approximately one Chinese-Origin epoch under 65,536 tokens/update |
| Diacritic matched-token | 6794 | 445,251,584 | same token budget as Chinese-Origin |
| Diacritic matched-content | 7441 | 487,653,376 | approximately one full Diacritic content epoch, just below 487,665,889 train tokens |

Note on matched-content:

- `7441 * 65,536 = 487,653,376`
- Diacritic train tokens: `487,665,889`
- leaves `12,513` tokens unconsumed
- `7442` would overshoot the train-token count by about `53,023` tokens, so `7441` was chosen.

## Final Run Results

### Chinese-Origin Oneepoch

Output:

```bash
outputs/chinese_125m_b1024_oneepoch_seed42
```

Summary:

```json
{
  "final_step": 6794,
  "tokens_seen": 445251584,
  "wall_clock_seconds": 3528.349571943283,
  "device": "cuda",
  "dtype": "bf16",
  "parameter_count": 119948544,
  "tokens_per_update": 65536,
  "train_rows": 434846,
  "valid_rows": 4374
}
```

Final log row:

```json
{"step": 6794, "train_loss": 4.449406623840332, "eval_loss": 4.370832562881665, "learning_rate": 0.0, "tokens_seen": 445251584}
```

Final checkpoint:

```bash
checkpoint-6794
```

### Diacritic Matched-Token

Output:

```bash
outputs/diacritic_125m_b1024_matched_token_seed42
```

Summary:

```json
{
  "final_step": 6794,
  "tokens_seen": 445251584,
  "wall_clock_seconds": 4538.559595108032,
  "device": "cuda",
  "dtype": "bf16",
  "parameter_count": 119948544,
  "tokens_per_update": 65536,
  "train_rows": 476236,
  "valid_rows": 4798
}
```

Final log row:

```json
{"step": 6794, "train_loss": 4.291114926338196, "eval_loss": 4.264621394872665, "learning_rate": 0.0, "tokens_seen": 445251584}
```

Final checkpoint:

```bash
checkpoint-6794
```

### Diacritic Matched-Content

Output:

```bash
outputs/diacritic_125m_b1024_matched_content_seed42
```

Summary:

```json
{
  "final_step": 7441,
  "tokens_seen": 487653376,
  "wall_clock_seconds": 4966.707597732544,
  "device": "cuda",
  "dtype": "bf16",
  "parameter_count": 119948544,
  "tokens_per_update": 65536,
  "train_rows": 476236,
  "valid_rows": 4798
}
```

Final log row:

```json
{"step": 7441, "train_loss": 4.202487528324127, "eval_loss": 4.210126048326492, "learning_rate": 0.0, "tokens_seen": 487653376}
```

Final checkpoint:

```bash
checkpoint-7441
```

## Instance Destruction Safety Check

Before destroying the Vast instance, local checks were performed:

- all three archives exist under `server_outputs`
- each archive is readable with `tar -tzf`
- each archive contains the final checkpoint
- each final checkpoint contains expected model/training state files
- each archive contains `run_summary.json`

The server was also checked after completion:

- no `tmux` training session
- GPU utilization `0%`
- GPU memory `0 MiB`

Codex did not run any Vast stop/destroy command. Instance stop/destroy is manual.

## Paper Notes

This stage validates and produces the first 125M-class Experiment 2 pretraining outputs:

- same architecture for Chinese-Origin and Pinyin-Diacritic
- same tokenizer size after EOS addition: `32001`
- same EOS/pad id: `32000`
- same block size: `1024`
- same optimizer/scheduler/seed
- Chinese and matched-token Diacritic have identical token budgets
- matched-content Diacritic runs for the content-comparable full Diacritic token budget

Useful values for paper methods:

- train/valid/test split: `98:1:1`
- train line count: `1,310,755`
- valid line count: `13,375`
- test line count: `13,375`
- Chinese train tokens after EOS: `445,283,267`
- Diacritic train tokens after EOS: `487,665,889`
- Diacritic token expansion ratio: approximately `1.0952`
- model size: `119,948,544` parameters
- sequence length: `1024`
- effective batch tokens per update: `65,536`
- hardware: 1 x A100 40GB
- precision: bf16
- random seed: `42`

## Next Window Checklist

1. Use this file as the source of truth for Step 5 server-run context.
2. Do not rerun training unless explicitly needed.
3. If analyzing logs, extract `train_log.jsonl` from the local tar archives.
4. If evaluating checkpoints, use the final checkpoints:
   - Chinese: `checkpoint-6794`
   - Diacritic matched-token: `checkpoint-6794`
   - Diacritic matched-content: `checkpoint-7441`
5. If a new server run is needed, repeat the packaging/upload/extract/verify/smoke/tmux workflow above.

## 2026-05-05 Strict 4epoch Rerun Preparation

Status: local code and server bundles are prepared; server upload/run is pending.

Purpose:

- rerun Chinese-Origin and Pinyin-Diacritic matched-token only
- strict 4x Chinese one-epoch token budget
- use a 12-layer model instead of the previous 10-layer model
- reduce checkpoint/eval frequency to avoid excessive saving

New/updated local files:

```bash
configs/step5_chinese_125m_b1024_4epoch.yaml
configs/step5_diacritic_125m_b1024_matched_token_4epoch.yaml
scripts/step5_run_chinese_125m_4epoch.sh
scripts/step5_run_diacritic_125m_matched_token_4epoch.sh
```

The smoke scripts now use the 4epoch configs with `--max_steps 20`, so smoke tests validate the 12-layer architecture:

```bash
scripts/step5_run_chinese_125m_smoke.sh
scripts/step5_run_diacritic_125m_smoke.sh
```

Formal 4epoch settings:

```yaml
num_hidden_layers: 12
max_steps: 27176
per_device_train_batch_size: 16
gradient_accumulation_steps: 4
max_position_embeddings: 1024
tokens_per_update: 65536
eval_steps: 1000
save_steps: 1000
logging_steps: 20
save_total_limit: 3
seed: 42
```

Expected final seen-token budget:

```text
27176 * 65536 = 1,781,006,336 tokens_seen
```

This is exactly `4 * 6794` optimizer updates. Relative to the previous Chinese effective one-epoch budget, this is strict 4epoch. For Diacritic it is matched-token, not matched-content.

The training loop in `step5_train_lm_formal.py` was updated for multi-epoch training:

- removed `itertools.cycle(train_loader)`
- iterates over fresh DataLoader passes so shuffled order is refreshed each pass
- drops incomplete gradient-accumulation windows at the end of each pass
- records `train_pass` in `train_log.jsonl`
- records `updates_per_train_pass` and `tokens_per_train_pass` in run config/summary

Prepared local server bundle archives:

```bash
server_bundle_exp2_chinese_125m_b1024.tar.gz
server_bundle_exp2_diacritic_125m_b1024.tar.gz
```

Archive sizes after repackaging:

```text
Chinese bundle: 753M
Diacritic bundle: 833M
```

When the server is ready, upload these two archives, extract them, create/activate the Python environment, run bundle verification, run smoke tests, then start the two formal tmux jobs:

```bash
cd /workspace/Chinese_Latinization_NLP/server_bundle_exp2_chinese_125m_b1024
. .venv/bin/activate
bash scripts/step5_verify_server_bundle.sh
bash scripts/step5_run_chinese_125m_smoke.sh
tmux new -d -s chinese_4epoch \
  "cd /workspace/Chinese_Latinization_NLP/server_bundle_exp2_chinese_125m_b1024 && . .venv/bin/activate && bash scripts/step5_run_chinese_125m_4epoch.sh 2>&1 | tee outputs/chinese_125m_b1024_4epoch_seed42.log"
```

```bash
cd /workspace/Chinese_Latinization_NLP/server_bundle_exp2_diacritic_125m_b1024
. .venv/bin/activate
bash scripts/step5_verify_diacritic_server_bundle.sh
bash scripts/step5_run_diacritic_125m_smoke.sh
tmux new -d -s diacritic_matched_token_4epoch \
  "cd /workspace/Chinese_Latinization_NLP/server_bundle_exp2_diacritic_125m_b1024 && . .venv/bin/activate && bash scripts/step5_run_diacritic_125m_matched_token_4epoch.sh 2>&1 | tee outputs/diacritic_125m_b1024_matched_token_4epoch_seed42.log"
```

## 2026-05-06 Strict 4epoch A100 Results

Status: Chinese-Origin and Pinyin-Diacritic matched-token strict 4epoch reruns completed, compressed, downloaded locally, and verified with `tar -tzf`.

Server:

```bash
ssh -i ~/.ssh/vast_ai_nopass -o IdentitiesOnly=yes -p 1581 root@199.126.203.145
```

Observed hardware/software:

- GPU: `NVIDIA A100-SXM4-40GB`
- precision: bf16
- torch: `2.11.0+cu130`
- CUDA available: yes
- bf16 supported: yes
- transformers: `5.8.0`
- datasets: `4.8.5`
- tokenizers: `0.22.2`

Completed 12-layer strict 4epoch runs:

| Variant | Output dir in archive | Final step | Tokens seen | Final eval loss | Wall clock |
|---|---|---:|---:|---:|---:|
| Chinese-Origin | `outputs/chinese_125m_b1024_4epoch_seed42` | 27176 | 1,781,006,336 | 3.779643 | 19262.25 s |
| Pinyin-Diacritic matched-token | `outputs/diacritic_125m_b1024_matched_token_4epoch_seed42` | 27176 | 1,781,006,336 | 3.660185 | 19315.62 s |

Model/training settings used:

```yaml
num_hidden_layers: 12
parameter_count: 134107392
max_steps: 27176
tokens_per_update: 65536
eval_steps: 1000
save_steps: 1000
logging_steps: 20
save_total_limit: 3
```

Local downloaded archives:

```bash
server_outputs/chinese_125m_b1024_4epoch_seed42_outputs.tar.gz
server_outputs/diacritic_125m_b1024_matched_token_4epoch_seed42_outputs.tar.gz
```

Each archive was locally verified with `tar -tzf` and contains:

- final checkpoint `checkpoint-27176`
- `model.safetensors`
- `optimizer.pt`
- `scheduler.pt`
- `trainer_state.json`
- `training_args.json`
- `rng_state.pth`
- `config.json`
- `generation_config.json`
- `train_log.jsonl`
- `run_summary.json`
- outer `.log`

Final retained checkpoints:

- Chinese-Origin: `checkpoint-26000`, `checkpoint-27000`, `checkpoint-27176`
- Pinyin-Diacritic matched-token: `checkpoint-26000`, `checkpoint-27000`, `checkpoint-27176`

Server archives also remained on the instance after local verification:

```bash
/workspace/chinese_125m_b1024_4epoch_seed42_outputs.tar.gz
/workspace/diacritic_125m_b1024_matched_token_4epoch_seed42_outputs.tar.gz
```
