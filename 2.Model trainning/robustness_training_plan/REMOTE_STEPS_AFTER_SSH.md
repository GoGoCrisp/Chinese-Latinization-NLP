# Remote Steps After SSH: Matched-Data Extra Seeds

This checklist is for the new instance that will train only:

1. `diacritic_125m_b1024_matched_data_4epoch_seed43`
2. `diacritic_125m_b1024_matched_data_4epoch_seed44`

Both runs use the Pinyin-Diacritic tokenizer, `max_steps=29764`, `tokens_per_update=65536`, and expected `tokens_seen=1950613504`.

## 1. Upload And Extract Bundle

From the local project directory:

```bash
bash scripts/robustness/make_remote_training_bundle.sh --overwrite
scp -P <PORT> robustness_training_plan/robustness_training_bundle.tar.gz root@<HOST>:/workspace/
ssh -p <PORT> root@<HOST>
cd /workspace
mkdir -p Chinese_Latinization_NLP
tar -xzf robustness_training_bundle.tar.gz -C Chinese_Latinization_NLP
cd Chinese_Latinization_NLP/robustness_training_bundle
```

The bundle intentionally contains only the two matched-data configs, the Pinyin-Diacritic tokenizer, the Pinyin-Diacritic tokenized train/valid datasets, training scripts, and robustness helper scripts.

## 2. Mandatory GPU/System Checks

These are run by `remote_preflight_check.sh`, and can also be inspected manually:

```bash
nvidia-smi
df -h .
free -h
lscpu | head
du -sh .
```

The instance may have a 30GB disk. The scripts default to `MIN_FREE_DISK_GB=10`.

## 3. Dependency And CUDA Checks

Use the server image's CUDA/PyTorch environment if it is already available. Do not blindly reinstall `torch`.

The preflight checks:

- `torch.__version__`
- `torch.cuda.is_available()`
- `torch.version.cuda`
- CUDA device name and VRAM
- bf16 support
- bf16 CUDA matrix multiplication smoke test
- imports for `torch`, `transformers`, `tokenizers`, `datasets`, `yaml`, `numpy`, `pandas`, `tqdm`, `pypinyin`, and `jieba`

Missing non-torch packages are installed automatically. Torch is not reinstalled by the script.

```bash
MIN_FREE_DISK_GB=10 bash scripts/robustness/remote_preflight_check.sh
python3 scripts/robustness/validate_robustness_configs.py
```

Resolve any `ERROR` line before training.

## 4. Start tmux

```bash
tmux new -s matched_data_extra
```

## 5. Smoke Test And Full Training

`run_one_robustness_training.sh` runs the mandatory preflight and a 2-step smoke training before the full run. The smoke must show finite train/eval loss, CUDA use, parameter count `134107392`, and checkpoint writing. Smoke output is deleted after passing.

Run the one-at-a-time batch:

```bash
MIN_FREE_DISK_GB=10 bash scripts/robustness/run_batch_matched_data_extra_seeds.sh
```

The batch defaults to one-at-a-time mode. After a run completes, it archives the output, writes sha256, and stops so the tarball can be downloaded and verified locally before the next model.

## 6. Monitoring During Training

During formal training, report:

- elapsed wall time
- current step / total steps
- train/eval loss when available
- GPU utilization, VRAM, temperature, and power
- disk free/used

Manual commands:

```bash
nvidia-smi
df -h .
tail -f outputs/logs/<run_id>.log
tail -f outputs/<run_id>/train_log.jsonl
```

## 7. After Each Model

The script prints and stores:

- final step
- tokens seen
- parameter count
- final train loss
- final eval loss
- unpacked output size
- tarball size
- disk growth before/after tar
- sha256

Download and verify locally:

```bash
scp -P <PORT> root@<HOST>:/workspace/Chinese_Latinization_NLP/robustness_training_bundle/outputs/<run_id>.tar.gz ./server_outputs/robustness/
scp -P <PORT> root@<HOST>:/workspace/Chinese_Latinization_NLP/robustness_training_bundle/outputs/<run_id>.tar.gz.sha256 ./server_outputs/robustness/
cd server_outputs/robustness
shasum -a 256 -c <run_id>.tar.gz.sha256
```

After local verification, mark the remote artifact:

```bash
ssh -p <PORT> root@<HOST>
cd /workspace/Chinese_Latinization_NLP/robustness_training_bundle
mkdir -p outputs/completed_runs
touch outputs/completed_runs/<run_id>.done outputs/<run_id>.tar.gz.local_verified
```

Then rerun the batch script for the next pending model:

```bash
MIN_FREE_DISK_GB=10 bash scripts/robustness/run_batch_matched_data_extra_seeds.sh
```

## 8. Disk Policy

Although the 30GB disk should be enough for two models by manual judgment, the script still refuses to start a full run below `MIN_FREE_DISK_GB=10`.

Remote deletion is not automatic unless `AUTO_DELETE_VERIFIED=1` is set. It is acceptable to keep remote artifacts if disk remains healthy. Only delete remote outputs after the local tarball hash has passed.

## 9. Required Run Order

1. `diacritic_125m_b1024_matched_data_4epoch_seed43`
2. `diacritic_125m_b1024_matched_data_4epoch_seed44`
