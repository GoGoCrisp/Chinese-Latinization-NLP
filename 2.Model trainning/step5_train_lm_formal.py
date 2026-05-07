#!/usr/bin/env python3
# Step 5 program: formal server training for Experiment 2 causal LM runs.
"""Formal Llama-style causal LM training for Experiment 2."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
from transformers.optimization import get_cosine_schedule_with_warmup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a formal Llama-style causal LM from scratch."
    )
    parser.add_argument("--config", default=None, help="Optional YAML or JSON config.")
    parser.add_argument("--tokenizer", default=None, help="Local tokenizer directory.")
    parser.add_argument("--train_data", default=None, help="HF Dataset train path.")
    parser.add_argument("--valid_data", default=None, help="HF Dataset validation path.")
    parser.add_argument("--output_dir", default=None, help="Output directory.")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=None)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--adam_beta1", type=float, default=None)
    parser.add_argument("--adam_beta2", type=float, default=None)
    parser.add_argument("--adam_epsilon", type=float, default=None)
    parser.add_argument("--scheduler", default=None, choices=[None, "cosine"])
    parser.add_argument("--warmup_ratio", type=float, default=None)
    parser.add_argument("--max_grad_norm", type=float, default=None)
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument("--logging_steps", type=int, default=None)
    parser.add_argument("--save_total_limit", type=int, default=None)
    parser.add_argument("--dataloader_drop_last", action="store_true", default=None)
    parser.add_argument("--no_dataloader_drop_last", action="store_false", dest="dataloader_drop_last")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--eval_max_batches", type=int, default=None)
    parser.add_argument(
        "--stop_after_checkpoint_step",
        type=int,
        default=None,
        help=(
            "Exit cleanly immediately after saving this checkpoint step. "
            "The scheduler still uses --max_steps for its full training budget."
        ),
    )
    parser.add_argument("--dry_run", action="store_true", help="Validate setup and exit.")

    parser.add_argument("--hidden_size", type=int, default=None)
    parser.add_argument("--intermediate_size", type=int, default=None)
    parser.add_argument("--num_hidden_layers", type=int, default=None)
    parser.add_argument("--num_attention_heads", type=int, default=None)
    parser.add_argument("--max_position_embeddings", type=int, default=None)
    parser.add_argument("--rms_norm_eps", type=float, default=None)
    parser.add_argument("--hidden_act", default=None)
    parser.add_argument("--rope_theta", type=float, default=None)
    parser.add_argument("--tie_word_embeddings", action="store_true", default=None)
    parser.add_argument("--no_tie_word_embeddings", action="store_false", dest="tie_word_embeddings")
    parser.add_argument("--expected_vocab_size", type=int, default=None)
    parser.add_argument("--expected_eos_id", type=int, default=None)
    parser.add_argument("--expected_pad_id", type=int, default=None)
    return parser.parse_args()


DEFAULTS: dict[str, Any] = {
    "max_steps": 20,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 16,
    "gradient_accumulation_steps": 4,
    "learning_rate": 3e-4,
    "weight_decay": 0.1,
    "adam_beta1": 0.9,
    "adam_beta2": 0.95,
    "adam_epsilon": 1e-8,
    "scheduler": "cosine",
    "warmup_ratio": 0.03,
    "max_grad_norm": 1.0,
    "eval_steps": 500,
    "save_steps": 500,
    "logging_steps": 10,
    "save_total_limit": 3,
    "dataloader_drop_last": True,
    "seed": 42,
    "resume_from_checkpoint": None,
    "eval_max_batches": None,
    "stop_after_checkpoint_step": None,
    "dry_run": False,
    "hidden_size": 768,
    "intermediate_size": 2048,
    "num_hidden_layers": 10,
    "num_attention_heads": 12,
    "max_position_embeddings": 1024,
    "rms_norm_eps": 1e-5,
    "hidden_act": "silu",
    "rope_theta": 10000.0,
    "tie_word_embeddings": False,
    "expected_vocab_size": 32001,
    "expected_eos_id": 32000,
    "expected_pad_id": 32000,
}


def load_config(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    config_path = Path(path).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Config does not exist: {config_path}")
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to load YAML configs: pip install pyyaml") from exc
    payload = yaml.safe_load(text)
    return payload or {}


def merge_args(args: argparse.Namespace) -> argparse.Namespace:
    config = load_config(args.config)
    merged = dict(DEFAULTS)
    merged.update(config)
    for key, value in vars(args).items():
        if key == "config":
            merged[key] = value
        elif value is not None:
            merged[key] = value
    return argparse.Namespace(**merged)


def resolve_path(path: str | None, label: str) -> Path:
    if not path:
        raise ValueError(f"--{label} is required, either on CLI or in config")
    return Path(path).expanduser()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_tokenizer(tokenizer_dir: Path) -> PreTrainedTokenizerFast:
    tokenizer_json = tokenizer_dir / "tokenizer.json"
    if not tokenizer_json.exists():
        raise FileNotFoundError(f"Expected tokenizer.json in {tokenizer_dir}")
    return PreTrainedTokenizerFast.from_pretrained(str(tokenizer_dir), local_files_only=True)


def validate_tokenizer(tokenizer: PreTrainedTokenizerFast, args: argparse.Namespace) -> None:
    if len(tokenizer) != args.expected_vocab_size:
        raise ValueError(f"Tokenizer vocab size must be {args.expected_vocab_size}; got {len(tokenizer)}")
    if tokenizer.eos_token_id != args.expected_eos_id:
        raise ValueError(f"Tokenizer eos_token_id must be {args.expected_eos_id}; got {tokenizer.eos_token_id}")
    if tokenizer.pad_token_id != args.expected_pad_id:
        raise ValueError(f"Tokenizer pad_token_id must be {args.expected_pad_id}; got {tokenizer.pad_token_id}")


def choose_device_and_dtype() -> tuple[torch.device, torch.dtype, str]:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return torch.device("cuda"), torch.bfloat16, "bf16"
    return torch.device("cpu"), torch.float32, "fp32"


def print_device_info(device: torch.device, dtype_name: str) -> None:
    print("Device")
    print(f"  device: {device.type}")
    print(f"  dtype: {dtype_name}")
    if device.type == "cuda":
        index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        print(f"  gpu_name: {props.name}")
        print(f"  gpu_memory_gb: {props.total_memory / 1024**3:.2f}")
        print(f"  cuda_version: {torch.version.cuda}")
        print("  tf32: true")


def collate_lm(batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
    input_ids = torch.tensor([example["input_ids"] for example in batch], dtype=torch.long)
    return {"input_ids": input_ids, "labels": input_ids.clone()}


def validate_dataset(dataset, label: str, block_size: int) -> None:
    if "input_ids" not in dataset.column_names:
        raise ValueError(f"{label} dataset has no input_ids column: {dataset.column_names}")
    if len(dataset) == 0:
        raise ValueError(f"{label} dataset is empty")
    first_len = len(dataset[0]["input_ids"])
    if first_len != block_size:
        raise ValueError(
            f"{label} dataset block length is {first_len}, but model max_position_embeddings is {block_size}"
        )


def make_model_config(tokenizer: PreTrainedTokenizerFast, args: argparse.Namespace) -> LlamaConfig:
    return LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        max_position_embeddings=args.max_position_embeddings,
        rms_norm_eps=args.rms_norm_eps,
        hidden_act=args.hidden_act,
        rope_theta=args.rope_theta,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        tie_word_embeddings=args.tie_word_embeddings,
    )


def count_parameters(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def rng_state_payload() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return payload


def save_checkpoint(
    checkpoint_dir: Path,
    model: LlamaForCausalLM,
    optimizer: torch.optim.Optimizer,
    scheduler,
    global_step: int,
    tokens_seen: int,
    args: argparse.Namespace,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir, safe_serialization=True)
    torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
    torch.save(scheduler.state_dict(), checkpoint_dir / "scheduler.pt")
    torch.save(rng_state_payload(), checkpoint_dir / "rng_state.pth")
    save_json(checkpoint_dir / "trainer_state.json", {
        "global_step": global_step,
        "tokens_seen": tokens_seen,
        "max_steps": args.max_steps,
    })
    save_json(checkpoint_dir / "training_args.json", vars(args))


def enforce_save_total_limit(output_dir: Path, save_total_limit: int | None) -> None:
    if save_total_limit is None or save_total_limit <= 0:
        return
    checkpoints = []
    for path in output_dir.glob("checkpoint-*"):
        if path.is_dir():
            try:
                checkpoints.append((int(path.name.split("-")[-1]), path))
            except ValueError:
                pass
    checkpoints.sort()
    for _, path in checkpoints[: max(0, len(checkpoints) - save_total_limit)]:
        shutil.rmtree(path)


def load_checkpoint(
    checkpoint_dir: Path,
    model: LlamaForCausalLM,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
) -> tuple[int, int]:
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_dir}")
    loaded = LlamaForCausalLM.from_pretrained(str(checkpoint_dir))
    model.load_state_dict(loaded.state_dict())
    optimizer_path = checkpoint_dir / "optimizer.pt"
    scheduler_path = checkpoint_dir / "scheduler.pt"
    state_path = checkpoint_dir / "trainer_state.json"
    if not optimizer_path.exists() or not scheduler_path.exists() or not state_path.exists():
        raise FileNotFoundError(f"Checkpoint is incomplete: {checkpoint_dir}")
    optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))
    scheduler.load_state_dict(torch.load(scheduler_path, map_location=device))
    state = json.loads(state_path.read_text(encoding="utf-8"))
    return int(state["global_step"]), int(state["tokens_seen"])


@torch.no_grad()
def evaluate(model: LlamaForCausalLM, valid_loader: DataLoader, device: torch.device, max_batches: int | None) -> float:
    model.eval()
    losses: list[float] = []
    for batch_idx, batch in enumerate(valid_loader, start=1):
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(**batch)
        losses.append(float(outputs.loss.detach().cpu()))
        if max_batches is not None and batch_idx >= max_batches:
            break
    model.train()
    return float(np.mean(losses)) if losses else math.nan


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    start_time = time.time()
    raw_args = parse_args()
    args = merge_args(raw_args)

    for key in ["max_steps", "per_device_train_batch_size", "gradient_accumulation_steps", "eval_steps", "save_steps", "logging_steps"]:
        if getattr(args, key) <= 0:
            raise ValueError(f"{key} must be positive")

    set_seed(args.seed)
    tokenizer_dir = resolve_path(args.tokenizer, "tokenizer")
    train_data_path = resolve_path(args.train_data, "train_data")
    valid_data_path = resolve_path(args.valid_data, "valid_data")
    output_dir = resolve_path(args.output_dir, "output_dir")
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(tokenizer_dir)
    validate_tokenizer(tokenizer, args)
    train_dataset = load_from_disk(str(train_data_path))
    valid_dataset = load_from_disk(str(valid_data_path))
    validate_dataset(train_dataset, "train", args.max_position_embeddings)
    validate_dataset(valid_dataset, "valid", args.max_position_embeddings)

    device, dtype, dtype_name = choose_device_and_dtype()
    print_device_info(device, dtype_name)
    config = make_model_config(tokenizer, args)
    model = LlamaForCausalLM(config).to(device=device, dtype=dtype)
    param_count = count_parameters(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
        weight_decay=args.weight_decay,
    )
    warmup_steps = int(args.max_steps * args.warmup_ratio)
    if args.scheduler != "cosine":
        raise ValueError(f"Unsupported scheduler: {args.scheduler}")
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=args.max_steps,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collate_lm,
        drop_last=args.dataloader_drop_last,
        num_workers=2 if device.type == "cuda" else 0,
        pin_memory=device.type == "cuda",
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=collate_lm,
        drop_last=False,
        num_workers=2 if device.type == "cuda" else 0,
        pin_memory=device.type == "cuda",
    )
    train_batches_per_pass = len(train_loader)
    usable_train_batches_per_pass = (
        train_batches_per_pass // args.gradient_accumulation_steps
    ) * args.gradient_accumulation_steps
    updates_per_train_pass = usable_train_batches_per_pass // args.gradient_accumulation_steps
    if updates_per_train_pass <= 0:
        raise ValueError(
            "Training loader must provide at least one full gradient accumulation "
            f"window; got {train_batches_per_pass} batches and "
            f"gradient_accumulation_steps={args.gradient_accumulation_steps}"
        )

    global_step = 0
    tokens_seen = 0
    if args.resume_from_checkpoint:
        global_step, tokens_seen = load_checkpoint(
            Path(args.resume_from_checkpoint).expanduser(), model, optimizer, scheduler, device
        )
        model.to(device=device, dtype=dtype)

    tokens_per_update = (
        args.per_device_train_batch_size
        * args.gradient_accumulation_steps
        * args.max_position_embeddings
    )
    run_info = {
        "args": vars(args),
        "parameter_count": param_count,
        "tokens_per_update": tokens_per_update,
        "train_rows": len(train_dataset),
        "valid_rows": len(valid_dataset),
        "train_batches_per_pass": train_batches_per_pass,
        "usable_train_batches_per_pass": usable_train_batches_per_pass,
        "updates_per_train_pass": updates_per_train_pass,
        "tokens_per_train_pass": updates_per_train_pass * tokens_per_update,
        "device": device.type,
        "dtype": dtype_name,
        "warmup_steps": warmup_steps,
    }
    save_json(output_dir / "training_args.json", vars(args))
    save_json(output_dir / "run_config_resolved.json", run_info)
    config.save_pretrained(output_dir)

    print("Training setup")
    print(f"  tokenizer_vocab_size: {len(tokenizer)}")
    print(f"  eos_token_id: {tokenizer.eos_token_id}")
    print(f"  pad_token_id: {tokenizer.pad_token_id}")
    print(f"  train_rows: {len(train_dataset)}")
    print(f"  valid_rows: {len(valid_dataset)}")
    print(f"  parameters: {param_count}")
    print(f"  max_position_embeddings: {args.max_position_embeddings}")
    print(f"  tokens_per_update: {tokens_per_update}")
    print(f"  train_batches_per_pass: {train_batches_per_pass}")
    print(f"  usable_train_batches_per_pass: {usable_train_batches_per_pass}")
    print(f"  updates_per_train_pass: {updates_per_train_pass}")
    print(f"  tokens_per_train_pass: {updates_per_train_pass * tokens_per_update}")
    print(f"  warmup_steps: {warmup_steps}")
    print(f"  starting_step: {global_step}")
    print(f"  output_dir: {output_dir}")

    if args.dry_run:
        print("Dry run complete; no training started.")
        return 0

    log_path = output_dir / "train_log.jsonl"
    model.train()
    optimizer.zero_grad(set_to_none=True)

    progress = tqdm(total=args.max_steps, initial=global_step, desc="steps")
    stop_requested = False
    train_pass = 0
    while global_step < args.max_steps and not stop_requested:
        train_pass += 1
        running_loss = 0.0
        accum_count = 0
        optimizer.zero_grad(set_to_none=True)

        for batch_index, batch in enumerate(train_loader):
            if batch_index >= usable_train_batches_per_pass:
                break
            if global_step >= args.max_steps or stop_requested:
                break

            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()

            batch_tokens = int(batch["input_ids"].numel())
            tokens_seen += batch_tokens
            running_loss += float(outputs.loss.detach().cpu())
            accum_count += 1

            if accum_count % args.gradient_accumulation_steps != 0:
                continue

            if args.max_grad_norm is not None and args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            progress.update(1)

            train_loss = running_loss / args.gradient_accumulation_steps
            running_loss = 0.0
            accum_count = 0

            eval_loss = None
            if global_step % args.eval_steps == 0 or global_step == args.max_steps:
                eval_loss = evaluate(model, valid_loader, device, args.eval_max_batches)

            row = {
                "step": global_step,
                "train_pass": train_pass,
                "train_loss": train_loss,
                "eval_loss": eval_loss,
                "learning_rate": get_lr(optimizer),
                "tokens_seen": tokens_seen,
            }
            append_jsonl(log_path, row)
            if global_step % args.logging_steps == 0 or eval_loss is not None or global_step <= 5:
                message = (
                    f"step={global_step} train_pass={train_pass} "
                    f"train_loss={train_loss:.6f} "
                    f"lr={get_lr(optimizer):.6g} tokens_seen={tokens_seen}"
                )
                if eval_loss is not None:
                    message += f" eval_loss={eval_loss:.6f}"
                print(message)

            if global_step % args.save_steps == 0 or global_step == args.max_steps:
                save_checkpoint(
                    output_dir / f"checkpoint-{global_step}",
                    model,
                    optimizer,
                    scheduler,
                    global_step,
                    tokens_seen,
                    args,
                )
                enforce_save_total_limit(output_dir, args.save_total_limit)
                if args.stop_after_checkpoint_step is not None and global_step >= args.stop_after_checkpoint_step:
                    print(
                        "Stopping after requested checkpoint "
                        f"checkpoint-{global_step} was saved."
                    )
                    stop_requested = True
                    break

        if accum_count:
            optimizer.zero_grad(set_to_none=True)

    progress.close()
    wall_clock_seconds = time.time() - start_time
    save_json(
        output_dir / "run_summary.json",
        {
            "final_step": global_step,
            "tokens_seen": tokens_seen,
            "wall_clock_seconds": wall_clock_seconds,
            "device": device.type,
            "dtype": dtype_name,
            "parameter_count": param_count,
            "tokens_per_update": tokens_per_update,
            "train_rows": len(train_dataset),
            "valid_rows": len(valid_dataset),
            "train_passes_started": train_pass,
            "updates_per_train_pass": updates_per_train_pass,
            "tokens_per_train_pass": updates_per_train_pass * tokens_per_update,
        },
    )
    print("Training complete")
    print(f"  final_step: {global_step}")
    print(f"  tokens_seen: {tokens_seen}")
    print(f"  wall_clock_seconds: {wall_clock_seconds:.2f}")
    print(f"  log: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
