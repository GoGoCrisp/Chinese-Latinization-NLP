#!/usr/bin/env python3
# Step 3 program: local debug training for a tiny causal LM.
"""Local debug training for a tiny Llama causal LM."""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tiny causal LM debug model.")
    parser.add_argument("--tokenizer", required=True, help="Local tokenizer directory.")
    parser.add_argument("--train_data", required=True, help="HF Dataset train path.")
    parser.add_argument("--valid_data", required=True, help="HF Dataset validation path.")
    parser.add_argument("--output_dir", required=True, help="Output directory.")
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=None)
    parser.add_argument("--dataloader_drop_last", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--eval_max_batches", type=int, default=None)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--intermediate_size", type=int, default=1024)
    parser.add_argument("--num_hidden_layers", type=int, default=4)
    parser.add_argument("--num_attention_heads", type=int, default=4)
    parser.add_argument("--max_position_embeddings", type=int, default=512)
    parser.add_argument("--rms_norm_eps", type=float, default=1e-5)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def resolve_path(path: str) -> Path:
    return Path(path).expanduser()


def load_tokenizer(tokenizer_dir: Path) -> PreTrainedTokenizerFast:
    tokenizer_json = tokenizer_dir / "tokenizer.json"
    if not tokenizer_json.exists():
        raise FileNotFoundError(f"Expected tokenizer.json in {tokenizer_dir}")
    return PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_json))


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def collate_lm(batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
    input_ids = torch.tensor([example["input_ids"] for example in batch], dtype=torch.long)
    return {"input_ids": input_ids, "labels": input_ids.clone()}


def make_model_config(tokenizer, args: argparse.Namespace) -> LlamaConfig:
    return LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        max_position_embeddings=args.max_position_embeddings,
        rms_norm_eps=args.rms_norm_eps,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        tie_word_embeddings=True,
    )


def count_parameters(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_checkpoint(
    checkpoint_dir: Path,
    model: LlamaForCausalLM,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    tokens_seen: int,
    args: argparse.Namespace,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "global_step": global_step,
            "tokens_seen": tokens_seen,
            "args": vars(args),
        },
        checkpoint_dir / "training_state.pt",
    )


def enforce_save_total_limit(output_dir: Path, save_total_limit: int | None) -> None:
    if save_total_limit is None or save_total_limit <= 0:
        return

    checkpoints = []
    for path in output_dir.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        try:
            step = int(path.name.split("-")[-1])
        except ValueError:
            continue
        checkpoints.append((step, path))

    checkpoints.sort()
    excess = len(checkpoints) - save_total_limit
    if excess <= 0:
        return

    for _, path in checkpoints[:excess]:
        shutil.rmtree(path)


def load_checkpoint(
    checkpoint_dir: Path,
    model: LlamaForCausalLM,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, int]:
    model_state_path = checkpoint_dir
    training_state_path = checkpoint_dir / "training_state.pt"
    if not training_state_path.exists():
        raise FileNotFoundError(f"Missing training state: {training_state_path}")

    loaded = LlamaForCausalLM.from_pretrained(model_state_path)
    model.load_state_dict(loaded.state_dict())
    state = torch.load(training_state_path, map_location=device)
    optimizer.load_state_dict(state["optimizer"])
    return int(state["global_step"]), int(state["tokens_seen"])


@torch.no_grad()
def evaluate(
    model: LlamaForCausalLM,
    valid_loader: DataLoader,
    device: torch.device,
    max_batches: int | None,
) -> float:
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


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    start_time = time.time()
    args = parse_args()
    if args.max_steps <= 0:
        raise ValueError("--max_steps must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    if args.grad_accum <= 0:
        raise ValueError("--grad_accum must be positive")
    if args.eval_steps <= 0:
        raise ValueError("--eval_steps must be positive")
    if args.save_steps <= 0:
        raise ValueError("--save_steps must be positive")

    set_seed(args.seed)
    tokenizer_dir = resolve_path(args.tokenizer)
    train_data_path = resolve_path(args.train_data)
    valid_data_path = resolve_path(args.valid_data)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(tokenizer_dir)
    train_dataset = load_from_disk(str(train_data_path))
    valid_dataset = load_from_disk(str(valid_data_path))
    if "input_ids" not in train_dataset.column_names:
        raise ValueError(f"train dataset has no input_ids column: {train_dataset.column_names}")
    if "input_ids" not in valid_dataset.column_names:
        raise ValueError(f"valid dataset has no input_ids column: {valid_dataset.column_names}")

    device = choose_device()
    config = make_model_config(tokenizer, args)
    model = LlamaForCausalLM(config).to(device=device, dtype=torch.float32)
    param_count = count_parameters(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_lm,
        drop_last=args.dataloader_drop_last,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_lm,
        drop_last=False,
    )

    global_step = 0
    tokens_seen = 0
    if args.resume_from_checkpoint:
        global_step, tokens_seen = load_checkpoint(
            resolve_path(args.resume_from_checkpoint), model, optimizer, device
        )
        model.to(device=device, dtype=torch.float32)

    save_json(output_dir / "training_args.json", vars(args))
    config.save_pretrained(output_dir)

    print("Training setup")
    print(f"  device: {device.type}")
    print(f"  dtype: fp32")
    print(f"  tokenizer_vocab_size: {len(tokenizer)}")
    print(f"  train_rows: {len(train_dataset)}")
    print(f"  valid_rows: {len(valid_dataset)}")
    print(f"  parameters: {param_count}")
    print(f"  starting_step: {global_step}")
    print(f"  output_dir: {output_dir}")

    log_path = output_dir / "train_log.jsonl"
    model.train()
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    accum_count = 0

    progress = tqdm(total=args.max_steps, initial=global_step, desc="steps")
    for batch in cycle(train_loader):
        if global_step >= args.max_steps:
            break

        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss / args.grad_accum
        loss.backward()

        batch_tokens = int(batch["input_ids"].numel())
        tokens_seen += batch_tokens
        running_loss += float(outputs.loss.detach().cpu())
        accum_count += 1

        if accum_count % args.grad_accum != 0:
            continue

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1
        progress.update(1)

        train_loss = running_loss / args.grad_accum
        running_loss = 0.0
        accum_count = 0

        eval_loss = None
        if global_step % args.eval_steps == 0 or global_step == args.max_steps:
            eval_loss = evaluate(model, valid_loader, device, args.eval_max_batches)

        row = {
            "step": global_step,
            "train_loss": train_loss,
            "eval_loss": eval_loss,
            "learning_rate": get_lr(optimizer),
            "tokens_seen": tokens_seen,
        }
        append_jsonl(log_path, row)
        if eval_loss is None:
            print(
                f"step={global_step} train_loss={train_loss:.6f} "
                f"lr={get_lr(optimizer):.6g} tokens_seen={tokens_seen}"
            )
        else:
            print(
                f"step={global_step} train_loss={train_loss:.6f} "
                f"eval_loss={eval_loss:.6f} lr={get_lr(optimizer):.6g} "
                f"tokens_seen={tokens_seen}"
            )

        if global_step % args.save_steps == 0 or global_step == args.max_steps:
            save_checkpoint(
                output_dir / f"checkpoint-{global_step}",
                model,
                optimizer,
                global_step,
                tokens_seen,
                args,
            )
            enforce_save_total_limit(output_dir, args.save_total_limit)

    progress.close()
    wall_clock_seconds = time.time() - start_time
    save_json(
        output_dir / "run_summary.json",
        {
            "final_step": global_step,
            "tokens_seen": tokens_seen,
            "wall_clock_seconds": wall_clock_seconds,
            "device": device.type,
            "parameter_count": param_count,
            "train_rows": len(train_dataset),
            "valid_rows": len(valid_dataset),
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
