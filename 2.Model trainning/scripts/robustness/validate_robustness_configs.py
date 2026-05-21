#!/usr/bin/env python3
"""Validate offline configs for the matched-data Pinyin-Diacritic extra seeds."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs" / "robustness"
EXPECTED_PARAMETER_COUNT = 134_107_392
EXPECTED_TOKENS_PER_UPDATE = 65_536
EXPECTED_TOKENIZER = {
    "expected_vocab_size": 32_001,
    "expected_eos_id": 32_000,
    "expected_pad_id": 32_000,
}
EXPECTED_ARCH = {
    "hidden_size": 768,
    "intermediate_size": 2048,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "max_position_embeddings": 1024,
    "rms_norm_eps": 1.0e-5,
    "hidden_act": "silu",
    "rope_theta": 10000.0,
    "tie_word_embeddings": False,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 16,
    "gradient_accumulation_steps": 4,
    "learning_rate": 3.0e-4,
    "weight_decay": 0.1,
    "adam_beta1": 0.9,
    "adam_beta2": 0.95,
    "adam_epsilon": 1.0e-8,
    "scheduler": "cosine",
    "warmup_ratio": 0.03,
    "max_grad_norm": 1.0,
    "eval_steps": 1000,
    "save_steps": 1000,
    "logging_steps": 20,
    "save_total_limit": 3,
    "dataloader_drop_last": True,
}

EXPECTED_CONFIGS = {
    "diacritic_125m_b1024_matched_data_4epoch_seed43.yaml": {
        "seed": 43,
        "max_steps": 29764,
        "tokenizer": "tokenizers/pinyin_diacritic_32k_eos",
        "train_data": "data/tokenized/diacritic_train_full_eos_1024",
        "valid_data": "data/tokenized/diacritic_valid_full_eos_1024",
        "output_dir": "outputs/diacritic_125m_b1024_matched_data_4epoch_seed43",
    },
    "diacritic_125m_b1024_matched_data_4epoch_seed44.yaml": {
        "seed": 44,
        "max_steps": 29764,
        "tokenizer": "tokenizers/pinyin_diacritic_32k_eos",
        "train_data": "data/tokenized/diacritic_train_full_eos_1024",
        "valid_data": "data/tokenized/diacritic_valid_full_eos_1024",
        "output_dir": "outputs/diacritic_125m_b1024_matched_data_4epoch_seed44",
    },
}


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value.strip("\"'")


def load_yaml_simple(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError:
        payload: dict[str, Any] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            clean = line.split("#", 1)[0].strip()
            if not clean or ":" not in clean:
                continue
            key, value = clean.split(":", 1)
            payload[key.strip()] = parse_scalar(value)
        return payload
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data or {}


def rel(path_text: str) -> Path:
    return ROOT / path_text


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def check_tokenizer(tokenizer_dir: Path, errors: list[str], warnings: list[str]) -> None:
    if not tokenizer_dir.exists():
        errors.append(f"missing tokenizer directory: {tokenizer_dir.relative_to(ROOT)}")
        return
    for name in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "eos_update_report.json"]:
        if not (tokenizer_dir / name).exists():
            errors.append(f"missing tokenizer file: {(tokenizer_dir / name).relative_to(ROOT)}")
    report_path = tokenizer_dir / "eos_update_report.json"
    if report_path.exists():
        report = read_json(report_path)
        checks = {
            "new_vocab_size": EXPECTED_TOKENIZER["expected_vocab_size"],
            "eos_token_id": EXPECTED_TOKENIZER["expected_eos_id"],
            "pad_token_id": EXPECTED_TOKENIZER["expected_pad_id"],
        }
        for key, expected in checks.items():
            if report.get(key) != expected:
                errors.append(
                    f"{report_path.relative_to(ROOT)} {key}={report.get(key)!r}, expected {expected!r}"
                )
    else:
        warnings.append(f"cannot verify tokenizer ids without {report_path.relative_to(ROOT)}")


def almost_equal(left: Any, right: Any) -> bool:
    if isinstance(right, float):
        try:
            return abs(float(left) - right) < 1e-12
        except (TypeError, ValueError):
            return False
    return left == right


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []
    rows: list[dict[str, Any]] = []

    for filename, expected in EXPECTED_CONFIGS.items():
        path = CONFIG_DIR / filename
        if not path.exists():
            errors.append(f"missing config: {path.relative_to(ROOT)}")
            continue
        cfg = load_yaml_simple(path)

        for key, expected_value in {**expected, **EXPECTED_TOKENIZER, **EXPECTED_ARCH}.items():
            if key not in cfg:
                errors.append(f"{path.relative_to(ROOT)} missing {key}")
            elif not almost_equal(cfg[key], expected_value):
                errors.append(
                    f"{path.relative_to(ROOT)} {key}={cfg[key]!r}, expected {expected_value!r}"
                )

        for key in ["tokenizer", "train_data", "valid_data"]:
            if key in cfg and not rel(str(cfg[key])).exists():
                errors.append(f"{path.relative_to(ROOT)} {key} path missing: {cfg[key]}")

        if "output_dir" in cfg and rel(str(cfg["output_dir"])).exists():
            warnings.append(f"output_dir already exists: {cfg['output_dir']}")

        if {"per_device_train_batch_size", "gradient_accumulation_steps", "max_position_embeddings"} <= set(cfg):
            tokens_per_update = (
                int(cfg["per_device_train_batch_size"])
                * int(cfg["gradient_accumulation_steps"])
                * int(cfg["max_position_embeddings"])
            )
            if tokens_per_update != EXPECTED_TOKENS_PER_UPDATE:
                errors.append(
                    f"{path.relative_to(ROOT)} tokens_per_update={tokens_per_update}, "
                    f"expected {EXPECTED_TOKENS_PER_UPDATE}"
                )
        else:
            tokens_per_update = None

        if "tokenizer" in cfg:
            check_tokenizer(rel(str(cfg["tokenizer"])), errors, warnings)

        rows.append(
            {
                "config": str(path.relative_to(ROOT)),
                "seed": cfg.get("seed"),
                "max_steps": cfg.get("max_steps"),
                "tokens_per_update": tokens_per_update,
                "output_dir": cfg.get("output_dir"),
            }
        )

    print("Matched-data extra seed config validation")
    for row in rows:
        print(
            f"  OK candidate: {row['config']} seed={row['seed']} "
            f"max_steps={row['max_steps']} tokens/update={row['tokens_per_update']} "
            f"output={row['output_dir']}"
        )
    print(f"  expected_parameter_count: {EXPECTED_PARAMETER_COUNT}")
    print(f"  warnings: {len(warnings)}")
    for warning in warnings:
        print(f"  WARN: {warning}")
    print(f"  errors: {len(errors)}")
    for error in errors:
        print(f"  ERROR: {error}")

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
