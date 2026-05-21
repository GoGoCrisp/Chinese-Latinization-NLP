#!/usr/bin/env python3
"""Prepare robustness checkpoints for Eval 1/2/4 without running model scoring."""

from __future__ import annotations

import argparse
import json
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TAR_DIR = ROOT / "server_outputs" / "robustness"
DEFAULT_UNPACK_DIR = DEFAULT_TAR_DIR / "unpacked"
DEFAULT_MANIFEST = ROOT / "configs" / "robustness" / "eval_runs_robustness.json"
REQUIRED_CHECKPOINT_FILES = [
    "model.safetensors",
    "optimizer.pt",
    "scheduler.pt",
    "trainer_state.json",
    "training_args.json",
    "rng_state.pth",
    "config.json",
]


@dataclass(frozen=True)
class RunSpec:
    tar_name: str
    run_name: str
    representation: str
    script: str
    regime: str
    seed: int
    tokenizer: str
    ppl_text_key: str
    eval4_text_key: str


RUN_SPECS = [
    RunSpec(
        "chinese_125m_b1024_matched_token_seed43.tar.gz",
        "chinese_125m_b1024_matched_token_seed43",
        "Chinese-Origin",
        "chinese_origin",
        "matched-token",
        43,
        "tokenizers/chinese_origin_32k_eos",
        "zh_text",
        "zh",
    ),
    RunSpec(
        "chinese_125m_b1024_matched_token_seed44.tar.gz",
        "chinese_125m_b1024_matched_token_seed44",
        "Chinese-Origin",
        "chinese_origin",
        "matched-token",
        44,
        "tokenizers/chinese_origin_32k_eos",
        "zh_text",
        "zh",
    ),
    RunSpec(
        "diacritic_125m_b1024_matched_data_4epoch_seed42.tar.gz",
        "diacritic_125m_b1024_matched_data_4epoch_seed42",
        "Pinyin-Diacritic",
        "pinyin_diacritic",
        "matched-data",
        42,
        "tokenizers/pinyin_diacritic_32k_eos",
        "diacritic_text",
        "diacritic",
    ),
    RunSpec(
        "diacritic_125m_b1024_matched_token_seed43.tar.gz",
        "diacritic_125m_b1024_matched_token_seed43",
        "Pinyin-Diacritic",
        "pinyin_diacritic",
        "matched-token",
        43,
        "tokenizers/pinyin_diacritic_32k_eos",
        "diacritic_text",
        "diacritic",
    ),
    RunSpec(
        "diacritic_125m_b1024_matched_token_seed44.tar.gz",
        "diacritic_125m_b1024_matched_token_seed44",
        "Pinyin-Diacritic",
        "pinyin_diacritic",
        "matched-token",
        44,
        "tokenizers/pinyin_diacritic_32k_eos",
        "diacritic_text",
        "diacritic",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tar-dir", default=str(DEFAULT_TAR_DIR))
    parser.add_argument("--unpack-dir", default=str(DEFAULT_UNPACK_DIR))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--extract", action="store_true", help="Actually extract tarballs into --unpack-dir.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow extraction into a non-empty destination directory.",
    )
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="Write the Eval 1/2/4 model-run manifest. Safe in dry-run mode.",
    )
    return parser.parse_args()


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def checkpoint_number(path: str) -> int:
    match = re.search(r"/checkpoint-(\d+)(?:/|$)", path)
    return int(match.group(1)) if match else -1


def latest_checkpoint_members(members: list[str], run_name: str) -> tuple[str, list[str]]:
    checkpoints = sorted(
        {
            member.rsplit("/", 1)[0]
            for member in members
            if f"outputs/{run_name}/checkpoint-" in member and member.rsplit("/", 1)[-1] in REQUIRED_CHECKPOINT_FILES
        },
        key=checkpoint_number,
    )
    if not checkpoints:
        raise ValueError(f"No checkpoints found for {run_name}")
    checkpoint = checkpoints[-1]
    checkpoint_members = [member for member in members if member.startswith(checkpoint + "/")]
    missing = [
        filename
        for filename in REQUIRED_CHECKPOINT_FILES
        if f"{checkpoint}/{filename}" not in checkpoint_members
    ]
    return checkpoint, missing


def safe_extract(tar: tarfile.TarFile, destination: Path) -> None:
    destination_resolved = destination.resolve()
    for member in tar.getmembers():
        target = (destination / member.name).resolve()
        if destination_resolved not in target.parents and target != destination_resolved:
            raise ValueError(f"Unsafe tar member path: {member.name}")
    tar.extractall(destination)


def build_manifest_entry(spec: RunSpec, tar_dir: Path, unpack_dir: Path, checkpoint_member: str) -> dict[str, object]:
    checkpoint = unpack_dir / spec.run_name / checkpoint_member
    checkpoint_rel = checkpoint.relative_to(ROOT).as_posix()
    return {
        "run_name": spec.run_name,
        "model": spec.run_name,
        "representation": spec.representation,
        "script": spec.script,
        "regime": spec.regime,
        "seed": spec.seed,
        "checkpoint": checkpoint_rel,
        "tokenizer": spec.tokenizer,
        "ppl_text_key": spec.ppl_text_key,
        "eval4_text_key": spec.eval4_text_key,
        "output_json": f"{spec.run_name}.json",
        "tarball": (tar_dir / spec.tar_name).relative_to(ROOT).as_posix(),
    }


def main() -> int:
    args = parse_args()
    tar_dir = project_path(args.tar_dir)
    unpack_dir = project_path(args.unpack_dir)
    manifest_path = project_path(args.manifest)

    manifest_entries = []
    report_rows = []
    for spec in RUN_SPECS:
        tar_path = tar_dir / spec.tar_name
        if not tar_path.exists():
            raise FileNotFoundError(f"Missing tarball: {tar_path}")
        with tarfile.open(tar_path, "r:gz") as tar:
            members = tar.getnames()
            checkpoint_member, missing_in_tar = latest_checkpoint_members(members, spec.run_name)
            if args.extract:
                destination = unpack_dir / spec.run_name
                if destination.exists() and any(destination.iterdir()) and not args.force:
                    raise FileExistsError(
                        f"Refusing to extract into non-empty directory without --force: {destination}"
                    )
                destination.mkdir(parents=True, exist_ok=True)
                safe_extract(tar, destination)
        checkpoint_path = unpack_dir / spec.run_name / checkpoint_member
        missing_on_disk = [
            filename
            for filename in REQUIRED_CHECKPOINT_FILES
            if not (checkpoint_path / filename).exists()
        ]
        manifest_entries.append(build_manifest_entry(spec, tar_dir, unpack_dir, checkpoint_member))
        report_rows.append(
            {
                "run": spec.run_name,
                "tarball": str(tar_path.relative_to(ROOT)),
                "checkpoint": str(checkpoint_path.relative_to(ROOT)),
                "missing_in_tar": ",".join(missing_in_tar) or "none",
                "missing_on_disk": ",".join(missing_on_disk) if args.extract else "not_checked_dry_run",
            }
        )

    if args.write_manifest:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "notes": [
                "Generated by scripts/robustness/prepare_robustness_eval.py.",
                "Checkpoint paths assume tarballs are extracted with --extract into server_outputs/robustness/unpacked.",
                "This manifest is for Eval 1 PPL, Eval 2 probes/controls, and Eval 4 ZhoBLiMP-style evaluation.",
            ],
            "required_checkpoint_files": REQUIRED_CHECKPOINT_FILES,
            "model_runs": manifest_entries,
        }
        manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote manifest: {manifest_path}")

    print("Robustness eval preparation report")
    print(f"mode: {'extract' if args.extract else 'dry-run'}")
    for row in report_rows:
        print(
            f"- {row['run']}: checkpoint={row['checkpoint']} "
            f"missing_in_tar={row['missing_in_tar']} missing_on_disk={row['missing_on_disk']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
