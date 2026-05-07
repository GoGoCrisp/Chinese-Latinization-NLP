#!/usr/bin/env python3
"""Shared build utilities for eval2 probes."""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

from pypinyin import Style, lazy_pinyin


DEFAULT_COLLISION_CSV = (
    "../1.Tokenization/decoded_superTokenizers_2048_subset100k/table2/"
    "table2_ab_overlap_superBPE_outputs/table2_ab_overlap_superBPE_details.csv"
)
DEFAULT_RAW_TEST = "data/raw/test.zh.txt"
DEFAULT_EMBEDDING_MODEL = "shibing624/text2vec-base-chinese"
DEFAULT_EXCLUDE_CHARS = "的地得了着过之"
DEFAULT_EXCLUDE_TOKENS = "以及,而是,任何,是以,是一,由于,于是,但是,所以,如果,或者,并且,而且"
NUMERAL_CHARS = set("零〇一二三四五六七八九十百千万亿两甲乙丙丁戊己庚辛壬癸")
BOUNDARY_CHARS = set("。！？!?；;\n")


class TokenTrie:
    def __init__(self, tokens: set[str]) -> None:
        self.root: dict[str, Any] = {}
        self.max_len = 0
        for token in tokens:
            node = self.root
            for ch in token:
                node = node.setdefault(ch, {})
            node.setdefault("_tokens", []).append(token)
            self.max_len = max(self.max_len, len(token))

    def find(self, text: str) -> list[tuple[int, str]]:
        hits: list[tuple[int, str]] = []
        for start in range(len(text)):
            node = self.root
            for ch in text[start : start + self.max_len]:
                node = node.get(ch)
                if node is None:
                    break
                for token in node.get("_tokens", []):
                    hits.append((start, token))
        return hits


def project_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def path_looks_like_train_split(path: Path | str) -> bool:
    path = Path(path)
    parts = {part.lower() for part in path.parts}
    name = path.name.lower()
    return "train" in parts or name.startswith("train.") or "_train" in name or "train_" in name


def cjk_count(text: str) -> int:
    return sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")


def token_is_clean_lexical(token: str, args: Any) -> bool:
    token = token.strip()
    if len(token) < args.min_cjk_chars or len(token) > args.max_cjk_chars:
        return False
    if cjk_count(token) != len(token):
        return False
    if any(ch in token for ch in args.exclude_chars):
        return False
    exclude_tokens = {item.strip() for item in args.exclude_tokens.split(",") if item.strip()}
    if token in exclude_tokens:
        return False
    if args.exclude_numeral_tokens and any(ch in NUMERAL_CHARS for ch in token):
        return False
    return True


def to_diacritic(text: str) -> str:
    parts = lazy_pinyin(
        text,
        style=Style.TONE,
        neutral_tone_with_five=False,
        errors=lambda chunk: list(chunk),
    )
    return re.sub(r"\s+", " ", " ".join(part.strip() for part in parts if part.strip())).strip()


def to_toneless(text: str) -> str:
    parts = lazy_pinyin(text, style=Style.NORMAL, errors=lambda chunk: list(chunk))
    return re.sub(r"\s+", " ", " ".join(part.strip() for part in parts if part.strip())).strip()


def join_diacritic(*parts: str) -> str:
    return re.sub(r"\s+", " ", " ".join(part for part in parts if part)).strip()


def resolve_local_hf_snapshot(model_name: str) -> str:
    if "/" not in model_name:
        return model_name
    owner, repo = model_name.split("/", 1)
    snapshots_dir = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{owner}--{repo}" / "snapshots"
    if not snapshots_dir.exists():
        return model_name
    snapshots = [
        path
        for path in snapshots_dir.iterdir()
        if path.is_dir() and (path / "modules.json").exists()
    ]
    if not snapshots:
        return model_name
    snapshots.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return str(snapshots[0])


def cosine_distance(left: Any, right: Any) -> float:
    import numpy as np

    value = float(1.0 - np.dot(left, right))
    if not math.isfinite(value):
        raise ValueError("Non-finite embedding distance")
    return max(0.0, min(2.0, value))


def summary_stats(values: list[float]) -> dict[str, float | int | str]:
    if not values:
        return {"count": 0, "mean": "", "median": "", "std": "", "min": "", "max": ""}
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {
        "count": len(values),
        "mean": mean,
        "median": median(values),
        "std": math.sqrt(variance),
        "min": min(values),
        "max": max(values),
    }


def read_semantic_pair_distances(path: Path) -> dict[tuple[str, str], float]:
    distances: dict[tuple[str, str], float] = {}
    if not path.exists():
        return distances
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            for pair_col, dist_col in (("max_pair", "max_dist"), ("min_pair", "min_dist")):
                pair = row.get(pair_col, "")
                dist = row.get(dist_col, "")
                if "|" not in pair or not dist:
                    continue
                left, right = pair.split("|", 1)
                try:
                    distances[tuple(sorted((left, right)))] = float(dist)
                except ValueError:
                    continue
    return distances


def load_embeddings(tokens: list[str], args: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    meta = {
        "model": args.embedding_model,
        "model_path": None,
        "token_count": len(tokens),
        "status": "not_started",
        "warning": None,
    }
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        if getattr(args, "allow_no_embedding", False):
            meta["status"] = "import_failed"
            meta["warning"] = str(exc)
            return {}, meta
        raise

    model_path = resolve_local_hf_snapshot(args.embedding_model) if args.prefer_local_cache else args.embedding_model
    meta["model_path"] = model_path
    print(f"embedding model: {args.embedding_model}")
    if model_path != args.embedding_model:
        print(f"using local cache snapshot: {model_path}")
    try:
        model = SentenceTransformer(model_path)
        vectors = model.encode(
            tokens,
            normalize_embeddings=True,
            batch_size=args.embedding_batch_size,
            show_progress_bar=not args.no_progress,
        )
    except Exception as exc:
        if getattr(args, "allow_no_embedding", False):
            meta["status"] = "load_or_encode_failed"
            meta["warning"] = str(exc)
            return {}, meta
        raise

    meta["status"] = "ok"
    vectors = np.asarray(vectors, dtype=np.float32)
    return dict(zip(tokens, vectors)), meta
