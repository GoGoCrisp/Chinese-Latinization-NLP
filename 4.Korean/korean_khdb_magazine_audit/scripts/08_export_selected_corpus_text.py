from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from khdb_common import count_chars, read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export selected KHDB diagnostic corpus text files.")
    parser.add_argument(
        "--selected-index",
        type=Path,
        default=Path("4.Korean/korean_khdb_magazine_audit/data/filtered/selected_diagnostic_article_index.jsonl"),
    )
    parser.add_argument(
        "--filtered-jsonl",
        type=Path,
        default=Path("4.Korean/korean_khdb_magazine_audit/data/filtered/all_articles_with_filter_flags_bounded.jsonl"),
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path(
            "4.Korean/korean_khdb_magazine_audit/data/corpus/article_level/selected_diagnostic_mixed_articles.jsonl"
        ),
    )
    parser.add_argument(
        "--output-text",
        type=Path,
        default=Path(
            "4.Korean/korean_khdb_magazine_audit/data/corpus/article_level/selected_diagnostic_mixed_article_texts.txt"
        ),
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path(
            "4.Korean/korean_khdb_magazine_audit/data/corpus/summaries/selected_diagnostic_mixed_corpus_summary.json"
        ),
    )
    return parser.parse_args()


def compact_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def main() -> None:
    args = parse_args()
    selected_rows = list(read_jsonl(args.selected_index))
    filtered_by_id = {row.get("khdb_id"): row for row in read_jsonl(args.filtered_jsonl)}
    exported: list[dict] = []
    missing_body_ids: list[str] = []

    for selected in sorted(selected_rows, key=lambda row: int(row.get("selection_rank", 0) or 0)):
        khdb_id = selected.get("khdb_id", "")
        source = filtered_by_id.get(khdb_id, {})
        body_text = source.get("body_text", "")
        if not body_text:
            missing_body_ids.append(khdb_id)
        exported.append(
            {
                **selected,
                "body_text": body_text,
                "body_text_compact": compact_text(body_text),
            }
        )

    write_jsonl(args.output_jsonl, exported)
    args.output_text.parent.mkdir(parents=True, exist_ok=True)
    args.output_text.write_text(
        "\n".join(row["body_text_compact"] for row in exported if row["body_text_compact"]) + "\n",
        encoding="utf-8",
    )

    joined_text = "\n".join(row["body_text"] for row in exported)
    counts = count_chars(joined_text)
    summary = {
        "selected_article_count": len(exported),
        "jsonl_output": str(args.output_jsonl),
        "text_output": str(args.output_text),
        "text_format": "one selected article per line, internal whitespace compacted, original mixed script preserved",
        "body_text_in_jsonl": True,
        "missing_body_count": len(missing_body_ids),
        "missing_body_ids": missing_body_ids[:100],
        **counts,
    }
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Exported articles: {len(exported)}")
    print(f"Missing body texts: {len(missing_body_ids)}")
    print(f"Source chars: {counts['body_length_chars']}")
    print(f"JSONL: {args.output_jsonl}")
    print(f"Text: {args.output_text}")
    print(f"Summary: {args.output_summary}")


if __name__ == "__main__":
    main()
