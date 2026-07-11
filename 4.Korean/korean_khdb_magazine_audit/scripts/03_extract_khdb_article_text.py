from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

from khdb_common import (
    TARGET_MAGAZINES,
    UI_TEXT_LINES,
    candidate_content_nodes,
    count_chars,
    find_metadata,
    iter_nodes,
    khdb_id_from_url_or_text,
    normalize_text,
    node_text,
    parse_html_tree,
    read_jsonl,
    visible_lines,
    write_jsonl,
)


JAPANESE_OMISSION_RE = re.compile(
    r"(이하\s*(?:\d+|[一二三四五六七八九十百千]+|숫자)?\s*줄?\s*일본문"
    r"|이하.{0,40}일본문"
    r"|以下.{0,40}日文"
    r"|日文\s*省略"
    r"|일본어\s*생략"
    r"|일본문\s*생략"
    r"|원문\s*(?:일본문|日文)\s*생략)",
    re.I,
)
LABEL_RE = re.compile(r"^(잡지명|발행일|기사제목|제목|필자|기사형태|형태분류|호수)\s*[:：]?.*$")
META_LABEL_MAP = {
    "잡지명": "magazine_title",
    "발행일": "publication_date",
    "기사제목": "article_title",
    "제목": "article_title",
    "필자": "author",
    "기사형태": "article_type",
    "형태분류": "article_type",
    "호수": "issue_title",
}

CANONICAL_MAGAZINES = sorted(TARGET_MAGAZINES, key=len, reverse=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract KHDB article metadata and body text.")
    parser.add_argument(
        "--articles-index",
        type=Path,
        default=Path("4.Korean/korean_khdb_magazine_audit/data/index/articles_index.jsonl"),
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("4.Korean/korean_khdb_magazine_audit/data/extracted/articles_extracted.jsonl"),
    )
    parser.add_argument(
        "--output-samples",
        type=Path,
        default=Path("4.Korean/korean_khdb_magazine_audit/results/samples/extraction_samples.md"),
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max-articles", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def remove_ui_lines(lines: list[str]) -> list[str]:
    cleaned: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped in UI_TEXT_LINES:
            continue
        if LABEL_RE.match(stripped):
            continue
        if re.match(r"^(이전글|다음글|관련사이트|자료일람|전체목록)\b", stripped):
            continue
        cleaned.append(stripped)
    return cleaned


def choose_body_text(full_text: str, candidates: list[dict]) -> tuple[str, list[str]]:
    notes: list[str] = []
    source_text = full_text
    if any("본문" in item["label_hits"] for item in candidates):
        notes.append("body_label_seen")
    lines = visible_lines(source_text)
    start = 0
    for i, line in enumerate(lines):
        if line.strip() == "본문" or line.strip().startswith("본문 "):
            start = i + 1
            notes.append("body_started_after_label")
            break
    body_lines = remove_ui_lines(lines[start:])
    return normalize_text("\n".join(body_lines)), notes


def find_node_by_id(root, node_id: str):
    for node in iter_nodes(root):
        if node.attrs.get("id") == node_id:
            return node
    return None


def node_class_tokens(node) -> set[str]:
    return set(node.attrs.get("class", "").split())


def extract_section_meta(root) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for node in iter_nodes(root):
        if "item" not in node_class_tokens(node):
            continue
        title = ""
        value = ""
        for child in node.children:
            classes = node_class_tokens(child)
            if "tit" in classes:
                title = normalize_text(node_text(child))
            elif "cont" in classes:
                value = normalize_text(node_text(child))
        field_name = META_LABEL_MAP.get(title)
        if field_name and value:
            metadata[field_name] = value
    return metadata


def split_magazine_issue(raw_value: str, fallback_magazine: str = "") -> tuple[str, str, list[str]]:
    notes: list[str] = []
    raw_value = normalize_text(raw_value)
    for title in CANONICAL_MAGAZINES:
        if raw_value == title:
            return title, "", notes
        if raw_value.startswith(title):
            issue = raw_value[len(title) :].strip()
            return title, issue, notes
    if fallback_magazine in TARGET_MAGAZINES:
        notes.append("used_index_magazine_title_fallback")
        issue = raw_value
        if issue.startswith(fallback_magazine):
            issue = issue[len(fallback_magazine) :].strip()
        return fallback_magazine, issue, notes
    notes.append("uncertain_magazine_title")
    return raw_value or fallback_magazine, "", notes


def extract_one(row: dict) -> dict:
    path = Path(row.get("raw_html_path", ""))
    if not path.exists():
        return {
            **row,
            "body_text": "",
            "extraction_success": False,
            "extraction_notes": ["missing_raw_html_path"],
        }
    html = path.read_text(encoding="utf-8", errors="replace")
    root = parse_html_tree(html)
    full_text = node_text(root)
    lines = visible_lines(full_text)
    metadata = extract_section_meta(root) or find_metadata(lines)
    raw_magazine_field = metadata.get("magazine_title", "")
    raw_issue_field = metadata.get("issue_title", "")
    raw_title_field = metadata.get("article_title", "")
    canonical_magazine, inferred_issue, metadata_notes = split_magazine_issue(
        raw_magazine_field, row.get("magazine_title", "")
    )
    issue_title = raw_issue_field or inferred_issue
    candidates = candidate_content_nodes(root)
    cont_view = find_node_by_id(root, "cont_view")
    if cont_view is not None:
        body_text = normalize_text("\n".join(remove_ui_lines(visible_lines(node_text(cont_view)))))
        notes = ["used_cont_view"]
    else:
        body_text, notes = choose_body_text(full_text, candidates)
    marker_lines = [line for line in visible_lines(body_text) if JAPANESE_OMISSION_RE.search(line)]
    counts = count_chars(body_text)
    counts.update(
        {
            "japanese_omission_marker_count": sum(1 for _ in JAPANESE_OMISSION_RE.finditer(body_text)),
            "japanese_omission_marker_lines": len(marker_lines),
            "japanese_omission_marker_ratio": len(marker_lines) / max(1, counts["line_count"]),
        }
    )
    result = {
        "khdb_id": row.get("khdb_id") or khdb_id_from_url_or_text(row.get("url", "")),
        "url": row.get("url", ""),
        "magazine_title": canonical_magazine,
        "issue_title": issue_title,
        "raw_magazine_field": raw_magazine_field,
        "raw_issue_field": raw_issue_field,
        "raw_title_field": raw_title_field,
        "publication_date": metadata.get("publication_date", ""),
        "author": metadata.get("author", ""),
        "article_title": metadata.get("article_title") or row.get("title_from_html", ""),
        "article_type": metadata.get("article_type", ""),
        "source_page_title": row.get("title_from_html", ""),
        "raw_text_preview": normalize_text(full_text)[:500],
        "body_text": body_text,
        "raw_html_path": str(path),
        "extraction_success": len(body_text) >= 20,
        "extraction_notes": notes + metadata_notes,
    }
    result.update(counts)
    return result


def sample_rows(rows: list[dict], seed: int) -> str:
    successful = [row for row in rows if row.get("extraction_success")]
    rng = random.Random(seed)
    random_rows = rng.sample(successful, min(20, len(successful))) if successful else []
    high_hanja = sorted(successful, key=lambda row: row.get("hanja_ratio", 0), reverse=True)[:20]
    low_hanja = sorted(successful, key=lambda row: row.get("hanja_ratio", 0))[:20]
    sections = [("Random Extracted Articles", random_rows), ("High Hanja Ratio", high_hanja), ("Low Hanja Ratio", low_hanja)]
    lines = ["# KHDB Extraction Samples", ""]
    for title, section_rows in sections:
        lines.extend([f"## {title}", ""])
        for row in section_rows:
            lines.extend(
                [
                    f"### {row.get('magazine_title', '')} / {row.get('article_title', '')}",
                    "",
                    f"- url: {row.get('url', '')}",
                    f"- date: {row.get('publication_date', '')}",
                    f"- chars: {row.get('body_length_chars', 0)}; hangul: {row.get('hangul_count', 0)}; hanja: {row.get('hanja_count', 0)}; hanja_ratio: {row.get('hanja_ratio', 0):.4f}",
                    "",
                    "```text",
                    row.get("body_text", "")[:500],
                    "```",
                    "",
                ]
            )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    rows = list(read_jsonl(args.articles_index))
    if args.max_articles is not None:
        rows = rows[: args.max_articles]
    extracted = [extract_one(row) for row in rows]
    write_jsonl(args.output_jsonl, extracted)
    args.output_samples.parent.mkdir(parents=True, exist_ok=True)
    args.output_samples.write_text(sample_rows(extracted, args.seed), encoding="utf-8")
    success_count = sum(1 for row in extracted if row.get("extraction_success"))
    print(f"Extracted articles: {len(extracted)}")
    print(f"Extraction success: {success_count}")
    print(f"Output: {args.output_jsonl}")
    print(f"Samples: {args.output_samples}")


if __name__ == "__main__":
    main()
