from __future__ import annotations

import argparse
import json
from pathlib import Path

from khdb_common import (
    ROOT_URL,
    TARGET_MAGAZINES,
    extract_links,
    extract_title,
    fetch_url,
    is_allowed_khdb_magazine_url,
    khdb_id_from_url_or_text,
    node_text,
    parse_html_tree,
    visible_lines,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discover KHDB target full-text magazine indexes.")
    parser.add_argument("--root-url", default=ROOT_URL)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("4.Korean/korean_khdb_magazine_audit/data/index"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("4.Korean/korean_khdb_magazine_audit/data/raw_html"),
    )
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def period_hint(lines: list[str], title: str) -> str:
    for i, line in enumerate(lines):
        if title in line:
            window = " / ".join(lines[max(0, i - 2) : i + 3])
            return window[:500]
    return ""


def find_magazine_rows(root_text: str, links: list[dict], cache_dir: Path, delay: float, max_pages: int | None) -> list[dict]:
    lines = visible_lines(root_text)
    rows: list[dict] = []
    seen_titles: set[str] = set()
    page_count = 0
    for title in TARGET_MAGAZINES:
        matches = [
            link
            for link in links
            if title in link["text"] and is_allowed_khdb_magazine_url(link["url"])
        ]
        if not matches:
            rows.append(
                {
                    "magazine_title": title,
                    "khdb_id": "",
                    "magazine_url": "",
                    "period_date_text": period_hint(lines, title),
                    "text_says_original_provided": "원문제공잡지" in root_text and title in root_text,
                    "raw_html_cache_path": "",
                    "discovery_status": "missing_link",
                }
            )
            continue
        link = matches[0]
        page_count += 1
        if max_pages is not None and page_count > max_pages:
            break
        fetched = fetch_url(link["url"], cache_dir=cache_dir, delay=delay)
        page_text = node_text(parse_html_tree(fetched["text"]))
        rows.append(
            {
                "magazine_title": title,
                "khdb_id": khdb_id_from_url_or_text(link["url"]),
                "magazine_url": link["url"],
                "period_date_text": period_hint(lines, title),
                "text_says_original_provided": "원문제공잡지" in page_text or "원문제공잡지" in root_text,
                "raw_html_cache_path": str(fetched["path"]),
                "discovery_status": "found",
                "source_link_text": link["text"],
                "source_page_title": extract_title(parse_html_tree(fetched["text"])),
            }
        )
        seen_titles.add(title)
    return rows


def render_report(rows: list[dict]) -> str:
    found = [row for row in rows if row.get("discovery_status") == "found"]
    missing = [row["magazine_title"] for row in rows if row.get("discovery_status") != "found"]
    lines = [
        "# KHDB Magazine Discovery Report",
        "",
        f"- target magazines: {len(TARGET_MAGAZINES)}",
        f"- found: {len(found)}",
        f"- missing: {len(missing)}",
        "",
        "## Missing Target Titles",
        "",
    ]
    lines.extend(f"- {title}" for title in missing)
    lines.extend(["", "## Title to KHDB ID Mapping", ""])
    for row in rows:
        lines.append(f"- {row['magazine_title']}: `{row.get('khdb_id', '')}` {row.get('magazine_url', '')}")
    lines.extend(["", "## Example URLs", ""])
    for row in found[:20]:
        lines.append(f"- {row['magazine_title']}: {row['magazine_url']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    fetched = fetch_url(args.root_url, cache_dir=args.cache_dir, delay=args.delay)
    root = parse_html_tree(fetched["text"])
    root_text = node_text(root)
    rows = find_magazine_rows(root_text, extract_links(root, args.root_url), args.cache_dir, args.delay, args.max_pages)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "magazines.jsonl", rows)
    summary = {
        "target_magazine_count": len(TARGET_MAGAZINES),
        "found_count": sum(1 for row in rows if row.get("discovery_status") == "found"),
        "missing_titles": [row["magazine_title"] for row in rows if row.get("discovery_status") != "found"],
        "root_url": args.root_url,
        "root_cache_path": str(fetched["path"]),
    }
    (args.output_dir / "magazines_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = Path("4.Korean/korean_khdb_magazine_audit/results/reports/magazine_discovery_report.md")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render_report(rows), encoding="utf-8")
    print(f"Found {summary['found_count']} / {summary['target_magazine_count']} target magazines")
    print(f"Index: {args.output_dir / 'magazines.jsonl'}")
    print(f"Report: {report}")


if __name__ == "__main__":
    main()
