from __future__ import annotations

import argparse
from pathlib import Path

from khdb_common import (
    ROOT_URL,
    candidate_content_nodes,
    extract_links,
    extract_title,
    fetch_url,
    node_text,
    parse_html_tree,
    visible_lines,
)


EXAMPLE_URLS = [
    "https://db.history.go.kr/id/ma_002_0050_0330",
    "https://db.history.go.kr/id/ma_016_0020_0220",
    "https://db.history.go.kr/id/ma_016_0840_0480",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect KHDB magazine HTML before crawling.")
    parser.add_argument("--root-url", default=ROOT_URL)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("4.Korean/korean_khdb_magazine_audit/data/raw_html/debug"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("4.Korean/korean_khdb_magazine_audit/results/reports/html_structure_inspection.md"),
    )
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def inspect_page(url: str, cache_dir: Path, delay: float, force: bool) -> dict:
    fetched = fetch_url(url, cache_dir=cache_dir, delay=delay, force=force)
    root = parse_html_tree(fetched["text"])
    links = extract_links(root, url)
    text = node_text(root)
    labels = ["잡지명", "발행일", "기사제목", "필자", "기사형태", "본문"]
    return {
        "url": url,
        "cache_path": str(fetched["path"]),
        "download_status": fetched["status"],
        "encoding": fetched["encoding"],
        "title": extract_title(root),
        "id_links": [link for link in links if "/id/ma_" in link["url"]][:100],
        "level_links": [link for link in links if "levelId=ma_" in link["url"]][:100],
        "candidate_containers": candidate_content_nodes(root)[:10],
        "visible_text_preview": "\n".join(visible_lines(text)[:40]),
        "metadata_label_presence": {label: label in text for label in labels},
    }


def render_report(results: list[dict]) -> str:
    lines = ["# KHDB HTML Structure Inspection", ""]
    for result in results:
        lines.extend(
            [
                f"## {result['url']}",
                "",
                f"- title: {result['title']}",
                f"- encoding: {result['encoding']}",
                f"- cache: `{result['cache_path']}`",
                f"- status: {result['download_status']}",
                f"- `/id/ma_` links: {len(result['id_links'])}",
                f"- `levelId=ma_` links: {len(result['level_links'])}",
                f"- metadata labels: {result['metadata_label_presence']}",
                "",
                "### Example `/id/ma_` Links",
                "",
            ]
        )
        for link in result["id_links"][:20]:
            lines.append(f"- {link['text'] or '(no text)'}: {link['url']}")
        lines.extend(["", "### Example `levelId=ma_` Links", ""])
        for link in result["level_links"][:20]:
            lines.append(f"- {link['text'] or '(no text)'}: {link['url']}")
        lines.extend(["", "### Candidate Content Containers", ""])
        for i, candidate in enumerate(result["candidate_containers"], start=1):
            lines.extend(
                [
                    f"#### Candidate {i}",
                    "",
                    f"- tag: `{candidate['tag']}`",
                    f"- class/id: `{candidate['class_id']}`",
                    f"- text length: {candidate['text_length']}",
                    f"- label hits: {candidate['label_hits']}",
                    "",
                    "```text",
                    candidate["preview"],
                    "```",
                    "",
                ]
            )
        lines.extend(["### Visible Text Preview", "", "```text", result["visible_text_preview"], "```", ""])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    urls = [args.root_url] + EXAMPLE_URLS
    results = [inspect_page(url, args.cache_dir, args.delay, args.force) for url in urls]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_report(results), encoding="utf-8")
    print(f"Inspected {len(results)} pages")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
