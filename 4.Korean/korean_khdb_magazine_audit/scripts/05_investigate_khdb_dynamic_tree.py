from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.parse import urljoin, urlparse

from khdb_common import (
    ROOT_URL,
    TARGET_MAGAZINES,
    canonical_url,
    extract_links,
    fetch_url,
    khdb_id_from_url_or_text,
    node_text,
    parse_html_tree,
    read_jsonl,
)


KNOWN_ARTICLE_URLS = [
    "https://db.history.go.kr/id/ma_002_0050_0330",
    "https://db.history.go.kr/id/ma_016_0020_0220",
    "https://db.history.go.kr/id/ma_016_0840_0480",
]

PATTERNS = {
    "ajax_endpoint": re.compile(r"""["']([^"']*(?:ajax|Ajax|get|search|level|tree|children|downloadItemLevel)[^"']*?\.do[^"']*)["']""", re.I),
    "js_function": re.compile(r"\bfunction\s+([A-Za-z0-9_]+)\s*\(([^)]*)\)"),
    "jquery_ajax": re.compile(r"\$\.ajax\s*\(\s*\{(.{0,1200}?)\}\s*\)", re.S),
    "empty_ul": re.compile(r"<ul\s+id=[\"'](ul-ma_\d+(?:_\d+)*)[\"'][^>]*>\s*</ul>", re.S),
    "khdb_id": re.compile(r"ma_\d+(?:_\d+)*"),
    "request_params": re.compile(r"\b(itemId|levelId|parentId|id|type|types|depth|tree|node|child|isLeaf|searchItemId|orderColumn)\b"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Investigate KHDB dynamic tree/article discovery.")
    parser.add_argument("--root-url", default=ROOT_URL)
    parser.add_argument(
        "--magazines-index",
        type=Path,
        default=Path("4.Korean/korean_khdb_magazine_audit/data/index/magazines.jsonl"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("4.Korean/korean_khdb_magazine_audit/data/raw_html"),
    )
    parser.add_argument(
        "--js-cache-dir",
        type=Path,
        default=Path("4.Korean/korean_khdb_magazine_audit/data/raw_html/js"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("4.Korean/korean_khdb_magazine_audit/data/index/dynamic_endpoint_candidates.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("4.Korean/korean_khdb_magazine_audit/results/reports/khdb_dynamic_tree_investigation.md"),
    )
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--max-magazines", type=int, default=None)
    parser.add_argument("--force-refresh", action="store_true")
    return parser.parse_args()


def safe_name(url: str) -> str:
    parsed = urlparse(url)
    stem = re.sub(r"[^A-Za-z0-9]+", "_", parsed.path.strip("/") or "root").strip("_")
    query = re.sub(r"[^A-Za-z0-9]+", "_", parsed.query).strip("_")
    return (stem + ("_" + query if query else ""))[:160] + ".js"


def script_sources(html: str, base_url: str) -> list[str]:
    sources = []
    for src in re.findall(r"<script[^>]+src=[\"']([^\"']+)[\"']", html, flags=re.I):
        url = canonical_url(urljoin(base_url, src))
        parsed = urlparse(url)
        if parsed.netloc == "db.history.go.kr" and "/resources/" in parsed.path:
            sources.append(url)
    return sorted(set(sources))


def download_js(urls: list[str], args: argparse.Namespace) -> dict[str, str]:
    args.js_cache_dir.mkdir(parents=True, exist_ok=True)
    contents: dict[str, str] = {}
    for url in urls:
        path = args.js_cache_dir / safe_name(url)
        if path.exists() and not args.force_refresh:
            contents[url] = path.read_text(encoding="utf-8", errors="replace")
            continue
        fetched = fetch_url(url, cache_dir=args.js_cache_dir, delay=args.delay, force=args.force_refresh)
        contents[url] = fetched["text"]
    return contents


def infer_parent_ids(article_urls: list[str]) -> list[dict]:
    rows = []
    for url in article_urls:
        khdb_id = khdb_id_from_url_or_text(url)
        parts = khdb_id.split("_")
        parent = "_".join(parts[:-1]) if len(parts) > 2 else ""
        rows.append(
            {
                "article_url": url,
                "article_id": khdb_id,
                "inferred_parent_id": parent,
                "parent_level_url": f"https://db.history.go.kr/modern/level.do?levelId={parent}" if parent else "",
                "id_depth": len(parts),
            }
        )
    return rows


def scan_text(label: str, text: str, url: str) -> dict:
    endpoints = []
    for match in PATTERNS["ajax_endpoint"].finditer(text):
        candidate = match.group(1)
        endpoints.append(canonical_url(urljoin(url, candidate)))
    functions = [
        {"name": match.group(1), "args": match.group(2)}
        for match in PATTERNS["js_function"].finditer(text)
        if any(token in match.group(0).lower() for token in ["level", "tree", "item", "search", "data", "download"])
    ]
    ajax_blocks = [normalize_snippet(match.group(0)) for match in PATTERNS["jquery_ajax"].finditer(text)]
    return {
        "label": label,
        "url": url,
        "candidate_endpoints": sorted(set(endpoints)),
        "candidate_functions": functions[:100],
        "ajax_blocks": ajax_blocks[:50],
        "empty_tree_containers": sorted(set(PATTERNS["empty_ul"].findall(text)))[:200],
        "request_parameters": sorted(set(PATTERNS["request_params"].findall(text))),
        "khdb_id_examples": sorted(set(PATTERNS["khdb_id"].findall(text)))[:200],
    }


def normalize_snippet(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()[:1200]


def render_report(payload: dict) -> str:
    lines = ["# KHDB Dynamic Tree Investigation", ""]
    lines.extend(
        [
            "## Summary",
            "",
            f"- pages inspected: {len(payload['pages'])}",
            f"- JavaScript files inspected: {len(payload['javascript_files'])}",
            f"- unique candidate endpoints: {len(payload['unique_candidate_endpoints'])}",
            f"- article ID examples: {len(payload['inferred_parent_child_patterns'])}",
            "",
            "## Candidate Endpoints",
            "",
        ]
    )
    for endpoint in payload["unique_candidate_endpoints"][:100]:
        lines.append(f"- {endpoint}")
    lines.extend(["", "## Candidate JavaScript Functions", ""])
    for item in payload["candidate_functions"][:100]:
        lines.append(f"- `{item['name']}({item['args']})` from {item['source']}")
    lines.extend(["", "## Request Parameters Observed", ""])
    for param in payload["request_parameters"]:
        lines.append(f"- `{param}`")
    lines.extend(["", "## Empty Tree Containers", ""])
    for item in payload["empty_tree_containers"][:80]:
        lines.append(f"- `{item['container_id']}` in {item['source']}")
    lines.extend(["", "## Inferred Parent-Child ID Patterns", ""])
    for row in payload["inferred_parent_child_patterns"]:
        lines.append(
            f"- article `{row['article_id']}` depth {row['id_depth']} -> parent `{row['inferred_parent_id']}` ({row['parent_level_url']})"
        )
    lines.extend(["", "## AJAX Blocks", ""])
    for block in payload["ajax_blocks"][:40]:
        lines.extend(["```javascript", block["snippet"], "```", f"source: {block['source']}", ""])
    lines.extend(
        [
            "## Current Assessment",
            "",
            payload["assessment"],
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    pages: list[dict] = []
    root = fetch_url(args.root_url, args.cache_dir, delay=args.delay, force=args.force_refresh)
    pages.append({"label": "root", "url": args.root_url, "text": root["text"], "path": str(root["path"])})

    if args.magazines_index.exists():
        magazines = list(read_jsonl(args.magazines_index))
        if args.max_magazines is not None:
            magazines = magazines[: args.max_magazines]
        for magazine in magazines:
            url = magazine.get("magazine_url")
            if not url:
                continue
            fetched = fetch_url(url, args.cache_dir, delay=args.delay, force=args.force_refresh)
            pages.append({"label": magazine.get("magazine_title", "magazine"), "url": url, "text": fetched["text"], "path": str(fetched["path"])})

    for article_url in KNOWN_ARTICLE_URLS:
        fetched = fetch_url(article_url, args.cache_dir, delay=args.delay, force=args.force_refresh)
        pages.append({"label": "known_article", "url": article_url, "text": fetched["text"], "path": str(fetched["path"])})
        parent = infer_parent_ids([article_url])[0]["parent_level_url"]
        if parent:
            parent_fetched = fetch_url(parent, args.cache_dir, delay=args.delay, force=args.force_refresh)
            pages.append({"label": "known_parent_issue", "url": parent, "text": parent_fetched["text"], "path": str(parent_fetched["path"])})

    js_urls = sorted(set(src for page in pages for src in script_sources(page["text"], page["url"])))
    js_contents = download_js(js_urls, args)

    scans = [scan_text(page["label"], page["text"], page["url"]) for page in pages]
    js_scans = [scan_text("javascript", text, url) for url, text in js_contents.items()]
    all_scans = scans + js_scans

    unique_endpoints = sorted(set(endpoint for scan in all_scans for endpoint in scan["candidate_endpoints"]))
    functions = [
        {**function, "source": scan["url"]}
        for scan in all_scans
        for function in scan["candidate_functions"]
    ]
    empty_containers = [
        {"container_id": container, "source": scan["url"]}
        for scan in scans
        for container in scan["empty_tree_containers"]
    ]
    ajax_blocks = [
        {"snippet": block, "source": scan["url"]}
        for scan in all_scans
        for block in scan["ajax_blocks"]
    ]
    request_parameters = sorted(set(param for scan in all_scans for param in scan["request_parameters"]))
    inferred = infer_parent_ids(KNOWN_ARTICLE_URLS)
    assessment = (
        "The inspected static issue pages expose issue-level `ma_xxx_yyyy` nodes but the corresponding "
        "`ul-ma_...` containers are empty. The likely public dynamic tree mechanism is "
        "`/modern/getChildItemLevelListAjax.do?parentId=...`, which appears among same-domain candidates. "
        "A bounded Stage 2 test can validate this by reproducing article URLs from a magazine root without "
        "seed article URLs. Known article pages also expose their own ID, parent issue ID, and prev/next "
        "article IDs. Full 19-magazine crawling should still wait for manual confirmation."
    )
    payload = {
        "pages": [{k: v for k, v in page.items() if k != "text"} for page in pages],
        "javascript_files": sorted(js_contents),
        "unique_candidate_endpoints": unique_endpoints,
        "candidate_functions": functions,
        "request_parameters": request_parameters,
        "empty_tree_containers": empty_containers,
        "ajax_blocks": ajax_blocks,
        "inferred_parent_child_patterns": inferred,
        "scans": all_scans,
        "assessment": assessment,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_report(payload), encoding="utf-8")
    print(f"Inspected pages: {len(pages)}")
    print(f"JavaScript files: {len(js_contents)}")
    print(f"Candidate endpoints: {len(unique_endpoints)}")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
