from __future__ import annotations

import argparse
import json
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

from khdb_common import (
    canonical_url,
    extract_links,
    extract_title,
    fetch_url,
    is_allowed_khdb_magazine_url,
    khdb_id_from_url_or_text,
    parse_html_tree,
    read_jsonl,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discover and cache KHDB magazine issue/article pages.")
    parser.add_argument(
        "--magazines-index",
        type=Path,
        default=Path("4.Korean/korean_khdb_magazine_audit/data/index/magazines.jsonl"),
    )
    parser.add_argument(
        "--output-index",
        type=Path,
        default=Path("4.Korean/korean_khdb_magazine_audit/data/index/articles_index.jsonl"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("4.Korean/korean_khdb_magazine_audit/data/raw_html"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("4.Korean/korean_khdb_magazine_audit/results/reports/article_discovery_report.md"),
    )
    parser.add_argument("--discovery-mode", choices=["static", "dynamic", "auto"], default="auto")
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--max-magazines", type=int, default=None)
    parser.add_argument("--max-pages-per-magazine", type=int, default=None)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--max-nodes", type=int, default=None)
    parser.add_argument("--max-articles", type=int, default=None)
    parser.add_argument("--magazine-title", default=None)
    parser.add_argument("--seed-article-url", action="append", default=[])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument(
        "--index-only",
        action="store_true",
        help="Discover article IDs/URLs without downloading article body HTML pages.",
    )
    parser.add_argument(
        "--nodes-index",
        type=Path,
        default=None,
        help="Optional path for discovered node records.",
    )
    parser.add_argument(
        "--graph-index",
        type=Path,
        default=None,
        help="Optional path for crawl graph records.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Optional path for JSON discovery summary.",
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def same_magazine_prefix(khdb_id: str, prefix: str) -> bool:
    return bool(khdb_id and prefix and (khdb_id == prefix or khdb_id.startswith(prefix + "_")))


def id_depth(khdb_id: str) -> int:
    return len(khdb_id.split("_")) if khdb_id else 0


def node_type(khdb_id: str) -> str:
    depth = id_depth(khdb_id)
    if depth == 2:
        return "magazine"
    if depth == 3:
        return "issue"
    if depth >= 4:
        return "article"
    return "unknown"


def article_url(khdb_id: str) -> str:
    return f"https://db.history.go.kr/id/{khdb_id}"


def level_url(khdb_id: str) -> str:
    return f"https://db.history.go.kr/modern/level.do?levelId={khdb_id}"


def make_node(
    *,
    khdb_id: str,
    url: str,
    magazine_title: str,
    issue_title: str = "",
    parent_id: str = "",
    parent_url: str = "",
    depth: int = 0,
    discovery_strategy: str,
    raw_path: str = "",
    download_status: str = "",
    fetched_at: str = "",
    title_from_html: str = "",
) -> dict:
    return {
        "khdb_id": khdb_id,
        "url": url,
        "node_type": node_type(khdb_id),
        "magazine_title": magazine_title,
        "issue_title": issue_title,
        "parent_id": parent_id,
        "parent_url": parent_url,
        "depth": depth,
        "discovery_strategy": discovery_strategy,
        "raw_html_path": raw_path if raw_path.endswith(".html") else "",
        "raw_response_path": raw_path if raw_path and not raw_path.endswith(".html") else "",
        "download_status": download_status,
        "fetched_at": fetched_at,
        "title_from_html": title_from_html,
    }


def dynamic_children(khdb_id: str, args: argparse.Namespace) -> tuple[list[dict], str, str]:
    url = f"https://db.history.go.kr/modern/getChildItemLevelListAjax.do?parentId={khdb_id}"
    try:
        fetched = fetch_url(url, cache_dir=args.cache_dir / "dynamic", delay=args.delay, force=args.force_refresh)
    except Exception as exc:
        return [], "", f"error: {exc}"
    root = parse_html_tree(fetched["text"])
    rows: list[dict] = []
    seen_ids: set[str] = set()
    for link in extract_links(root, url):
        child_id = khdb_id_from_url_or_text(link["url"])
        if not child_id or child_id in seen_ids:
            continue
        seen_ids.add(child_id)
        rows.append(
            {
                "khdb_id": child_id,
                "url": article_url(child_id) if node_type(child_id) == "article" else level_url(child_id),
                "article_title": link["text"],
            }
        )
    return rows, str(fetched["path"]), fetched["status"]


def enqueue(queue: deque, seen_or_queued: set[str], item: tuple) -> bool:
    url = item[0]
    if url not in seen_or_queued:
        seen_or_queued.add(url)
        khdb_id = khdb_id_from_url_or_text(url)
        if node_type(khdb_id) == "article":
            queue.appendleft(item)
        else:
            queue.append(item)
        return True
    return False


def crawl_magazine(magazine: dict, args: argparse.Namespace) -> tuple[list[dict], list[dict], list[dict], list[str]]:
    magazine_url = magazine.get("magazine_url", "")
    magazine_prefix = magazine.get("khdb_id", "") or khdb_id_from_url_or_text(magazine_url)
    queue: deque = deque()
    queued: set[str] = set()
    enqueue(queue, queued, (magazine_url, "", "", 0, "magazine_root", "static"))
    for seed_url in args.seed_article_url:
        seed_id = khdb_id_from_url_or_text(seed_url)
        if not magazine_prefix or same_magazine_prefix(seed_id, magazine_prefix):
            enqueue(queue, queued, (seed_url, "", "", 0, "seed_article_url", "static_seed"))

    seen: set[str] = set()
    article_records: list[dict] = []
    node_records: list[dict] = []
    graph: list[dict] = []
    unresolved: list[str] = []
    discovered_article_ids: set[str] = set()
    pages_seen = 0
    articles_seen = 0
    duplicate_enqueue_count = 0
    cycle_prevented_count = 0

    while queue:
        url, parent_url, parent_id, depth, discovered_from, strategy = queue.popleft()
        if not url or url in seen:
            if url in seen:
                cycle_prevented_count += 1
            continue
        if args.max_depth is not None and depth > args.max_depth:
            continue
        if args.max_pages_per_magazine is not None and pages_seen >= args.max_pages_per_magazine:
            break
        if args.max_nodes is not None and len(node_records) >= args.max_nodes:
            break
        if args.max_articles is not None and articles_seen >= args.max_articles:
            break

        khdb_id = khdb_id_from_url_or_text(url)
        if khdb_id and magazine_prefix and not same_magazine_prefix(khdb_id, magazine_prefix):
            continue

        current_type = node_type(khdb_id)
        fetch_url_value = article_url(khdb_id) if current_type == "article" else url
        should_fetch_html = not (
            args.index_only
            and (
                current_type == "article"
                or (args.discovery_mode == "dynamic" and current_type in {"magazine", "issue"})
            )
        )
        if should_fetch_html:
            try:
                fetched = fetch_url(fetch_url_value, cache_dir=args.cache_dir, delay=args.delay, force=args.force_refresh)
                root = parse_html_tree(fetched["text"])
                title = extract_title(root)
                raw_path = str(fetched["path"])
                status = fetched["status"]
                fetched_at = fetched["fetched_at"]
            except Exception as exc:
                root = None
                title = ""
                raw_path = ""
                status = f"error: {exc}"
                fetched_at = datetime.now(timezone.utc).isoformat()
        else:
            root = None
            title = discovered_from if current_type == "article" else ""
            raw_path = ""
            status = "index_only_not_downloaded"
            fetched_at = ""

        seen.add(url)
        pages_seen += 1
        node = make_node(
            khdb_id=khdb_id,
            url=fetch_url_value,
            magazine_title=magazine.get("magazine_title", ""),
            parent_id=parent_id,
            parent_url=parent_url,
            depth=depth,
            discovery_strategy=strategy,
            raw_path=raw_path,
            download_status=status,
            fetched_at=fetched_at,
            title_from_html=title,
        )
        node_records.append(node)
        if node["node_type"] == "article":
            articles_seen += 1
            article_records.append(node)

        if root is not None and args.discovery_mode in {"static", "auto"}:
            for link in extract_links(root, fetch_url_value):
                child_url = link["url"]
                child_id = khdb_id_from_url_or_text(child_url)
                if not is_allowed_khdb_magazine_url(child_url):
                    continue
                if child_id and magazine_prefix and not same_magazine_prefix(child_id, magazine_prefix):
                    continue
                if child_id and node_type(child_id) == "article":
                    if args.max_articles is not None and len(discovered_article_ids) >= args.max_articles:
                        continue
                    discovered_article_ids.add(child_id)
                    child_url = article_url(child_id)
                graph.append(
                    {
                        "parent_url": fetch_url_value,
                        "child_url": child_url,
                        "parent_khdb_id": khdb_id,
                        "child_khdb_id": child_id,
                        "magazine_title": magazine.get("magazine_title", ""),
                        "link_text": link["text"],
                        "discovery_strategy": "static",
                    }
                )
                if not enqueue(queue, queued, (child_url, fetch_url_value, khdb_id, depth + 1, link["text"], "static")):
                    duplicate_enqueue_count += 1

        if khdb_id and args.discovery_mode in {"dynamic", "auto"} and node_type(khdb_id) in {"magazine", "issue"}:
            rows, response_path, dynamic_status = dynamic_children(khdb_id, args)
            if not rows and node_type(khdb_id) == "issue":
                unresolved.append(f"No dynamic rows for issue {khdb_id} via getChildItemLevelListAjax.do ({dynamic_status})")
            for row in rows:
                child_id = row["khdb_id"]
                if child_id == khdb_id:
                    continue
                child_url = row["url"]
                if child_id and magazine_prefix and not same_magazine_prefix(child_id, magazine_prefix):
                    continue
                if child_id and node_type(child_id) == "article":
                    if args.max_articles is not None and len(discovered_article_ids) >= args.max_articles:
                        continue
                    discovered_article_ids.add(child_id)
                graph.append(
                    {
                        "parent_url": fetch_url_value,
                        "child_url": child_url,
                        "parent_khdb_id": khdb_id,
                        "child_khdb_id": child_id,
                        "magazine_title": magazine.get("magazine_title", ""),
                        "link_text": row.get("article_title") or row.get("magazine_field", ""),
                        "discovery_strategy": "dynamic_getChildItemLevelListAjax",
                        "raw_response_path": response_path,
                    }
                )
                if not enqueue(
                    queue,
                    queued,
                    (
                        child_url,
                        fetch_url_value,
                        khdb_id,
                        depth + 1,
                        row.get("article_title") or "getChildItemLevelListAjax.do",
                        "dynamic_getChildItemLevelListAjax",
                    ),
                ):
                    duplicate_enqueue_count += 1

    if duplicate_enqueue_count:
        unresolved.append(f"Duplicate queued URLs skipped: {duplicate_enqueue_count}")
    if cycle_prevented_count:
        unresolved.append(f"Already-seen URLs skipped as cycles: {cycle_prevented_count}")
    return article_records, node_records, graph, unresolved


def render_report(article_records: list[dict], node_records: list[dict], graph: list[dict], unresolved: list[str]) -> str:
    by_magazine = Counter(row.get("magazine_title", "") for row in article_records)
    by_strategy = Counter(row.get("discovery_strategy", "") for row in node_records)
    by_depth = Counter(row.get("depth", 0) for row in node_records)
    dynamic_response_paths = {
        row.get("raw_response_path", "")
        for row in graph
        if row.get("discovery_strategy") == "dynamic_getChildItemLevelListAjax" and row.get("raw_response_path")
    }
    dynamic_request_count = len(dynamic_response_paths)
    article_ids = [row.get("khdb_id", "") for row in article_records if row.get("khdb_id")]
    duplicate_article_ids = sorted([item for item, count in Counter(article_ids).items() if count > 1])
    node_ids = [row.get("khdb_id", "") for row in node_records if row.get("khdb_id")]
    duplicate_node_ids = sorted([item for item, count in Counter(node_ids).items() if count > 1])
    zero_magazines = sorted(
        {
            row.get("magazine_title", "")
            for row in node_records
            if row.get("node_type") == "magazine"
        }
        - set(by_magazine)
    )
    lines = [
        "# KHDB Article Discovery Report",
        "",
        f"- nodes discovered: {len(node_records)}",
        f"- candidate article pages discovered: {len(article_records)}",
        f"- crawl graph edges: {len(graph)}",
        "",
        "## Counts by Magazine",
        "",
    ]
    for magazine, count in sorted(by_magazine.items()):
        lines.append(f"- {magazine}: {count}")
    lines.extend(["", "## Counts by Discovery Strategy", ""])
    for strategy, count in sorted(by_strategy.items()):
        lines.append(f"- {strategy}: {count}")
    lines.extend(["", "## Nodes by Depth", ""])
    for depth, count in sorted(by_depth.items()):
        lines.append(f"- depth {depth}: {count}")
    lines.extend(
        [
            "",
            "## Dynamic Endpoint Requests",
            "",
            f"- dynamic child links recorded: {dynamic_request_count}",
            "",
            "## Duplicate IDs",
            "",
            f"- duplicate article IDs: {len(duplicate_article_ids)}",
            f"- duplicate node IDs: {len(duplicate_node_ids)}",
        ]
    )
    if duplicate_article_ids:
        lines.append(f"- duplicate article ID examples: {', '.join(duplicate_article_ids[:20])}")
    if duplicate_node_ids:
        lines.append(f"- duplicate node ID examples: {', '.join(duplicate_node_ids[:20])}")
    lines.extend(["", "## Example Article URLs", ""])
    seen_magazines: set[str] = set()
    diverse_examples: list[dict] = []
    for row in article_records:
        magazine = row.get("magazine_title", "")
        if magazine not in seen_magazines:
            diverse_examples.append(row)
            seen_magazines.add(magazine)
        if len(diverse_examples) >= 20:
            break
    if len(diverse_examples) < 20:
        for row in article_records:
            if row not in diverse_examples:
                diverse_examples.append(row)
            if len(diverse_examples) >= 20:
                break
    for row in diverse_examples[:20]:
        lines.append(f"- {row.get('khdb_id')}: {row.get('url')}")
    lines.extend(["", "## Magazines with Zero Article Pages in This Run", ""])
    for magazine in zero_magazines:
        lines.append(f"- {magazine}")
    lines.extend(["", "## Unresolved Issues", ""])
    if unresolved:
        for issue in unresolved[:100]:
            lines.append(f"- {issue}")
    else:
        lines.append("- None recorded in this bounded run.")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    magazines = [row for row in read_jsonl(args.magazines_index) if row.get("magazine_url")]
    if args.magazine_title:
        magazines = [row for row in magazines if row.get("magazine_title") == args.magazine_title]
    if args.max_magazines is not None:
        magazines = magazines[: args.max_magazines]
    if not magazines and args.seed_article_url:
        magazines = [{"magazine_title": "", "khdb_id": "", "magazine_url": ""}]

    all_articles: list[dict] = []
    all_nodes: list[dict] = []
    all_graph: list[dict] = []
    all_unresolved: list[str] = []
    for magazine in magazines:
        records, nodes, graph, unresolved = crawl_magazine(magazine, args)
        all_articles.extend(records)
        all_nodes.extend(nodes)
        all_graph.extend(graph)
        all_unresolved.extend(unresolved)
        if args.debug:
            print(
                f"{magazine.get('magazine_title', '(seed-only)')}: "
                f"nodes={len(nodes)} candidate_articles={len(records)} links={len(graph)}"
            )

    write_jsonl(args.output_index, all_articles)
    nodes_path = args.nodes_index or args.output_index.parent / "nodes_index.jsonl"
    graph_path = args.graph_index or args.output_index.parent / "crawl_graph.jsonl"
    write_jsonl(nodes_path, all_nodes)
    write_jsonl(graph_path, all_graph)
    summary = {
        "magazines_processed": len(magazines),
        "nodes_discovered": len(all_nodes),
        "candidate_article_pages": len(all_articles),
        "crawl_edges": len(all_graph),
        "discovery_mode": args.discovery_mode,
        "index_only": args.index_only,
        "output_index": str(args.output_index),
        "nodes_index": str(nodes_path),
        "crawl_graph": str(graph_path),
        "unresolved_issue_count": len(all_unresolved),
    }
    summary_path = args.summary or args.output_index.parent / "download_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path = args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(all_articles, all_nodes, all_graph, all_unresolved), encoding="utf-8")
    print(f"Nodes discovered: {len(all_nodes)}")
    print(f"Candidate article pages: {len(all_articles)}")
    print(f"Index: {args.output_index}")
    print(f"Nodes: {nodes_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
