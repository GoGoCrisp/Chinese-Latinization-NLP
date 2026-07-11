from __future__ import annotations

import csv
import hashlib
import html
import json
import re
import time
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qs, urlencode, urljoin, urlparse, urlunparse
from urllib.request import Request, urlopen


ROOT_URL = "https://db.history.go.kr/modern/level.do?itemId=ma"
KHDB_HOST = "db.history.go.kr"

TARGET_MAGAZINES = [
    "기호흥학회월보",
    "대동학회월보",
    "대조선독립협회회보",
    "대한유학생회학보",
    "대한자강회월보",
    "대한학회월보",
    "대한협회회보",
    "대한흥학보",
    "서북학회월보",
    "서우",
    "태극학보",
    "호남학보",
    "개벽",
    "대동아",
    "동광",
    "만국부인",
    "별건곤",
    "삼천리",
    "삼천리문학",
]

UI_TEXT_LINES = {
    "검색",
    "목록",
    "이전",
    "다음",
    "오류신고",
    "URL 복사",
    "이미지",
    "원문이미지",
}

BLOCK_TAGS = {
    "address",
    "article",
    "aside",
    "blockquote",
    "br",
    "dd",
    "div",
    "dl",
    "dt",
    "figcaption",
    "figure",
    "footer",
    "form",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hr",
    "li",
    "main",
    "nav",
    "ol",
    "p",
    "pre",
    "section",
    "table",
    "td",
    "th",
    "tr",
    "ul",
}

DROP_TEXT_TAGS = {"script", "style", "noscript", "svg"}

HANGUL_RE = re.compile(r"[\uac00-\ud7a3]")
HANJA_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
HIRAGANA_RE = re.compile(r"[\u3040-\u309f]")
KATAKANA_RE = re.compile(r"[\u30a0-\u30ff]")
LATIN_RE = re.compile(r"[A-Za-z]")
DIGIT_RE = re.compile(r"\d")
PUNCT_RE = re.compile(r"[^\w\s\uac00-\ud7a3\u3400-\u4dbf\u4e00-\u9fff]", re.UNICODE)
KHDB_ID_RE = re.compile(r"ma_\d+(?:_\d+)*")


@dataclass
class Node:
    tag: str
    attrs: dict[str, str] = field(default_factory=dict)
    parent: "Node | None" = None
    children: list["Node"] = field(default_factory=list)
    text_parts: list[str] = field(default_factory=list)

    def append_text(self, text: str) -> None:
        if text:
            self.text_parts.append(text)


class TreeBuilder(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = Node("document")
        self.current = self.root

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        node = Node(tag=tag, attrs={k.lower(): v or "" for k, v in attrs}, parent=self.current)
        self.current.children.append(node)
        if tag not in {"br", "img", "meta", "link", "input"}:
            self.current = node
        if tag == "br":
            self.current.append_text("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        cursor = self.current
        while cursor.parent is not None:
            if cursor.tag == tag:
                self.current = cursor.parent
                return
            cursor = cursor.parent

    def handle_data(self, data: str) -> None:
        self.current.append_text(data)


def parse_html_tree(source: str) -> Node:
    parser = TreeBuilder()
    parser.feed(source)
    parser.close()
    return parser.root


def iter_nodes(root: Node) -> Iterable[Node]:
    yield root
    for child in root.children:
        yield from iter_nodes(child)


def node_text(node: Node, include_children: bool = True) -> str:
    pieces: list[str] = []

    def walk(item: Node, muted: bool = False) -> None:
        muted = muted or item.tag in DROP_TEXT_TAGS
        if not muted:
            if item.tag in BLOCK_TAGS:
                pieces.append("\n")
            pieces.extend(item.text_parts)
        if include_children:
            for child in item.children:
                walk(child, muted)
        if not muted and item.tag in BLOCK_TAGS:
            pieces.append("\n")

    walk(node)
    return normalize_text("".join(pieces))


def normalize_text(text: str) -> str:
    text = html.unescape(unicodedata.normalize("NFKC", text or ""))
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def visible_lines(text: str) -> list[str]:
    return [line.strip() for line in normalize_text(text).splitlines() if line.strip()]


def extract_title(root: Node) -> str:
    for node in iter_nodes(root):
        if node.tag == "title":
            return node_text(node)
    for tag in ("h1", "h2"):
        for node in iter_nodes(root):
            if node.tag == tag:
                title = node_text(node)
                if title:
                    return title
    return ""


def extract_links(root: Node, base_url: str) -> list[dict]:
    links: list[dict] = []
    seen: set[str] = set()
    for node in iter_nodes(root):
        if node.tag == "a":
            href = node.attrs.get("href", "").strip()
            if href and not href.startswith("#") and not href.lower().startswith("javascript:"):
                url = canonical_url(urljoin(base_url, href))
                if url not in seen:
                    links.append({"url": url, "text": node_text(node), "href": href})
                    seen.add(url)
        onclick = node.attrs.get("onclick", "")
        for match in re.finditer(r"moveDetailList\(['\"](ma_\d+(?:_\d+)*)['\"]\)", onclick):
            khdb_id = match.group(1)
            url = canonical_url(urljoin(base_url, f"/modern/level.do?levelId={khdb_id}"))
            if url not in seen:
                links.append({"url": url, "text": node_text(node), "href": onclick})
                seen.add(url)
        for match in re.finditer(r"goData\(['\"][^'\"]+['\"],\s*['\"](ma_\d+(?:_\d+)*)['\"]\)", onclick):
            khdb_id = match.group(1)
            path = f"/id/{khdb_id}" if len(khdb_id.split("_")) >= 4 else f"/modern/level.do?levelId={khdb_id}"
            url = canonical_url(urljoin(base_url, path))
            if url not in seen:
                links.append({"url": url, "text": node_text(node), "href": onclick})
                seen.add(url)
    return links


def canonical_url(url: str) -> str:
    parsed = urlparse(url)
    scheme = parsed.scheme or "https"
    netloc = parsed.netloc or KHDB_HOST
    path = parsed.path or "/"
    query_pairs = parse_qs(parsed.query, keep_blank_values=True)
    query = urlencode(sorted((k, v) for k, values in query_pairs.items() for v in values))
    return urlunparse((scheme, netloc, path, "", query, ""))


def is_allowed_khdb_magazine_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.netloc and parsed.netloc != KHDB_HOST:
        return False
    if re.search(r"(login|search|newspaper|image|img|pdf|download)", url, re.I):
        return False
    return "/id/ma_" in parsed.path or ("modern/level.do" in parsed.path and "ma" in parsed.query)


def khdb_id_from_url_or_text(value: str) -> str:
    match = KHDB_ID_RE.search(value or "")
    if match:
        return match.group(0)
    parsed = urlparse(value or "")
    query = parse_qs(parsed.query)
    for key in ("levelId", "itemId", "setId"):
        for candidate in query.get(key, []):
            match = KHDB_ID_RE.search(candidate)
            if match:
                return match.group(0)
    return ""


def cache_path_for_url(cache_dir: Path, url: str) -> Path:
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    khdb_id = khdb_id_from_url_or_text(url)
    stem = khdb_id or re.sub(r"[^A-Za-z0-9]+", "_", urlparse(url).path).strip("_") or "page"
    return cache_dir / f"{stem}_{digest}.html"


def decode_bytes(payload: bytes, content_type: str = "") -> tuple[str, str]:
    encodings: list[str] = []
    header_match = re.search(r"charset=([^;\s]+)", content_type or "", re.I)
    if header_match:
        encodings.append(header_match.group(1).strip("\"'"))
    head = payload[:2048].decode("ascii", errors="ignore")
    meta_match = re.search(r"charset=[\"']?([A-Za-z0-9_-]+)", head, re.I)
    if meta_match:
        encodings.append(meta_match.group(1))
    encodings.extend(["utf-8", "cp949", "euc-kr"])
    seen: set[str] = set()
    for encoding in encodings:
        encoding = encoding.lower()
        if encoding in seen:
            continue
        seen.add(encoding)
        try:
            return payload.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return payload.decode("utf-8", errors="replace"), "utf-8-replace"


def fetch_url(url: str, cache_dir: Path, delay: float = 1.0, force: bool = False) -> dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    url = canonical_url(url)
    path = cache_path_for_url(cache_dir, url)
    if path.exists() and not force:
        text = path.read_text(encoding="utf-8", errors="replace")
        return {
            "url": url,
            "status": "cached",
            "content_type": "text/html",
            "encoding": "utf-8",
            "text": text,
            "path": path,
            "fetched_at": "",
        }

    if delay > 0:
        time.sleep(delay)
    request = Request(
        url,
        headers={
            "User-Agent": "EMNLP-Chinese-Latinization-KHDB-audit/0.1 (polite research crawler)",
            "Accept": "text/html,application/xhtml+xml",
        },
    )
    with urlopen(request, timeout=30) as response:
        payload = response.read()
        content_type = response.headers.get("content-type", "")
        text, encoding = decode_bytes(payload, content_type)
    path.write_text(text, encoding="utf-8")
    return {
        "url": url,
        "status": "downloaded",
        "content_type": content_type,
        "encoding": encoding,
        "text": text,
        "path": path,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def count_chars(text: str) -> dict:
    chars = [ch for ch in text if not ch.isspace()]
    body_length = len(chars)
    hangul_count = sum(1 for ch in chars if HANGUL_RE.match(ch))
    hanja_count = sum(1 for ch in chars if HANJA_RE.match(ch))
    hiragana_count = sum(1 for ch in chars if HIRAGANA_RE.match(ch))
    katakana_count = sum(1 for ch in chars if KATAKANA_RE.match(ch))
    latin_count = sum(1 for ch in chars if LATIN_RE.match(ch))
    digit_count = sum(1 for ch in chars if DIGIT_RE.match(ch))
    punctuation_count = sum(1 for ch in chars if PUNCT_RE.match(ch))
    script_denominator = max(1, hanja_count + hangul_count)
    return {
        "body_length_chars": body_length,
        "hangul_count": hangul_count,
        "hanja_count": hanja_count,
        "hiragana_count": hiragana_count,
        "katakana_count": katakana_count,
        "latin_count": latin_count,
        "digit_count": digit_count,
        "punctuation_count": punctuation_count,
        "hanja_ratio": hanja_count / script_denominator,
        "hangul_ratio": hangul_count / script_denominator,
        "japanese_kana_ratio": (hiragana_count + katakana_count) / max(1, body_length),
        "line_count": len(visible_lines(text)),
        "paragraph_count": len([p for p in re.split(r"\n\s*\n", normalize_text(text)) if p.strip()]),
    }


def find_metadata(lines: list[str]) -> dict[str, str]:
    label_map = {
        "잡지명": "magazine_title",
        "발행일": "publication_date",
        "기사제목": "article_title",
        "제목": "article_title",
        "필자": "author",
        "기사형태": "article_type",
        "형태분류": "article_type",
        "호수": "issue_title",
    }
    metadata: dict[str, str] = {}
    for i, line in enumerate(lines):
        compact = re.sub(r"\s+", "", line)
        for label, field_name in label_map.items():
            if field_name in metadata:
                continue
            if compact == label and i + 1 < len(lines):
                metadata[field_name] = lines[i + 1]
            else:
                match = re.match(rf"^{re.escape(label)}\s*[:：]?\s*(.+)$", line)
                if match:
                    metadata[field_name] = match.group(1).strip()
    return metadata


def candidate_content_nodes(root: Node) -> list[dict]:
    candidates: list[dict] = []
    labels = ("잡지명", "발행일", "기사제목", "필자", "기사형태", "본문")
    for node in iter_nodes(root):
        if node.tag not in {"article", "main", "section", "div", "td", "table"}:
            continue
        text = node_text(node)
        clean_len = len(re.sub(r"\s+", "", text))
        if clean_len < 80:
            continue
        label_hits = [label for label in labels if label in text]
        class_id = " ".join(
            value for key, value in node.attrs.items() if key in {"class", "id"} and value
        )
        candidates.append(
            {
                "tag": node.tag,
                "class_id": class_id,
                "text_length": clean_len,
                "label_hits": label_hits,
                "preview": normalize_text(text)[:500],
            }
        )
    return sorted(candidates, key=lambda item: (len(item["label_hits"]), item["text_length"]), reverse=True)


def grouped_counts(rows: Iterable[dict], key: str) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row.get(key, ""))] += 1
    return dict(counts)
