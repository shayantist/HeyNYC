"""Corpus builder: fetch module seed pages → clean → chunk → embed → store.

Uses httpx (not automate's Playwright fetch) to keep HeyNYC dependency-light and
OSS-isolated. nyc.gov / nyctourism content pages are server-rendered, so a plain
GET + HTML strip is enough for the curated "skeleton" index.
"""
from __future__ import annotations

import re
from typing import Optional

import httpx

from ..registry import Registry
from .embedder import Embedder
from .store import IndexDoc, VectorStore

_SCRIPT_RE = re.compile(r"<(script|style|noscript|svg)[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.DOTALL | re.IGNORECASE)
_WS_RE = re.compile(r"\s+")


def clean_html(html: str) -> tuple[str, str]:
    """Return (title, plain_text) from raw HTML."""
    title_match = _TITLE_RE.search(html)
    title = _WS_RE.sub(" ", _TAG_RE.sub("", title_match.group(1))).strip() if title_match else ""
    body = _SCRIPT_RE.sub(" ", html)
    body = _TAG_RE.sub(" ", body)
    body = re.sub(r"&[a-z]+;", " ", body)
    body = _WS_RE.sub(" ", body).strip()
    return title, body


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 150) -> list[str]:
    """Split text into overlapping char windows on whitespace boundaries."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            space = text.rfind(" ", start + max_chars - overlap, end)
            if space > start:
                end = space
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]


async def fetch_clean(url: str, client: httpx.AsyncClient) -> tuple[str, str]:
    response = await client.get(
        url,
        follow_redirects=True,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; HeyNYC/0.1; +https://reach4help.org)",
        },
    )
    response.raise_for_status()
    return clean_html(response.text)


async def build_index(
    registry: Registry,
    store: VectorStore,
    embedder: Embedder,
    client: Optional[httpx.AsyncClient] = None,
) -> dict:
    """Fetch every module seed, chunk, embed, and replace the store corpus.

    Returns a summary {urls, ok, failed, chunks}. Failures are recorded, not raised,
    so one dead seed doesn't sink the whole build.
    """
    own = client is None
    client = client or httpx.AsyncClient(timeout=30.0)
    summary = {"urls": 0, "ok": 0, "failed": [], "chunks": 0}
    documents: list[IndexDoc] = []
    try:
        for module in registry.modules:
            for url in module.seeds:
                summary["urls"] += 1
                try:
                    title, text = await fetch_clean(url, client)
                except Exception as exc:  # dead/blocked seed, record and move on
                    summary["failed"].append({"url": url, "error": str(exc)})
                    continue
                chunks = chunk_text(text)
                if not chunks:
                    summary["failed"].append({"url": url, "error": "no text extracted"})
                    continue
                vectors = embedder.embed(chunks)
                if len(vectors) != len(chunks):
                    raise RuntimeError(
                        f"embedder returned {len(vectors)} vector(s) for {len(chunks)} chunk(s)"
                    )
                docs = [
                    IndexDoc(
                        id=f"{module.name}::{url}::{i}",
                        text=chunk,
                        url=url,
                        title=title or module.name,
                        module=module.name,
                        vector=vec,
                    )
                    for i, (chunk, vec) in enumerate(zip(chunks, vectors))
                ]
                documents.extend(docs)
                summary["ok"] += 1
                summary["chunks"] += len(docs)
        if documents and not summary["failed"]:
            store.replace(documents)
    finally:
        if own:
            await client.aclose()
    return summary
