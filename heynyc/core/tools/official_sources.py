"""Direct retrieval from the official pages declared as module seeds."""
from __future__ import annotations

import asyncio
from io import BytesIO
import re

import httpx
from pypdf import PdfReader

from ..index.corpus import chunk_text, clean_html
from .base import Tool, ToolContext

_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_STOPWORDS = {
    "and", "are", "can", "current", "for", "from", "how", "new", "nyc", "official",
    "the", "their", "this", "what", "with", "york",
}


def _terms(text: str) -> set[str]:
    return {word.lower() for word in _WORD_RE.findall(text) if len(word) > 2} - _STOPWORDS


def _relevant_chunks(text: str, query: str, limit: int = 2) -> list[str]:
    chunks = chunk_text(text, max_chars=1800, overlap=180)
    wanted = _terms(query)
    scored = [
        (index, chunk, len(_terms(chunk) & wanted))
        for index, chunk in enumerate(chunks)
    ]
    ranked = sorted(
        (item for item in scored if item[2] > 0),
        key=lambda item: (item[2], -item[0]),
        reverse=True,
    )[:limit]
    return [chunk for _, chunk, _score in ranked]


async def _fetch_official(url: str, client) -> tuple[str, str]:
    response = await client.get(
        url, follow_redirects=True, headers={"User-Agent": "HeyNYC/0.1"},
    )
    response.raise_for_status()
    content_type = response.headers.get("content-type", "").lower()
    if "application/pdf" in content_type or response.content.startswith(b"%PDF"):
        reader = PdfReader(BytesIO(response.content))
        text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
        return "Official PDF", text
    return clean_html(response.text)


def official_source_tools() -> list[Tool]:
    async def _handler(args: dict, ctx: ToolContext) -> str:
        approved = set(ctx.registry.seeds())
        urls = list(dict.fromkeys(args["urls"]))[:4]
        rejected = [url for url in urls if url not in approved]
        if rejected:
            return "That URL is not an approved official source declared by a HeyNYC module."

        own_client = ctx.http is None
        client = ctx.http or httpx.AsyncClient(timeout=20.0)
        try:
            fetched = await asyncio.gather(
                *(_fetch_official(url, client) for url in urls), return_exceptions=True,
            )
        finally:
            if own_client:
                await client.aclose()

        blocks: list[str] = []
        for url, result in zip(urls, fetched):
            if isinstance(result, Exception):
                continue
            title, text = result
            chunks = _relevant_chunks(text, args["query"])
            if not chunks:
                continue
            evidence = "\n\n".join(chunks)
            cite = ctx.citations.register(
                url, snippet=evidence, title=title or "Official source", kind="WEB",
            )
            blocks.append(f"{title or 'Official source'} ({url})\n{evidence} {{cite:{cite}}}")
        if not blocks:
            return "The approved official pages could not be retrieved. Do not guess; route to 311."
        return "\n\n".join(blocks)

    return [
        Tool(
            name="official_sources",
            description=(
                "Fetch current text directly from official pages already declared by HeyNYC modules. "
                "Use it when a known civic rule or workflow needs current source text without relying "
                "on search ranking. URLs outside the curated module seeds are rejected."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "urls": {
                        "type": "array", "items": {"type": "string"}, "minItems": 1,
                        "maxItems": 4, "description": "Official module seed URLs to retrieve.",
                    },
                    "query": {"type": "string", "description": "Claims to find on those pages."},
                },
                "required": ["urls", "query"],
            },
            handler=_handler,
            open_world=True,
        )
    ]
