"""Direct retrieval from the official pages declared as module seeds."""
from __future__ import annotations

import asyncio
import unicodedata
from io import BytesIO
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx
from pypdf import PdfReader

from ..citations import canonical_source_url
from ..index.corpus import chunk_text, clean_html
from .base import Tool, ToolContext
from .web_search import archive_warning

_STOPWORDS = {
    "and", "are", "can", "current", "for", "from", "how", "new", "nyc", "official",
    "the", "their", "this", "what", "with", "york",
}


def _words(text: str) -> list[str]:
    words: list[str] = []
    current: list[str] = []
    for char in text:
        if char.isalnum() or (
            current and unicodedata.category(char).startswith("M")
        ):
            current.append(char)
        elif current:
            words.append("".join(current))
            current = []
    if current:
        words.append("".join(current))
    return words


def _terms(text: str) -> set[str]:
    return {
        word.lower()
        for word in _words(text)
        if len(word) > 2 or not word.isascii()
    } - _STOPWORDS


def _relevant_chunks(text: str, query: str, limit: int = 2) -> list[str]:
    chunks = chunk_text(text, max_chars=1800, overlap=180)
    wanted = _terms(query)
    scored = [
        (
            index,
            chunk,
            sum(
                term in _terms(chunk)
                or (not term.isascii() and term in chunk.lower())
                for term in wanted
            ),
        )
        for index, chunk in enumerate(chunks)
    ]
    ranked = sorted(
        (item for item in scored if item[2] > 0),
        key=lambda item: (item[2], -item[0]),
        reverse=True,
    )[:limit]
    return [chunk for _, chunk, _score in ranked]


def _approval_key(url: str) -> str:
    parts = urlsplit(canonical_source_url(url))
    path = parts.path[:-1] if parts.path.endswith("/") else parts.path
    return urlunsplit(parts._replace(path=path))


def _host_matches(host: str, domains: set[str]) -> bool:
    return any(host == domain or host.endswith("." + domain) for domain in domains)


def _url_approved(url: str, seeded: set[str], domains: set[str]) -> bool:
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    if parts.scheme != "https" or parts.username is not None or parts.password is not None:
        return False
    return _approval_key(url) in seeded or _host_matches(host, domains)


async def _fetch_official(
    url: str,
    client,
    seeded: set[str],
    domains: set[str],
) -> tuple[str, str, str]:
    current = url
    for _ in range(4):
        response = await client.get(
            current, follow_redirects=False, headers={"User-Agent": "HeyNYC/0.1"},
        )
        if getattr(response, "status_code", 200) in {301, 302, 303, 307, 308}:
            location = response.headers.get("location")
            target = urljoin(current, location or "")
            if not location or not _url_approved(target, seeded, domains):
                raise ValueError("official source redirected outside the curated sources")
            current = target
            continue
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").lower()
        if "application/pdf" in content_type or response.content.startswith(b"%PDF"):
            reader = PdfReader(BytesIO(response.content))
            text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
            return current, "Official PDF", text
        title, text = clean_html(response.text)
        return current, title, text
    raise ValueError("official source redirected too many times")


def official_source_tools() -> list[Tool]:
    async def _handler(args: dict, ctx: ToolContext) -> str:
        seeds = ctx.registry.seeds()
        source_tiers = ctx.registry.source_tiers()
        non_authoritative_domains = {
            domain
            for domain, (tier, _module) in source_tiers.items()
            if tier != "authoritative"
        }
        approved = {
            _approval_key(url)
            for url in seeds
            if not _host_matches(
                (urlsplit(url).hostname or "").lower(),
                non_authoritative_domains,
            )
        }
        domains = {
            domain
            for domain, (tier, _module) in source_tiers.items()
            if tier == "authoritative"
        }
        domains.update(
            domain.lower()
            for domain in ctx.registry.allowlist()
            if domain.lower().endswith(".gov")
            and not _host_matches(domain.lower(), non_authoritative_domains)
        )
        urls = list(dict.fromkeys(args["urls"]))[:4]
        rejected = [
            url for url in urls if not _url_approved(url, approved, domains)
        ]
        if rejected:
            return "That URL is not an approved official source declared by a HeyNYC module."

        own_client = ctx.http is None
        client = ctx.http or httpx.AsyncClient(timeout=20.0)
        try:
            fetched = await asyncio.gather(
                *(
                    _fetch_official(url, client, approved, domains)
                    for url in urls
                ),
                return_exceptions=True,
            )
        finally:
            if own_client:
                await client.aclose()

        blocks: list[str] = []
        emitted: set[tuple[str, str]] = set()
        for _url, result in zip(urls, fetched):
            if isinstance(result, Exception):
                continue
            final_url, title, text = result
            source_text = f"{title}\n{text}".lower()
            if any(
                marker in source_text
                for marker in (
                    "access denied",
                    "complete the security challenge",
                    "enable javascript and cookies to continue",
                )
            ):
                continue
            chunks = _relevant_chunks(text, args["query"])
            if not chunks:
                continue
            evidence = "\n\n".join(chunks)
            warning = archive_warning(final_url, title)
            if warning:
                evidence = f"{warning}\n\n{evidence}"
            block_key = (_approval_key(final_url), evidence)
            if block_key in emitted:
                continue
            emitted.add(block_key)
            cite = ctx.citations.register(
                final_url,
                snippet=evidence,
                title=title or "Official source",
                kind="WEB",
                provenance={
                    "evidence_grade": "discovery" if warning else "authoritative",
                },
            )
            blocks.append(
                f"{title or 'Official source'} ({final_url})\n{evidence} {{cite:{cite}}}"
            )
        if not blocks:
            return "The approved official pages could not be retrieved. Do not guess; route to 311."
        return "\n\n".join(blocks)

    return [
        Tool(
            name="official_sources",
            description=(
                "Fetch current text directly from official module seeds or pages on curated official "
                "domains. Use it when a known civic rule or workflow needs current source text without "
                "relying on search ranking. Other URLs are rejected."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "urls": {
                        "type": "array", "items": {"type": "string"}, "minItems": 1,
                        "maxItems": 4,
                        "description": (
                            "Official module seed URLs or discovered HTTPS pages on curated "
                            "official domains to retrieve."
                        ),
                    },
                    "query": {"type": "string", "description": "Claims to find on those pages."},
                },
                "required": ["urls", "query"],
            },
            handler=_handler,
            open_world=True,
        )
    ]
