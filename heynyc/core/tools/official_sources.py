"""Direct retrieval from the official pages declared as module seeds."""
from __future__ import annotations

import asyncio
import unicodedata
from io import BytesIO
from pathlib import Path
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx
from pypdf import PdfReader
from trafilatura import html2txt

from ..citations import canonical_source_url
from ..index.corpus import chunk_text, clean_html
from .base import Tool, ToolContext
from .web_search import archive_warning

_STOPWORDS = {
    "and", "are", "can", "current", "for", "from", "how", "new", "nyc", "official",
    "the", "their", "this", "what", "with", "york",
}
_ACCESS_WALL_MARKERS = (
    "access denied",
    "complete the security challenge",
    "enable javascript and cookies to continue",
    "javascript is required",
)
_BROWSER_EXECUTABLE_CANDIDATES = (
    Path("/Applications/Brave Browser.app/Contents/MacOS/Brave Browser"),
)


class _RenderedFetchNeeded(Exception):
    pass


def _browser_launch_options() -> dict:
    options = {"headless": True}
    executable = next(
        (path for path in _BROWSER_EXECUTABLE_CANDIDATES if path.is_file()),
        None,
    )
    if executable is not None:
        options["executable_path"] = str(executable)
    return options


def _browser_context_options() -> dict:
    return {
        "java_script_enabled": True,
        "timezone_id": "America/New_York",
        "user_agent": "Mozilla/5.0 (compatible; HeyNYC/0.1; +https://reach4help.org)",
    }


def _rendered_page_text(html: str, visible_text: str) -> tuple[str, str]:
    title, text = clean_html(html)
    if visible_text.strip():
        return title, visible_text.strip()
    full_text = html2txt(html).strip()
    return title, full_text if full_text and full_text != text else text


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


def _is_access_wall(title: str, text: str) -> bool:
    source_text = f"{title}\n{text}".lower()
    return any(marker in source_text for marker in _ACCESS_WALL_MARKERS)


async def _fetch_rendered_official(
    url: str,
    seeded: set[str],
    domains: set[str],
) -> tuple[str, str, str]:
    from playwright.async_api import TimeoutError as PlaywrightTimeoutError
    from playwright.async_api import async_playwright

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(**_browser_launch_options())
        try:
            context = await browser.new_context(**_browser_context_options())
            page = await context.new_page()

            async def allow_approved_requests(route, request):
                parts = urlsplit(request.url)
                if parts.scheme in {"about", "blob", "data"}:
                    await route.continue_()
                elif _url_approved(request.url, seeded, domains):
                    await route.continue_()
                else:
                    await route.abort()

            await page.route("**/*", allow_approved_requests)
            response = await page.goto(url, wait_until="domcontentloaded", timeout=15_000)
            if response is not None and response.status >= 400:
                raise ValueError(f"rendered official source returned HTTP {response.status}")
            try:
                await page.wait_for_function(
                    "document.body && document.body.innerText.trim().length > 80",
                    timeout=3_000,
                )
            except PlaywrightTimeoutError:
                pass
            final_url = page.url
            if not _url_approved(final_url, seeded, domains):
                raise ValueError("rendered official source left the curated sources")
            visible_text = await page.locator("body").inner_text()
            html = await page.content()
        finally:
            await browser.close()

    title, text = _rendered_page_text(html, visible_text)
    if not text.strip() or _is_access_wall(title, text):
        raise ValueError("rendered official source did not expose usable text")
    return final_url, title, text


async def _fetch_official(
    url: str,
    client,
    seeded: set[str],
    domains: set[str],
) -> tuple[str, str, str]:
    current = url
    for _ in range(4):
        response = await client.get(
            current,
            follow_redirects=False,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (compatible; HeyNYC/0.1; +https://reach4help.org)"
                ),
            },
        )
        if getattr(response, "status_code", 200) in {301, 302, 303, 307, 308}:
            location = response.headers.get("location")
            target = urljoin(current, location or "")
            if not location or not _url_approved(target, seeded, domains):
                raise ValueError("official source redirected outside the curated sources")
            current = target
            continue
        if getattr(response, "status_code", 200) == 403:
            raise _RenderedFetchNeeded
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").lower()
        if "application/pdf" in content_type or response.content.startswith(b"%PDF"):
            reader = PdfReader(BytesIO(response.content))
            text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
            return current, "Official PDF", text
        title, text = clean_html(response.text)
        full_text = html2txt(response.text).strip()
        if full_text and full_text != text:
            text = f"{text}\n{full_text}"
        if not text.strip() or _is_access_wall(title, text):
            raise _RenderedFetchNeeded
        return current, title, text
    raise ValueError("official source redirected too many times")


async def _fetch_official_with_browser(
    url: str,
    client,
    seeded: set[str],
    domains: set[str],
) -> tuple[str, str, str]:
    try:
        return await _fetch_official(url, client, seeded, domains)
    except _RenderedFetchNeeded:
        return await _fetch_rendered_official(url, seeded, domains)


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
        requested_urls = list(dict.fromkeys(args["urls"]))[:4]
        rejected = [
            url for url in requested_urls if not _url_approved(url, approved, domains)
        ]
        urls = [url for url in requested_urls if url not in rejected]
        if not urls:
            return "That URL is not an approved official source declared by a HeyNYC module."

        own_client = ctx.http is None
        client = ctx.http or httpx.AsyncClient(timeout=20.0)
        try:
            fetched = await asyncio.gather(
                *(
                    _fetch_official_with_browser(url, client, approved, domains)
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
            if _is_access_wall(title, text):
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
            return (
                "The approved official pages could not be retrieved. Do not guess. Preserve other "
                "verified results and state which requested claim could not be verified; route to "
                "the relevant official service when no useful result remains."
            )
        if rejected:
            blocks.insert(0, "One requested URL was not approved and was not fetched.")
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
