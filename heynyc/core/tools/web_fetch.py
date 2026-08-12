"""Secure local retrieval for one known public URL."""
from __future__ import annotations

import ipaddress
import unicodedata
from io import BytesIO
from pathlib import Path
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx
from pydantic_ai._ssrf import safe_download, validate_and_resolve_url
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
_MAX_RESPONSE_BYTES = 5_000_000
_MIN_STATIC_TEXT_CHARS = 200


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


def _url_safe_shape(url: str) -> bool:
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    if (
        parts.scheme != "https"
        or not host
        or parts.username is not None
        or parts.password is not None
    ):
        return False
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return True
    return address.is_global


async def _validate_public_url(url: str) -> None:
    if not _url_safe_shape(url):
        raise ValueError("URL must be public HTTPS without credentials")
    await validate_and_resolve_url(url, allow_local=False)


def _is_access_wall(title: str, text: str) -> bool:
    source_text = f"{title}\n{text}".lower()
    return any(marker in source_text for marker in _ACCESS_WALL_MARKERS)


async def _route_public_request(route, request) -> None:
    if urlsplit(request.url).scheme in {"about", "blob", "data"}:
        await route.continue_()
        return
    try:
        await _validate_public_url(request.url)
    except ValueError:
        await route.abort()
    else:
        await route.continue_()


async def _fetch_rendered_page(
    url: str,
) -> tuple[str, str, str]:
    from playwright.async_api import TimeoutError as PlaywrightTimeoutError
    from playwright.async_api import async_playwright

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(**_browser_launch_options())
        page = None
        try:
            context = await browser.new_context(**_browser_context_options())
            page = await context.new_page()
            await page.route("**/*", _route_public_request)
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
            await _validate_public_url(final_url)
            visible_text = await page.locator("body").inner_text()
            html = await page.content()
        finally:
            if page is not None:
                await page.unroute_all(behavior="ignoreErrors")
            await browser.close()

    title, text = _rendered_page_text(html, visible_text)
    if not text.strip() or _is_access_wall(title, text):
        raise ValueError("rendered official source did not expose usable text")
    return final_url, title, text


async def _fetch_page(
    url: str,
    client,
) -> tuple[str, str, str]:
    if not _url_safe_shape(url):
        raise ValueError("URL must be public HTTPS without credentials")
    if client is None:
        response = await safe_download(
            url,
            allow_local=False,
            timeout=20,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (compatible; HeyNYC/0.1; +https://reach4help.org)"
                ),
            },
        )
        current = str(getattr(response, "url", "") or url)
        current_parts = urlsplit(current)
        try:
            ipaddress.ip_address(current_parts.hostname or "")
        except ValueError:
            pass
        else:
            request = getattr(response, "request", None)
            logical_host = request.headers.get("host") if request is not None else ""
            if logical_host:
                current = urlunsplit(current_parts._replace(netloc=logical_host))
        content_length = response.headers.get("content-length")
        if content_length and int(content_length) > _MAX_RESPONSE_BYTES:
            raise ValueError("page exceeds the fetch size limit")
        if len(response.content) > _MAX_RESPONSE_BYTES:
            raise ValueError("page exceeds the fetch size limit")
        return _extract_response(current, response)

    current = url
    for _ in range(4):
        await _validate_public_url(current)
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
            if not location or not _url_safe_shape(target):
                raise ValueError("page redirected outside the public web")
            await _validate_public_url(target)
            current = target
            continue
        if getattr(response, "status_code", 200) == 403:
            raise _RenderedFetchNeeded
        response.raise_for_status()
        if len(response.content) > _MAX_RESPONSE_BYTES:
            raise ValueError("page exceeds the fetch size limit")
        return _extract_response(current, response)
    raise ValueError("official source redirected too many times")


def _extract_response(url: str, response) -> tuple[str, str, str]:
    content_type = response.headers.get("content-type", "").lower()
    if "application/pdf" in content_type or response.content.startswith(b"%PDF"):
        reader = PdfReader(BytesIO(response.content))
        text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
        return url, "Official PDF", text
    title, text = clean_html(response.text)
    full_text = html2txt(response.text).strip()
    if full_text and full_text != text:
        text = f"{text}\n{full_text}"
    if not text.strip() or _is_access_wall(title, text):
        raise _RenderedFetchNeeded
    return url, title, text


async def _fetch_page_with_browser(
    url: str,
    client,
    query: str,
) -> tuple[str, str, str]:
    try:
        fetched = await _fetch_page(url, client)
    except _RenderedFetchNeeded:
        return await _fetch_rendered_page(url)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code != 403:
            raise
        return await _fetch_rendered_page(url)
    if (client is None and len(fetched[2].strip()) < _MIN_STATIC_TEXT_CHARS) or (
        query and not _relevant_chunks(fetched[2], query)
    ):
        return await _fetch_rendered_page(url)
    return fetched


def web_fetch_tools() -> list[Tool]:
    async def _handler(args: dict, ctx: ToolContext) -> str:
        url = str(args["url"]).strip()
        query = str(args.get("query") or "").strip()
        try:
            final_url, title, text = await _fetch_page_with_browser(
                url,
                ctx.http,
                query,
            )
        except Exception:
            return (
                "The page could not be fetched. Do not guess. Preserve other "
                "verified results and state which requested claim could not be verified; route to "
                "the relevant official service when no useful result remains."
            )
        chunks = _relevant_chunks(text, query) if query else chunk_text(text)[:2]
        if not chunks:
            return (
                "The page was fetched but did not contain text relevant to the requested claim. "
                "Do not guess from the page title or navigation."
            )
        evidence = "\n\n".join(chunks)
        warning = archive_warning(final_url, title)
        from .web_search import _tier_of

        tier = _tier_of(
            final_url, ctx.registry.source_tiers(), ctx.registry.news_tier(),
        )
        authoritative = tier == "authoritative" and not warning
        if warning:
            evidence = f"{warning}\n\n{evidence}"
        provenance = {
            "evidence_grade": "authoritative" if authoritative else "discovery",
            "source_tier": tier,
        }
        cite = ctx.citations.register(
            final_url,
            snippet=evidence,
            title=title or "Fetched page",
            kind="WEB",
            provenance=provenance,
        )
        return f"{title or 'Fetched page'} ({final_url})\n{evidence} {{cite:{cite}}}"

    return [
        Tool(
            name="web_fetch",
            description=(
                "Fetch and extract one known public web page. Use it after search when the page itself "
                "is needed as evidence. Source trust is graded separately; fetching a page does not "
                "make it authoritative."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Public HTTPS URL to fetch."},
                    "query": {"type": "string", "description": "Optional claim or detail to find."},
                },
                "required": ["url"],
            },
            handler=_handler,
            open_world=True,
        )
    ]
