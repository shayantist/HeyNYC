"""Secure local retrieval for one known public URL."""
from __future__ import annotations

import ipaddress
import re
import unicodedata
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Literal
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx
from pydantic import BaseModel, ConfigDict
from pydantic_ai._ssrf import safe_download, validate_and_resolve_url
from pypdf import PdfReader
from trafilatura import extract, html2txt

from .. import config
from ..citations import canonical_source_url
from ..index.corpus import chunk_text, clean_html
from .base import Tool, ToolContext
from .web_search import archive_warning, web_search_tools

_STOPWORDS = {
    "and", "are", "can", "current", "for", "from", "how", "new", "nyc", "official",
    "site", "the", "their", "this", "what", "with", "york",
}
_ACCESS_WALL_MARKERS = (
    "access denied",
    "complete the security challenge",
    "enable javascript to run this app",
    "enable javascript and cookies to continue",
    "javascript is required",
)
_BROWSER_EXECUTABLE_CANDIDATES = (
    Path("/Applications/Brave Browser.app/Contents/MacOS/Brave Browser"),
)
_MAX_RESPONSE_BYTES = 5_000_000
_MAX_FULL_PAGE_TOKENS = 2_000
_MIN_STATIC_TEXT_CHARS = 200


class _RenderedFetchNeeded(Exception):
    pass


class _UnsafePublicUrl(ValueError):
    pass


class WebFetchAcquisition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requested_url: str
    final_url: str
    citation_url: str
    route: Literal["http", "browser"]
    fetched_at: datetime
    status_code: int | None = None
    content_type: str | None = None
    body_bytes: int | None = None
    etag: str | None = None
    last_modified: str | None = None
    cache_control: str | None = None
    response_date: str | None = None


class _FetchedPage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    final_url: str
    title: str
    text: str
    acquisition: WebFetchAcquisition


def _acquisition(
    requested_url: str,
    final_url: str,
    route: Literal["http", "browser"],
    *,
    response=None,
    headers=None,
    body_bytes: int | None = None,
) -> WebFetchAcquisition:
    headers = headers or getattr(response, "headers", {}) or {}
    return WebFetchAcquisition(
        requested_url=requested_url,
        final_url=final_url,
        citation_url=canonical_source_url(final_url),
        route=route,
        fetched_at=datetime.now().astimezone(),
        status_code=getattr(response, "status_code", None) or getattr(response, "status", None),
        content_type=headers.get("content-type") or None,
        body_bytes=body_bytes,
        etag=headers.get("etag") or None,
        last_modified=headers.get("last-modified") or None,
        cache_control=headers.get("cache-control") or None,
        response_date=headers.get("date") or None,
    )


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
    title, text = _extract_html(html)
    if visible_text.strip():
        return title, visible_text.strip()
    return title, text


def _extract_html(html: str, url: str = "") -> tuple[str, str]:
    title, cleaned = clean_html(html)
    text = (
        extract(
            html,
            url=url or None,
            output_format="markdown",
            include_comments=False,
            include_tables=True,
            include_links=True,
            favor_recall=True,
        )
        or ""
    ).strip()
    return title, text or html2txt(html).strip() or cleaned.strip()


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
    wanted = _terms(query)
    chunks = []
    for index, chunk in enumerate(chunk_text(text, max_chars=1800, overlap=180)):
        if index:
            boundary = re.search(r'[.!?]["\'’”]?\s+', chunk)
            if boundary:
                tail = chunk[boundary.end():]
                lost_terms = (wanted & _terms(chunk[:boundary.end()])) - _terms(tail)
                chunk = chunk.partition(" ")[2] or chunk if lost_terms else tail
            else:
                chunk = chunk.partition(" ")[2] or chunk
        chunks.append(chunk)
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


def _text_tokens(text: str) -> int:
    import litellm

    return int(litellm.token_counter(model=config.HEYNYC_MODEL, text=text))


def _evidence_chunks(
    text: str, query: str, *, force_focus: bool = False,
) -> list[str]:
    if force_focus and query:
        return _relevant_chunks(text, query)
    if _text_tokens(text) <= _MAX_FULL_PAGE_TOKENS:
        return [text]
    return _relevant_chunks(text, query) if query else chunk_text(text)[:2]


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
        raise _UnsafePublicUrl("URL must be public HTTPS without credentials")
    try:
        await validate_and_resolve_url(url, allow_local=False)
    except Exception as exc:
        raise _UnsafePublicUrl("URL did not resolve to a public address") from exc


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
) -> _FetchedPage:
    from playwright.async_api import Error as PlaywrightError
    from playwright.async_api import TimeoutError as PlaywrightTimeoutError
    from playwright.async_api import async_playwright

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(**_browser_launch_options())
        page = None
        try:
            context = await browser.new_context(**_browser_context_options())
            page = await context.new_page()
            await page.route("**/*", _route_public_request)
            response = await page.goto(url, wait_until="load", timeout=15_000)
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
            response_headers = (
                await response.all_headers()
                if response is not None and hasattr(response, "all_headers")
                else {}
            )
            try:
                response_body = await response.body() if response is not None else None
            except PlaywrightError:
                response_body = None
        finally:
            if page is not None:
                await page.unroute_all(behavior="ignoreErrors")
            await browser.close()

    title, text = _rendered_page_text(html, visible_text)
    if not text.strip() or _is_access_wall(title, text):
        raise ValueError("rendered official source did not expose usable text")
    return _FetchedPage(
        final_url=final_url,
        title=title,
        text=text,
        acquisition=_acquisition(
            url,
            final_url,
            "browser",
            response=response,
            headers=response_headers,
            body_bytes=len(response_body) if response_body is not None else None,
        ),
    )


async def _fetch_page(
    url: str,
    client,
) -> _FetchedPage:
    if not _url_safe_shape(url):
        raise ValueError("URL must be public HTTPS without credentials")
    if client is None:
        await _validate_public_url(url)
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
        final_url, title, text = _extract_response(current, response)
        return _FetchedPage(
            final_url=final_url,
            title=title,
            text=text,
            acquisition=_acquisition(
                url,
                final_url,
                "http",
                response=response,
                body_bytes=len(response.content),
            ),
        )

    requested_url = url
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
        final_url, title, text = _extract_response(current, response)
        return _FetchedPage(
            final_url=final_url,
            title=title,
            text=text,
            acquisition=_acquisition(
                requested_url,
                final_url,
                "http",
                response=response,
                body_bytes=len(response.content),
            ),
        )
    raise ValueError("official source redirected too many times")


def _extract_response(url: str, response) -> tuple[str, str, str]:
    content_type = response.headers.get("content-type", "").lower()
    if "application/pdf" in content_type or response.content.startswith(b"%PDF"):
        reader = PdfReader(BytesIO(response.content))
        text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
        return url, "Official PDF", text
    title, text = _extract_html(response.text, url)
    if not text.strip() or _is_access_wall(title, text):
        raise _RenderedFetchNeeded
    return url, title, text


async def _fetch_page_with_browser(
    url: str,
    client,
    query: str,
    *,
    render: bool = False,
) -> _FetchedPage:
    if render:
        return await _fetch_rendered_page(url)
    try:
        fetched = await _fetch_page(url, client)
    except _RenderedFetchNeeded:
        return await _fetch_rendered_page(url)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code != 403:
            raise
        return await _fetch_rendered_page(url)
    if client is None and len(fetched.text.strip()) < _MIN_STATIC_TEXT_CHARS:
        return await _fetch_rendered_page(url)
    if "showing 0 events" in fetched.text.lower():
        return await _fetch_rendered_page(url)
    wanted = _terms(query)
    if wanted and not wanted & _terms(fetched.text):
        return await _fetch_rendered_page(url)
    return fetched


def web_fetch_tools() -> list[Tool]:
    async def _handler(args: dict, ctx: ToolContext) -> str:
        url = str(args["url"]).strip()
        try:
            safe_shape = _url_safe_shape(url)
            urlsplit(url).port
        except ValueError:
            safe_shape = False
        if not safe_shape:
            return "The page could not be fetched."
        query = str(args.get("query") or ctx.query).strip()
        evidence_scope = str(args.get("evidence_scope") or "").strip()
        render = bool(args.get("render", False))
        render_key = _approval_key(url)
        if render and render_key in ctx.rendered_fetch_urls:
            return "Rendered acquisition was already attempted for this URL in this turn."
        if render:
            ctx.rendered_fetch_urls.add(render_key)
        try:
            fetched = await _fetch_page_with_browser(
                url,
                ctx.http,
                query,
                render=render,
            )
        except _UnsafePublicUrl:
            return "The page could not be fetched."
        except Exception:
            cite = ctx.citations.register(
                url,
                snippet="No page content was retrieved.",
                title="Unavailable source",
                kind="WEB",
                provenance={"evidence_grade": "unavailable"},
            )
            failure = (
                f"The page could not be fetched: {url}\n"
                "No page content was retrieved. Any claim attributed only to this source is "
                f"unverified; preserve the URL so the resident can check it. {{cite:{cite}}}\n"
            )
            fallback_query = evidence_scope or query
            if not fallback_query:
                return failure
            host = urlsplit(url).hostname or ""
            try:
                search = web_search_tools(
                    ctx.registry.allowlist(),
                    ctx.registry.source_tiers(),
                    ctx.registry.news_tier(),
                )[0]
                fallback = await search.handler(
                    {
                        "query": fallback_query,
                        "prefer": [host],
                        "count": 5,
                    },
                    ctx,
                )
            except Exception:
                return failure
            if not fallback or fallback.startswith("No results from the live web"):
                return failure
            return f"{failure}\nFocused search fallback:\n{fallback}"
        final_url, title, text = fetched.final_url, fetched.title, fetched.text
        chunks = _evidence_chunks(text, query, force_focus=bool(evidence_scope))
        if not chunks:
            return "The page was fetched but did not contain text relevant to the requested query."
        content_scope = (
            "full extracted page" if chunks == [text] else "query-selected excerpts"
        )
        evidence = "\n\n".join(chunks)
        warning = archive_warning(final_url, title)
        from .web_search import _TIER_LABELS, _tier_of

        tier = _tier_of(
            final_url, ctx.registry.source_tiers(), ctx.registry.news_tier(),
        )
        authoritative = tier == "authoritative" and not warning
        if tier != "authoritative":
            label = (
                "unverified source, check before relying on it"
                if tier == "unverified"
                else _TIER_LABELS.get(tier, tier)
            )
            evidence = f"SOURCE TRUST: {label}\n\n{evidence}"
        if warning:
            evidence = f"{warning}\n\n{evidence}"
        citation_evidence = evidence
        provenance = {
            "evidence_grade": (
                "discovery" if warning else "authoritative" if authoritative else "fetched"
            ),
            "source_tier": tier,
            "acquisition": fetched.acquisition.model_dump(mode="json"),
        }
        cite = ctx.citations.register(
            final_url,
            snippet=citation_evidence,
            title=title or "Fetched page",
            kind="WEB",
            provenance=provenance,
        )
        scope_prefix = f"EVIDENCE SCOPE: {evidence_scope}\n" if evidence_scope else ""
        return (
            f"{scope_prefix}SOURCE {cite}: {title or 'Fetched page'} ({final_url})\n"
            f"CONTENT SCOPE: {content_scope}\n"
            f"{evidence} {{cite:{cite}}}"
        )

    return [
        Tool(
            name="web_fetch",
            description=(
                "Fetch and extract one known public web page. Use it after search when the page itself "
                "is needed as evidence. Source trust is graded separately; fetching a page does not "
                "make it authoritative. A full extracted page will not reveal different text when the "
                "same URL is fetched with another query. If static text omits JavaScript-loaded content, "
                "retry the same URL once with render=true. Otherwise search another page or state that "
                "the missing detail could not be verified."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Public HTTPS URL to fetch."},
                    "query": {
                        "type": "string",
                        "description": (
                            "Claim or detail to find. Omit only when the resident's current request "
                            "already states it clearly."
                        ),
                    },
                    "render": {
                        "type": "boolean",
                        "description": (
                            "Use true only after a static fetch omitted content likely loaded by "
                            "JavaScript. This uses the browser and is slower."
                        ),
                        "default": False,
                    },
                    "evidence_scope": {
                        "type": "string",
                        "description": (
                            "Optional short label for the claim this fetch is meant to support. "
                            "Copy the label unchanged when a coordinator supplies one."
                        ),
                    },
                },
                "required": ["url"],
            },
            handler=_handler,
            open_world=True,
        )
    ]
