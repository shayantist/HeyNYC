"""Secure local retrieval for one known public URL."""
from __future__ import annotations

import ipaddress
import json
from datetime import datetime
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
from typing import Literal
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx
from playwright.async_api import Error as PlaywrightError
from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field
from pydantic_ai._ssrf import safe_download, validate_and_resolve_url
from pypdf import PdfReader
from trafilatura import extract, html2txt

from .. import config
from ..citations import canonical_source_url
from ..index.corpus import clean_html
from .base import Tool, ToolContext, ToolFailure, ToolInput
from .web_search import archive_warning

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


class WebFetchInput(ToolInput):
    url: AnyHttpUrl = Field(description="Public HTTPS URL")
    find: str | None = Field(default=None, description="Text to locate")


class _FetchedPage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    final_url: str
    title: str
    text: str
    acquisition: WebFetchAcquisition
    child_links: list[dict[str, str]] = Field(default_factory=list)
    structured_data: list[dict] = Field(default_factory=list)


class _LinkParser(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.current: tuple[str, list[str]] | None = None
        self.links: list[dict[str, str]] = []
        self.content_links: list[dict[str, str]] = []
        self.blocked_depth = 0
        self.content_depth = 0
        self.current_is_content = False

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.casefold()
        if tag in {"header", "nav", "footer", "aside"}:
            self.blocked_depth += 1
        if tag in {"main", "article"} and self.blocked_depth == 0:
            self.content_depth += 1
        if tag != "a" or self.blocked_depth:
            return
        href = dict(attrs).get("href")
        self.current = (href, []) if href else None
        self.current_is_content = self.content_depth > 0

    def handle_data(self, data: str) -> None:
        if self.current is not None:
            self.current[1].append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.casefold()
        if tag != "a" or self.current is None:
            if tag in {"main", "article"} and self.blocked_depth == 0:
                self.content_depth = max(0, self.content_depth - 1)
            if tag in {"header", "nav", "footer", "aside"}:
                self.blocked_depth = max(0, self.blocked_depth - 1)
            return
        href, chunks = self.current
        url = urljoin(self.base_url, href)
        parts = urlsplit(url)
        title = " ".join("".join(chunks).split())
        if parts.scheme == "https" and parts.hostname and title:
            link = {
                "url": urlunsplit(parts._replace(fragment="")),
                "title": title,
            }
            self.links.append(link)
            if self.current_is_content:
                self.content_links.append(link)
        self.current = None
        self.current_is_content = False


def _page_links(html: str, base_url: str) -> list[dict[str, str]]:
    parser = _LinkParser(base_url)
    parser.feed(html)
    links = parser.content_links or parser.links
    return list({link["url"]: link for link in reversed(links)}.values())[::-1]


class _JsonLdParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.current: list[str] | None = None
        self.values: list[dict] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.casefold() == "script" and dict(attrs).get("type", "").casefold() == (
            "application/ld+json"
        ):
            self.current = []

    def handle_data(self, data: str) -> None:
        if self.current is not None:
            self.current.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() != "script" or self.current is None:
            return
        try:
            value = json.loads("".join(self.current))
        except (TypeError, ValueError):
            value = None
        if isinstance(value, dict):
            self.values.append(value)
        elif isinstance(value, list):
            self.values.extend(item for item in value if isinstance(item, dict))
        self.current = None


def _json_ld(html: str) -> list[dict]:
    parser = _JsonLdParser()
    parser.feed(html)
    return parser.values


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


def _text_tokens(text: str, model: str | None = None) -> int:
    import litellm

    return int(litellm.token_counter(model=model or config.HEYNYC_MODEL, text=text))


def _line_addressable(
    text: str,
    *,
    find: str | None = None,
) -> str | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    first = 1
    if find:
        needle = find.casefold()
        match = next(
            (number for number, line in enumerate(lines, 1) if needle in line.casefold()),
            None,
        )
        if match is None:
            return None
        first = match
    return "\n".join(
        f"L{number}: {line}"
        for number, line in enumerate(lines, 1)
        if number >= first
    )


def _fit_text(text: str, token_budget: int | None, model: str) -> str:
    if token_budget is None or _text_tokens(text, model) <= token_budget:
        return text
    selected: list[str] = []
    for line in text.splitlines():
        candidate = "\n".join([*selected, line])
        if _text_tokens(candidate, model) > token_budget:
            break
        selected.append(line)
    return "\n".join(selected)


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
        child_links=_page_links(html, final_url),
        structured_data=_json_ld(html),
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
            child_links=(
                _page_links(response.text, final_url)
                if "html" in response.headers.get("content-type", "").lower()
                else []
            ),
            structured_data=(
                _json_ld(response.text)
                if "html" in response.headers.get("content-type", "").lower()
                else []
            ),
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
            child_links=(
                _page_links(response.text, final_url)
                if "html" in response.headers.get("content-type", "").lower()
                else []
            ),
            structured_data=(
                _json_ld(response.text)
                if "html" in response.headers.get("content-type", "").lower()
                else []
            ),
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
    return fetched


def web_fetch_tools() -> list[Tool]:
    async def _handler(args: WebFetchInput, ctx: ToolContext) -> str | ToolFailure:
        url = str(args["url"]).strip()
        try:
            safe_shape = _url_safe_shape(url)
            urlsplit(url).port
        except ValueError:
            safe_shape = False
        if not safe_shape:
            return ToolFailure(status="rejected", reason="URL must be public HTTPS.", retryable=False)
        try:
            fetched = await _fetch_page_with_browser(url, ctx.http)
        except _UnsafePublicUrl:
            return ToolFailure(status="rejected", reason="URL must be public HTTPS.", retryable=False)
        except (httpx.HTTPError, PlaywrightError, ValueError):
            return ToolFailure(
                status="unavailable",
                reason="The page could not be retrieved by HTTP or browser.",
                retryable=False,
                source_url=url,
            )
        final_url, title = fetched.final_url, fetched.title
        text = _line_addressable(
            fetched.text,
            find=args.get("find"),
        )
        if text is None:
            return ToolFailure(
                status="partial",
                reason=f"The requested text was not found on the fetched page: {args['find']}",
                retryable=False,
                source_url=final_url,
            )
        model = ctx.evidence_model or config.HEYNYC_MODEL
        available = ctx.evidence_token_budget
        if available is None:
            from ..memory import context_capacity

            available = context_capacity(model, None, True) or None
        fitted_text = _fit_text(text, available, model)
        truncated = fitted_text != text
        text = fitted_text
        if not text:
            cite = ctx.citations.register(
                final_url,
                snippet="The page was fetched, but no page text fit in the model context.",
                title=title or "Fetched page",
                kind="WEB",
                provenance={
                    "evidence_grade": "unavailable",
                    "acquisition": fetched.acquisition.model_dump(mode="json"),
                },
            )
            return f"The page was fetched, but its text did not fit. {final_url} {{cite:{cite}}}"
        warning = archive_warning(final_url, title)
        from .web_search import _TIER_LABELS, _tier_of

        tier = _tier_of(
            final_url, ctx.registry.source_tiers(), ctx.registry.news_tier(),
        )
        authoritative = tier == "authoritative" and not warning
        relevant_links = fetched.child_links
        direct_links = "\n".join(
            f"Related URL only, not event evidence: {link['title']}: {link['url']}"
            for link in relevant_links
        )
        def rendered_evidence(*, include_links: bool = True) -> str:
            evidence = "\n\n".join(filter(None, (
                text,
                direct_links if include_links else "",
            )))
            if tier != "authoritative":
                label = (
                    "unverified source, check before relying on it"
                    if tier == "unverified"
                    else _TIER_LABELS.get(tier, tier)
                )
                evidence = f"SOURCE TRUST: {label}\n\n{evidence}"
            return f"{warning}\n\n{evidence}" if warning else evidence

        content_scope = "partial numbered page" if truncated else "numbered extracted page"
        evidence = rendered_evidence()
        citation_evidence = rendered_evidence(include_links=False)
        provenance = {
            "evidence_grade": (
                "discovery" if warning else "authoritative" if authoritative else "fetched"
            ),
            "source_tier": tier,
            "content_complete": not truncated,
            "acquisition": fetched.acquisition.model_dump(mode="json"),
            **({"links": relevant_links} if relevant_links else {}),
            **({"structured_data": fetched.structured_data} if fetched.structured_data else {}),
        }
        cite = ctx.citations.register(
            final_url,
            snippet=citation_evidence,
            title=title or "Fetched page",
            kind="WEB",
            provenance=provenance,
        )
        result = (
            f"SOURCE {cite}: {title or 'Fetched page'} ({final_url})\n"
            f"CONTENT SCOPE: {content_scope}\n"
            f"{evidence} {{cite:{cite}}}"
        )
        if available is not None:
            used = min(available, _text_tokens(result, model))
            ctx.evidence_tokens_used += used
            ctx.evidence_token_budget = max(0, available - used)
        return result

    return [
        Tool(
            name="web_fetch",
            description="Fetch one public page as numbered evidence; optionally find exact text",
            input_type=WebFetchInput,
            handler=_handler,
            open_world=True,
        )
    ]
