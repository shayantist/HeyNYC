from __future__ import annotations

from datetime import datetime
from io import BytesIO

import httpx
import pytest
from reportlab.pdfgen import canvas

import heynyc.core.tools.web_fetch as web_fetch_module
from heynyc.core import config
from heynyc.core.citations import CitationRegistry
from heynyc.core.manifest import ServiceModule
from heynyc.core.registry import Registry
from heynyc.core.tools.base import Tool, ToolContext
from heynyc.core.tools.web_fetch import (
    _evidence_chunks,
    _extract_response,
    _relevant_chunks,
    _route_public_request,
    web_fetch_tools,
)
from heynyc.core.tools.web_search import web_search_tools


class _Response:
    def __init__(
        self,
        text: str = "",
        *,
        content: bytes | None = None,
        content_type: str = "text/html",
        status_code: int = 200,
        location: str | None = None,
        headers: dict[str, str] | None = None,
    ):
        self.text = text
        self.content = content if content is not None else text.encode()
        self.headers = {"content-type": content_type, **(headers or {})}
        if location is not None:
            self.headers["location"] = location
        self.status_code = status_code

    def raise_for_status(self):
        return None


class _Client:
    def __init__(self, pages: dict[str, str | _Response]):
        self.pages = pages
        self.urls: list[str] = []
        self.requests: list[tuple[str, dict]] = []

    async def get(self, url, **kwargs):
        self.urls.append(url)
        self.requests.append((url, kwargs))
        page = self.pages[url]
        return page if isinstance(page, _Response) else _Response(page)


def _rendered_result(url: str, title: str, text: str):
    return web_fetch_module._FetchedPage(
        final_url=url,
        title=title,
        text=text,
        acquisition=web_fetch_module.WebFetchAcquisition(
            requested_url=url,
            final_url=url,
            citation_url=url,
            route="browser",
            fetched_at=datetime.now().astimezone(),
        ),
    )


def _assert_fetch_provenance(citation: dict, evidence_grade: str, source_tier: str) -> None:
    provenance = citation["provenance"]
    assert provenance["evidence_grade"] == evidence_grade
    assert provenance["source_tier"] == source_tier
    assert provenance["acquisition"]["final_url"] == citation["url"]


class _BrowserRoute:
    def __init__(self):
        self.action = ""

    async def abort(self):
        self.action = "abort"

    async def continue_(self):
        self.action = "continue"


class _BrowserRequest:
    def __init__(self, url: str):
        self.url = url


@pytest.fixture(autouse=True)
def _stub_public_url_validation(monkeypatch):
    async def validate(url: str) -> None:
        return None

    async def empty_search(_args, _ctx):
        return "No results from the live web search."

    monkeypatch.setattr(web_fetch_module, "_validate_public_url", validate)
    monkeypatch.setattr(
        web_fetch_module,
        "web_search_tools",
        lambda *_args, **_kwargs: [
            Tool(
                name="web_search",
                description="Search",
                parameters={"type": "object", "properties": {}},
                handler=empty_search,
            )
        ],
    )


def test_relevant_chunks_put_strongest_evidence_first():
    text = (
        "Kitchen navigation. "
        + ("Navigation services programs contact us. " * 60)
        + "Kitchen cooks are not eligible for a tip credit and must receive the full minimum wage."
    )

    chunks = _relevant_chunks(text, "kitchen cooks tip credit full minimum wage", limit=2)

    assert "full minimum wage" in chunks[0]


def test_focused_retrieval_stops_when_selected_chunks_cover_the_scope():
    text = (
        ("Tenant program history without contact details. " * 120)
        + "For other questions, call 311 and ask for the Tenant Helpline."
    )

    chunks = _relevant_chunks(text, "call 311 ask Tenant Helpline", limit=None)

    assert len(chunks) == 1
    assert "Tenant Helpline" in chunks[0]


def test_evidence_chunks_use_the_token_budget_instead_of_a_fixed_count(monkeypatch):
    text = "\n\n".join(
        f"Target section {index}. " + ("context " * 210)
        for index in range(6)
    )
    monkeypatch.setattr(web_fetch_module, "_text_tokens", lambda value, *_: len(value) // 4)
    budget = 2_000

    chunks = _evidence_chunks(text, "target section", token_budget=budget)

    assert len(chunks) > 2
    assert len("\n\n".join(chunks)) // 4 <= budget


def test_evidence_chunks_expand_with_available_context(monkeypatch):
    text = "\n\n".join(
        f"Target section {index}. " + ("context " * 210)
        for index in range(6)
    )
    monkeypatch.setattr(web_fetch_module, "_text_tokens", lambda value, *_: len(value) // 4)

    smaller = _evidence_chunks(text, "target section", token_budget=900)
    larger = _evidence_chunks(text, "target section", token_budget=2_000)

    assert len(larger) > len(smaller)


def test_evidence_chunks_emit_nothing_when_no_context_remains(monkeypatch):
    monkeypatch.setattr(web_fetch_module, "_text_tokens", lambda value, *_: len(value))

    assert _evidence_chunks("Target evidence.", "target", token_budget=0) == []


async def test_web_fetch_preserves_the_source_when_no_page_text_fits(
    monkeypatch,
) -> None:
    url = "https://www.nyc.gov/example"

    async def fetched(*_args, **_kwargs):
        return _rendered_result(
            url,
            "Official page",
            "The requested detail is present on this page.",
        )

    monkeypatch.setattr(web_fetch_module, "_text_tokens", lambda value, *_: len(value))
    monkeypatch.setattr(
        web_fetch_module,
        "_fetch_page_with_browser",
        fetched,
    )
    ctx = ToolContext(
        citations=CitationRegistry(),
        registry=Registry([ServiceModule(name="benefits", seeds=[url])]),
        evidence_token_budget=1,
        evidence_model="test/model",
    )

    out = await web_fetch_tools()[0].handler({"url": url}, ctx)

    assert url in out
    assert "did not fit" in out
    assert ctx.citations.mapping()["S1"]["provenance"]["evidence_grade"] == "unavailable"


async def test_browser_blocks_private_subresources(monkeypatch):
    async def validate(url: str) -> None:
        if url == "http://127.0.0.1/private":
            raise ValueError("private network")

    monkeypatch.setattr(web_fetch_module, "_validate_public_url", validate)
    route = _BrowserRoute()

    await _route_public_request(route, _BrowserRequest("http://127.0.0.1/private"))

    assert route.action == "abort"


def test_relevant_chunks_fail_closed_when_the_page_has_no_query_overlap():
    chunks = _relevant_chunks(
        "Official page with only unrelated navigation text. " * 80,
        "quantum banana eligibility",
        limit=2,
    )

    assert chunks == []


def test_relevant_chunks_support_non_latin_query_terms():
    chunks = _relevant_chunks(
        "সহায়তা সম্পর্কে অন্য তথ্য। স্ন্যাপ সুবিধা নবায়নের সরকারি নির্দেশনা এখানে আছে।",
        "স্ন্যাপ সুবিধা নবায়ন",
    )

    assert chunks == [
        "সহায়তা সম্পর্কে অন্য তথ্য। স্ন্যাপ সুবিধা নবায়নের সরকারি নির্দেশনা এখানে আছে।"
    ]


def test_relevant_chunks_keep_short_non_ascii_terms():
    chunks = _relevant_chunks(
        "官方福利更新说明。",
        "福利 更新",
    )

    assert chunks == ["官方福利更新说明。"]


def test_html_extraction_uses_one_canonical_result(monkeypatch):
    monkeypatch.setattr(
        web_fetch_module,
        "extract",
        lambda *_args, **_kwargs: "Canonical article text.",
        raising=False,
    )
    monkeypatch.setattr(
        web_fetch_module,
        "html2txt",
        lambda _html: pytest.fail("fallback extractor should not run"),
    )

    _url, title, text = _extract_response(
        "https://example.com/article",
        _Response("<title>Article</title><p>Canonical article text.</p>"),
    )

    assert title == "Article"
    assert text == "Canonical article text."


def test_html_extraction_favors_recall_for_hidden_page_sections(monkeypatch):
    options = {}

    def capture(_html, **kwargs):
        options.update(kwargs)
        return "Complete page text."

    monkeypatch.setattr(web_fetch_module, "extract", capture, raising=False)

    _extract_response(
        "https://example.com/guide",
        _Response("<title>Guide</title><p>Complete page text.</p>"),
    )

    assert options["favor_recall"] is True


def test_html_extraction_preserves_absolute_page_links():
    _url, _title, text = _extract_response(
        "https://example.com/guide",
        _Response(
            "<html><head><title>Guide</title></head><body><main>"
            + (
                "<p>Current public application guidance for residents. Use the "
                "<a href='/apply'>official application</a> now. The guide explains "
                "the available route and where to get help.</p>" * 8
            )
            + "</main></body></html>"
        ),
    )

    assert "[official application](https://example.com/apply)" in text


def test_html_extraction_uses_last_resort_fallback_when_main_text_is_empty(
    monkeypatch,
):
    monkeypatch.setattr(
        web_fetch_module,
        "extract",
        lambda *_args, **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        web_fetch_module,
        "html2txt",
        lambda _html: "Fallback page text.",
    )

    _url, _title, text = _extract_response(
        "https://example.com/article",
        _Response("<title>Article</title>"),
    )

    assert text == "Fallback page text."


def test_short_page_returns_full_extracted_text_without_keyword_filtering(
    monkeypatch,
):
    text = "SNAP applications and case management are available through ACCESS HRA."
    monkeypatch.setattr(web_fetch_module, "_text_tokens", lambda _text, *_: 400)

    assert _evidence_chunks(text, "como solicito ayuda para comprar comida") == [text]


def test_medium_page_keeps_its_lead_instead_of_dropping_context(monkeypatch):
    text = "Service is available at every location.\n" + ("Instructions and details. " * 250)
    monkeypatch.setattr(web_fetch_module, "_text_tokens", lambda _text, *_: 1_577)

    assert _evidence_chunks(text, "instructions details") == [text]


def test_long_page_keeps_query_focused_selection(monkeypatch):
    text = (
        "Navigation and unrelated material. " * 120
        + "Food service workers must receive the full minimum wage."
    )
    monkeypatch.setattr(web_fetch_module, "_text_tokens", lambda _text, *_: 3_001)

    chunks = _evidence_chunks(text, "food service full minimum wage")

    assert len(chunks) <= 2
    assert "full minimum wage" in "\n".join(chunks)


async def test_web_fetch_uses_the_resident_turn_when_query_is_omitted(monkeypatch):
    url = "https://www.nyc.gov/site/hra/help/snap-application-faq.page"
    text = (
        "Navigation and unrelated program information. " * 120
        + "At a SNAP walk-in center, staff can help complete an online or paper application."
    )
    monkeypatch.setattr(web_fetch_module, "_text_tokens", lambda _text, *_: 3_001)
    ctx = ToolContext(
        citations=CitationRegistry(),
        registry=Registry([ServiceModule(name="benefits", seeds=[url])]),
        http=_Client({url: text}),
        query="Where can I apply for SNAP in person?",
    )

    out = await web_fetch_tools()[0].handler({"url": url}, ctx)

    assert "help complete an online or paper application" in out


async def test_web_fetch_registers_the_same_full_page_evidence_shown_to_the_model(
    monkeypatch,
):
    url = "https://www.nyc.gov/example"
    text = (
        "Official program overview. " * 180
        + "Tenants have the right to organize and may call the Tenant Helpline at "
        "(800) 342-3334."
    )
    monkeypatch.setattr(web_fetch_module, "_text_tokens", lambda _text, *_: 2_000)
    ctx = ToolContext(
        citations=CitationRegistry(),
        registry=Registry([
            ServiceModule(
                name="housing",
                source_tiers={"authoritative": ["nyc.gov"]},
            )
        ]),
        http=_Client({url: text}),
    )

    out = await web_fetch_tools()[0].handler(
        {"url": url, "query": "tenant right organize Tenant Helpline"}, ctx,
    )

    assert "Official program overview" in out
    snippet = ctx.citations.mapping()["S1"]["snippet"]
    assert "right to organize" in snippet
    assert "(800) 342-3334" in snippet
    assert f"{snippet} {{cite:S1}}" in out


def test_web_fetch_schema_accepts_one_public_url():
    tool = web_fetch_tools()[0]

    assert tool.name == "web_fetch"
    assert set(tool.parameters["properties"]) == {"url", "query", "render", "evidence_scope"}
    assert tool.parameters["required"] == ["url"]


def test_rendered_fetch_uses_the_browser_native_user_agent():
    assert "user_agent" not in web_fetch_module._browser_context_options()


async def test_web_fetch_accepts_an_unlisted_public_url_as_unverified():
    url = "https://example.com/current-event"
    client = _Client({url: "Current event details and schedule."})
    ctx = ToolContext(
        citations=CitationRegistry(), registry=Registry([]), http=client,
    )

    out = await web_fetch_tools()[0].handler(
        {"url": url, "query": "current event details schedule"}, ctx,
    )

    assert client.urls == [url]
    assert "Current event details" in out
    assert "unverified source, check before relying on it" in out
    _assert_fetch_provenance(ctx.citations.mapping()["S1"], "fetched", "unverified")


async def test_web_fetch_labels_the_source_before_its_evidence():
    url = "https://example.com/current-guidance"
    ctx = ToolContext(
        citations=CitationRegistry(),
        registry=Registry([]),
        http=_Client({url: "Payment agreements may prevent service shutoff."}),
    )

    out = await web_fetch_tools()[0].handler(
        {"url": url, "query": "payment agreement shutoff"}, ctx,
    )

    assert out.startswith(f"SOURCE S1: Fetched page ({url})\n")
    assert out.index("SOURCE S1:") < out.index("Payment agreements")
    assert out.endswith("{cite:S1}")


async def test_web_fetch_preserves_caller_evidence_scope():
    url = "https://example.com/utility-rights"
    ctx = ToolContext(
        citations=CitationRegistry(),
        registry=Registry([]),
        http=_Client({url: "A payment agreement may prevent service shutoff."}),
    )

    out = await web_fetch_tools()[0].handler(
        {
            "url": url,
            "query": "payment agreement shutoff",
            "evidence_scope": "SHUTOFF PROTECTIONS",
        },
        ctx,
    )

    assert out.startswith("EVIDENCE SCOPE: SHUTOFF PROTECTIONS\n")
    assert "SOURCE S1:" in out


async def test_web_fetch_evidence_scope_limits_full_page_output(monkeypatch):
    url = "https://example.com/tenant-help"
    text = (
        "Unrelated program history and partner directory. " * 80
        + "For other questions, call 311 and ask for the Tenant Helpline."
    )
    monkeypatch.setattr(web_fetch_module, "_text_tokens", lambda _text, *_: 2_000)
    ctx = ToolContext(
        citations=CitationRegistry(),
        registry=Registry([]),
        http=_Client({url: text}),
    )

    out = await web_fetch_tools()[0].handler(
        {
            "url": url,
            "query": "call 311 ask Tenant Helpline",
            "evidence_scope": "Tenant Helpline route",
        },
        ctx,
    )

    assert "Tenant Helpline" in out
    assert out.count("Unrelated program history") < 80
    assert "CONTENT SCOPE: query-selected excerpts" in out


def test_relevant_web_excerpts_do_not_start_inside_a_word():
    words = [f"distinctword{index}" for index in range(500)]
    text = " ".join(words)

    chunks = web_fetch_module._relevant_chunks(text, "distinctword300")

    assert all(chunk.split()[0] in words for chunk in chunks)


def test_relevant_web_excerpts_start_at_a_complete_sentence():
    text = (
        "Background " + ("context " * 230) + ". "
        "For organizing questions, call 311 and ask for the Tenant Helpline."
    )

    chunks = web_fetch_module._relevant_chunks(text, "call 311 Tenant Helpline")

    assert chunks[0].startswith("For organizing questions")


async def test_web_fetch_rejects_userinfo_before_network():
    client = _Client({})
    ctx = ToolContext(
        citations=CitationRegistry(), registry=Registry([]), http=client,
    )

    out = await web_fetch_tools()[0].handler(
        {"url": "https://user:password@example.com/private"}, ctx,
    )

    assert "could not be fetched" in out.lower()
    assert client.urls == []


async def test_web_fetch_rejects_plain_http_before_network():
    url = "http://www.nyc.gov/current-guidance"
    client = _Client({url: "Current official guidance."})
    registry = Registry([
        ServiceModule(
            name="guidance",
            source_tiers={"authoritative": ["nyc.gov"]},
        )
    ])
    ctx = ToolContext(
        citations=CitationRegistry(), registry=registry, http=client,
    )

    out = await web_fetch_tools()[0].handler(
        {"url": url, "query": "current official guidance"}, ctx,
    )

    assert "could not be fetched" in out.lower()
    assert client.urls == []
    assert ctx.citations.mapping() == {}


async def test_web_fetch_validates_dns_before_using_an_injected_client(monkeypatch):
    url = "https://rebinding.example/private"
    client = _Client({url: "Private service content."})
    checked: list[str] = []

    async def reject_private_resolution(candidate: str, **kwargs) -> None:
        checked.append(candidate)
        raise web_fetch_module._UnsafePublicUrl("URL resolved to a private address")

    monkeypatch.setattr(
        web_fetch_module,
        "_validate_public_url",
        reject_private_resolution,
    )
    ctx = ToolContext(
        citations=CitationRegistry(), registry=Registry([]), http=client,
    )

    out = await web_fetch_tools()[0].handler(
        {"url": url, "query": "private service"}, ctx,
    )

    assert checked == [url]
    assert client.urls == []
    assert "could not be fetched" in out.lower()


async def test_web_fetch_uses_pydantic_ssrf_download_without_an_injected_client(
    monkeypatch,
):
    url = "https://example.com/current-guidance"
    calls = []

    async def fake_safe_download(requested_url, **kwargs):
        calls.append((requested_url, kwargs))
        return _Response("Current public guidance for residents. " * 8)

    monkeypatch.setattr(
        web_fetch_module, "safe_download", fake_safe_download, raising=False,
    )
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]))

    out = await web_fetch_tools()[0].handler(
        {"url": url, "query": "current public guidance residents"}, ctx,
    )

    assert calls == [(url, {
        "allow_local": False,
        "timeout": 20,
        "headers": {
            "User-Agent": "Mozilla/5.0 (compatible; HeyNYC/0.1; +https://reach4help.org)",
        },
    })]
    assert "Current public guidance" in out


async def test_production_fetch_attributes_a_safe_redirect_to_its_final_url(monkeypatch):
    requested = "https://example.com/old"
    final = "https://example.com/current"
    response = _Response("Current public guidance for residents. " * 8)
    response.url = httpx.URL(final)

    async def fake_safe_download(_requested_url, **_kwargs):
        return response

    monkeypatch.setattr(web_fetch_module, "safe_download", fake_safe_download)
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]))

    out = await web_fetch_tools()[0].handler(
        {"url": requested, "query": "current public guidance residents"}, ctx,
    )

    assert final in out
    assert ctx.citations.mapping()["S1"]["url"] == final


async def test_web_fetch_preserves_http_acquisition_metadata():
    url = "https://example.com/current"
    response = _Response(
        "Current public guidance for residents. " * 8,
        headers={
            "etag": 'W/"version-7"',
            "last-modified": "Fri, 14 Aug 2026 16:00:00 GMT",
            "cache-control": "max-age=300",
            "date": "Sat, 15 Aug 2026 10:00:00 GMT",
        },
    )
    ctx = ToolContext(
        citations=CitationRegistry(),
        registry=Registry([]),
        http=_Client({url: response}),
    )

    await web_fetch_tools()[0].handler(
        {"url": url, "query": "current public guidance residents"}, ctx,
    )

    acquisition = ctx.citations.mapping()["S1"]["provenance"]["acquisition"]
    assert acquisition == {
        "requested_url": url,
        "final_url": url,
        "citation_url": url,
        "route": "http",
        "fetched_at": acquisition["fetched_at"],
        "status_code": 200,
        "content_type": "text/html",
        "body_bytes": len(response.content),
        "etag": 'W/"version-7"',
        "last_modified": "Fri, 14 Aug 2026 16:00:00 GMT",
        "cache_control": "max-age=300",
        "response_date": "Sat, 15 Aug 2026 10:00:00 GMT",
    }
    assert datetime.fromisoformat(acquisition["fetched_at"]).tzinfo is not None


async def test_production_fetch_restores_the_logical_host_after_ssrf_resolution(
    monkeypatch,
):
    requested = "https://www.bklynlibrary.org/locations/sunset-park"
    response = httpx.Response(
        200,
        text="Sunset Park Library is open Wednesday from 10 am to 6 pm. " * 8,
        request=httpx.Request(
            "GET",
            "https://104.18.21.164/locations/sunset-park",
            headers={"Host": "www.bklynlibrary.org"},
        ),
    )

    async def fake_safe_download(_requested_url, **_kwargs):
        return response

    monkeypatch.setattr(web_fetch_module, "safe_download", fake_safe_download)
    registry = Registry([
        ServiceModule(
            name="libraries",
            source_tiers={"authoritative": ["bklynlibrary.org"]},
        )
    ])
    ctx = ToolContext(citations=CitationRegistry(), registry=registry)

    out = await web_fetch_tools()[0].handler(
        {"url": requested, "query": "Sunset Park Library Wednesday hours"},
        ctx,
    )

    assert requested in out
    assert ctx.citations.mapping()["S1"]["url"] == requested
    _assert_fetch_provenance(
        ctx.citations.mapping()["S1"], "authoritative", "authoritative",
    )


async def test_web_fetch_uses_browser_after_a_production_403(monkeypatch):
    url = "https://www.nba.com/knicks/schedule"
    request = httpx.Request("GET", url)
    response = httpx.Response(403, request=request)
    browser_calls: list[str] = []

    async def blocked_download(_requested_url, **_kwargs):
        raise httpx.HTTPStatusError(
            "blocked",
            request=request,
            response=response,
        )

    async def rendered(requested_url):
        browser_calls.append(requested_url)
        return _rendered_result(
            requested_url,
            "Knicks schedule",
            "The Knicks next game date and time is Tuesday October 20 at 7:00 PM EDT. "
            "The opponent is Philadelphia.",
        )

    monkeypatch.setattr(web_fetch_module, "safe_download", blocked_download)
    monkeypatch.setattr(web_fetch_module, "_fetch_rendered_page", rendered)
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]))

    out = await web_fetch_tools()[0].handler(
        {"url": url, "query": "next Knicks game date time opponent"}, ctx,
    )

    assert browser_calls == [url]
    assert "October 20" in out


async def test_web_fetch_uses_browser_after_a_production_javascript_shell(
    monkeypatch,
):
    url = "https://www.nba.com/knicks/schedule"
    browser_calls: list[str] = []

    async def shell_download(_requested_url, **_kwargs):
        return _Response(
            "<title>Schedule | New York Knicks</title>"
            "<div id='__next'>" + ("Teams Tickets Schedule Shop " * 30) + "</div>"
            "<script>window.pageData={date:'October 20',time:'7 PM',"
            "opponent:'Philadelphia'}</script>"
        )

    async def rendered(requested_url):
        browser_calls.append(requested_url)
        return _rendered_result(
            requested_url,
            "Knicks schedule",
            "The Knicks next game date and time is Tuesday October 20 at 7:00 PM EDT. "
            "The opponent is Philadelphia.",
        )

    monkeypatch.setattr(web_fetch_module, "safe_download", shell_download)
    monkeypatch.setattr(web_fetch_module, "_fetch_rendered_page", rendered)
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]))

    out = await web_fetch_tools()[0].handler(
        {
            "url": url,
            "query": "next Knicks game date time opponent",
            "render": True,
        },
        ctx,
    )

    assert browser_calls == [url]
    assert "October 20" in out


async def test_web_fetch_keeps_usable_static_text_without_inferring_a_visual_gap(
    monkeypatch,
):
    url = "https://www.bklynlibrary.org/locations/sunset-park"
    browser_calls: list[str] = []

    async def static_download(_requested_url, **_kwargs):
        return _Response(
            "<title>Sunset Park Library</title>"
            "<main><h1>Hours</h1><p>Wednesday 10 am - 6 pm</p>"
            "<p>5108 Fourth Avenue</p><p>718.230.2255</p>"
            "<p>Current branch information for Sunset Park residents. "
            "The branch offers books, programs, community space, and staff help. "
            "Contact the branch for service availability.</p></main>"
            "<script>window.services=['printing']</script>"
        )

    async def rendered(requested_url):
        browser_calls.append(requested_url)
        return _rendered_result(requested_url, "Rendered", "Rendered content")

    monkeypatch.setattr(web_fetch_module, "safe_download", static_download)
    monkeypatch.setattr(web_fetch_module, "_fetch_rendered_page", rendered)
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]))

    out = await web_fetch_tools()[0].handler(
        {"url": url, "query": "hours address phone printing"},
        ctx,
    )

    assert browser_calls == []
    assert "Wednesday 10 am - 6 pm" in out


async def test_web_fetch_renders_when_static_text_misses_the_query_target(monkeypatch):
    url = "https://www.queenslibrary.org/about-us/locations/corona"
    browser_calls: list[str] = []

    async def static_download(_requested_url, **_kwargs):
        return _Response(
            "<title>Corona Library</title><main><p>Site navigation.</p>"
            + ("<p>Branch hours and general library services.</p>" * 12)
            + "</main>"
        )

    async def rendered(requested_url):
        browser_calls.append(requested_url)
        return _rendered_result(requested_url, "Corona Library", "Wheelchair accessible")

    monkeypatch.setattr(web_fetch_module, "safe_download", static_download)
    monkeypatch.setattr(web_fetch_module, "_fetch_rendered_page", rendered)
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]))

    out = await web_fetch_tools()[0].handler(
        {"url": url, "query": "site wheelchair accessibility"}, ctx,
    )

    assert browser_calls == [url]
    assert "Wheelchair accessible" in out


async def test_web_fetch_renders_a_static_calendar_that_reports_zero_events(monkeypatch):
    url = "https://www.example.com/calendar"
    browser_calls: list[str] = []

    async def static_download(_requested_url, **_kwargs):
        return _Response(
            "<title>Event Calendar</title><main><h1>Upcoming Events</h1>"
            "<p>Showing 0 Events</p>"
            + ("<p>Music sports family tickets venue calendar.</p>" * 12)
            + "</main>"
        )

    async def rendered(requested_url):
        browser_calls.append(requested_url)
        return _rendered_result(
            requested_url,
            "Event Calendar",
            "Joe Hisaishi at Radio City Music Hall on August 17 at 8 PM.",
        )

    monkeypatch.setattr(web_fetch_module, "safe_download", static_download)
    monkeypatch.setattr(web_fetch_module, "_fetch_rendered_page", rendered)
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]))

    out = await web_fetch_tools()[0].handler(
        {"url": url, "query": "music events NYC Monday August 17 2026"},
        ctx,
    )

    assert browser_calls == [url]
    assert "Joe Hisaishi" in out


async def test_web_fetch_can_explicitly_render_a_page_with_usable_static_text(
    monkeypatch,
):
    url = "https://example.com/schedule"
    browser_calls: list[str] = []

    async def static_download(_requested_url, **_kwargs):
        return _Response("<title>Schedule</title><p>Schedule navigation</p>")

    async def rendered(requested_url):
        browser_calls.append(requested_url)
        return _rendered_result(
            requested_url,
            "Schedule",
            "Tuesday at 7 PM against Philadelphia",
        )

    monkeypatch.setattr(web_fetch_module, "safe_download", static_download)
    monkeypatch.setattr(web_fetch_module, "_fetch_rendered_page", rendered)
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]))

    out = await web_fetch_tools()[0].handler(
        {"url": url, "query": "schedule", "render": True},
        ctx,
    )

    assert browser_calls == [url]
    assert "Philadelphia" in out


async def test_web_fetch_renders_the_same_url_at_most_once_per_turn(monkeypatch):
    url = "https://example.com/schedule"
    browser_calls: list[str] = []

    async def rendered(requested_url):
        browser_calls.append(requested_url)
        raise RuntimeError("browser failed")

    monkeypatch.setattr(web_fetch_module, "_fetch_rendered_page", rendered)
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]))
    tool = web_fetch_tools()[0]

    first = await tool.handler({"url": url, "render": True}, ctx)
    second = await tool.handler({"url": url, "render": True}, ctx)

    assert "could not be fetched" in first
    assert "already attempted" in second
    assert browser_calls == [url]


async def test_web_fetch_fetches_only_seeded_pages_and_returns_relevant_citations():
    cash = "https://www.nyc.gov/cash"
    schedule = "https://www.nyc.gov/schedule"
    registry = Registry([ServiceModule(name="law", seeds=[cash, schedule])])
    client = _Client({
        cash: "<title>Cash</title><p>Navigation filler.</p><p>Food stores must accept cash. "
              "The cashless rule has limited exceptions.</p>",
        schedule: "<title>Schedules</title><p>Retail employers must post schedules 72 hours "
                  "before a shift under Fair Workweek.</p>",
    })
    ctx = ToolContext(citations=CitationRegistry(), registry=registry, http=client)
    tool = web_fetch_tools()[0]

    await tool.handler({"url": cash, "query": "cashless rule"}, ctx)
    out = await tool.handler(
        {"url": schedule, "query": "retail schedule Fair Workweek notice"}, ctx,
    )

    assert set(client.urls) == {cash, schedule}
    assert "72 hours" in out
    assert "{cite:S" in out
    assert all(cite["kind"] == "WEB" for cite in ctx.citations.mapping().values())
    assert all(
        cite["provenance"]["evidence_grade"] == "authoritative"
        for cite in ctx.citations.mapping().values()
    )
    assert all(
        request["headers"]["User-Agent"].startswith("Mozilla/5.0 (compatible; HeyNYC/")
        for _url, request in client.requests
    )


async def test_web_fetch_rejects_access_wall_content():
    url = "https://www.uscis.gov/current-guidance"
    registry = Registry([ServiceModule(name="immigration", seeds=[url])])
    client = _Client({
        url: "<title>Access Denied</title><p>Access denied. Complete the security challenge.</p>",
    })
    ctx = ToolContext(citations=CitationRegistry(), registry=registry, http=client)

    out = await web_fetch_tools()[0].handler(
        {"url": url, "query": "current guidance access"},
        ctx,
    )

    assert "could not be fetched" in out
    assert ctx.citations.mapping()["S1"]["provenance"] == {
        "evidence_grade": "unavailable"
    }


async def test_web_fetch_uses_browser_after_an_access_wall(monkeypatch):
    url = "https://www.nba.com/knicks/news/team-captain"
    registry = Registry([
        ServiceModule(
            name="events",
            allowlist=["nba.com"],
            source_tiers={"authoritative": ["nba.com"]},
        )
    ])
    client = _Client({
        url: "<title>Access Denied</title><p>Enable JavaScript and cookies to continue.</p>",
    })
    browser_calls: list[str] = []

    async def fake_browser_fetch(requested_url):
        browser_calls.append(requested_url)
        assert requested_url == url
        return _rendered_result(
            url,
            "Knicks Name Jalen Brunson Team Captain",
            "The New York Knicks named Jalen Brunson the 36th captain in franchise history.",
        )

    monkeypatch.setattr(
        web_fetch_module,
        "_fetch_rendered_page",
        fake_browser_fetch,
        raising=False,
    )
    ctx = ToolContext(citations=CitationRegistry(), registry=registry, http=client)

    out = await web_fetch_tools()[0].handler(
        {"url": url, "query": "current Knicks captain Jalen Brunson"},
        ctx,
    )

    assert browser_calls == [url]
    assert "Jalen Brunson" in out
    _assert_fetch_provenance(
        ctx.citations.mapping()["S1"], "authoritative", "authoritative",
    )


def test_browser_fallback_uses_an_installed_brave_executable(monkeypatch, tmp_path):
    brave = tmp_path / "Brave Browser"
    brave.touch()
    monkeypatch.setattr(
        web_fetch_module,
        "_BROWSER_EXECUTABLE_CANDIDATES",
        (brave,),
        raising=False,
    )

    assert web_fetch_module._browser_launch_options() == {
        "headless": True,
        "executable_path": str(brave),
    }


def test_browser_fallback_uses_new_york_time():
    assert web_fetch_module._browser_context_options()["timezone_id"] == (
        "America/New_York"
    )


async def test_browser_waits_for_route_cleanup_before_close(monkeypatch):
    class FakeResponse:
        status = 200

        async def all_headers(self):
            return {
                "content-type": "text/html; charset=utf-8",
                "etag": '"browser-v1"',
            }

        async def body(self):
            return b"rendered response body"

    class FakePage:
        url = "https://example.com/current"
        unrouted = False
        goto_kwargs = None

        async def route(self, _pattern, _handler):
            return None

        async def goto(self, _url, **_kwargs):
            self.goto_kwargs = _kwargs
            return FakeResponse()

        async def wait_for_function(self, _expression, **_kwargs):
            return None

        def locator(self, _selector):
            return self

        async def inner_text(self):
            return "Current public guidance for New Yorkers."

        async def content(self):
            return "<title>Current guidance</title>"

        async def unroute_all(self, **kwargs):
            assert kwargs == {"behavior": "ignoreErrors"}
            self.unrouted = True

    page = FakePage()

    class FakeContext:
        async def new_page(self):
            return page

    class FakeBrowser:
        async def new_context(self, **_kwargs):
            return FakeContext()

        async def close(self):
            assert page.unrouted

    class FakeChromium:
        async def launch(self, **_kwargs):
            return FakeBrowser()

    class FakePlaywright:
        chromium = FakeChromium()

    class FakeManager:
        async def __aenter__(self):
            return FakePlaywright()

        async def __aexit__(self, *_args):
            return None

    monkeypatch.setattr(
        "playwright.async_api.async_playwright",
        lambda: FakeManager(),
    )

    result = await web_fetch_module._fetch_rendered_page("https://example.com/current")

    assert page.goto_kwargs["wait_until"] == "load"
    assert result.acquisition.status_code == 200
    assert result.acquisition.content_type == "text/html; charset=utf-8"
    assert result.acquisition.etag == '"browser-v1"'
    assert result.acquisition.body_bytes == len(b"rendered response body")


def test_rendered_page_prefers_visible_text_over_hidden_responsive_markup():
    _title, text = web_fetch_module._rendered_page_text(
        "<title>Schedule</title><p>Time (PDT): 7:00 PM vs. 76ers</p>",
        "Tuesday Oct 20 7:00 PM EDT Philadelphia 76ers Madison Square Garden",
    )

    chunks = _relevant_chunks(text, "Knicks 76ers game date time", limit=1)

    assert "7:00 PM EDT" in chunks[0]
    assert "PDT" not in text


async def test_web_fetch_keeps_article_body_when_navigation_dominates_extraction(
    monkeypatch,
):
    url = "https://portal.311.nyc.gov/article/?kanumber=KA-02518"
    navigation = " ".join(
        f"<a href='/help/{index}'>Browser settings and navigation help {index}</a>"
        for index in range(120)
    )
    client = _Client({
        url: (
            "<html><head><title>Illegal Eviction or Lockout</title></head><body>"
            f"<nav>{navigation}</nav>"
            "<div id='offlineNotificationBar'>You're offline. This is a read only version.</div>"
            "<div class='panel-expand'><div class='divTableCell1'>Call 911</div>"
            "<div class='divTableCell2'>Call 911 to report landlords who lock out tenants. "
            "Only a City Marshal or Sheriff may carry out a Warrant of Eviction.</div></div>"
            "</body></html>"
        ),
    })
    monkeypatch.setattr(
        web_fetch_module,
        "clean_html",
        lambda _html: (
            "Illegal Eviction or Lockout",
            "You're offline. Browser settings and navigation help.",
        ),
    )
    ctx = ToolContext(
        citations=CitationRegistry(),
        registry=Registry([ServiceModule(name="housing", seeds=[url])]),
        http=client,
    )

    out = await web_fetch_tools()[0].handler(
        {"url": url, "query": "landlord lock out tenant call 911 warrant eviction"},
        ctx,
    )

    assert "Call 911 to report landlords who lock out tenants" in out
    assert "City Marshal or Sheriff" in ctx.citations.mapping()["S1"]["snippet"]


async def test_web_fetch_marks_archived_content_in_the_evidence():
    url = "https://www.uscis.gov/archive/current-guidance"
    registry = Registry([ServiceModule(name="immigration", seeds=[url])])
    client = _Client({
        url: (
            "<title>Archived guidance</title>"
            "<p>Archived Content. The information on this page is out of date.</p>"
            "<p>Temporary Protected Status guidance for Haiti.</p>"
        ),
    })
    ctx = ToolContext(citations=CitationRegistry(), registry=registry, http=client)

    out = await web_fetch_tools()[0].handler(
        {"url": url, "query": "Temporary Protected Status Haiti"}, ctx,
    )

    assert "SOURCE STATUS: ARCHIVED" in out
    assert "SOURCE STATUS: ARCHIVED" in ctx.citations.mapping()["S1"]["snippet"]
    _assert_fetch_provenance(
        ctx.citations.mapping()["S1"], "discovery", "authoritative",
    )


async def test_web_fetch_marks_archive_urls_without_banner_wording():
    url = "https://www.uscis.gov/archive/current-guidance"
    registry = Registry([ServiceModule(name="immigration", seeds=[url])])
    client = _Client({
        url: (
            "<title>Historical guidance</title>"
            "<p>Temporary Protected Status guidance for Haiti.</p>"
        ),
    })
    ctx = ToolContext(citations=CitationRegistry(), registry=registry, http=client)

    out = await web_fetch_tools()[0].handler(
        {"url": url, "query": "Temporary Protected Status Haiti"}, ctx,
    )

    assert "SOURCE STATUS: ARCHIVED" in out


async def test_web_fetch_does_not_mark_current_page_from_unrelated_archive_footer():
    url = "https://www.uscis.gov/current-guidance"
    registry = Registry([ServiceModule(name="immigration", seeds=[url])])
    client = _Client({
        url: (
            "<title>Current guidance</title>"
            "<p>Temporary Protected Status guidance for Haiti is current.</p>"
            "<p>Historical content is available elsewhere in our archive.</p>"
        ),
    })
    ctx = ToolContext(citations=CitationRegistry(), registry=registry, http=client)

    out = await web_fetch_tools()[0].handler(
        {"url": url, "query": "Temporary Protected Status Haiti current"},
        ctx,
    )

    assert "SOURCE STATUS: ARCHIVED" not in out


async def test_web_fetch_does_not_require_a_url_declared_by_a_module():
    registry = Registry([ServiceModule(name="law", seeds=["https://www.nyc.gov/allowed"])])
    url = "https://evil.example/page"
    client = _Client({url: "Public page with cash guidance."})
    ctx = ToolContext(citations=CitationRegistry(), registry=registry, http=client)
    tool = web_fetch_tools()[0]

    out = await tool.handler(
        {"url": url, "query": "cash"}, ctx,
    )

    assert "Public page with cash guidance" in out
    assert client.urls == [url]
    _assert_fetch_provenance(ctx.citations.mapping()["S1"], "fetched", "unverified")


async def test_web_fetch_calls_are_atomic_when_one_url_fails():
    approved = "https://www.nyc.gov/allowed"
    rejected = "https://evil.example/page"
    registry = Registry([ServiceModule(name="law", seeds=[approved])])
    client = _Client({approved: "Official cash assistance guidance."})
    ctx = ToolContext(citations=CitationRegistry(), registry=registry, http=client)

    failed = await web_fetch_tools()[0].handler(
        {"url": rejected, "query": "cash assistance guidance"}, ctx,
    )
    out = await web_fetch_tools()[0].handler(
        {"url": approved, "query": "cash assistance guidance"}, ctx,
    )

    assert client.urls == [rejected, approved]
    assert "Official cash assistance guidance" in out
    assert "could not be fetched" in failed


async def test_web_fetch_keeps_editorial_source_tier_on_fetched_evidence():
    url = "https://timeout.com/newyork/news/example"
    registry = Registry([
        ServiceModule(
            name="events",
            official_only=False,
            allowlist=["timeout.com"],
            source_tiers={"editorial": ["timeout.com"]},
        )
    ])
    client = _Client({url: "Editorial event details."})
    ctx = ToolContext(citations=CitationRegistry(), registry=registry, http=client)

    out = await web_fetch_tools()[0].handler(
        {"url": url, "query": "event details"},
        ctx,
    )

    assert "Editorial event details" in out
    _assert_fetch_provenance(ctx.citations.mapping()["S1"], "fetched", "editorial")


async def test_web_fetch_trust_metadata_overrides_a_gov_suffix():
    url = "https://sub.events.example.gov/weekend"
    registry = Registry([
        ServiceModule(
            name="events",
            official_only=False,
            allowlist=["sub.events.example.gov"],
            source_tiers={"editorial": ["events.example.gov"]},
        )
    ])
    client = _Client({url: "Editorial government event details."})
    ctx = ToolContext(citations=CitationRegistry(), registry=registry, http=client)

    out = await web_fetch_tools()[0].handler(
        {"url": url, "query": "event details"},
        ctx,
    )

    assert "Editorial government event details" in out
    assert ctx.citations.mapping()["S1"]["provenance"]["source_tier"] == "editorial"


async def test_web_fetch_marks_an_editorial_seed_as_fetched():
    url = "https://worldcup.example/events"
    registry = Registry([
        ServiceModule(
            name="events",
            official_only=False,
            seeds=[url],
            allowlist=["worldcup.example"],
            source_tiers={"editorial": ["worldcup.example"]},
        )
    ])
    client = _Client({url: "Editorial tournament event details."})
    ctx = ToolContext(citations=CitationRegistry(), registry=registry, http=client)

    out = await web_fetch_tools()[0].handler(
        {"url": url, "query": "event details"},
        ctx,
    )

    assert "Editorial tournament event details" in out
    assert ctx.citations.mapping()["S1"]["provenance"]["evidence_grade"] == "fetched"
    assert ctx.citations.mapping()["S1"]["provenance"]["source_tier"] == "editorial"


async def test_web_fetch_accepts_explicit_authoritative_domain():
    url = "https://nycgovparks.org/events/example"
    registry = Registry([
        ServiceModule(
            name="events",
            official_only=False,
            allowlist=["nycgovparks.org"],
            source_tiers={"authoritative": ["nycgovparks.org"]},
        )
    ])
    client = _Client({url: "Official event details and schedule."})
    ctx = ToolContext(citations=CitationRegistry(), registry=registry, http=client)

    out = await web_fetch_tools()[0].handler(
        {"url": url, "query": "event details schedule"},
        ctx,
    )

    assert "{cite:S1}" in out
    assert client.urls == [url]


async def test_web_fetch_fetches_an_unseeded_page_on_a_curated_domain():
    url = "https://otda.ny.gov/programs/snap"
    registry = Registry([
        ServiceModule(
            name="benefits",
            seeds=["https://otda.ny.gov/programs/snap/work-requirements.asp"],
            allowlist=["otda.ny.gov"],
        )
    ])
    client = _Client({
        url: "SNAP recipients can manage their case and recertification through official channels.",
    })
    ctx = ToolContext(citations=CitationRegistry(), registry=registry, http=client)

    out = await web_fetch_tools()[0].handler(
        {"url": url, "query": "SNAP case recertification"},
        ctx,
    )

    assert client.urls == [url]
    assert "{cite:S1}" in out


async def test_discovered_result_can_be_fetched_into_answer_grade_evidence():
    url = "https://otda.ny.gov/programs/snap"

    async def fake_search(query, domains, published_after=None, published_before=None, count=5):
        return [{
            "title": "SNAP",
            "url": url,
            "snippet": "Short SNAP discovery result.",
        }]

    registry = Registry([
        ServiceModule(name="benefits", allowlist=["otda.ny.gov"])
    ])
    client = _Client({
        url: "Current SNAP recertification and case-management guidance.",
    })
    ctx = ToolContext(
        citations=CitationRegistry(),
        registry=registry,
        http=client,
    )

    search_out = await web_search_tools(
        ["otda.ny.gov"],
        search_fn=fake_search,
    )[0].handler({"query": "SNAP recertification"}, ctx)
    source_out = await web_fetch_tools()[0].handler(
        {"url": url, "query": "SNAP recertification case guidance"},
        ctx,
    )

    assert "Short SNAP discovery result" in search_out
    assert "Current SNAP recertification" in source_out
    assert ctx.citations.mapping()["S1"]["provenance"] == {
        "evidence_grade": "authoritative_excerpt",
        "source_tier": "authoritative",
    }
    assert ctx.citations.mapping()["S2"]["url"] == url
    _assert_fetch_provenance(
        ctx.citations.mapping()["S2"], "authoritative", "authoritative",
    )
    assert "Current SNAP recertification" in ctx.citations.mapping()["S2"]["snippet"]


async def test_failed_source_fetch_preserves_official_excerpt_history():
    url = "https://otda.ny.gov/hearings/faq.asp"

    async def fake_search(query, domains, published_after=None, published_before=None, count=5):
        return [{
            "title": "Fair Hearings",
            "url": url,
            "snippet": "You may ask within 90 days from the",
        }]

    registry = Registry([
        ServiceModule(name="benefits", allowlist=["otda.ny.gov"])
    ])
    ctx = ToolContext(
        citations=CitationRegistry(),
        registry=registry,
        http=_Client({url: _Response(status_code=503)}),
    )

    await web_search_tools(
        ["otda.ny.gov"],
        search_fn=fake_search,
    )[0].handler({"query": "SNAP fair hearing deadline"}, ctx)
    out = await web_fetch_tools()[0].handler(
        {"url": url, "query": "SNAP fair hearing deadline"},
        ctx,
    )

    assert "could not be fetched" in out
    assert "preserve other verified results" not in out.lower()
    assert "do not guess" not in out.lower()
    assert ctx.citations.mapping()["S1"]["provenance"] == {
        "evidence_grade": "authoritative_excerpt",
        "source_tier": "authoritative",
    }


async def test_failed_fetch_fallback_search_preserves_the_page_identity(monkeypatch):
    url = "https://www.nyc.gov/content/organize/pages/talk-to-tenants-nav"
    captured = {}

    async def unavailable(*_args, **_kwargs):
        raise ValueError("blocked")

    async def search(args, _ctx):
        captured.update(args)
        return "Alternate official source"

    monkeypatch.setattr(web_fetch_module, "_fetch_page_with_browser", unavailable)
    monkeypatch.setattr(
        web_fetch_module,
        "web_search_tools",
        lambda *_args, **_kwargs: [
            Tool(
                name="web_search",
                description="Search",
                parameters={"type": "object", "properties": {}},
                handler=search,
            )
        ],
    )
    ctx = ToolContext(
        citations=CitationRegistry(),
        registry=Registry([ServiceModule(name="housing", allowlist=["nyc.gov"])]),
    )

    out = await web_fetch_tools()[0].handler(
        {"url": url, "query": "tenant organizing resources"},
        ctx,
    )

    assert captured["query"] == f"{url} tenant organizing resources"
    assert captured["prefer"] == ["www.nyc.gov"]
    assert "Alternate official source" in out


async def test_failed_source_fetch_preserves_canonical_city_discovery_history():
    legacy_url = "https://www1.nyc.gov/site/hra/help/snap.page"
    canonical_url = "https://www.nyc.gov/site/hra/help/snap.page"
    registry = Registry([ServiceModule(name="benefits", seeds=[legacy_url])])
    citations = CitationRegistry()
    citations.register(
        legacy_url,
        snippet="Incomplete discovery evidence",
        kind="WEB",
        provenance={"evidence_grade": "discovery"},
    )
    ctx = ToolContext(
        citations=citations,
        registry=registry,
        http=_Client({legacy_url: _Response(status_code=503)}),
    )

    out = await web_fetch_tools()[0].handler(
        {"url": legacy_url, "query": "SNAP guidance"},
        ctx,
    )

    assert "could not be fetched" in out
    assert ctx.citations.mapping()["S1"] == {
        "id": "S1",
        "url": canonical_url,
        "title": "",
        "snippet": "Incomplete discovery evidence",
        "kind": "WEB",
        "valid_as_of": "",
        "provenance": {"evidence_grade": "discovery"},
    }


async def test_web_fetch_preserves_network_and_canonical_citation_urls():
    network_url = "https://www1.nyc.gov/site/hra/help/snap.page"
    citation_url = "https://www.nyc.gov/site/hra/help/snap.page"
    ctx = ToolContext(
        citations=CitationRegistry(),
        registry=Registry([]),
        http=_Client({network_url: "Current SNAP application guidance."}),
    )

    await web_fetch_tools()[0].handler(
        {"url": network_url, "query": "SNAP application guidance"}, ctx,
    )

    citation = ctx.citations.mapping()["S1"]
    acquisition = citation["provenance"]["acquisition"]
    assert citation["url"] == citation_url
    assert acquisition["final_url"] == network_url
    assert acquisition["citation_url"] == citation_url


async def test_web_fetch_does_not_follow_a_redirect_off_curated_domains():
    url = "https://otda.ny.gov/programs/snap"
    registry = Registry([
        ServiceModule(name="benefits", allowlist=["otda.ny.gov"])
    ])
    client = _Client({
        url: _Response(
            status_code=302,
            location="http://127.0.0.1/private",
        ),
    })
    ctx = ToolContext(citations=CitationRegistry(), registry=registry, http=client)

    out = await web_fetch_tools()[0].handler(
        {"url": url, "query": "SNAP recertification"},
        ctx,
    )

    assert client.urls == [url]
    assert client.requests[0][1]["follow_redirects"] is False
    assert "could not be fetched" in out


async def test_web_fetch_rejects_unsafe_url_shapes_before_network():
    tool = web_fetch_tools()[0]

    for url in (
        "https://otda.ny.gov@evil.example/programs/snap",
        "http://127.0.0.1/private",
    ):
        registry = Registry([
            ServiceModule(name="benefits", allowlist=["otda.ny.gov"])
        ])
        client = _Client({})
        ctx = ToolContext(citations=CitationRegistry(), registry=registry, http=client)

        out = await tool.handler({"url": url, "query": "SNAP"}, ctx)

        assert "could not be fetched" in out
        assert client.urls == []


async def test_web_fetch_validates_each_redirect_target():
    start = "https://otda.ny.gov/programs/snap"
    second = "https://otda.ny.gov/programs/snap/current"
    client = _Client({
        start: _Response(status_code=302, location="/programs/snap/current"),
        second: _Response(status_code=302, location="http://127.0.0.1/private"),
    })
    registry = Registry([
        ServiceModule(name="benefits", allowlist=["otda.ny.gov"])
    ])
    ctx = ToolContext(citations=CitationRegistry(), registry=registry, http=client)

    out = await web_fetch_tools()[0].handler(
        {"url": start, "query": "SNAP"},
        ctx,
    )

    assert client.urls == [start, second]
    assert all(request["follow_redirects"] is False for _, request in client.requests)
    assert "could not be fetched" in out


async def test_web_fetch_attributes_redirected_evidence_to_final_url():
    start = "https://otda.ny.gov/programs/snap"
    final = "https://otda.ny.gov/programs/snap/current"
    client = _Client({
        start: _Response(status_code=302, location="/programs/snap/current"),
        final: "Current SNAP recertification guidance.",
    })
    registry = Registry([
        ServiceModule(name="benefits", allowlist=["otda.ny.gov"])
    ])
    ctx = ToolContext(citations=CitationRegistry(), registry=registry, http=client)

    out = await web_fetch_tools()[0].handler(
        {"url": start, "query": "SNAP recertification guidance"},
        ctx,
    )

    assert final in out
    assert ctx.citations.mapping()["S1"]["url"] == final


async def test_redirected_fetch_reuses_prior_authoritative_evidence():
    start = "https://otda.ny.gov/programs/snap"
    final = "https://otda.ny.gov/programs/snap/current"
    registry = Registry([
        ServiceModule(name="benefits", allowlist=["otda.ny.gov"])
    ])
    citations = CitationRegistry()
    old_id = citations.register(
        final,
        snippet="Current SNAP recertification guidance.",
        kind="WEB",
        provenance={"evidence_grade": "authoritative"},
    )
    ctx = ToolContext(
        citations=citations,
        registry=registry,
        http=_Client({
            start: _Response(status_code=302, location="/programs/snap/current"),
            final: "Current SNAP recertification guidance.",
        }),
    )

    out = await web_fetch_tools()[0].handler(
        {"url": start, "query": "SNAP recertification guidance"},
        ctx,
    )

    assert old_id in ctx.citations.mapping()
    assert "{cite:S1}" in out
    assert list(ctx.citations.mapping()) == ["S1"]


async def test_duplicate_redirect_targets_reuse_one_citation_across_atomic_calls():
    first = "https://otda.ny.gov/programs/snap"
    second = "https://otda.ny.gov/programs/snap-guide"
    final = "https://otda.ny.gov/programs/snap/current"
    ctx = ToolContext(
        citations=CitationRegistry(),
        registry=Registry([
            ServiceModule(name="benefits", allowlist=["otda.ny.gov"])
        ]),
        http=_Client({
            first: _Response(status_code=302, location=final),
            second: _Response(status_code=302, location=final),
            final: "Current SNAP recertification guidance.",
        }),
    )

    first_out = await web_fetch_tools()[0].handler(
        {"url": first, "query": "SNAP recertification guidance"}, ctx,
    )
    second_out = await web_fetch_tools()[0].handler(
        {"url": second, "query": "SNAP recertification guidance"}, ctx,
    )

    assert "Current SNAP recertification guidance." in first_out
    assert "Current SNAP recertification guidance." in second_out
    assert list(ctx.citations.mapping()) == ["S1"]


def test_web_fetch_schema_explains_acquisition_is_not_trust():
    tool = web_fetch_tools()[0]

    assert "trust is graded separately" in tool.description
    assert "same URL once with render=true" in tool.description
    assert "Public HTTPS" in tool.parameters["properties"]["url"]["description"]


async def test_web_fetch_accepts_only_a_trailing_slash_variant():
    declared = "https://access.nyc.gov/snap-work-requirements/"
    requested = declared.rstrip("/")
    registry = Registry([ServiceModule(name="benefits", seeds=[declared])])
    client = _Client({
        requested: "SNAP work requirements include exemptions and qualifying activities.",
    })
    ctx = ToolContext(citations=CitationRegistry(), registry=registry, http=client)
    tool = web_fetch_tools()[0]

    out = await tool.handler(
        {"url": requested, "query": "SNAP work requirements exemptions activities"},
        ctx,
    )

    assert client.urls == [requested]
    assert "{cite:S1}" in out
    assert ctx.citations.mapping()["S1"]["url"] == requested

    for public_url in (
        "https://access.nyc.gov/snap-work-requirements//",
        "https://access.nyc.gov/other-page",
    ):
        client.pages[public_url] = "Public SNAP work requirements guidance."
        fetched = await tool.handler(
            {"url": public_url, "query": "SNAP work requirements"}, ctx,
        )
        assert "Public SNAP work requirements guidance" in fetched

async def test_benefits_declares_current_snap_work_rule_pages():
    city = "https://www.nyc.gov/main/services/snap-benefits/abawd"
    state = "https://otda.ny.gov/programs/snap/work-requirements.asp"
    registry = Registry.discover(config.MODULES_DIR)
    client = _Client({
        city: "SNAP work requirements include work, volunteering, or training.",
        state: "ABAWD work rules include exemptions and an 80-hour monthly requirement.",
    })
    ctx = ToolContext(citations=CitationRegistry(), registry=registry, http=client)

    first = await web_fetch_tools()[0].handler(
        {"url": state, "query": "SNAP work rules exemptions 80 hours"}, ctx,
    )
    second = await web_fetch_tools()[0].handler(
        {"url": city, "query": "SNAP work rules volunteering training"}, ctx,
    )

    assert set(client.urls) == {city, state}
    assert "80-hour monthly requirement" in first
    assert "volunteering, or training" in second
    assert len(ctx.citations) == 2


async def test_advisories_recovers_authoritative_notify_cost_evidence():
    faq = "https://a858-nycnotify.nyc.gov/Home/FAQ"
    terms = (
        "https://www.nyc.gov/site/em/resources/notify_nyc/"
        "notify-nyc-short-code-terms-conditions-privacy-policy-information.page"
    )
    registry = Registry.discover(config.MODULES_DIR)
    client = _Client({
        faq: (
            "<title>Notify NYC FAQ</title>"
            "<p>The Notify NYC mobile app is completely free to use.</p>"
        ),
        terms: (
            "<title>Notify NYC Short Code Terms</title>"
            "<p>Message and Data Rates May Apply. You are responsible for these charges "
            "to your wireless provider.</p>"
        ),
    })
    ctx = ToolContext(citations=CitationRegistry(), registry=registry, http=client)

    faq_out = await web_fetch_tools()[0].handler(
        {"url": faq, "query": "Notify NYC cost free"}, ctx,
    )
    terms_out = await web_fetch_tools()[0].handler(
        {"url": terms, "query": "message data rates wireless provider"}, ctx,
    )

    assert "completely free" in faq_out
    assert "Message and Data Rates May Apply" in terms_out
    assert len(ctx.citations) == 2
    assert all(
        citation["provenance"]["evidence_grade"] == "authoritative"
        for citation in ctx.citations.mapping().values()
    )


async def test_web_fetch_extracts_text_from_an_approved_pdf():
    url = "https://www.nycourts.gov/decision-list.pdf"
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer)
    pdf.drawString(72, 720, "Commons West motion for leave to appeal denied")
    pdf.save()
    registry = Registry([ServiceModule(name="law", seeds=[url])])
    client = _Client({
        url: _Response(content=buffer.getvalue(), content_type="application/pdf"),
    })
    ctx = ToolContext(citations=CitationRegistry(), registry=registry, http=client)
    tool = web_fetch_tools()[0]

    out = await tool.handler(
        {"url": url, "query": "Commons West leave appeal denied"}, ctx,
    )

    assert "leave to appeal denied" in out
    assert "{cite:S1}" in out


async def test_web_fetch_combines_relevant_excerpts_from_one_page_into_one_citation():
    url = "https://dol.ny.gov/minimum-wage-tipped-workers"
    text = (
        "<title>Tipped wage</title><p>Food service workers receive an $11.35 cash wage.</p>"
        + ("<p>Navigation and unrelated material.</p>" * 120)
        + "<p>The full minimum wage is $17.00 per hour.</p>"
    )
    registry = Registry([ServiceModule(name="law", seeds=[url])])
    client = _Client({url: text})
    ctx = ToolContext(citations=CitationRegistry(), registry=registry, http=client)
    tool = web_fetch_tools()[0]

    out = await tool.handler(
        {"url": url, "query": "food service cash wage full minimum wage"}, ctx,
    )

    citations = ctx.citations.mapping()
    assert len(citations) == 1
    assert "$11.35" in citations["S1"]["snippet"]
    assert "$17.00" in citations["S1"]["snippet"]
    assert out.count("{cite:S1}") == 1
