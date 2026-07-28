from __future__ import annotations

from io import BytesIO

from reportlab.pdfgen import canvas

from heynyc.core import config
from heynyc.core.citations import CitationRegistry
from heynyc.core.manifest import ServiceModule
from heynyc.core.registry import Registry
from heynyc.core.tools.base import ToolContext
from heynyc.core.tools.official_sources import _relevant_chunks, official_source_tools
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
    ):
        self.text = text
        self.content = content if content is not None else text.encode()
        self.headers = {"content-type": content_type}
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


def test_relevant_chunks_put_strongest_evidence_first():
    text = (
        "Kitchen navigation. "
        + ("Navigation services programs contact us. " * 60)
        + "Kitchen cooks are not eligible for a tip credit and must receive the full minimum wage."
    )

    chunks = _relevant_chunks(text, "kitchen cooks tip credit full minimum wage", limit=2)

    assert "full minimum wage" in chunks[0]


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


async def test_official_sources_fetches_only_seeded_pages_and_returns_relevant_citations():
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
    tool = official_source_tools()[0]

    out = await tool.handler(
        {"urls": [cash, schedule], "query": "retail schedule Fair Workweek notice"}, ctx,
    )

    assert set(client.urls) == {cash, schedule}
    assert "72 hours" in out
    assert "{cite:S" in out
    assert all(cite["kind"] == "WEB" for cite in ctx.citations.mapping().values())
    assert all(
        cite["provenance"] == {"evidence_grade": "authoritative"}
        for cite in ctx.citations.mapping().values()
    )


async def test_official_sources_rejects_access_wall_content():
    url = "https://www.uscis.gov/current-guidance"
    registry = Registry([ServiceModule(name="immigration", seeds=[url])])
    client = _Client({
        url: "<title>Access Denied</title><p>Access denied. Complete the security challenge.</p>",
    })
    ctx = ToolContext(citations=CitationRegistry(), registry=registry, http=client)

    out = await official_source_tools()[0].handler(
        {"urls": [url], "query": "current guidance access"},
        ctx,
    )

    assert "could not be retrieved" in out
    assert ctx.citations.mapping() == {}


async def test_official_sources_marks_archived_content_in_the_evidence():
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

    out = await official_source_tools()[0].handler(
        {"urls": [url], "query": "Temporary Protected Status Haiti"}, ctx,
    )

    assert "SOURCE STATUS: ARCHIVED" in out
    assert "SOURCE STATUS: ARCHIVED" in ctx.citations.mapping()["S1"]["snippet"]
    assert ctx.citations.mapping()["S1"]["provenance"] == {
        "evidence_grade": "discovery",
    }


async def test_official_sources_marks_archive_urls_without_banner_wording():
    url = "https://www.uscis.gov/archive/current-guidance"
    registry = Registry([ServiceModule(name="immigration", seeds=[url])])
    client = _Client({
        url: (
            "<title>Historical guidance</title>"
            "<p>Temporary Protected Status guidance for Haiti.</p>"
        ),
    })
    ctx = ToolContext(citations=CitationRegistry(), registry=registry, http=client)

    out = await official_source_tools()[0].handler(
        {"urls": [url], "query": "Temporary Protected Status Haiti"}, ctx,
    )

    assert "SOURCE STATUS: ARCHIVED" in out


async def test_official_sources_does_not_mark_current_page_from_unrelated_archive_footer():
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

    out = await official_source_tools()[0].handler(
        {"urls": [url], "query": "Temporary Protected Status Haiti current"},
        ctx,
    )

    assert "SOURCE STATUS: ARCHIVED" not in out


async def test_official_sources_rejects_a_url_not_declared_by_a_module():
    registry = Registry([ServiceModule(name="law", seeds=["https://www.nyc.gov/allowed"])])
    client = _Client({})
    ctx = ToolContext(citations=CitationRegistry(), registry=registry, http=client)
    tool = official_source_tools()[0]

    out = await tool.handler(
        {"urls": ["https://evil.example/page"], "query": "cash"}, ctx,
    )

    assert "not an approved official source" in out
    assert client.urls == []


async def test_official_sources_rejects_editorial_allowlist_domains():
    url = "https://timeout.com/newyork/news/example"
    registry = Registry([
        ServiceModule(
            name="events",
            official_only=False,
            allowlist=["timeout.com"],
            source_tiers={"editorial": ["timeout.com"]},
        )
    ])
    client = _Client({})
    ctx = ToolContext(citations=CitationRegistry(), registry=registry, http=client)

    out = await official_source_tools()[0].handler(
        {"urls": [url], "query": "event details"},
        ctx,
    )

    assert "not an approved official source" in out
    assert client.urls == []


async def test_official_sources_rejects_explicitly_editorial_gov_domain():
    url = "https://sub.events.example.gov/weekend"
    registry = Registry([
        ServiceModule(
            name="events",
            official_only=False,
            allowlist=["sub.events.example.gov"],
            source_tiers={"editorial": ["events.example.gov"]},
        )
    ])
    client = _Client({})
    ctx = ToolContext(citations=CitationRegistry(), registry=registry, http=client)

    out = await official_source_tools()[0].handler(
        {"urls": [url], "query": "event details"},
        ctx,
    )

    assert "not an approved official source" in out
    assert client.urls == []


async def test_official_sources_rejects_explicitly_editorial_seed():
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
    client = _Client({})
    ctx = ToolContext(citations=CitationRegistry(), registry=registry, http=client)

    out = await official_source_tools()[0].handler(
        {"urls": [url], "query": "event details"},
        ctx,
    )

    assert "not an approved official source" in out
    assert client.urls == []


async def test_official_sources_accepts_explicit_authoritative_domain():
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

    out = await official_source_tools()[0].handler(
        {"urls": [url], "query": "event details schedule"},
        ctx,
    )

    assert "{cite:S1}" in out
    assert client.urls == [url]


async def test_official_sources_fetches_an_unseeded_page_on_a_curated_domain():
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

    out = await official_source_tools()[0].handler(
        {"urls": [url], "query": "SNAP case recertification"},
        ctx,
    )

    assert client.urls == [url]
    assert "{cite:S1}" in out


async def test_discovered_result_can_be_fetched_into_answer_grade_evidence():
    url = "https://otda.ny.gov/programs/snap"

    async def fake_search(query, domains, recency=None):
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
    source_out = await official_source_tools()[0].handler(
        {"urls": [url], "query": "SNAP recertification case guidance"},
        ctx,
    )

    assert "call official_sources" in search_out
    assert "Current SNAP recertification" in source_out
    assert ctx.citations.mapping()["S1"]["provenance"] == {
        "evidence_grade": "discovery",
    }
    assert ctx.citations.mapping()["S2"]["url"] == url
    assert ctx.citations.mapping()["S2"]["provenance"] == {
        "evidence_grade": "authoritative",
    }
    assert "Current SNAP recertification" in ctx.citations.mapping()["S2"]["snippet"]


async def test_failed_source_fetch_preserves_discovery_history():
    url = "https://otda.ny.gov/hearings/faq.asp"

    async def fake_search(query, domains, recency=None):
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
    out = await official_source_tools()[0].handler(
        {"urls": [url], "query": "SNAP fair hearing deadline"},
        ctx,
    )

    assert "could not be retrieved" in out
    assert ctx.citations.mapping()["S1"]["provenance"] == {
        "evidence_grade": "discovery",
    }


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

    out = await official_source_tools()[0].handler(
        {"urls": [legacy_url], "query": "SNAP guidance"},
        ctx,
    )

    assert "could not be retrieved" in out
    assert ctx.citations.mapping()["S1"] == {
        "id": "S1",
        "url": canonical_url,
        "title": "",
        "snippet": "Incomplete discovery evidence",
        "kind": "WEB",
        "valid_as_of": "",
        "provenance": {"evidence_grade": "discovery"},
    }


async def test_official_sources_does_not_follow_a_redirect_off_curated_domains():
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

    out = await official_source_tools()[0].handler(
        {"urls": [url], "query": "SNAP recertification"},
        ctx,
    )

    assert client.urls == [url]
    assert client.requests[0][1]["follow_redirects"] is False
    assert "approved official pages could not be retrieved" in out


async def test_official_sources_rejects_curated_domain_bypasses():
    tool = official_source_tools()[0]

    for url in (
        "http://otda.ny.gov/programs/snap",
        "https://otda.ny.gov@evil.example/programs/snap",
        "https://evil.example/otda.ny.gov/programs/snap",
        "https://evilotda.ny.gov/programs/snap",
    ):
        registry = Registry([
            ServiceModule(name="benefits", allowlist=["otda.ny.gov"])
        ])
        client = _Client({})
        ctx = ToolContext(citations=CitationRegistry(), registry=registry, http=client)

        out = await tool.handler({"urls": [url], "query": "SNAP"}, ctx)

        assert "not an approved official source" in out
        assert client.urls == []

    for url in (
        "http://otda.ny.gov/programs/snap",
        "https://user:password@otda.ny.gov/programs/snap",
    ):
        registry = Registry([ServiceModule(name="benefits", seeds=[url])])
        client = _Client({})
        ctx = ToolContext(citations=CitationRegistry(), registry=registry, http=client)

        out = await tool.handler({"urls": [url], "query": "SNAP"}, ctx)

        assert "not an approved official source" in out
        assert client.urls == []


async def test_official_sources_validates_each_redirect_target():
    start = "https://otda.ny.gov/programs/snap"
    second = "https://otda.ny.gov/programs/snap/current"
    client = _Client({
        start: _Response(status_code=302, location="/programs/snap/current"),
        second: _Response(status_code=302, location="https://evil.example/private"),
    })
    registry = Registry([
        ServiceModule(name="benefits", allowlist=["otda.ny.gov"])
    ])
    ctx = ToolContext(citations=CitationRegistry(), registry=registry, http=client)

    out = await official_source_tools()[0].handler(
        {"urls": [start], "query": "SNAP"},
        ctx,
    )

    assert client.urls == [start, second]
    assert all(request["follow_redirects"] is False for _, request in client.requests)
    assert "approved official pages could not be retrieved" in out


async def test_official_sources_attributes_redirected_evidence_to_final_url():
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

    out = await official_source_tools()[0].handler(
        {"urls": [start], "query": "SNAP recertification guidance"},
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

    out = await official_source_tools()[0].handler(
        {"urls": [start], "query": "SNAP recertification guidance"},
        ctx,
    )

    assert old_id in ctx.citations.mapping()
    assert "{cite:S1}" in out
    assert list(ctx.citations.mapping()) == ["S1"]


async def test_duplicate_redirect_targets_emit_one_evidence_block():
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

    out = await official_source_tools()[0].handler(
        {
            "urls": [first, second],
            "query": "SNAP recertification guidance",
        },
        ctx,
    )

    assert out.count("Current SNAP recertification guidance.") == 1
    assert list(ctx.citations.mapping()) == ["S1"]


def test_official_sources_schema_accepts_discovered_curated_urls():
    tool = official_source_tools()[0]

    assert "curated official domains" in tool.description
    assert "discovered" in tool.parameters["properties"]["urls"]["description"]


async def test_official_sources_accepts_only_a_trailing_slash_variant():
    declared = "https://access.nyc.gov/snap-work-requirements/"
    requested = declared.rstrip("/")
    registry = Registry([ServiceModule(name="benefits", seeds=[declared])])
    client = _Client({
        requested: "SNAP work requirements include exemptions and qualifying activities.",
    })
    ctx = ToolContext(citations=CitationRegistry(), registry=registry, http=client)
    tool = official_source_tools()[0]

    out = await tool.handler(
        {"urls": [requested], "query": "SNAP work requirements exemptions activities"},
        ctx,
    )

    assert client.urls == [requested]
    assert "{cite:S1}" in out
    assert ctx.citations.mapping()["S1"]["url"] == requested

    for unapproved in (
        "https://access.nyc.gov/snap-work-requirements//",
        "https://access.nyc.gov/other-page",
    ):
        rejected = await tool.handler(
            {
                "urls": [unapproved],
                "query": "SNAP work requirements",
            },
            ctx,
        )
        assert "not an approved official source" in rejected

async def test_benefits_declares_current_snap_work_rule_pages():
    city = "https://www.nyc.gov/main/services/snap-benefits/abawd"
    state = "https://otda.ny.gov/programs/snap/work-requirements.asp"
    registry = Registry.discover(config.MODULES_DIR)
    client = _Client({
        city: "SNAP work requirements include work, volunteering, or training.",
        state: "ABAWD work rules include exemptions and an 80-hour monthly requirement.",
    })
    ctx = ToolContext(citations=CitationRegistry(), registry=registry, http=client)

    out = await official_source_tools()[0].handler(
        {
            "urls": [state, city],
            "query": "SNAP work rules exemptions volunteering training 80 hours",
        },
        ctx,
    )

    assert set(client.urls) == {city, state}
    assert "not an approved official source" not in out
    assert len(ctx.citations) == 2


async def test_official_sources_extracts_text_from_an_approved_pdf():
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
    tool = official_source_tools()[0]

    out = await tool.handler(
        {"urls": [url], "query": "Commons West leave appeal denied"}, ctx,
    )

    assert "leave to appeal denied" in out
    assert "{cite:S1}" in out


async def test_official_sources_combines_relevant_excerpts_from_one_page_into_one_citation():
    url = "https://dol.ny.gov/minimum-wage-tipped-workers"
    text = (
        "<title>Tipped wage</title><p>Food service workers receive an $11.35 cash wage.</p>"
        + ("<p>Navigation and unrelated material.</p>" * 120)
        + "<p>The full minimum wage is $17.00 per hour.</p>"
    )
    registry = Registry([ServiceModule(name="law", seeds=[url])])
    client = _Client({url: text})
    ctx = ToolContext(citations=CitationRegistry(), registry=registry, http=client)
    tool = official_source_tools()[0]

    out = await tool.handler(
        {"urls": [url], "query": "food service cash wage full minimum wage"}, ctx,
    )

    citations = ctx.citations.mapping()
    assert len(citations) == 1
    assert "$11.35" in citations["S1"]["snippet"]
    assert "$17.00" in citations["S1"]["snippet"]
    assert out.count("{cite:S1}") == 1
