from __future__ import annotations

from io import BytesIO

from reportlab.pdfgen import canvas

from heynyc.core import config
from heynyc.core.citations import CitationRegistry
from heynyc.core.manifest import ServiceModule
from heynyc.core.registry import Registry
from heynyc.core.tools.base import ToolContext
from heynyc.core.tools.official_sources import _relevant_chunks, official_source_tools


class _Response:
    def __init__(self, text: str = "", *, content: bytes | None = None, content_type: str = "text/html"):
        self.text = text
        self.content = content if content is not None else text.encode()
        self.headers = {"content-type": content_type}

    def raise_for_status(self):
        return None


class _Client:
    def __init__(self, pages: dict[str, str | _Response]):
        self.pages = pages
        self.urls: list[str] = []

    async def get(self, url, **kwargs):
        self.urls.append(url)
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
