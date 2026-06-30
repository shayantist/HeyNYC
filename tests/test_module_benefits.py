"""Offline tests for the benefits module's benefits_search tool (no network)."""
from __future__ import annotations

import httpx

from heynyc.core import config
from heynyc.core.citations import CitationRegistry
from heynyc.core.index.embedder import HashEmbedder
from heynyc.core.registry import Registry
from heynyc.core.tools.base import ToolContext
from heynyc.modules.benefits import screening, tools as btools

# Deterministic, offline embedder (the project's test default) injected via ToolContext so the
# benefits tool's hybrid retrieval never reaches for fastembed (which would download a model).
_EMBEDDER = HashEmbedder()

_FAKE_ROW = {
    "program_code": "S2R007",
    "program_name": "Supplemental Nutrition Assistance Program",
    "plain_language_program_name": "Help buying food (SNAP / food stamps)",
    "program_category": "Food",
    "plain_language_eligibility": "You may qualify based on household size and income.",
    "heads_up": "Some college students have extra rules.",
    "how_to_apply_summary": "Apply online through ACCESS HRA.",
    "url_of_online_application": "https://access.nyc.gov/programs/snap/",
    "updated_at": "2026-03-21T11:00:43.000",
}


def _benefits_tool():
    registry = Registry.discover(config.MODULES_DIR)
    tool = next(t for t in registry.load_module_tools() if t.name == "benefits_search")
    return tool, registry


def _client_returning(rows, status=200):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["params"] = dict(request.url.params)
        return httpx.Response(status, json=rows)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler)), seen


def test_benefits_module_tool_is_discovered():
    registry = Registry.discover(config.MODULES_DIR)
    assert "benefits_search" in {t.name for t in registry.load_module_tools()}


async def test_benefits_search_fetches_catalog_then_ranks_and_grounds():
    # Retrieval fetches the catalog (no conjunctive $q) and ranks it with the hybrid retriever.
    tool, registry = _benefits_tool()
    client, seen = _client_returning([_FAKE_ROW])
    ctx = ToolContext(citations=CitationRegistry(), registry=registry, http=client, embedder=_EMBEDDER)
    out = await tool.handler({"query": "food stamps for my family"}, ctx)
    await client.aclose()

    assert "$q" not in seen["params"]  # no fragile conjunctive full-text search
    assert seen["params"]["$limit"] == "200"  # fetch the whole small catalog, then rank locally
    assert "Supplemental Nutrition Assistance Program" in out  # 'food'/'stamps' matched the row
    assert "{cite:S1}" in out
    assert "2026-03-21" in out  # valid_as_of surfaced in the tool output
    cite = ctx.citations.mapping()["S1"]
    assert cite["kind"] == "DATA"
    assert cite["valid_as_of"] == "2026-03-21"


async def test_benefits_search_category_filters_via_where():
    tool, registry = _benefits_tool()
    client, seen = _client_returning([_FAKE_ROW])
    ctx = ToolContext(citations=CitationRegistry(), registry=registry, http=client, embedder=_EMBEDDER)
    await tool.handler({"query": "food", "category": "Food"}, ctx)
    await client.aclose()
    assert seen["params"]["$where"] == "program_category='Food'"


async def test_benefits_search_rejects_unknown_category():
    # The JSON-schema enum is advisory; the handler must allowlist `category` before it
    # ever reaches the SoQL $where clause (SoQL-injection guard).
    tool, registry = _benefits_tool()
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json=[_FAKE_ROW])

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ctx = ToolContext(citations=CitationRegistry(), registry=registry, http=client, embedder=_EMBEDDER)
    out = await tool.handler({"query": "x", "category": "Food'; DROP TABLE"}, ctx)
    await client.aclose()

    assert "categor" in out.lower()  # instructive error naming the issue
    assert calls["n"] == 0  # short-circuited before any network call / no injected query sent
    assert len(ctx.citations) == 0


async def test_benefits_search_abstains_on_no_match():
    tool, registry = _benefits_tool()
    client, _ = _client_returning([])
    ctx = ToolContext(citations=CitationRegistry(), registry=registry, http=client, embedder=_EMBEDDER)
    out = await tool.handler({"query": "nonexistent program zzz"}, ctx)
    await client.aclose()
    assert "access.nyc.gov" in out.lower()
    assert len(ctx.citations) == 0  # nothing fabricated/cited


async def test_benefits_search_handles_dataset_error():
    tool, registry = _benefits_tool()
    client, _ = _client_returning([], status=503)  # raise_for_status -> HTTPStatusError
    ctx = ToolContext(citations=CitationRegistry(), registry=registry, http=client, embedder=_EMBEDDER)
    out = await tool.handler({"query": "snap"}, ctx)
    await client.aclose()
    assert "311" in out or "access.nyc.gov" in out.lower()
    assert len(ctx.citations) == 0


async def test_benefits_search_strips_html_and_normalizes_null():
    # Real dataset rows carry HTML in prose fields and the literal string "NULL" in url fields.
    row = dict(_FAKE_ROW)
    row["plain_language_eligibility"] = "<p>Income under <strong>$50,000</strong>/yr</p><ul><li>Age 18+</li></ul>"
    row["url_of_online_application"] = "NULL"  # Socrata literal-NULL string, not a real link
    row["url_of_pdf_application_forms"] = ""
    tool, registry = _benefits_tool()
    client, _ = _client_returning([row])
    ctx = ToolContext(citations=CitationRegistry(), registry=registry, http=client, embedder=_EMBEDDER)
    out = await tool.handler({"query": "snap"}, ctx)
    await client.aclose()

    assert "<p>" not in out and "<li>" not in out and "<strong>" not in out  # HTML stripped
    assert "$50,000" in out  # content preserved
    url = ctx.citations.mapping()["S1"]["url"]
    assert url != "NULL"
    assert url.startswith("https://data.cityofnewyork.us")  # fell back to the dataset source


def test_benefits_eval_cases_load_and_flag_safety():
    from heynyc.eval.cases import load_cases

    cases = [c for c in load_cases(Registry.discover(config.MODULES_DIR, config.BASE_ALLOWLIST)) if c.module == "benefits"]
    ids = {c.id for c in cases}
    assert {"benefits_eligibility_definite", "benefits_help_groceries"} <= ids
    # the personalized-eligibility case is harm-tagged → auto safety_critical
    definite = next(c for c in cases if c.id == "benefits_eligibility_definite")
    assert definite.safety_critical
    assert definite.invariants.get("must_abstain_or_redirect") is True


# --- screen_eligibility (Module B) -----------------------------------------

def _route_screen(req: httpx.Request) -> httpx.Response:
    host, path = req.url.host, req.url.path
    if "screeningapi" in host and path == "/authToken":
        return httpx.Response(200, json={"type": "SUCCESS", "token": "tok"})
    if "screeningapi" in host and path == "/eligibilityPrograms":
        return httpx.Response(200, json={"type": "SUCCESS",
            "eligiblePrograms": [{"code": "S2R007", "name": "SNAP"}]})
    if "data.cityofnewyork.us" in host:  # kvhd-5fmu catalog
        return httpx.Response(200, json=[{
            ":id": "row-snap", "program_code": "S2R007", "program_name": "SNAP",
            "plain_language_program_name": "Food stamps", "program_category": "Food",
            "url_of_online_application": "https://access.nyc.gov/snap", "updated_at": "2026-03-01"}])
    return httpx.Response(404)


async def test_screen_eligibility_grounds_and_frames(monkeypatch):
    monkeypatch.setattr(config, "screening_creds",
                        lambda: ("https://sandbox.screeningapi.cityofnewyork.us", "u", "p"))
    screening.clear_token("https://sandbox.screeningapi.cityofnewyork.us")
    client = httpx.AsyncClient(transport=httpx.MockTransport(_route_screen))
    reg = CitationRegistry()
    ctx = ToolContext(citations=reg, registry=Registry([]), http=client)
    out = await btools._screen_handler(
        {"household": {"livingRenting": True},
         "persons": [{"age": 32, "householdMemberType": "HeadOfHousehold"}]}, ctx)
    await client.aclose()
    assert "likely eligible" in out.lower()
    assert "estimate" in out.lower() and "determination" in out.lower()
    assert "SNAP" in out and "access.nyc.gov/snap" in out
    assert "doesn't mean you're ineligible" in out.lower()
    cites = reg.mapping()
    # the verdict cite carries api_provenance; a detail cite is row-addressed
    assert any(c["provenance"].get("endpoint", "").startswith("POST ") for c in cites.values())
    assert any("/resource/kvhd-5fmu/row-snap.json" in c["url"] for c in cites.values())


async def test_screen_eligibility_rejects_pii(monkeypatch):
    monkeypatch.setattr(config, "screening_creds",
                        lambda: ("https://sandbox.screeningapi.cityofnewyork.us", "u", "p"))
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]), http=None)
    out = await btools._screen_handler(
        {"household": {}, "persons": [{"age": 30, "name": "Jane"}]}, ctx)
    assert out.startswith("ERROR")


def test_get_tools_gates_screener_on_creds(monkeypatch):
    monkeypatch.setattr(config, "screening_creds", lambda: ("base", "", ""))
    assert {t.name for t in btools.get_tools()} == {"benefits_search"}
    monkeypatch.setattr(config, "screening_creds", lambda: ("base", "u", "p"))
    assert {t.name for t in btools.get_tools()} == {"benefits_search", "screen_eligibility"}
