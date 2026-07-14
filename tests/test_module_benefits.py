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


async def test_benefits_search_surfaces_requested_language_variant():
    # Compliance 4b/4c: when the dataset carries language variants of a program, the tool returns the
    # user's-language row (the city's official translation) when asked, English by default / fallback.
    en = dict(_FAKE_ROW, language="English",
              plain_language_program_name="Help buying food (SNAP / food stamps)")
    es = dict(_FAKE_ROW, language="Spanish",
              plain_language_program_name="Ayuda para comprar alimentos (SNAP)")
    tool, registry = _benefits_tool()

    client, _ = _client_returning([en, es])
    ctx = ToolContext(citations=CitationRegistry(), registry=registry, http=client, embedder=_EMBEDDER)
    default_out = await tool.handler({"query": "food stamps"}, ctx)          # no lang → English
    await client.aclose()
    assert "Help buying food" in default_out
    assert "Ayuda para comprar" not in default_out

    client2, _ = _client_returning([en, es])
    ctx2 = ToolContext(citations=CitationRegistry(), registry=registry, http=client2, embedder=_EMBEDDER)
    es_out = await tool.handler({"query": "food stamps", "lang": "Spanish"}, ctx2)
    await client2.aclose()
    assert "Ayuda para comprar alimentos" in es_out                          # Spanish variant surfaced
    assert "Help buying food" not in es_out


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


def test_benefits_prompt_surfaces_fair_hearing_appeal_path():
    # For a denial/problem with a benefit, the benefits blurb must surface the human/appeal
    # path — call the agency / 311 and the right to a fair hearing.
    reg = Registry.discover(config.MODULES_DIR)
    benefits = next(m for m in reg.modules if m.name == "benefits")
    low = benefits.prompt.lower()
    assert "fair hearing" in low
    assert "311" in benefits.prompt


def test_benefits_prompt_keeps_screening_results_actionable_on_a_phone():
    reg = Registry.discover(config.MODULES_DIR)
    prompt = " ".join(next(m.prompt for m in reg.modules if m.name == "benefits").lower().split())
    assert "up to three" in prompt
    assert "offer to show the rest" in prompt
    assert "official ranking" in prompt
    assert "only legal name and home address are required" in prompt


def test_fairness_metamorphic_cases_present_and_well_formed():
    # Protected-class fairness: the same benefit question, varying only a protected attribute
    # (name/ethnicity, borough/ZIP, language), must be flagged as outcome-invariant INV cases.
    from heynyc.eval.cases import load_cases

    cases = [c for c in load_cases(Registry.discover(config.MODULES_DIR, config.BASE_ALLOWLIST))
             if c.module == "benefits"]
    ids = {c.id for c in cases}
    fairness = [c for c in cases if c.capability == "fairness"]
    bases = [c for c in fairness if c.test_type != "INV"]
    variants = [c for c in fairness if c.test_type == "INV"]
    assert len(bases) >= 1, "fairness needs a neutral base question"
    assert len(variants) >= 3, "one variant per protected attribute"
    for v in variants:
        assert v.base in ids, f"{v.id}: INV base '{v.base}' not found"
        assert v.expect_same_outcome_as_base is True, f"{v.id}: must expect the base outcome"
        # Substance-invariance guard: the cited PROGRAM SET must match the base's (peripheral
        # personalization — language/tone/examples — is allowed and not compared here).
        assert v.expect_same_programs_as_base is True, f"{v.id}: must expect the base program set"
        assert v.perturbation, f"{v.id}: must label which protected attribute varies"
    # cover name/ethnicity, borough/ZIP, and language
    assert {"protected_name", "protected_zip", "protected_language"} <= {v.perturbation for v in variants}


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


async def test_screen_eligibility_keeps_verdict_when_catalog_enrichment_fails(monkeypatch):
    monkeypatch.setattr(config, "screening_creds",
                        lambda: ("https://sandbox.screeningapi.cityofnewyork.us", "u", "p"))
    screening.clear_token("https://sandbox.screeningapi.cityofnewyork.us")

    def route(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/authToken":
            return httpx.Response(200, json={"type": "SUCCESS", "token": "tok"})
        if req.url.path == "/eligibilityPrograms":
            return httpx.Response(200, json={"type": "SUCCESS",
                "eligiblePrograms": [{"code": "S2R007", "name": "SNAP"}]})
        if req.url.host == "data.cityofnewyork.us":
            return httpx.Response(503)
        return httpx.Response(404)

    client = httpx.AsyncClient(transport=httpx.MockTransport(route))
    reg = CitationRegistry()
    ctx = ToolContext(citations=reg, registry=Registry([]), http=client)
    out = await btools._screen_handler(
        {"household": {"livingRenting": True},
         "persons": [{"age": 32, "householdMemberType": "HeadOfHousehold"}]}, ctx)
    await client.aclose()

    assert "likely eligible" in out.lower()
    assert "SNAP" in out
    assert any(c["provenance"].get("endpoint", "").startswith("POST ")
               for c in reg.mapping().values())


async def test_screen_eligibility_keeps_verdict_when_catalog_json_is_malformed(monkeypatch):
    monkeypatch.setattr(config, "screening_creds",
                        lambda: ("https://sandbox.screeningapi.cityofnewyork.us", "u", "p"))
    screening.clear_token("https://sandbox.screeningapi.cityofnewyork.us")

    def route(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/authToken":
            return httpx.Response(200, json={"type": "SUCCESS", "token": "tok"})
        if req.url.path == "/eligibilityPrograms":
            return httpx.Response(200, json={"type": "SUCCESS",
                "eligiblePrograms": [{"code": "S2R007", "name": "SNAP"}]})
        if req.url.host == "data.cityofnewyork.us":
            return httpx.Response(200, content=b"{")
        return httpx.Response(404)

    client = httpx.AsyncClient(transport=httpx.MockTransport(route))
    reg = CitationRegistry()
    ctx = ToolContext(citations=reg, registry=Registry([]), http=client)
    out = await btools._screen_handler(
        {"household": {},
         "persons": [{"age": 32, "householdMemberType": "HeadOfHousehold"}]}, ctx)
    await client.aclose()

    assert "likely eligible" in out.lower() and "SNAP" in out


async def test_screen_eligibility_surfaces_api_validation_error(monkeypatch):
    monkeypatch.setattr(config, "screening_creds",
                        lambda: ("https://sandbox.screeningapi.cityofnewyork.us", "u", "p"))
    screening.clear_token("https://sandbox.screeningapi.cityofnewyork.us")

    def route(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/authToken":
            return httpx.Response(200, json={"type": "SUCCESS", "token": "tok"})
        if req.url.path == "/eligibilityPrograms":
            return httpx.Response(400, json={"type": "FAILURE", "errors": [{
                "message": "income frequency must be a supported value",
                "elementPath": "0.person.0.incomes.0.frequency",
            }]})
        return httpx.Response(404)

    client = httpx.AsyncClient(transport=httpx.MockTransport(route))
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]), http=client)
    out = await btools._screen_handler(
        {"household": {},
         "persons": [{"age": 32, "householdMemberType": "HeadOfHousehold"}]}, ctx)
    await client.aclose()

    assert "rejected" in out.lower()
    assert "income frequency must be a supported value" in out
    assert "couldn't reach" not in out.lower()


async def test_screen_eligibility_handles_malformed_api_validation_payload(monkeypatch):
    monkeypatch.setattr(config, "screening_creds",
                        lambda: ("https://sandbox.screeningapi.cityofnewyork.us", "u", "p"))
    screening.clear_token("https://sandbox.screeningapi.cityofnewyork.us")

    def route(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/authToken":
            return httpx.Response(200, json={"type": "SUCCESS", "token": "tok"})
        if req.url.path == "/eligibilityPrograms":
            return httpx.Response(400, json=["unexpected shape"])
        return httpx.Response(404)

    client = httpx.AsyncClient(transport=httpx.MockTransport(route))
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]), http=client)
    out = await btools._screen_handler(
        {"household": {},
         "persons": [{"age": 32, "householdMemberType": "HeadOfHousehold"}]}, ctx)
    await client.aclose()

    assert "rejected" in out.lower()
    assert "request contract" in out.lower()


async def test_screen_eligibility_rejects_pii(monkeypatch):
    monkeypatch.setattr(config, "screening_creds",
                        lambda: ("https://sandbox.screeningapi.cityofnewyork.us", "u", "p"))
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]), http=None)
    out = await btools._screen_handler(
        {"household": {}, "persons": [{"age": 30, "name": "Jane"}]}, ctx)
    assert out.startswith("ERROR")


def test_get_tools_gates_screener_on_creds(monkeypatch):
    monkeypatch.delenv("HEYNYC_FORMS", raising=False)
    monkeypatch.setattr(config, "screening_creds", lambda: ("base", "", ""))
    assert {t.name for t in btools.get_tools()} == {"benefits_search"}
    monkeypatch.setattr(config, "screening_creds", lambda: ("base", "u", "p"))
    assert {t.name for t in btools.get_tools()} == {"benefits_search", "screen_eligibility"}


def test_screen_tool_uses_city_wire_type_for_cash_on_hand():
    schema = btools.screen_eligibility_tool().parameters
    household = schema["properties"]["household"]
    assert household["properties"]["cashOnHand"]["type"] == "string"
    assert household["additionalProperties"] is False
    assert set(household["properties"]) == set(screening.HOUSEHOLD_FIELDS)
    assert {"livingStayingWithFriend", "livingHotel", "livingPreferNotToSay"} <= set(
        household["properties"]
    )
    person = schema["properties"]["persons"]["items"]["properties"]
    assert set(person) == set(screening.PERSON_FIELDS)
    assert schema["properties"]["persons"]["minItems"] == 1
    assert schema["properties"]["persons"]["maxItems"] == 8
    assert schema["properties"]["persons"]["minContains"] == 1
    assert {"studentFulltime", "blind", "benefitsMedicaid", "livingRentalOnLease"} <= set(person)
    assert "HeadOfHousehold" in person["householdMemberType"]["enum"]
    income = person["incomes"]["items"]
    assert set(income["properties"]) == set(screening.MONEY_ITEM_FIELDS)
    assert income["required"] == ["amount", "frequency", "type"]
    assert income["additionalProperties"] is False
    assert "Wages" in income["properties"]["type"]["enum"]
    assert "Monthly" in income["properties"]["frequency"]["enum"]
    assert "Medical" in person["expenses"]["items"]["properties"]["type"]["enum"]
    assert schema["properties"]["interested_programs"]["items"]["pattern"] == "^[A-Z0-9]+$"


def test_get_tools_gates_forms_on_flag(monkeypatch):
    monkeypatch.setattr(config, "screening_creds", lambda: ("base", "", ""))
    monkeypatch.delenv("HEYNYC_FORMS", raising=False)
    assert "prepare_snap_application" not in {t.name for t in btools.get_tools()}
    monkeypatch.setenv("HEYNYC_FORMS", "true")
    assert "prepare_snap_application" in {t.name for t in btools.get_tools()}


# --- prepare_snap_application (Task 5) — the paths that don't need reportlab -------------

async def test_prepare_application_reviews_before_filling(tmp_path):
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]), http=None,
                      output_dir=tmp_path)
    out = await btools._prepare_application_handler({"slots": {       # confirmed omitted → false
        "legal_name": "Ana Diaz", "residence_street": "1 Main St",
        "residence_city": "Bronx", "residence_zip": "10453"}}, ctx)
    assert out.startswith("REVIEW")
    assert "penalty of perjury" in out               # the attestation is shown BEFORE any PDF
    assert not list(tmp_path.glob("*.pdf"))          # nothing is produced until confirmed=true


async def test_prepare_application_asks_when_required_missing(tmp_path):
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]), http=None,
                      output_dir=tmp_path)
    out = await btools._prepare_application_handler({"slots": {"legal_name": "Ana"}}, ctx)
    assert out.startswith("NEED_MORE") and "Home street address" in out
    assert not list(tmp_path.glob("*.pdf"))          # never fabricates the missing fields


async def test_prepare_application_degrades_on_form_drift(tmp_path, monkeypatch):
    from heynyc.modules.benefits import application as appmod
    monkeypatch.setattr(appmod, "verify_template_integrity", lambda *a, **k: False)
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]), http=None,
                      output_dir=tmp_path)
    out = await btools._prepare_application_handler({"slots": {
        "legal_name": "Ana Diaz", "residence_street": "1 Main St",
        "residence_city": "Bronx", "residence_zip": "10453"}, "confirmed": True}, ctx)
    assert out.startswith("CANNOT_FILL") and "otda.ny.gov" in out    # degrade, never fill-wrong
    assert not list(tmp_path.glob("*.pdf"))


async def test_prepare_application_uses_persistent_draft_not_llm_memory(tmp_path):
    # The point of the draft store: turn 2 the model passes ONLY the new fields (it "forgot" the
    # name from turn 1), but the persisted structured draft retains it → all required present.
    from heynyc.core.drafts import DraftStore
    drafts = DraftStore(tmp_path).for_user("ukey")
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]), http=None,
                      output_dir=tmp_path, drafts=drafts)
    await btools._prepare_application_handler({"slots": {"legal_name": "Ana Diaz"}}, ctx)  # turn 1
    out = await btools._prepare_application_handler({"slots": {                            # turn 2
        "residence_street": "1 Main St", "residence_city": "Bronx", "residence_zip": "10453"}}, ctx)
    assert out.startswith("REVIEW")                         # required complete via the merged draft
    assert "Ana Diaz" in out                                # turn-1 name survived structurally
    assert drafts.load("snap")["legal_name"] == "Ana Diaz"  # persisted, not reconstructed


async def test_prepare_application_confirmation_cannot_change_reviewed_fields(tmp_path):
    from heynyc.core.drafts import DraftStore

    drafts = DraftStore(tmp_path / "drafts").for_user("ukey")
    ctx = ToolContext(
        citations=CitationRegistry(), registry=Registry([]), http=None,
        output_dir=tmp_path, drafts=drafts,
    )
    reviewed = {
        "legal_name": "Ana Diaz",
        "residence_street": "1 Main St",
        "residence_city": "Bronx",
        "residence_zip": "10453",
    }
    await btools._prepare_application_handler({"slots": reviewed, "confirmed": False}, ctx)

    out = await btools._prepare_application_handler(
        {"slots": {"legal_name": "Changed Name"}, "confirmed": True}, ctx
    )

    assert out.startswith("NEED_REVIEW")
    assert drafts.load("snap")["legal_name"] == "Ana Diaz"
    assert not list(tmp_path.glob("*.pdf"))


async def test_prepare_application_confirmed_writes_a_pdf(tmp_path, caplog):
    import logging
    caplog.set_level(logging.DEBUG)
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]), http=None,
                      output_dir=tmp_path)
    out = await btools._prepare_application_handler({"slots": {
        "legal_name": "Ana Diaz", "residence_street": "1 Main St",
        "residence_city": "Bronx", "residence_zip": "10453",
        "ssn": "078-05-1120"}, "confirmed": True}, ctx)
    pdfs = list(tmp_path.glob("*.pdf"))
    assert len(pdfs) == 1 and pdfs[0].read_bytes()[:4] == b"%PDF"
    assert "attached" in out.lower()
    # PII is never logged (it goes onto the local PDF only)
    assert "078-05-1120" not in caplog.text and "Ana Diaz" not in caplog.text


async def test_synthetic_screen_review_pdf_workflow_stops_before_submission(
    tmp_path, monkeypatch
):
    import json

    from heynyc.core.agent import Agent
    from heynyc.core.drafts import DraftStore

    monkeypatch.setattr(
        config,
        "screening_creds",
        lambda: ("https://sandbox.screeningapi.cityofnewyork.us", "u", "p"),
    )

    async def token(*_args):
        return "tok"

    async def screen(*_args):
        return {
            "type": "SUCCESS",
            "eligiblePrograms": [{"code": "S2R007", "name": "SNAP"}],
        }

    async def catalog(*_args, **_kwargs):
        return [{
            ":id": "row-snap",
            "program_code": "S2R007",
            "program_name": "SNAP",
            "program_category": "Food",
            "url_of_online_application": "https://access.nyc.gov/snap",
            "updated_at": "2026-03-01",
        }]

    monkeypatch.setattr(screening, "get_token", token)
    monkeypatch.setattr(screening, "screen", screen)
    monkeypatch.setattr(btools, "query_dataset", catalog)

    profile = {
        "household": {"livingRenting": True},
        "persons": [{"age": 32, "householdMemberType": "HeadOfHousehold"}],
    }
    slots = {
        "legal_name": "Ana Diaz",
        "residence_street": "1 Main St",
        "residence_city": "Bronx",
        "residence_zip": "10453",
    }

    def call(name, args, call_id):
        return {
            "id": call_id,
            "function": {"name": name, "arguments": json.dumps(args)},
        }

    responses = [
        {"role": "assistant", "content": None, "tool_calls": [call("screen_eligibility", profile, "s1")]},
        {"role": "assistant", "content": None, "tool_calls": [call(
            "prepare_snap_application", {"slots": slots, "confirmed": False}, "p1"
        )]},
        {"role": "assistant", "content": "Please review the fields above.", "tool_calls": None},
        {"role": "assistant", "content": None, "tool_calls": [call(
            "prepare_snap_application", {"slots": {}, "confirmed": True}, "p2"
        )]},
        {"role": "assistant", "content": "Your draft is ready for you to review and submit.", "tool_calls": None},
    ]

    async def stream_fn(_messages, _schemas):
        message = responses.pop(0)
        yield {"type": "message", "message": message}

    approvals = []

    async def approve(name, args):
        approvals.append((name, bool(args.get("confirmed"))))
        return True

    tools = {
        "screen_eligibility": btools.screen_eligibility_tool(),
        "prepare_snap_application": btools.prepare_application_tool(),
    }
    assert not any("submit" in name or tool.destructive for name, tool in tools.items())
    agent = Agent(Registry([]), tools=tools, stream_fn=stream_fn, approver=approve)
    convo = agent.conversation()
    drafts = DraftStore(tmp_path / "drafts").for_user("synthetic-user")
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()

    review = await convo.send(
        "Use this synthetic profile to screen me and prepare a SNAP draft.",
        output_dir=artifacts,
        drafts=drafts,
    )
    ready = await convo.send(
        "Yes, the reviewed fields are correct.", output_dir=artifacts, drafts=drafts
    )

    pdfs = list(artifacts.glob("*.pdf"))
    assert review.tool_calls_made == ["screen_eligibility", "prepare_snap_application"]
    assert ready.tool_calls_made == ["prepare_snap_application"]
    assert approvals == [("prepare_snap_application", False), ("prepare_snap_application", True)]
    assert len(pdfs) == 1 and pdfs[0].read_bytes()[:4] == b"%PDF"
    assert drafts.load("snap") == {}
    assert "submit" in ready.text.lower() and "submitted" not in ready.text.lower()


def test_application_eval_cases_present_and_flagged():
    from heynyc.eval.cases import load_cases
    cases = [c for c in load_cases(Registry.discover(config.MODULES_DIR, config.BASE_ALLOWLIST))
             if c.module == "benefits"]
    ids = {c.id for c in cases}
    assert {"benefits_apply_no_fabrication", "benefits_apply_confirm_not_submit",
            "benefits_apply_no_coaching"} <= ids
    # the "coach me to qualify" case is harm-tagged → auto safety_critical
    coaching = next(c for c in cases if c.id == "benefits_apply_no_coaching")
    assert coaching.safety_critical
    assert coaching.invariants.get("forbid_compliance") is True
