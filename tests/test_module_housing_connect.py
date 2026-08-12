"""Offline tests for the housing_connect module's `find_housing_connect_lotteries` finder.

Grounded in one NYC Open Data (Socrata) dataset, Advertised Lotteries on Housing
Connect by Lottery (vy5i-a666), filtered to the currently-open slice
(`lottery_status='Active' AND lottery_end_date >= <today>`). Every HTTP call is
mocked/injected: `_today` is monkeypatched to a fixed date (so "open now" is
deterministic) and the Socrata query is served by a MockTransport routed by the
dataset's 4x4 id. No live Socrata call, no model.

Covers: the grounded happy path (borough names, unit count, unit mix, AMI income
bands, set-aside preferences, deadline, one DATA citation per lottery addressed
to its lottery_id), that the query is filtered server-side to Active + future
deadline, the borough filter, graceful abstention when nothing is open (still
hands off the portal deep-link, never fabricates a listing), a reachability
failure that points to the portal without inventing data, and that the shipped
module loads with its tool + eval cases.
"""
from __future__ import annotations

import httpx

from heynyc.core import config
from heynyc.core.citations import CitationRegistry
from heynyc.core.registry import Registry
from heynyc.core.tools.base import ToolContext
from heynyc.modules.housing_connect import tools as hc
from heynyc.modules.housing_connect.tools import DATASET_ID, PORTAL, get_tools

# Canned rows mirror the real vy5i-a666 schema (values are strings, borough is a
# 2-letter code, AMI fields are per-band unit counts, set-aside fields are percents,
# Socrata omits null columns). Verified live 2026-07-11.
_ROW_SMALL = {
    "lottery_id": "7541", "lottery_name": "1004 Summit Avenue Apartments",
    "lottery_status": "Active", "development_type": "Rental",
    "lottery_start_date": "2026-06-18T00:00:00.000",
    "lottery_end_date": "2026-07-13T00:00:00.000",
    "building_count": "1", "unit_count": "3",
    "unit_distribution_studio": "2", "unit_distribution_1bed": "1",
    "applied_income_ami_low": "3", "borough": "BX", "postcode": "10452",
    ":updated_at": "2026-07-06T12:00:00.000",
}
_ROW_LARGE = {
    "lottery_id": "7500", "lottery_name": "RIALTO WEST",
    "lottery_status": "Active", "development_type": "Rental",
    "lottery_start_date": "2026-06-20T00:00:00.000",
    "lottery_end_date": "2026-07-20T00:00:00.000",
    "building_count": "1", "unit_count": "133",
    "unit_distribution_studio": "32", "unit_distribution_1bed": "36",
    "unit_distribution_2bed": "53", "unit_distribution_3bed": "12",
    "applied_income_ami_very_low": "39", "applied_income_ami_low": "31",
    "applied_income_ami_moderate": "47", "applied_income_ami_middle": "16",
    "lottery_mobility_percent": "5", "lottery_vision_hearing_percent": "2",
    "lottery_community_board_percent": "20", "lottery_municipal_employee_percent": "10",
    "borough": "MN", "postcode": "10039", ":updated_at": "2026-07-06T12:00:00.000",
}


def _client(rows, *, status=200, captured=None) -> httpx.AsyncClient:
    """MockTransport serving the vy5i-a666 query by its 4x4 id, optionally recording
    the request URL so a test can assert the server-side filter that was applied."""
    def handler(request: httpx.Request) -> httpx.Response:
        if captured is not None:
            captured.append(request.url)
        if DATASET_ID in request.url.path:
            if status != 200:
                return httpx.Response(status, text="upstream error")
            return httpx.Response(200, json=rows)
        return httpx.Response(404)
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def _run(monkeypatch, rows, *, today="2026-07-11", borough=None, status=200, captured=None):
    monkeypatch.setattr(hc, "_today", lambda: today)
    citations = CitationRegistry()
    client = _client(rows, status=status, captured=captured)
    ctx = ToolContext(citations=citations, registry=Registry([]), http=client)
    args = {} if borough is None else {"borough": borough}
    out = await get_tools()[0].handler(args, ctx)
    await client.aclose()
    return out, citations


# --- 1. happy path: grounded listing + a DATA citation per lottery ----------

async def test_finder_lists_open_lotteries_grounded_and_cited(monkeypatch):
    captured: list = []
    out, citations = await _run(monkeypatch, [_ROW_SMALL, _ROW_LARGE], captured=captured)

    # both lotteries surfaced with mapped borough names, unit counts, deadlines
    assert "1004 Summit Avenue Apartments" in out
    assert "RIALTO WEST" in out
    assert "Bronx" in out and "Manhattan" in out
    assert "3 unit" in out and "133 unit" in out
    assert "2026-07-13" in out and "2026-07-20" in out

    # unit mix + AMI income bands + set-asides come from the row, never invented
    assert "2 studio" in out and "1 1BR" in out
    assert "53 2BR" in out
    assert "very low" in out and "moderate" in out
    assert "mobility" in out.lower()          # a set-aside preference surfaced

    # one DATA citation per lottery, each addressed to its own lottery_id
    mapping = citations.mapping()
    assert len(mapping) == 2
    assert "{cite:S1}" in out and "{cite:S2}" in out
    for c in mapping.values():
        assert c["kind"] == "DATA"
        assert DATASET_ID in c["url"]
    urls = " ".join(c["url"] for c in mapping.values())
    assert "7541" in urls and "7500" in urls

    # provenance grounded in the actual rows; as-of is the row's refresh date, not fetch time
    s1 = mapping["S1"]
    assert s1["provenance"]["record_id"] == "7541"
    assert s1["provenance"]["snapshot"]["unit_count"] == 3
    assert s1["valid_as_of"] == "2026-07-06"

    # the query is filtered server-side to Active + future deadline
    where = captured[0].params["$where"]
    assert "lottery_status='Active'" in where
    assert "2026-07-11" in where

    # deep-link handoff + honest caveats + no auto-submit
    assert PORTAL in out
    low = out.lower()
    assert "as of" in low                      # reporting-feed, not real-time
    assert "can't submit" in low or "cannot submit" in low


# --- 2. abstain gracefully when nothing is open (still hand off, never fake) -

async def test_finder_abstains_when_nothing_open_but_still_hands_off(monkeypatch):
    out, citations = await _run(monkeypatch, [])
    low = out.lower()
    assert "don't see any" in low or "no affordable-housing lotteries" in low
    assert PORTAL in out
    # no fabricated listing details
    assert "deadline to apply" not in low
    # still grounded: a DATA citation to the (empty) filtered query
    mapping = citations.mapping()
    assert len(mapping) == 1
    assert mapping["S1"]["kind"] == "DATA"
    assert DATASET_ID in mapping["S1"]["url"]
    assert "{cite:S1}" in out


# --- 3. borough filter narrows the server-side query ------------------------

async def test_finder_borough_filter_narrows_query(monkeypatch):
    captured: list = []
    out, _ = await _run(monkeypatch, [_ROW_SMALL], borough="Bronx", captured=captured)
    where = captured[0].params["$where"]
    assert "borough='BX'" in where
    assert "Bronx" in out


# --- 4. reachability failure: point to the portal, fabricate nothing --------

async def test_finder_http_error_points_to_portal_without_fabricating(monkeypatch):
    out, citations = await _run(monkeypatch, None, status=500)
    assert PORTAL in out
    assert len(citations) == 0
    low = out.lower()
    assert "couldn't reach" in low or "could not reach" in low
    assert "deadline to apply" not in low


# --- 5. the shipped module stays valid --------------------------------------

def test_housing_connect_module_loads_with_tool_and_eval():
    registry = Registry.discover(config.MODULES_DIR)
    module = next((m for m in registry.modules if m.name == "housing_connect"), None)
    assert module is not None
    tool_names = {t.name for t in registry.load_module_tools()}
    assert "find_housing_connect_lotteries" in tool_names

    from heynyc.eval.cases import load_cases
    cases = [c for c in load_cases(registry) if c.module == "housing_connect"]
    assert cases, "housing_connect should ship eval cases"
    assert any(c.abstain for c in cases)
