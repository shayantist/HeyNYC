from __future__ import annotations

import httpx

from heynyc.core.citations import CitationRegistry
from heynyc.core import config
from heynyc.core.registry import Registry
from heynyc.core.tools.base import ToolContext
from heynyc.modules.transit.tools import get_tools


def test_f234_expects_retrieval_without_inferred_station_checks():
    from heynyc.eval.cases import load_cases

    registry = Registry.discover(config.MODULES_DIR)
    case = next(case for case in load_cases(registry) if case.id == "transit_f234_flushing_met_wheelchair")

    assert case.expect_tools == ["web_fetch"]


def test_transit_treats_a_named_landmark_as_a_supplied_endpoint():
    registry = Registry.discover(config.MODULES_DIR)
    transit = next(module for module in registry.modules if module.name == "transit")
    prompt = " ".join(transit.prompt.lower().split())

    assert "named venue or landmark is a supplied endpoint" in prompt
    assert "do not ask for its exact entrance or street address" in prompt
    assert "do not ask for a finer endpoint" in prompt
    assert "endpoint is otherwise too broad" not in prompt


async def test_mta_elevator_status_reports_matching_outages_without_calling_other_tools():
    rows = [
        {
            "station": "61 St-Woodside",
            "trainno": "7/LIRR",
            "equipment": "ES448",
            "equipmenttype": "ES",
            "serving": "61 St and Roosevelt Ave to mezzanine",
            "outagedate": "09/30/2024 12:05:00 PM",
            "estimatedreturntoservice": "08/31/2026 11:59:00 PM",
            "reason": "Capital Replacement",
            "isupcomingoutage": "N",
        },
        {
            "station": "14 St-Union Sq",
            "trainno": "4/5/6/L/N/Q/R/W",
            "equipment": "EL999",
            "equipmenttype": "EL",
            "serving": "mezzanine to platform",
            "outagedate": "08/11/2026 08:00:00 AM",
            "estimatedreturntoservice": "08/11/2026 05:00:00 PM",
            "reason": "Maintenance",
            "isupcomingoutage": "N",
        },
        {
            "station": "86 St",
            "trainno": "Q",
            "equipment": "ES276",
            "equipmenttype": "ES",
            "serving": "83 St and 2nd Ave to lower mezzanine",
            "reason": "Under Investigation",
        },
        {
            "station": "86 St",
            "trainno": "4/5/6",
            "equipment": "EL860",
            "equipmenttype": "EL",
            "serving": "Lexington Ave entrance to mezzanine",
            "reason": "Maintenance",
        },
    ]

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "api-endpoint.mta.info"
        return httpx.Response(200, json=rows)

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    citations = CitationRegistry()
    ctx = ToolContext(citations=citations, registry=Registry([]), http=client)
    tool = get_tools()[0]

    output = await tool.handler(
        {"stations": ["Flushing-Main St", "61 St-Woodside", "86 St-4/5/6"]},
        ctx,
    )
    await client.aclose()

    assert "61 St-Woodside" in output
    assert "ES448" in output
    assert "Capital Replacement" in output
    assert "Flushing-Main St" in output
    assert "not a guarantee" in output
    assert "14 St-Union Sq" not in output
    assert "EL860" in output
    assert "ES276" not in output
    citation = citations.mapping()["S1"]
    assert citation["url"].startswith("https://api-endpoint.mta.info/")
    assert citation["kind"] == "DATA"
    assert citation["provenance"]["snapshot"] == {"outages": rows}
    assert citation["provenance"]["derivation"]["requested_stations"] == [
        "Flushing-Main St",
        "61 St-Woodside",
        "86 St-4/5/6",
    ]
    assert any(
        item["url"] == "https://www.mta.info/elevator-escalator-status"
        for item in citations.mapping().values()
    )


async def test_mta_elevator_status_fails_closed_when_feed_is_unavailable():
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(503))
    )
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]), http=client)

    output = await get_tools()[0].handler({"stations": ["86 St"]}, ctx)
    await client.aclose()

    assert "could not verify" in output.lower()
    assert "mta.info/elevator-escalator-status" in output
    assert "{cite:S1}" in output
    assert ctx.citations.mapping()["S1"]["url"] == (
        "https://www.mta.info/elevator-escalator-status"
    )
