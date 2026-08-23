from __future__ import annotations

import httpx

from heynyc.core import config
from heynyc.core.citations import CitationRegistry
from heynyc.core.registry import Registry
from heynyc.core.tools.base import ToolContext
from heynyc.core.tools.geo import GeoPoint
from heynyc.modules.libraries import tools as library_tools


def test_library_systemwide_service_search_waits_for_a_location() -> None:
    module = next(
        module for module in Registry.discover(config.MODULES_DIR).modules
        if module.name == "libraries"
    )
    prompt = " ".join(module.prompt.split())

    assert "site:queenslibrary.org SERVICE" in prompt
    assert "Do not call search_tools" in prompt
    assert "stop and ask for a branch or neighborhood" in prompt
    assert "branch or neighborhood" in prompt
    assert "straight-line" in prompt
    assert "travel distance" in prompt
    assert "https://www.bklynlibrary.org/use-the-library/print" in prompt
    assert "do not search for the printing page" in prompt.lower()
    assert "report the nearest returned branches in order" in prompt.lower()
    assert "do not skip a nearer returned branch" in prompt.lower()
    assert "do not fetch individual branch pages for printing" in prompt.lower()
    assert "any neighborhood library" in prompt.lower()
    assert "latest returned closing time" in prompt.lower()
    assert "no later returned branch" in prompt.lower()


async def test_find_bpl_branches_ranks_the_official_feed_by_resident_location(monkeypatch):
    rows = [
        {
            "title": "Central Library",
            "address": "10 Grand Army Plaza Brooklyn, NY 11238",
            "position": "40.67250, -73.96810",
            "phone": "718.230.2100",
            "path": "https://www.bklynlibrary.org/locations/central",
            "hours": "<i></i>Today's Hours:  9 am - 8 pm",
            "tags": "computer reservations;open late",
            "branchid": "1",
        },
        {
            "title": "Sunset Park Library",
            "address": "5108 Fourth Avenue Brooklyn, NY 11220",
            "position": "40.6459117, -74.0136256",
            "phone": "718.230.2255",
            "path": "https://www.bklynlibrary.org/locations/sunset-park",
            "hours": "<i></i>Today's Hours:  10 am - 8 pm",
            "closingmsg": "Closed today because of a building problem.",
            "tags": "computer reservations;open late;notary",
            "branchid": "55",
        },
    ]

    async def fake_geocode(text, *, client=None):
        assert text == "Sunset Park"
        return GeoPoint(
            lat=40.6460,
            lon=-74.0136,
            label="Sunset Park, Brooklyn",
            confidence=1.0,
            match_type="nta",
        )

    def respond(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://www.bklynlibrary.org/api/locations/v1/map"
        return httpx.Response(200, json=rows)

    monkeypatch.setattr("heynyc.core.tools.geo.geocode", fake_geocode)
    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    citations = CitationRegistry()
    ctx = ToolContext(citations=citations, registry=Registry([]), http=client)

    output = await library_tools.get_tools()[0].handler(
        {"near": "Sunset Park", "max_results": 2},
        ctx,
    )
    await client.aclose()

    assert output.index("Sunset Park Library") < output.index("Central Library")
    assert "today's listed hours: 10 am - 8 pm" in output
    assert "listed hours unconfirmed because the feed also says: Closed today because of a building problem." in output
    assert "718.230.2255" in output
    assert "services:" not in output
    assert "; closed;" not in output
    assert "https://www.bklynlibrary.org/locations/sunset-park" in output
    assert "google.com/maps" not in output
    citation = citations.mapping()["S1"]
    assert citation["kind"] == "DATA"
    assert citation["provenance"]["snapshot"] == rows[1]
    assert "Closed today because of a building problem." in citation["snippet"]
    assert citation["provenance"]["derivation"]["origin"] == [40.646, -74.0136]
    assert citation["provenance"]["derivation"]["point"] == [40.6459117, -74.0136256]
