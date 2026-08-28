import httpx

from heynyc.core import config
from heynyc.core.citations import CitationRegistry
from heynyc.core.grounding import check_grounding
from heynyc.core.registry import Registry
from heynyc.core.tools import geo
from heynyc.core.tools.base import ToolContext
from heynyc.core.tools.geo import GeoPoint, travel_distance


async def test_public_osrm_never_labels_driving_data_as_walking(monkeypatch):
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={"routes": [{"distance": 1_000, "duration": 600}]},
        )

    monkeypatch.setattr(
        "heynyc.core.tools.geo.OSRM_BASE",
        "https://router.project-osrm.org",
    )
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    result = await travel_distance(
        GeoPoint(40.75, -73.99),
        GeoPoint(40.76, -73.98),
        mode="walking",
        client=client,
    )
    await client.aclose()

    assert calls == 0
    assert result["source"] == "haversine"
    assert result["minutes"] is None
    assert result["mode"] == "straight-line"


async def test_custom_osrm_may_serve_a_walking_profile(monkeypatch):
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        return httpx.Response(
            200,
            json={"routes": [{"distance": 1_000, "duration": 600}]},
        )

    monkeypatch.setattr("heynyc.core.tools.geo.OSRM_BASE", "https://routing.example")
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    result = await travel_distance(
        GeoPoint(40.75, -73.99),
        GeoPoint(40.76, -73.98),
        mode="walking",
        client=client,
    )
    await client.aclose()

    assert paths == [
        "/route/v1/foot/-73.99,40.75;-73.98,40.76"
    ]
    assert result["source"] == "osrm"
    assert result["mode"] == "walking"
    assert result["minutes"] == 10.0


async def test_f182_distance_handler_returns_grounded_directions_link(monkeypatch):
    async def fake_geocode(text, **_kwargs):
        return (
            GeoPoint(40.75780, -73.82890, "Flushing Library")
            if text == "Flushing Library"
            else GeoPoint(40.78328, -73.83521, "Petco College Point")
        )

    async def fake_distance(*_args, **_kwargs):
        return {"meters": 2_880, "minutes": 8.0, "mode": "driving", "source": "osrm"}

    monkeypatch.setattr(geo, "geocode", fake_geocode)
    monkeypatch.setattr(geo, "travel_distance", fake_distance)
    ctx = ToolContext(
        citations=CitationRegistry(),
        registry=Registry.discover(config.MODULES_DIR),
    )

    output = await geo.geo_tools()[2].handler(
        {
            "origin": "Flushing Library",
            "destination": "Petco College Point",
            "mode": "transit",
        },
        ctx,
    )

    assert (
        "https://www.google.com/maps/dir/?api=1&origin=40.75780,-73.82890"
        "&destination=40.78328,-73.83521&travelmode=transit" in output
    )
    assert "{cite:S1}" in output
    citations = ctx.citations.mapping()
    assert citations["S1"]["provenance"]["derivation"]["point"] == [40.78328, -73.83521]
    assert check_grounding(output, citations).blocking is False
    changed = output.replace("destination=40.78328,-73.83521", "destination=40.70000,-73.90000")
    assert check_grounding(changed, citations).blocking is True
    changed = output.replace("origin=40.75780,-73.82890", "origin=40.70000,-73.90000")
    assert check_grounding(changed, citations).blocking is True
    assert "travelmode=bicycling" in geo.directions_link(
        GeoPoint(40.75780, -73.82890),
        GeoPoint(40.78328, -73.83521),
        mode="cycling",
    )
    assert "transit" in geo.geo_tools()[2]._input_schema()["properties"]["mode"]["enum"]
