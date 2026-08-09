import httpx

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
