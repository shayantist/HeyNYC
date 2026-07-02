from __future__ import annotations

import httpx
import pytest

from heynyc.core.citations import CitationRegistry
from heynyc.core.manifest import DatasetBinding, ServiceModule
from heynyc.core.registry import Registry
from heynyc.core.tools.base import ToolContext
from heynyc.core.tools.geo import (
    GeoPoint,
    _distance_handler,
    _geosearch_geocode,
    _in_nyc,
    _looks_like_intersection,
    _nearest_handler,
    _zip_centroid,
    geocode,
    haversine_m,
    miles,
    travel_distance,
)


def test_looks_like_intersection():
    assert _looks_like_intersection("116 St and Broadway")
    assert _looks_like_intersection("W 116 St & Broadway")
    assert _looks_like_intersection("5 Ave / 42 St")
    # No street numbers → treat as a place name, not an intersection
    assert not _looks_like_intersection("Union Square")
    assert not _looks_like_intersection("Fordham Road, the Bronx")

# --- BUG-1: bare NYC ZIP must resolve via the bundled ZCTA centroids, never
# GeoSearch (which has no postalcode layer and misparses "10453" as a house
# number → a confidently-wrong Queens street). ------------------------------

def _wrong_queens_client() -> httpx.AsyncClient:
    """A MockTransport that (wrongly) resolves anything to the BUG-1 Queens
    house-number match. If the ZIP guard works, this is never called."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"features": [{
            "geometry": {"coordinates": [-73.8310, 40.6816]},  # Richmond Hill, Queens
            "properties": {"label": "10453 109 Street, Richmond Hill, Queens",
                           "housenumber": "10453", "confidence": 1.0},
        }]})
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_zip_centroid_lookup():
    lat, lon = _zip_centroid("10453")
    assert round(lat, 4) == 40.8525
    assert round(lon, 4) == -73.9133
    assert _zip_centroid("00000") is None


def test_in_nyc_bbox():
    assert _in_nyc(40.8525, -73.9133)      # the Bronx
    assert not _in_nyc(34.1005, -118.4146)  # Beverly Hills


async def test_bare_zip_bypasses_geosearch_to_zcta_centroid():
    # GeoSearch would return the WRONG Queens venue; the ZIP guard must beat it.
    client = _wrong_queens_client()
    point = await geocode("10453", client=client, forgiving=_fake_forgiving(None))
    await client.aclose()
    assert point is not None
    assert point.match_type == "zcta"
    assert round(point.lat, 2) == 40.85        # Bronx centroid
    assert abs(point.lat - 40.68) > 0.1        # NOT the Queens misparse


async def test_zip_with_borough_word_resolves_same_centroid():
    for query in ("10453 Bronx", "Bronx 10453"):
        client = _wrong_queens_client()
        point = await geocode(query, client=client, forgiving=_fake_forgiving(None))
        await client.aclose()
        assert point.match_type == "zcta"
        assert round(point.lat, 4) == 40.8525
        assert round(point.lon, 4) == -73.9133


async def test_non_nyc_bare_zip_returns_none():
    client = _wrong_queens_client()
    assert await geocode("90210", client=client, forgiving=_fake_forgiving(None)) is None
    await client.aclose()


async def test_real_address_still_uses_geosearch():
    # No 5-digit token → falls through to GeoSearch, unaffected by the ZIP guard.
    def handler(request: httpx.Request) -> httpx.Response:
        return _geosearch_response(40.8536, -73.9010, "1910 Monterey Ave, Bronx")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    point = await geocode("1910 Monterey Ave Bronx", client=client, forgiving=_fake_forgiving(None))
    await client.aclose()
    assert point.match_type == "geosearch"
    assert "Monterey" in point.label


async def test_geosearch_rejects_zip_misparsed_as_housenumber():
    # Belt-and-suspenders: a GeoSearch result whose housenumber is a 5-digit ZIP
    # present in the input is rejected outright.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"features": [{
            "geometry": {"coordinates": [-73.8310, 40.6816]},
            "properties": {"label": "10453 109 Street, Richmond Hill, Queens",
                           "housenumber": "10453", "confidence": 1.0},
        }]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    assert await _geosearch_geocode("10453", client) is None
    await client.aclose()


FIELD_MAP = {"name": "propertyname", "lat": "y", "lon": "x", "status": "status", "borough": "borough"}


def test_haversine_known_distance():
    # Columbia University ↔ Times Square ≈ 5.8 km
    d = haversine_m(40.8075, -73.9626, 40.7580, -73.9855)
    assert 5500 < d < 6200


def test_miles():
    assert round(miles(1609.344), 3) == 1.0


def _geosearch_response(lat: float, lon: float, label: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={"features": [{"geometry": {"coordinates": [lon, lat]}, "properties": {"label": label}}]},
    )


async def test_geocode_parses_lon_lat_order():
    def handler(request):
        return _geosearch_response(40.8075, -73.9626, "Columbia University, Manhattan")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    point = await geocode("Columbia University", client=client)  # non-intersection → GeoSearch
    await client.aclose()
    assert point is not None
    assert round(point.lat, 4) == 40.8075
    assert round(point.lon, 4) == -73.9626
    assert "Columbia" in point.label


async def test_geocode_no_match_returns_none():
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"features": []})))
    # GeoSearch empty AND forgiving provider empty → None (inject to stay offline).
    assert await geocode("nowhere", client=client, forgiving=_fake_forgiving(None)) is None
    await client.aclose()


def _fake_forgiving(point):
    """Build an injectable forgiving-geocoder coroutine returning a fixed GeoPoint."""
    async def fn(text):
        return point
    return fn


async def test_high_confidence_match_not_flagged():
    # A high-confidence NYC-biased match is trusted regardless of phrasing.
    forg = _fake_forgiving(GeoPoint(40.8073, -73.9626, "Broadway & W 116 St, Manhattan",
                                    confidence=0.95, match_type="mapbox"))
    point = await geocode("116 St and Broadway", forgiving=forg)
    assert point.match_type == "mapbox"
    assert point.low_confidence is False


async def test_low_confidence_match_flagged():
    # A genuinely low provider confidence → flagged so the agent clarifies.
    forg = _fake_forgiving(GeoPoint(40.749, -73.988, "Broadway, New York", confidence=0.5, match_type="mapbox"))
    # Intersection routes to the forgiving provider first, so the injected stub is used (offline).
    point = await geocode("broadway and 116", forgiving=forg)
    assert point.low_confidence is True


async def test_forgiving_fallback_when_geosearch_empty():
    forg = _fake_forgiving(GeoPoint(40.75, -73.99, "Apollo Theater, Harlem", match_type="nominatim"))

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"features": []})  # GeoSearch whiffs on the POI

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    point = await geocode("the Apollo Theater", client=client, forgiving=forg)
    await client.aclose()
    assert point.match_type == "nominatim"
    assert "Apollo" in point.label


async def test_low_confidence_origin_makes_nearest_clarify(monkeypatch):
    # A low-confidence geocode must produce a clarify request, not (wrong) results.
    async def fake_geocode(text, **kwargs):
        return GeoPoint(40.7, -73.9, "ambiguous", low_confidence=True)

    monkeypatch.setattr("heynyc.core.tools.geo.geocode", fake_geocode)
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=[])))
    ctx = ToolContext(citations=CitationRegistry(), registry=_registry_with_cooling(), http=client)
    out = await _nearest_handler({"category": "cooling_center", "near": "Broadway and 116th"}, ctx)
    await client.aclose()
    assert "which borough" in out.lower()
    assert "- " not in out  # no location list emitted


def _registry_with_cooling() -> Registry:
    module = ServiceModule(
        name="cooling_centers",
        category="health",
        datasets=[DatasetBinding(id="h2bn-gu9k", category="cooling_center", field_map=FIELD_MAP, where="status='Activated'")],
    )
    return Registry([module])


async def test_nearest_handler_ranks_and_cites():
    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if "geosearch" in host:
            return _geosearch_response(40.7500, -73.9900, "Origin, Manhattan")
        if "cityofnewyork" in host:
            return httpx.Response(
                200,
                json=[
                    {"propertyname": "Far Site", "y": "40.8000", "x": "-73.9600", "status": "Activated", "borough": "Manhattan"},
                    {"propertyname": "Close Site", "y": "40.7510", "x": "-73.9910", "status": "Activated", "borough": "Manhattan"},
                    {"propertyname": "No Coords", "status": "Activated"},
                ],
            )
        return httpx.Response(404)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ctx = ToolContext(citations=CitationRegistry(), registry=_registry_with_cooling(), http=client)
    out = await _nearest_handler({"category": "cooling_center", "near": "origin", "k": 2}, ctx)
    await client.aclose()

    # Closest first, bad-coords row skipped
    lines = [l for l in out.splitlines() if l.startswith("- ")]
    assert len(lines) == 2
    assert "Close Site" in lines[0]
    assert "Far Site" in lines[1]
    # Citations registered as DATA, inline cite ids present
    assert ctx.citations.mapping()["S1"]["kind"] == "DATA"
    assert "{cite:S1}" in out
    # Transparency: the resolved origin label is surfaced
    assert "Resolved 'origin'" in out
    # A deterministic Google Maps link is offered per place (navigation handoff)
    assert "google.com/maps" in out


def _registry_with_arcgis_cooling() -> Registry:
    """The shipped (BUG-2-fixed) shape: cooling bound to the ArcGIS cooling-center finder."""
    module = ServiceModule(
        name="cooling_centers",
        category="health",
        datasets=[DatasetBinding(
            source="arcgis",
            url="https://services6.arcgis.com/yG5s3afENB5iO9fj/arcgis/rest/services/CoolingCenters_PROD_view/FeatureServer/0",
            category="cooling_center",
            record_id_field="NYCEM_ID",
            title="NYC Emergency Management - Cooling Centers",
            where="Finder_status='OPEN'",
            field_map={"name": "Facility_name", "lat": "lat", "lon": "lon", "address": "Address",
                       "borough": "Borough_name", "status": "Finder_status", "phone": "Phone"},
        )],
    )
    return Registry([module])


def _cooling_feature(lon, lat, **props) -> dict:
    return {"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat]}, "properties": props}


async def test_nearest_handler_arcgis_ranks_surfaces_phone_and_cites():
    # The ArcGIS declarative path: mock the Feature Service /query (GeoJSON) + the geocoder,
    # then assert distance ranking, phone surfacing, and an NYCEM_ID-addressed DATA citation.
    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if "geosearch" in host:
            return _geosearch_response(40.7500, -73.9900, "Origin, Manhattan")
        if "arcgis.com" in host:
            return httpx.Response(200, json={"type": "FeatureCollection", "features": [
                _cooling_feature(-73.9600, 40.8000, Facility_name="Far Library", Address="1 Far St",
                                 Borough_name="Manhattan", Phone="212-555-0001",
                                 Finder_status="OPEN", Facility_type="Library", NYCEM_ID="CC1001"),
                _cooling_feature(-73.9910, 40.7510, Facility_name="Close Senior Center", Address="2 Near Ave",
                                 Borough_name="Manhattan", Phone="212-555-0002",
                                 Finder_status="OPEN", Facility_type="Older Adult Center", NYCEM_ID="CC1043"),
                _cooling_feature(None, None, Facility_name="No Coords", Finder_status="OPEN", NYCEM_ID="CC9999"),
            ]})
        return httpx.Response(404)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ctx = ToolContext(citations=CitationRegistry(), registry=_registry_with_arcgis_cooling(), http=client)
    out = await _nearest_handler({"category": "cooling_center", "near": "origin", "k": 2}, ctx)
    await client.aclose()

    site_lines = [l for l in out.splitlines() if l.startswith("- ")]
    assert len(site_lines) == 2                       # bad-coords row dropped
    assert "Close Senior Center" in site_lines[0]     # nearest first
    assert "Far Library" in site_lines[1]
    assert "212-555-0002" in site_lines[0]            # phone surfaced from the ArcGIS Phone field
    # nearest site's citation is registered first (S1), row-addressed by NYCEM_ID
    c = ctx.citations.mapping()["S1"]
    assert c["kind"] == "DATA"
    assert "{cite:S1}" in out
    assert "NYCEM_ID" in c["url"]                     # feature_query_url row-address on NYCEM_ID
    assert "CC1043" in c["url"]
    assert c["title"] == "NYC Emergency Management - Cooling Centers"


async def test_nearest_handler_dedupes_repeated_sites():
    def handler(request: httpx.Request) -> httpx.Response:
        if "geosearch" in request.url.host:
            return _geosearch_response(40.7500, -73.9900, "Origin")
        return httpx.Response(
            200,
            json=[
                {"propertyname": "Dup Park", "y": "40.7510", "x": "-73.9910", "status": "Activated", "borough": "Manhattan"},
                {"propertyname": "Dup Park", "y": "40.7510", "x": "-73.9910", "status": "Activated", "borough": "Manhattan"},
                {"propertyname": "Other Park", "y": "40.7600", "x": "-73.9800", "status": "Activated", "borough": "Manhattan"},
            ],
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ctx = ToolContext(citations=CitationRegistry(), registry=_registry_with_cooling(), http=client)
    out = await _nearest_handler({"category": "cooling_center", "near": "origin", "k": 3}, ctx)
    await client.aclose()
    site_lines = [l for l in out.splitlines() if l.startswith("- ")]
    assert len(site_lines) == 2  # dup collapsed
    assert sum("Dup Park" in l for l in site_lines) == 1


async def test_nearest_handler_unknown_category():
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=[])))
    ctx = ToolContext(citations=CitationRegistry(), registry=_registry_with_cooling(), http=client)
    out = await _nearest_handler({"category": "bus_depot", "near": "x"}, ctx)
    await client.aclose()
    assert "No dataset for category 'bus_depot'" in out
    assert "cooling_center" in out


async def test_travel_distance_osrm_then_fallback():
    def ok(request):
        return httpx.Response(200, json={"routes": [{"distance": 1609.344, "duration": 600}]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(ok))
    res = await travel_distance(GeoPoint(40.75, -73.99), GeoPoint(40.76, -73.98), client=client)
    await client.aclose()
    assert round(miles(res["meters"]), 2) == 1.0
    assert res["minutes"] == 10.0
    assert res["source"] == "osrm"

    bad = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    res2 = await travel_distance(GeoPoint(40.75, -73.99), GeoPoint(40.76, -73.98), client=bad)
    await bad.aclose()
    assert res2["source"] == "haversine"
    assert res2["minutes"] is None


async def test_distance_handler_reports_route():
    def handler(request: httpx.Request):
        if "geosearch" in request.url.host:
            return _geosearch_response(40.75, -73.99, "A")
        return httpx.Response(200, json={"routes": [{"distance": 3218.69, "duration": 1200}]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ctx = ToolContext(citations=CitationRegistry(), registry=_registry_with_cooling(), http=client)
    out = await _distance_handler({"origin": "A", "destination": "B"}, ctx)
    await client.aclose()
    assert "2.00 mi" in out
    assert "20.0 min" in out


def test_place_citation_is_row_addressed_and_carries_recomputable_provenance():
    from heynyc.core.citations import content_hash
    from heynyc.core.tools.datasets import Place
    from heynyc.core.tools.geo import _place_citation

    reg = CitationRegistry()

    class _Binding:
        id = "h2bn-gu9k"

    class _Ctx:
        citations = reg

    place = Place(name="Marconi Park", lat=40.74, lon=-73.88, status="Activated",
                  borough="Queens", record_id="row-9", updated_at="2026-06-20",
                  raw={":id": "row-9", "propertyname": "Marconi Park", "status": "Activated"})
    cid = _place_citation(_Ctx(), place, _Binding(), origin_lat=40.75, origin_lon=-73.87, dist_mi=0.68)
    c = reg.mapping()[cid]
    assert c["kind"] == "DATA"
    assert c["url"] == "https://data.cityofnewyork.us/resource/h2bn-gu9k/row-9.json"
    assert c["valid_as_of"] == "2026-06-20"
    prov = c["provenance"]
    assert prov["record_id"] == "row-9"
    assert prov["content_hash"] == content_hash(place.raw)
    assert prov["derivation"] == {"origin": [40.75, -73.87], "point": [40.74, -73.88], "distance_mi": 0.68}
