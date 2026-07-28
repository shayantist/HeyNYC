from __future__ import annotations

import httpx

from heynyc.core import config
from heynyc.core.citations import CitationRegistry
from heynyc.core.manifest import DatasetBinding, ServiceModule
from heynyc.core.registry import Registry
from heynyc.core.tools.base import ToolContext
from heynyc.core.tools.geo import (
    GeoPoint,
    _borough_rect,
    _detect_borough,
    _distance_handler,
    _geosearch_geocode,
    _geosearch_params,
    _in_nyc,
    _looks_like_intersection,
    _nearest_handler,
    _point_in_named_borough,
    _resolution_note,
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
    assert _looks_like_intersection("Atlantic Avenue and Flatbush Avenue")
    assert not _looks_like_intersection("Food and Agriculture Building")
    assert not _looks_like_intersection("I need 2 food pantries and 3 cooling centers")
    # No street numbers → treat as a place name, not an intersection
    assert not _looks_like_intersection("Union Square")
    assert not _looks_like_intersection("Fordham Road, the Bronx")


# --- Borough-bias fix: the "125th Street Manhattan → College Point, Queens" bug is a
# QUERY bug. A borough named in the query must produce a borough-aware boundary.rect;
# a plain address with no borough must still get the citywide NYC rect as a floor.
# (Verified live 2026-07-05; see docs/internal/strategy/2026-07-05-geocoder-upgrade.md.) --------

def test_detect_borough_from_full_names():
    # The exact live-verified failing query, plus the doc's stated cases.
    assert _detect_borough("125th Street Manhattan") == "manhattan"
    assert _detect_borough("123 Main St, the Bronx") == "bronx"
    assert _detect_borough("350 5th Ave") is None
    assert _detect_borough("67 Atlantic Avenue, Brooklyn") == "brooklyn"
    assert _detect_borough("29-00 Northern Boulevard, Queens") == "queens"
    assert _detect_borough("457 Victory Boulevard, Staten Island") == "staten island"


def test_detect_borough_from_aliases_and_abbreviations():
    assert _detect_borough("125 W 125 St MN") == "manhattan"
    assert _detect_borough("1299 Grand Concourse BX") == "bronx"
    assert _detect_borough("67 Atlantic Ave BK") == "brooklyn"
    assert _detect_borough("29-00 Northern Blvd QN") == "queens"
    assert _detect_borough("457 Victory Blvd SI") == "staten island"
    assert _detect_borough("457 Victory Blvd, Staten Is") == "staten island"


def test_detect_borough_treats_citywide_words_as_no_borough():
    # "NYC" / "New York" are not a borough → return None so the citywide floor applies.
    assert _detect_borough("350 5th Ave, New York, NY") is None
    assert _detect_borough("City Hall NYC") is None


def test_detect_borough_does_not_false_positive_on_street_names():
    # Abbreviations must match as whole tokens, never as substrings inside a word.
    assert _detect_borough("123 Business Park Road") is None   # 'si' inside 'Business'
    assert _detect_borough("40 Simone Street") is None         # 'si' inside 'Simone'


def test_geosearch_params_attaches_borough_rect_when_borough_named():
    # A detected borough → the borough's hard boundary.rect (min/max lon/lat) is attached.
    rect = _borough_rect("125th Street Manhattan")
    params = _geosearch_params("125th Street Manhattan", rect)
    assert params["text"] == "125th Street Manhattan"  # borough word stays in text (helps scoring)
    assert params["boundary.rect.min_lon"] == -74.0479
    assert params["boundary.rect.max_lon"] == -73.9067
    assert params["boundary.rect.min_lat"] == 40.6829
    assert params["boundary.rect.max_lat"] == 40.8820


def test_geosearch_params_uses_citywide_floor_when_no_borough():
    # No borough named → the citywide NYC rect (config.NYC_BBOX, W,S,E,N) is the floor.
    rect = _borough_rect("350 5th Ave")
    assert rect is None
    params = _geosearch_params("350 5th Ave", rect)
    w, s, e, n = (float(x) for x in config.NYC_BBOX.split(","))
    assert params["boundary.rect.min_lon"] == w
    assert params["boundary.rect.min_lat"] == s
    assert params["boundary.rect.max_lon"] == e
    assert params["boundary.rect.max_lat"] == n

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
        point = await geocode(
            query, client=client, forgiving=_fake_forgiving(None),
            borough_contains=_fake_borough_contains(True),
        )
        await client.aclose()
        assert point.match_type == "zcta"
        assert round(point.lat, 4) == 40.8525
        assert round(point.lon, 4) == -73.9133


async def test_zip_with_contradictory_borough_fails_closed():
    client = _wrong_queens_client()
    point = await geocode(
        "10453 Queens", client=client, forgiving=_fake_forgiving(None),
        borough_contains=_fake_borough_contains(False),
    )
    await client.aclose()

    assert point is None


async def test_non_nyc_bare_zip_returns_none():
    client = _wrong_queens_client()
    assert await geocode("90210", client=client, forgiving=_fake_forgiving(None)) is None
    await client.aclose()


async def test_real_address_still_uses_geosearch():
    # No 5-digit token → falls through to GeoSearch, unaffected by the ZIP guard.
    def handler(request: httpx.Request) -> httpx.Response:
        return _geosearch_response(40.8536, -73.9010, "1910 Monterey Ave, Bronx")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    point = await geocode(
        "1910 Monterey Ave Bronx", client=client, forgiving=_fake_forgiving(None),
        borough_contains=_fake_borough_contains(True),
    )
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


async def test_geosearch_geocode_extracts_bbl_from_pad_addendum():
    # GeoSearch (NYC PAD) carries the building's Borough-Block-Lot under
    # properties.addendum.pad.bbl — the tax-lot key for building-level datasets.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"features": [{
            "geometry": {"coordinates": [-73.9010, 40.8536]},
            "properties": {"label": "1910 Monterey Ave, Bronx",
                           "addendum": {"pad": {"bbl": "2030600032"}}},
        }]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    point = await _geosearch_geocode("1910 Monterey Ave Bronx", client)
    await client.aclose()
    assert point is not None
    assert point.bbl == "2030600032"


async def test_geosearch_geocode_leaves_bbl_empty_without_addendum():
    # No PAD addendum (a non-addressed match) → bbl stays "", never fabricated.
    def handler(request: httpx.Request) -> httpx.Response:
        return _geosearch_response(40.8075, -73.9626, "Columbia University, Manhattan")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    point = await _geosearch_geocode("Columbia University", client)
    await client.aclose()
    assert point is not None
    assert point.bbl == ""


FIELD_MAP = {"name": "propertyname", "lat": "y", "lon": "x", "status": "status", "borough": "borough", "website": "website", "hours": "comments"}


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


async def test_geocode_accepts_grounded_nyc_coordinate_text_without_provider_lookup():
    async def should_not_run(_text):
        raise AssertionError("coordinate text reached the place-name geocoder")

    point = await geocode("40.75953,-73.97859", forgiving=should_not_run)

    assert point is not None
    assert point.match_type == "coordinates"
    assert point.lat == 40.75953
    assert point.lon == -73.97859


async def test_geocode_rejects_coordinate_text_outside_nyc():
    assert await geocode("-72.82204,0", forgiving=_fake_forgiving(None)) is None


async def test_geocode_no_match_returns_none():
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"features": []})))
    # GeoSearch empty AND forgiving provider empty → None (inject to stay offline).
    assert await geocode("nowhere", client=client, forgiving=_fake_forgiving(None)) is None
    await client.aclose()


async def test_geocode_rejects_a_deictic_location_without_device_coordinates():
    async def should_not_run(_text):
        raise AssertionError("deictic location reached the place-name geocoder")

    assert await geocode("near me", forgiving=should_not_run) is None
    assert await geocode("my current location", forgiving=should_not_run) is None


def _fake_forgiving(point):
    """Build an injectable forgiving-geocoder coroutine returning a fixed GeoPoint."""
    async def fn(text):
        return point
    return fn


def _fake_borough_contains(result):
    async def fn(_point, _borough, _client):
        return result
    return fn


async def test_high_confidence_match_not_flagged():
    # A high-confidence NYC-biased match is trusted regardless of phrasing.
    forg = _fake_forgiving(GeoPoint(40.8073, -73.9626, "Broadway & W 116 St, Manhattan",
                                    confidence=0.95, match_type="mapbox"))
    point = await geocode("116 St and Broadway", forgiving=forg)
    assert point.match_type == "mapbox"
    assert point.low_confidence is False


async def test_intersection_direction_mismatch_is_flagged_for_clarification():
    forg = _fake_forgiving(GeoPoint(
        40.7980, -73.9410, "Broadway & E 116th St, Manhattan",
        confidence=0.99, match_type="mapbox",
    ))

    point = await geocode("Broadway and West 116th Street", forgiving=forg)

    assert point is not None
    assert point.low_confidence is True


async def test_intersection_street_mismatch_is_flagged_for_clarification():
    forg = _fake_forgiving(GeoPoint(
        40.8073, -73.9626, "Amsterdam Ave & W 116th St, Manhattan",
        confidence=0.99, match_type="mapbox",
    ))

    point = await geocode("Broadway and West 116th Street", forgiving=forg)

    assert point is not None
    assert point.low_confidence is True


async def test_low_confidence_match_flagged():
    # A genuinely low provider confidence → flagged so the agent clarifies.
    forg = _fake_forgiving(GeoPoint(40.749, -73.988, "Broadway, New York", confidence=0.5, match_type="mapbox"))
    # Intersection routes to the forgiving provider first, so the injected stub is used (offline).
    point = await geocode("broadway and 116", forgiving=forg)
    assert point.low_confidence is True


async def test_named_neighborhood_low_provider_confidence_is_not_flagged():
    # F064: a borough-qualified neighborhood name resolves to a usable centroid. A provider
    # confidence below the address-tuned floor must NOT flag it low_confidence and push the
    # resident for cross streets. (Intersection confidence discipline is untouched.)
    forg = _fake_forgiving(
        GeoPoint(40.7654, -73.8318, "Flushing, Queens, NY", confidence=0.5, match_type="mapbox")
    )
    point = await geocode(
        "Flushing, Queens", forgiving=forg, borough_contains=_fake_borough_contains(True)
    )
    assert point is not None
    assert point.low_confidence is False


async def test_forgiving_fallback_when_geosearch_empty():
    forg = _fake_forgiving(GeoPoint(40.75, -73.99, "Apollo Theater, Harlem", match_type="nominatim"))

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"features": []})  # GeoSearch whiffs on the POI

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    point = await geocode("the Apollo Theater", client=client, forgiving=forg)
    await client.aclose()
    assert point.match_type == "nominatim"
    assert "Apollo" in point.label


async def test_geosearch_fallback_name_mismatch_uses_forgiving_landmark_match():
    correct = GeoPoint(
        40.7507, -73.8627, "Civic Plaza, Queens, NY", confidence=0.8, match_type="nominatim"
    )

    def wrong_fuzzy_match(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "features": [{
                    "geometry": {"coordinates": [-73.8459, 40.7530]},
                    "properties": {
                        "name": "CIVIC YARD",
                        "label": "CIVIC YARD, Queens, NY",
                        "confidence": 0.8,
                        "match_type": "fallback",
                    },
                }]
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(wrong_fuzzy_match))
    point = await geocode("Civic Plaza", client=client, forgiving=_fake_forgiving(correct))
    await client.aclose()

    assert point == correct


async def test_geosearch_fallback_rejects_a_different_numbered_address():
    def wrong_address(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "features": [{
                    "geometry": {"coordinates": [-73.99, 40.75]},
                    "properties": {
                        "label": "999 Main Street, Manhattan",
                        "housenumber": "999",
                        "confidence": 0.8,
                        "match_type": "fallback",
                    },
                }]
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(wrong_address))
    point = await _geosearch_geocode("123 Main Street", client)
    await client.aclose()

    assert point is None


async def test_geosearch_fallback_accepts_the_matching_numbered_address():
    def matching_address(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "features": [{
                    "geometry": {"coordinates": [-73.99, 40.75]},
                    "properties": {
                        "label": "123 Main St, Manhattan",
                        "housenumber": "123",
                        "confidence": 0.8,
                        "match_type": "fallback",
                    },
                }]
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(matching_address))
    point = await _geosearch_geocode("123 Main Street", client)
    await client.aclose()

    assert point is not None
    assert point.label == "123 Main St, Manhattan"


async def test_gazetteer_beats_both_providers_for_known_neighborhoods():
    """F079 supersedes the F064 forgiving-first route for neighborhoods the NTA gazetteer
    knows: "Jackson Heights, Queens" resolves from city data, and neither GeoSearch (which
    once fuzzy-matched TRCS JACKSON HEIGHTS in St. Albans) nor nominatim is consulted."""
    async def no_forgiving(_text):
        raise AssertionError("gazetteer-known neighborhood reached the fuzzy provider")

    def no_geosearch(request: httpx.Request) -> httpx.Response:
        raise AssertionError("gazetteer-known neighborhood reached GeoSearch")

    client = httpx.AsyncClient(transport=httpx.MockTransport(no_geosearch))
    point = await geocode("Jackson Heights, Queens", client=client, forgiving=no_forgiving)
    await client.aclose()

    assert point is not None and point.match_type == "nta"
    assert "Queens" in point.label
    assert 40.72 < point.lat < 40.78 and -73.92 < point.lon < -73.85


async def test_named_area_unknown_to_gazetteer_uses_forgiving_geocoder_first():
    """The F064 contract survives for named areas the gazetteer does NOT know: the
    forgiving provider is asked before GeoSearch's PAD fuzzy match can win."""
    correct = GeoPoint(
        40.7557, -73.8858, "Flurbville, Queens", confidence=0.8, match_type="nominatim"
    )
    forgiving = _fake_forgiving(correct)

    def wrong_pad_match(request: httpx.Request) -> httpx.Response:
        return _geosearch_response(40.7003, -73.7717, "TRCS FLURBVILLE, St. Albans")

    client = httpx.AsyncClient(transport=httpx.MockTransport(wrong_pad_match))
    point = await geocode(
        "Flurbville, Queens", client=client, forgiving=forgiving,
        borough_contains=_fake_borough_contains(True),
    )
    await client.aclose()

    assert point == correct


async def test_named_neighborhood_rejects_forgiving_result_in_wrong_borough():
    wrong = GeoPoint(40.8460, -73.9090, "Jackson Avenue, Bronx", match_type="nominatim")

    def correct_pad_fallback(request: httpx.Request) -> httpx.Response:
        return _geosearch_response(40.7557, -73.8858, "Jackson Heights, Queens")

    client = httpx.AsyncClient(transport=httpx.MockTransport(correct_pad_fallback))
    point = await geocode(
        "Jackson Heights, Queens", client=client, forgiving=_fake_forgiving(wrong),
        borough_contains=_fake_borough_contains(True),
    )
    await client.aclose()

    assert point is not None
    assert point.label == "Jackson Heights, Queens"


async def test_numbered_address_rejects_forgiving_fallback_in_wrong_borough():
    wrong = GeoPoint(40.7557, -73.8858, "125th Street, Queens", match_type="nominatim")

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"features": []}))
    )
    point = await geocode(
        "125th Street Manhattan", client=client, forgiving=_fake_forgiving(wrong)
    )
    await client.aclose()

    assert point is None


async def test_overlapping_borough_boxes_use_official_polygon_validation():
    overlap = GeoPoint(40.7000, -73.9000, "Queens result", match_type="nominatim")

    async def outside_brooklyn(_point, _borough, _client):
        return False

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"features": []}))
    )
    point = await geocode(
        "Brooklyn", client=client, forgiving=_fake_forgiving(overlap),
        borough_contains=outside_brooklyn,
    )
    await client.aclose()

    assert point is None


async def test_official_borough_polygon_query_uses_point_intersection():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(request.url.params)
        return httpx.Response(200, json={"count": 1})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    inside = await _point_in_named_borough(
        GeoPoint(40.7000, -73.9000), "brooklyn", client
    )
    await client.aclose()

    assert inside
    assert seen["geometry"] == "-73.9,40.7"
    assert seen["geometryType"] == "esriGeometryPoint"
    assert seen["spatialRel"] == "esriSpatialRelIntersects"
    assert seen["where"] == "BoroName='Brooklyn'"


async def test_official_borough_polygon_query_fails_closed_on_non_object_json():
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=["unexpected"]))
    )
    inside = await _point_in_named_borough(
        GeoPoint(40.7000, -73.9000), "brooklyn", client
    )
    await client.aclose()

    assert not inside


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


async def test_nearest_rejects_a_citywide_origin(monkeypatch):
    async def should_not_geocode(_text, **_kwargs):
        raise AssertionError("citywide placeholder reached the geocoder")

    monkeypatch.setattr("heynyc.core.tools.geo.geocode", should_not_geocode)
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=[])))
    ctx = ToolContext(citations=CitationRegistry(), registry=_registry_with_cooling(), http=client)

    out = await _nearest_handler(
        {"category": "cooling_center", "near": "New York City"},
        ctx,
    )

    await client.aclose()
    assert "neighborhood, address, or landmark" in out


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
                    {"propertyname": "Close Site", "y": "40.7510", "x": "-73.9910", "status": "Activated", "borough": "Manhattan", ":updated_at": "2025-06-27T13:37:17.684Z", "website": {"url": "https://example.nyc/close-site"}, "comments": "Monday-Friday 8:30am to 5:00pm"},
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
    assert "record updated=2025-06-27" in lines[0]
    assert "official info: https://example.nyc/close-site" in lines[0]
    assert "hours: Monday-Friday 8:30am to 5:00pm" in lines[0]


async def test_nearest_handler_defaults_to_three_when_model_overrequests():
    def handler(request: httpx.Request) -> httpx.Response:
        if "geosearch" in request.url.host:
            return _geosearch_response(40.7500, -73.9900, "Origin, Manhattan")
        return httpx.Response(
            200,
            json=[
                {
                    "propertyname": f"Site {index}",
                    "y": str(40.7500 + index / 1000),
                    "x": "-73.9900",
                    "status": "Activated",
                    "borough": "Manhattan",
                }
                for index in range(1, 6)
            ],
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ctx = ToolContext(
        citations=CitationRegistry(),
        registry=_registry_with_cooling(),
        query="Where are the nearest fountains to Rockefeller Center?",
        http=client,
    )
    out = await _nearest_handler(
        {"category": "cooling_center", "near": "Rockefeller Center", "k": 5}, ctx
    )
    await client.aclose()

    assert len([line for line in out.splitlines() if line.startswith("- ")]) == 3

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ctx = ToolContext(
        citations=CitationRegistry(),
        registry=_registry_with_cooling(),
        query="Show me five fountains near Rockefeller Center",
        http=client,
    )
    out = await _nearest_handler(
        {"category": "cooling_center", "near": "Rockefeller Center", "k": 5}, ctx
    )
    await client.aclose()

    assert len([line for line in out.splitlines() if line.startswith("- ")]) == 5

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ctx = ToolContext(
        citations=CitationRegistry(),
        registry=_registry_with_cooling(),
        query="I need 5 cooling centers",
        http=client,
    )
    out = await _nearest_handler(
        {"category": "cooling_center", "near": "Rockefeller Center", "k": 8}, ctx
    )
    await client.aclose()

    assert len([line for line in out.splitlines() if line.startswith("- ")]) == 5


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


# --- F079: the bundled NTA neighborhood gazetteer resolves before any fuzzy provider ---

def _no_network_client():
    def handler(request):
        raise AssertionError(f"gazetteer path must not reach the network: {request.url}")
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def _no_forgiving(_text):
    raise AssertionError("neighborhood reached the fuzzy provider")


async def test_neighborhood_resolves_from_bundled_gazetteer_before_any_provider():
    """F079: "Upper West Side" once fuzzy-matched to a Bronx playground at provider
    confidence 1.0. Famous neighborhoods resolve from the city's own NTA data, offline."""
    client = _no_network_client()
    point = await geocode("Upper West Side", client=client, forgiving=_no_forgiving)
    await client.aclose()
    assert point is not None
    assert point.match_type == "nta"
    assert "Manhattan" in point.label
    assert 40.75 < point.lat < 40.82 and -74.01 < point.lon < -73.93
    assert not point.low_confidence


async def test_neighborhood_normalization_and_aliases():
    cases = {
        "the upper west side": "Manhattan",
        "UWS": "Manhattan",
        "FiDi": "Manhattan",
        "upper west side, manhattan": "Manhattan",
        "bed-stuy": "Brooklyn",
        "Flushing": "Queens",
        "Queens Village": "Queens",
    }
    for query, borough in cases.items():
        client = _no_network_client()
        point = await geocode(query, client=client, forgiving=_no_forgiving)
        await client.aclose()
        assert point is not None and point.match_type == "nta", query
        assert borough in point.label, query


def test_neighborhood_resolution_note_names_official_nta_source():
    point = GeoPoint(
        40.760197,
        -73.832301,
        "Flushing, Queens",
        confidence=1.0,
        match_type="nta",
    )

    note = _resolution_note("Flushing", point)

    assert "official NYC neighborhood data" in note
    assert "map search" not in note


async def test_neighborhood_with_contradictory_borough_falls_through():
    """"Upper West Side Brooklyn" is not a gazetteer hit; it takes the existing provider
    path (which fails closed here) instead of confidently answering for Manhattan."""
    def handler(request):
        return httpx.Response(200, json={"features": []})
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    point = await geocode(
        "Upper West Side Brooklyn", client=client, forgiving=_fake_forgiving(None),
    )
    await client.aclose()
    assert point is None


async def test_unknown_area_still_uses_existing_path():
    def handler(request):
        return _geosearch_response(40.7, -73.9, "Somewhere, NYC")
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    point = await geocode("Flurbville", client=client, forgiving=_fake_forgiving(None))
    await client.aclose()
    assert point is not None and point.match_type == "geosearch"


async def test_missing_gazetteer_file_degrades_to_provider_path(monkeypatch):
    from pathlib import Path

    from heynyc.core.tools import geo

    monkeypatch.setattr(geo, "_NTA_PATH", Path("/nonexistent/nta.tsv"))
    monkeypatch.setattr(geo, "_NTA_GAZETTEER", None)
    def handler(request):
        return httpx.Response(200, json={"features": []})
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    point = await geocode("Upper West Side", client=client, forgiving=_fake_forgiving(None))
    await client.aclose()
    assert point is None
    monkeypatch.setattr(geo, "_NTA_GAZETTEER", None)  # do not poison other tests' cache
