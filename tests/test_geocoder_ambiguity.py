import httpx
import pytest

from heynyc.core.tools.geo import (
    GeoPoint,
    _geosearch_geocode,
    geocode,
    resident_supplied_location,
)


def _client() -> httpx.AsyncClient:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            json={
                "features": [
                    {
                        "geometry": {"coordinates": [-73.82374, 40.73740]},
                        "properties": {
                            "label": "6591 MAIN STREET, Flushing, NY, USA",
                            "housenumber": "6591",
                            "confidence": 1.0,
                        },
                    }
                ]
            },
        )

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_provider_added_house_number_marks_the_street_ambiguous() -> None:
    async with _client() as client:
        point = await _geosearch_geocode("Main Street, Flushing", client)

    assert point is not None
    assert point.low_confidence is True


@pytest.mark.asyncio
async def test_resident_supplied_house_number_remains_precise() -> None:
    async with _client() as client:
        point = await _geosearch_geocode("6591 Main Street, Flushing", client)

    assert point is not None
    assert point.low_confidence is False


@pytest.mark.asyncio
async def test_house_number_after_leading_context_remains_precise() -> None:
    async with _client() as client:
        point = await _geosearch_geocode("near 6591 Main Street, Flushing", client)

    assert point is not None
    assert point.low_confidence is False


@pytest.mark.asyncio
async def test_locality_qualified_street_uses_general_geocoder_before_geosearch() -> None:
    expected = GeoPoint(
        lat=40.756031,
        lon=-73.828535,
        label="Main Street, Flushing, Queens, New York",
        confidence=0.8,
        match_type="nominatim",
    )

    async def forgiving(_text: str) -> GeoPoint:
        return expected

    async with _client() as client:
        point = await geocode("Main Street, Flushing", client=client, forgiving=forgiving)

    assert point == expected
    assert point.low_confidence is False


@pytest.mark.asyncio
async def test_bare_place_does_not_silently_expand_to_a_street() -> None:
    async def forgiving(_text: str) -> GeoPoint:
        return GeoPoint(
            lat=40.60139,
            lon=-74.07006,
            label="Chicago Avenue, Arrochar, Staten Island, New York",
            confidence=0.7,
            match_type="nominatim",
        )

    point = await geocode("Chicago", forgiving=forgiving)

    assert point is not None
    assert point.low_confidence is True


@pytest.mark.asyncio
async def test_failed_place_lookup_does_not_fall_through_to_address_geosearch() -> None:
    async def no_match(_text: str) -> None:
        return None

    async def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"place lookup reached NYC address service: {request.url}")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        point = await geocode("Flushing Library", client=client, forgiving=no_match)

    assert point is None


@pytest.mark.asyncio
async def test_worded_intersection_uses_provider_intersection_syntax() -> None:
    seen = ""

    async def forgiving(text: str) -> GeoPoint:
        nonlocal seen
        seen = text
        return GeoPoint(
            lat=40.7596853,
            lon=-73.8303214,
            label="Main Street & Roosevelt Avenue, Queens, New York",
            confidence=0.8,
            match_type="nominatim",
        )

    point = await geocode(
        "Main Street and Roosevelt Avenue, Queens",
        forgiving=forgiving,
    )

    assert seen == "Main Street & Roosevelt Avenue, Queens"
    assert point is not None
    assert point.low_confidence is False


def test_multilingual_location_provenance_keeps_provider_friendly_candidate() -> None:
    assert resident_supplied_location(
        "Flushing Main Street, Queens, NY",
        "我在Flushing附近，靠近Main Street，但是不想说完整地址。",
        (),
    ) == "Flushing Main Street"


def test_location_provenance_combines_separately_supplied_street_and_locality() -> None:
    assert resident_supplied_location(
        "Main Street, Flushing, Queens, NY",
        "我在Flushing附近，靠近Main Street，但是不想说完整地址。",
        (),
    ) == "Main Street, Flushing"
