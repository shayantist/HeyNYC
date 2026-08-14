from __future__ import annotations

from heynyc.core import config
from heynyc.core.tools import geocoder
from heynyc.core.tools.geocoder import _confidence, forgiving_geocode


class _FakeLocation:
    """Mimics a geopy Location (latitude/longitude/address/raw)."""
    def __init__(self, lat, lon, address, raw):
        self.latitude = lat
        self.longitude = lon
        self.address = address
        self.raw = raw


def test_confidence_extraction_per_provider():
    assert _confidence("pelias", {"confidence": 0.9}) == 0.9
    assert _confidence("mapbox", {"relevance": 0.75}) == 0.75
    assert _confidence("nominatim", {"importance": 0.42}) == 0.42
    assert _confidence("nominatim", {}) == 0.0


async def test_forgiving_geocode_maps_location_to_geopoint():
    async def fake_fn(text):
        return _FakeLocation(40.808, -73.964, "Broadway & W 116 St, Manhattan", {"relevance": 0.93})

    point = await forgiving_geocode("116 & broadway, manhattan", geocode_fn=fake_fn)
    assert round(point.lat, 3) == 40.808
    assert round(point.lon, 3) == -73.964
    assert "Broadway" in point.label
    assert point.confidence == 0.93


async def test_forgiving_geocode_none_and_error_are_safe():
    async def none_fn(text):
        return None

    async def boom_fn(text):
        raise RuntimeError("provider down")

    assert await forgiving_geocode("x", geocode_fn=none_fn) is None
    assert await forgiving_geocode("x", geocode_fn=boom_fn) is None  # errors degrade to None, not crash


async def test_nominatim_reuses_an_in_process_result(monkeypatch):
    calls = 0

    async def fake_fn(text):
        nonlocal calls
        calls += 1
        return _FakeLocation(40.7578, -73.8289, "Flushing Library, Queens", {"importance": 0.8})

    monkeypatch.setattr(config, "HEYNYC_GEOCODER", "nominatim")
    monkeypatch.setattr(config, "MAPBOX_TOKEN", "")
    monkeypatch.setattr(geocoder, "_build_geocode_fn", lambda _provider: fake_fn)
    geocoder._NOMINATIM_CACHE.clear()
    first = await forgiving_geocode("Flushing Library")
    second = await forgiving_geocode(" Flushing Library ")

    assert first == second
    assert calls == 1


async def test_nominatim_miss_falls_back_to_configured_mapbox(monkeypatch):
    calls = []

    async def nominatim(_text):
        calls.append("nominatim")
        return None

    async def mapbox(_text):
        calls.append("mapbox")
        return _FakeLocation(40.7578, -73.8289, "Flushing Library, Queens", {"relevance": 0.9})

    monkeypatch.setattr(config, "HEYNYC_GEOCODER", "nominatim")
    monkeypatch.setattr(config, "MAPBOX_TOKEN", "configured")
    monkeypatch.setattr(
        geocoder,
        "_build_geocode_fn",
        lambda provider: nominatim if provider == "nominatim" else mapbox,
    )
    geocoder._NOMINATIM_CACHE.clear()

    point = await forgiving_geocode("Flushing Library")

    assert point is not None
    assert point.match_type == "mapbox"
    assert calls == ["nominatim", "mapbox"]
