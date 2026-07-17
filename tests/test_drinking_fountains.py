"""NYC Parks drinking-fountain module uses the shared nearest-place path."""
import pytest

from heynyc.core import config
from heynyc.core.citations import CitationRegistry
from heynyc.core.registry import Registry
from heynyc.core.tools.base import ToolContext
from heynyc.core.tools.geo import GeoPoint, _nearest_handler


def test_drinking_fountain_module_registers_official_parks_layer():
    registry = Registry.discover(config.MODULES_DIR)
    binding = registry.dataset_bindings()["drinking_fountain"]

    assert binding.source == "arcgis"
    assert binding.record_id_field == "FID"
    assert "NYC_Parks_Drinking_Fountains" in binding.url
    assert binding.where == "FeatureSta='Active'"
    assert "outdoor fountains in parks only" in binding.limitations


def test_drinking_fountain_module_explains_coverage_and_status_limits():
    registry = Registry.discover(config.MODULES_DIR)
    module = next(module for module in registry.modules if module.name == "drinking_fountains")

    prompt = module.prompt.lower()
    assert "outdoor fountains in nyc parks" in prompt
    assert "not a live guarantee" in prompt
    assert "directions" in prompt


@pytest.mark.asyncio
async def test_drinking_fountain_nearest_returns_row_citation_and_directions(monkeypatch):
    async def fake_geocode(text, **kwargs):
        return GeoPoint(40.7580, -73.9780, text)

    async def fake_query(url, **kwargs):
        return [{
            "FID": 4708,
            "PropName": "Central Park",
            "Borough": "M",
            "FeatureSta": "Active",
            "lat": 40.76484,
            "lon": -73.97366,
        }]

    monkeypatch.setattr("heynyc.core.tools.geo.geocode", fake_geocode)
    monkeypatch.setattr("heynyc.core.tools.geo.query_feature_service", fake_query)
    citations = CitationRegistry()
    ctx = ToolContext(citations=citations, registry=Registry.discover(config.MODULES_DIR))

    output = await _nearest_handler(
        {"category": "drinking_fountain", "near": "Rockefeller Center"}, ctx,
    )

    assert "Central Park" in output
    assert "status=Active" in output
    assert "directions: https://www.google.com/maps/search/?api=1&query=40.76484,-73.97366" in output
    citation = citations.mapping()["S1"]
    assert citation["url"].endswith("/query?where=FID%3D4708&outFields=*&f=geojson")
    assert citation["provenance"]["derivation"]["point"] == [40.76484, -73.97366]
    assert "not a live guarantee" in citation["provenance"]["derivation"]["limitations"]
    assert "Source limit:" in output
