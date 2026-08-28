"""DOT street-closure lane over NYC Open Data i6b5-j7bu.

Game-day / travel questions grounded in DOT's construction street-closure
schedule: "are streets closed near <place> on <date>?". The closure geometry is
a line (the_geom), so the query geo-filters with within_circle and date-filters
on the row's own work_start_date / work_end_date window. Every listed closure is
cited full-row with a re-fetchable permalink keyed on the dataset's own uniqueid.
Fully offline: query_dataset and geocode are injected, never a live Socrata /
geocoder call.
"""
from __future__ import annotations

import pytest

from heynyc.core import config
from heynyc.core.citations import CitationRegistry
from heynyc.core.registry import Registry
from heynyc.core.tools.base import ToolContext
from heynyc.core.tools.geo import GeoPoint
from heynyc.modules.street_closures import tools as closures


def _ctx() -> ToolContext:
    return ToolContext(
        citations=CitationRegistry(), registry=Registry.discover(config.MODULES_DIR)
    )


def _closure_row() -> dict:
    return {
        ":id": "row-qpqz-n89d~tpzi",
        ":updated_at": "2026-07-13T12:18:56.284Z",
        "the_geom": {
            "type": "MultiLineString",
            "coordinates": [[[-73.9266823583887, 40.73646157545949],
                             [-73.92612185142727, 40.736364686388704]]],
        },
        "segmentid": "129578",
        "onstreetname": "51 AVENUE",
        "fromstreetname": "39 PLACE",
        "tostreetname": "40 STREET",
        "borough_code": "Q",
        "work_start_date": "2026-08-03T00:00:00.000",
        "work_end_date": "2026-10-30T00:00:00.000",
        "uniqueid": "51898c98654217274bf0bea783451c16a5656c4ed0b2425f42ca8a7a00825946",
        "purpose": "DOT IN-HOUSE MILLING",
    }


def test_module_loads_custom_tool():
    names = {tool.name for tool in Registry.discover(config.MODULES_DIR).load_module_tools()}
    assert "find_street_closures" in names


def test_street_closure_date_is_typed_in_the_tool_schema():
    properties = closures.get_tools()[0]._input_schema()["properties"]
    assert {"format": "date", "type": "string"} in properties["visit_date"]["anyOf"]
    assert "on" not in properties
    assert "limit" not in properties


def test_module_declares_the_closures_dataset_binding():
    binding = Registry.discover(config.MODULES_DIR).dataset_bindings().get("street_closure")
    assert binding is not None
    assert binding.id == "i6b5-j7bu"


def test_permalink_is_refetchable_uniqueid_query():
    url = closures._closure_permalink("abc123")
    assert url.startswith("https://data.cityofnewyork.us/resource/i6b5-j7bu.json")
    # keyed on the dataset's OWN uniqueid field, not a prose match
    assert "uniqueid='abc123'" in url.replace("%27", "'").replace("%3D", "=")


def test_valid_as_of_uses_records_own_update_date():
    assert closures._valid_as_of(_closure_row()) == "2026-07-13T12:18:56.284Z"
    # no :updated_at -> falls back to the closure's own work_start_date
    assert (
        closures._valid_as_of({"work_start_date": "2026-08-03T00:00:00.000"})
        == "2026-08-03T00:00:00.000"
    )


def test_representative_point_reads_the_line_geometry():
    lat, lon = closures._representative_point(_closure_row())
    assert round(lat, 5) == 40.73646
    assert round(lon, 5) == -73.92668
    # a row with no usable geometry yields no point (never a fabricated coord)
    assert closures._representative_point({"the_geom": None}) is None


@pytest.mark.asyncio
async def test_invalid_explicit_date_requests_correction_before_lookup(monkeypatch):
    async def should_not_run(*args, **kwargs):
        raise AssertionError("invalid date reached a location or dataset lookup")

    monkeypatch.setattr("heynyc.core.tools.geo.geocode", should_not_run)
    monkeypatch.setattr(closures, "query_dataset", should_not_run)

    out = await closures.get_tools()[0].handler(
        {"near": "Union Square", "visit_date": "2026-02-30"},
        _ctx(),
    )

    assert "date" in out.lower()
    assert "YYYY-MM-DD" in out
    assert "today" not in out.lower()


@pytest.mark.asyncio
async def test_closures_near_filters_by_geo_and_date_and_cites_each(monkeypatch):
    captured: list[dict] = []

    async def fake_geocode(text, **kwargs):
        return GeoPoint(40.7365, -73.9260, "Sunnyside, Queens")

    async def fake_qd(dataset_id, **kwargs):
        captured.append({"dataset_id": dataset_id, **kwargs})
        return [_closure_row()]

    monkeypatch.setattr("heynyc.core.tools.geo.geocode", fake_geocode)
    monkeypatch.setattr(closures, "query_dataset", fake_qd)

    ctx = _ctx()
    out = await closures.get_tools()[0].handler(
        {"near": "Sunnyside", "visit_date": "2026-08-15"}, ctx
    )

    where = captured[0]["where"]
    assert captured[0]["dataset_id"] == "i6b5-j7bu"
    assert "within_circle(the_geom, 40.7365, -73.926," in where
    # date window keyed on the dataset's OWN columns, not a keyword scan
    assert "work_start_date <= '2026-08-15" in where
    assert "work_end_date >= '2026-08-15" in where

    assert "51 AVENUE" in out
    assert "DOT IN-HOUSE MILLING" in out
    assert "{cite:S1}" in out

    cites = ctx.citations.mapping()
    assert len(cites) == 1
    cite = cites["S1"]
    assert cite["kind"] == "DATA"
    assert cite["valid_as_of"] == "2026-07-13T12:18:56.284Z"
    # full-row snapshot keyed on the row's own uniqueid, not a hand-picked subset
    assert cite["provenance"]["record_id"] == _closure_row()["uniqueid"]
    assert cite["provenance"]["snapshot"]["purpose"] == "DOT IN-HOUSE MILLING"


@pytest.mark.asyncio
async def test_closure_count_comes_from_every_source_page(monkeypatch):
    async def fake_geocode(text, **kwargs):
        return GeoPoint(40.7365, -73.9260, "Sunnyside, Queens")

    async def fake_qd(_dataset_id, **kwargs):
        offset = kwargs["offset"]
        if offset == 0:
            return [{**_closure_row(), "uniqueid": "first"}]
        if offset == 1:
            return [{**_closure_row(), "uniqueid": "second"}]
        return []

    monkeypatch.setattr("heynyc.core.tools.geo.geocode", fake_geocode)
    monkeypatch.setattr(closures, "query_dataset", fake_qd)
    monkeypatch.setattr(closures, "_PAGE_SIZE", 1)

    out = await closures.get_tools()[0].handler(
        {"near": "Sunnyside", "max_results": 1}, _ctx()
    )

    assert "(2 found)" in out


@pytest.mark.asyncio
async def test_no_location_asks_for_one_and_never_queries(monkeypatch):
    async def boom_qd(*args, **kwargs):
        raise AssertionError("must not query without a location to bound closures")

    monkeypatch.setattr(closures, "query_dataset", boom_qd)

    out = await closures.get_tools()[0].handler({}, _ctx())
    assert "address" in out.lower() or "location" in out.lower()


@pytest.mark.asyncio
async def test_low_confidence_location_asks_to_clarify(monkeypatch):
    async def fake_geocode(text, **kwargs):
        return GeoPoint(40.0, -73.0, "somewhere", low_confidence=True)

    monkeypatch.setattr("heynyc.core.tools.geo.geocode", fake_geocode)

    out = await closures.get_tools()[0].handler({"near": "the bridge"}, _ctx())
    assert "specific NYC address" in out


@pytest.mark.asyncio
async def test_no_closures_is_honest_and_routes_without_fabricating(monkeypatch):
    async def fake_geocode(text, **kwargs):
        return GeoPoint(40.7365, -73.9260, "Sunnyside, Queens")

    async def fake_qd(dataset_id, **kwargs):
        return []

    monkeypatch.setattr("heynyc.core.tools.geo.geocode", fake_geocode)
    monkeypatch.setattr(closures, "query_dataset", fake_qd)

    out = await closures.get_tools()[0].handler({"near": "Sunnyside"}, _ctx())
    assert "no" in out.lower()
    assert "511" in out or "dot" in out.lower()  # route to live traffic source
    assert "{cite:" not in out  # nothing invented, nothing cited
