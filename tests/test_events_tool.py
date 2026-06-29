from __future__ import annotations

import httpx
import pytest

from heynyc.core.citations import CitationRegistry
from heynyc.core.registry import Registry
from heynyc.core.tools.base import ToolContext
from heynyc.modules.events.tools import (
    Event, _from_parks, _from_ticketmaster, _future_only, get_tools,
)


def test_from_ticketmaster_maps_fields():
    raw = {
        "name": "FIFA Final Watch Party",
        "url": "https://www.ticketmaster.com/event/abc",
        "dates": {"start": {"localDate": "2026-07-19", "localTime": "15:00:00"}},
        "_embedded": {"venues": [{"name": "Central Park", "city": {"name": "New York"}}]},
    }
    ev = _from_ticketmaster(raw)
    assert ev == Event(
        name="FIFA Final Watch Party", start_date="2026-07-19", start_time="15:00:00",
        venue="Central Park", borough="New York",
        url="https://www.ticketmaster.com/event/abc", source="Ticketmaster", tier="authoritative",
    )


def test_from_ticketmaster_drops_dateless():
    assert _from_ticketmaster({"name": "TBA", "dates": {"start": {}}}) is None


def test_from_parks_maps_nested_link():
    raw = {
        "title": "Summer Pickleball",
        "startdate": "2026-06-17T00:00:00.000",
        "starttime": "7:00 am",
        "parknames": "Blood Root Valley",
        "location": "Arts and Crafts Room",
        "link": {"url": "http://www.nycgovparks.org/events/2026/06/17/x"},
    }
    ev = _from_parks(raw)
    assert ev.name == "Summer Pickleball"
    assert ev.start_date == "2026-06-17"
    assert ev.start_time == "7:00 am"
    assert ev.venue == "Blood Root Valley"
    assert ev.url == "http://www.nycgovparks.org/events/2026/06/17/x"
    assert ev.source == "NYC Parks" and ev.tier == "authoritative"


def test_future_only_filters_past():
    past = Event("old", "2026-06-01", "", "", "", "u", "NYC Parks", "authoritative")
    future = Event("new", "2026-07-19", "", "", "", "u", "NYC Parks", "authoritative")
    kept = _future_only([past, future], today="2026-06-28")
    assert kept == [future]


@pytest.fixture(autouse=True)
def _force_tm_key(monkeypatch):
    # Force the TM branch to run offline (the handler reads config.TICKETMASTER_API_KEY).
    monkeypatch.setattr("heynyc.core.ticketmaster.config.TICKETMASTER_API_KEY", "test-key")


def _routed_client() -> httpx.AsyncClient:
    """One client routing Ticketmaster vs Socrata by host — both fully offline."""
    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if "ticketmaster" in host:
            return httpx.Response(200, json={"_embedded": {"events": [{
                "name": "Concert in the Park",
                "url": "https://www.ticketmaster.com/event/abc",
                "dates": {"start": {"localDate": "2099-07-19", "localTime": "20:00:00"}},
                "_embedded": {"venues": [{"name": "SummerStage", "city": {"name": "New York"}}]},
            }]}})
        # Socrata (Parks): one past row (filtered out) + one future row.
        return httpx.Response(200, json=[
            {"title": "Old Festival", "startdate": "2000-01-01T00:00:00.000",
             "starttime": "9:00 am", "parknames": "Old Park",
             "link": {"url": "http://www.nycgovparks.org/events/old"}},
            {"title": "Future Fair", "startdate": "2099-08-01T00:00:00.000",
             "starttime": "10:00 am", "parknames": "New Park",
             "link": {"url": "http://www.nycgovparks.org/events/new"}},
        ])
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_whats_on_events_merges_grounds_and_filters_future():
    [tool] = get_tools()
    citations = CitationRegistry()
    async with _routed_client() as client:
        ctx = ToolContext(citations=citations, registry=Registry([]), http=client)
        out = await tool.handler({"keyword": "music"}, ctx)

    assert "Concert in the Park" in out
    assert "Future Fair" in out
    assert "Old Festival" not in out          # past event filtered (§12)
    assert "{cite:" in out                     # everything is grounded + cited
    assert citations.mapping()                 # at least one DATA citation registered
