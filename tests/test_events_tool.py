from __future__ import annotations

import asyncio
from pathlib import Path
import httpx
import pytest

from heynyc.core.citations import CitationRegistry
from heynyc.core.registry import Registry
from heynyc.core.tools.base import Tool, ToolContext
from heynyc.modules.events import tools as events
from heynyc.modules.events.tools import (
    Event, _event_block, _explicitly_free, _from_parks, _from_ticketmaster, _future_only,
    _requested_window, get_tools,
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
    assert _from_ticketmaster({"name": "Bad", "dates": {"start": {"localDate": "later"}}}) is None


def test_from_ticketmaster_drops_cancelled_or_postponed_events():
    for status in ("canceled", "cancelled", "postponed"):
        raw = {
            "name": "No longer happening",
            "dates": {"start": {"localDate": "2026-07-19"}, "status": {"code": status}},
        }
        assert _from_ticketmaster(raw) is None


def test_from_ticketmaster_normalizes_null_name():
    event = _from_ticketmaster({
        "name": None, "dates": {"start": {"localDate": "2026-07-19"}},
    })
    assert event is not None
    assert event.name == ""


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


def test_from_parks_drops_cancelled_titles():
    for title in ("CANCELLED: Movie Night", "Canceled - Outdoor Concert", "POSTPONED: Movie"):
        assert _from_parks({"title": title, "startdate": "2026-07-19"}) is None


def test_from_parks_handles_null_title():
    event = _from_parks({"title": None, "startdate": "2026-07-19"})
    assert event is not None
    assert event.name == ""


def test_from_parks_drops_malformed_date():
    assert _from_parks({"title": "Bad upstream row", "startdate": "not-a-date"}) is None


def test_future_only_filters_past():
    past = Event("old", "2026-06-01", "", "", "", "u", "NYC Parks", "authoritative")
    future = Event("new", "2026-07-19", "", "", "", "u", "NYC Parks", "authoritative")
    kept = _future_only([past, future], today="2026-06-28")
    assert kept == [future]


def test_requested_window_resolves_this_weekend_from_nyc_date():
    assert _requested_window("free events this weekend", "2026-07-16") == (
        "2026-07-18", "2026-07-19",
    )
    assert _requested_window("events this weekend", "2026-07-18") == (
        "2026-07-18", "2026-07-19",
    )
    assert _requested_window("events today", "2026-07-16") == (
        "2026-07-16", "2026-07-16",
    )
    assert _requested_window("what is happening tonight", "2026-07-16") == (
        "2026-07-16", "2026-07-16",
    )
    assert _requested_window("things to do this week", "2026-07-16") == (
        "2026-07-16", "2026-07-19",
    )


def test_requested_window_resolves_tomorrow():
    assert _requested_window("what to prepare for tomorrows wc game", "2026-07-17") == (
        "2026-07-18", "2026-07-18",
    )
    assert _requested_window("what should i bring to the game tmrw", "2026-07-31") == (
        "2026-08-01", "2026-08-01",
    )
    assert _requested_window("what game is happening tomorow", "2026-07-17") == (
        "2026-07-18", "2026-07-18",
    )
    assert _requested_window("events at the department", "2026-07-17") == ("2026-07-17", None)


def test_requested_window_resolves_numeric_dates():
    assert _requested_window("what should i bring to the game on 7/18", "2026-07-17") == (
        "2026-07-18", "2026-07-18",
    )
    # A month/day earlier in the year means the next occurrence.
    assert _requested_window("the parade on 1/1", "2026-07-17") == ("2027-01-01", "2027-01-01")
    # Invalid calendar dates fall through to the default window.
    assert _requested_window("ratio is 19/32 exactly", "2026-07-17") == ("2026-07-17", None)


def test_editorial_query_includes_the_resolved_date_window():
    build = getattr(events, "_editorial_query", None)
    assert callable(build)
    query = build("free events this weekend", "2026-07-18", "2026-07-19")
    assert "July 18, 2026" in query
    assert "July 19, 2026" in query


def test_windowed_context_drops_explicitly_stale_event_blocks():
    filter_context = getattr(events, "_windowed_context", None)
    assert callable(filter_context)
    context = (
        "[S1] West Side Fest\nJuly 10-12, 2026.\n\n"
        "[S2] Rockefeller Center\nJuly 11-17, 2026.\n\n"
        "[S3] FIFA Museum\nOpen July 19, 2026."
    )

    filtered = filter_context(context, "2026-07-18", "2026-07-19")

    assert "FIFA Museum" in filtered
    assert "West Side Fest" not in filtered
    assert "Rockefeller Center" not in filtered


def test_nyc_for_free_rss_keeps_only_items_matching_the_requested_dates():
    select = getattr(events, "_nyc_for_free_items", None)
    assert callable(select)
    rss = """<rss><channel><lastBuildDate>Thu, 16 Jul 2026 16:25:39 +0000</lastBuildDate>
      <item><title>Weekend Pop-Up</title><link>https://www.nycforfree.co/events/weekend</link>
        <description><![CDATA[Free on July 18th and July 19th.]]></description></item>
      <item><title>August Event</title><link>https://www.nycforfree.co/events/august</link>
        <description><![CDATA[Free on August 8th.]]></description></item>
    </channel></rss>"""

    build_date, items = select(rss, "2026-07-18", "2026-07-19")

    assert build_date == "Thu, 16 Jul 2026 16:25:39 +0000"
    assert [item[0] for item in items] == ["Weekend Pop-Up"]
    assert items[0][2] == "July 18th"


def test_free_filter_requires_source_title_and_event_block_supplies_weekday():
    listed_free = Event(
        "Free Yoga", "2026-07-18", "9:00 AM", "Park", "", "u1", "NYC Parks", "authoritative",
    )
    unknown_cost = Event(
        "Open Run", "2026-07-18", "9:00 AM", "Park", "", "u2", "NYC Parks", "authoritative",
    )

    assert _explicitly_free([listed_free, unknown_cost], "free events") == [listed_free]
    assert _explicitly_free([listed_free, unknown_cost], "events") == [listed_free, unknown_cost]
    assert "Saturday, 2026-07-18" in _event_block(listed_free, "S1")
    assert "Details: u1" in _event_block(listed_free, "S1")


def test_tonight_filter_keeps_only_parseable_future_evening_events():
    morning = Event("Morning", "2026-07-16", "9:00 am", "", "", "u1", "NYC Parks", "authoritative")
    evening = Event("Evening", "2026-07-16", "7:00 pm", "", "", "u2", "NYC Parks", "authoritative")
    unknown = Event("Unknown", "2026-07-16", "", "", "", "u3", "NYC Parks", "authoritative")
    now = events.datetime(2026, 7, 16, 15, 0, tzinfo=events.NYC_TZ)

    assert events._tonight_only([morning, evening, unknown], now) == [evening]


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


async def test_broad_temporal_events_gather_current_city_context_concurrently(monkeypatch):
    started: set[str] = set()
    all_started = asyncio.Event()

    async def rendezvous(name: str):
        started.add(name)
        if len(started) == 5:
            all_started.set()
        await asyncio.wait_for(all_started.wait(), timeout=0.5)

    async def fake_ticketmaster(**kwargs):
        await rendezvous("ticketmaster")
        return [{
            "name": "Catalog Event",
            "url": "https://example.com/catalog",
            "dates": {"start": {"localDate": events.datetime.now(events.NYC_TZ).strftime("%Y-%m-%d")}},
        }]

    async def fake_parks(*args, **kwargs):
        await rendezvous("parks")
        return []

    async def web_handler(args, ctx):
        await rendezvous("official-web")
        current_date = events.datetime.now(events.NYC_TZ).strftime("%B %-d, %Y")
        cite = ctx.citations.register(
            "https://www.nyc.gov/current-event",
            snippet=f"Major current city event on {current_date}",
            title="NYC current event", kind="WEB",
        )
        return f"Major current city event on {current_date} {{cite:{cite}}}"

    async def index_handler(args, ctx):
        await rendezvous("official-index")
        cite = ctx.citations.register(
            "https://www.nynjfwc26.com/fan-events", snippet="Official seasonal event",
            title="Official seasonal event", kind="DOC",
        )
        return f"Official seasonal event {{cite:{cite}}}"

    async def editorial_context(ctx, window_start, window_end):
        await rendezvous("editorial-guides")
        cite = ctx.citations.register(
            "https://secretnyc.co/what-to-do-this-weekend-nyc/",
            snippet="Current weekend guide", title="Secret NYC weekend guide", kind="WEB",
        )
        return f"Current editorial weekend guide {{cite:{cite}}}"

    monkeypatch.setattr(events, "ticketmaster_events", fake_ticketmaster)
    monkeypatch.setattr(events, "query_dataset", fake_parks)
    monkeypatch.setattr(events, "_editorial_context", editorial_context, raising=False)
    monkeypatch.setattr(events, "_context_tools", lambda ctx: (
        Tool("index_search", "", {}, index_handler),
        Tool("web_search", "", {}, web_handler),
    ))
    ctx = ToolContext(
        citations=CitationRegistry(), registry=Registry([]),
        query="What events are happening in NYC today?",
    )

    output = await get_tools()[0].handler({}, ctx)

    assert started == {
        "ticketmaster", "parks", "official-index", "official-web", "editorial-guides",
    }
    assert "Catalog Event" in output
    assert "Official seasonal event" in output
    assert "Major current city event" in output
    assert "Current editorial weekend guide" in output
    assert "newly retrieved" in output.lower()
    assert "at most 5" in output
    assert "light emoji" in output
    assert "group the rest" in output
    assert "today-only advisory once" in output
    assert "merge sources" in output


async def test_tool_context_event_preparation_flag_activates_prep_synthesis(monkeypatch):
    """The semantic flag from the scope preflight reaches the tool through ToolContext, so a
    numeric-date phrasing the regex fallback misses still gets identity-first synthesis."""
    async def fake_ticketmaster(**kwargs):
        return []

    async def fake_parks(*args, **kwargs):
        return []

    async def quiet_editorial(ctx, window_start, window_end):
        return "Current editorial event guides unavailable for this lookup."

    monkeypatch.setattr(events, "ticketmaster_events", fake_ticketmaster)
    monkeypatch.setattr(events, "query_dataset", fake_parks)
    monkeypatch.setattr(events, "_editorial_context", quiet_editorial, raising=False)
    monkeypatch.setattr(events, "_context_tools", lambda ctx: ())
    ctx = ToolContext(
        citations=CitationRegistry(), registry=Registry([]),
        query="what should i bring to the game on 7/18",
        event_preparation=True,
    )

    output = await get_tools()[0].handler({}, ctx)

    low = output.lower()
    assert "identity" in low
    assert "resolve" in low
    assert "at most 5" not in output


async def test_empty_keyworded_catalog_retries_unkeyworded(monkeypatch):
    """Observed in the pre-commit eval: a stuffed keyword ('free NYC Parks weekend') matches
    no listing's full text, so the catalog came back empty while the window alone would have
    found the weekend rows. On empty, the catalog lanes retry once without the keyword."""
    keywords_seen = []

    async def fake_ticketmaster(**kwargs):
        keywords_seen.append(("tm", kwargs.get("keyword")))
        return []

    async def fake_parks(dataset_id, **kwargs):
        keywords_seen.append(("parks", kwargs.get("q")))
        if kwargs.get("q"):
            return []
        tomorrow = events.datetime.now(events.NYC_TZ) + events.timedelta(days=1)
        return [{
            "title": "Free Yoga on the Lawn", "startdate": tomorrow.strftime("%Y-%m-%d"),
            "starttime": "10:00 am", "parknames": "Fort Greene Park",
            "link": {"url": "http://www.nycgovparks.org/events/free-yoga"},
        }]

    monkeypatch.setattr(events, "ticketmaster_events", fake_ticketmaster)
    monkeypatch.setattr(events, "query_dataset", fake_parks)
    ctx = ToolContext(
        citations=CitationRegistry(), registry=Registry([]), query="events tomorrow",
    )

    output = await get_tools()[0].handler({"keyword": "free NYC Parks weekend"}, ctx)

    assert "Free Yoga on the Lawn" in output
    assert ("parks", "free NYC Parks weekend") in keywords_seen
    assert ("parks", None) in keywords_seen
    # A broadened result set must not silently substitute for what was actually asked:
    # the model is told to say so and route when the listings cannot satisfy the request.
    assert "broadened" in output.lower()
    assert "311" in output
    # The registered snapshot carries the full row the model will describe, time and source
    # included, so cited prose stays supported by its evidence.
    snapshot = next(iter(ctx.citations.mapping().values()))
    assert "10:00 am" in snapshot["snippet"]
    assert "NYC Parks" in snapshot["snippet"]


async def test_broad_shortlist_query_keeps_shortlist_rules_despite_prep_flag(monkeypatch):
    """A broad what's-happening query keeps the shortlist voice even when the semantic flag
    over-fires (observed in the pre-commit eval run)."""
    async def fake_ticketmaster(**kwargs):
        return []

    async def fake_parks(*args, **kwargs):
        return []

    async def quiet_editorial(ctx, window_start, window_end):
        return "Current editorial event guides unavailable for this lookup."

    monkeypatch.setattr(events, "ticketmaster_events", fake_ticketmaster)
    monkeypatch.setattr(events, "query_dataset", fake_parks)
    monkeypatch.setattr(events, "_editorial_context", quiet_editorial, raising=False)
    monkeypatch.setattr(events, "_context_tools", lambda ctx: ())
    ctx = ToolContext(
        citations=CitationRegistry(), registry=Registry([]),
        query="What free events are happening in NYC parks this weekend?",
        event_preparation=True,
    )

    output = await get_tools()[0].handler({}, ctx)

    assert "at most 5" in output
    assert "Event identity context" not in output


async def test_preparation_query_gathers_context_and_requires_event_resolution(monkeypatch):
    # This pins the tool contract AFTER the model has interpreted the abbreviation: the
    # scripted call passes keyword="world cup". Model-side interpretation of "WC" is pinned
    # by the live `events_abbreviated_game_preparation` eval case, not by this unit test.
    started: set[str] = set()
    all_started = asyncio.Event()

    async def rendezvous(name: str):
        started.add(name)
        if len(started) == 6:
            all_started.set()
        await asyncio.wait_for(all_started.wait(), timeout=0.5)

    async def fake_ticketmaster(**kwargs):
        await rendezvous("ticketmaster")
        tomorrow = events.datetime.now(events.NYC_TZ) + events.timedelta(days=1)
        return [{
            "name": "Catalog Event",
            "url": "https://example.com/catalog",
            "dates": {"start": {"localDate": tomorrow.strftime("%Y-%m-%d")}},
        }]

    async def fake_parks(*args, **kwargs):
        await rendezvous("parks")
        return []

    identity_queries = []

    async def web_handler(args, ctx):
        if "prefer" not in args:
            # The identity lane prefers the asked date's schedule rows; rows about other
            # dates (the famous final) must not drown the asked date out.
            identity_queries.append(args["query"])
            await rendezvous("identity-web")
            tomorrow = events.datetime.now(events.NYC_TZ) + events.timedelta(days=1)
            after = tomorrow + events.timedelta(days=1)
            return (
                f"France vs England bronze final schedule on {tomorrow.strftime('%B %-d, %Y')}"
                "\n\n"
                f"Famous final row on {after.strftime('%B %-d, %Y')}"
            )
        await rendezvous("official-web")
        tomorrow = events.datetime.now(events.NYC_TZ) + events.timedelta(days=1)
        cite = ctx.citations.register(
            "https://www.nyc.gov/current-event",
            snippet=f"Bronze final watch details on {tomorrow.strftime('%B %-d, %Y')}",
            title="NYC current event", kind="WEB",
        )
        return f"Bronze final watch details on {tomorrow.strftime('%B %-d, %Y')} {{cite:{cite}}}"

    async def index_handler(args, ctx):
        await rendezvous("official-index")
        return "Official seasonal event context"

    async def editorial_context(ctx, window_start, window_end):
        await rendezvous("editorial-guides")
        return "Current editorial guide context"

    monkeypatch.setattr(events, "ticketmaster_events", fake_ticketmaster)
    monkeypatch.setattr(events, "query_dataset", fake_parks)
    monkeypatch.setattr(events, "_editorial_context", editorial_context, raising=False)
    monkeypatch.setattr(events, "_context_tools", lambda ctx: (
        Tool("index_search", "", {}, index_handler),
        Tool("web_search", "", {}, web_handler),
    ))
    ctx = ToolContext(
        citations=CitationRegistry(), registry=Registry([]),
        query="What to prepare for tomorrows WC game",
    )

    output = await get_tools()[0].handler({"keyword": "world cup"}, ctx)

    assert started == {
        "ticketmaster", "parks", "official-index", "official-web", "editorial-guides",
        "identity-web",
    }
    assert "Catalog Event" in output
    assert "France vs England bronze final schedule" in output
    assert "Famous final row" not in output  # other-date rows filtered when dated rows exist
    # Audited live: searching the resident's prep-phrased sentence returns gardening events
    # ("prepare" matched a raised-beds workshop). The identity query must be schedule-shaped,
    # built from the model's keyword interpretation, never the raw prep phrasing.
    assert identity_queries, "identity lane must run for a preparation turn"
    assert "world cup" in identity_queries[0].lower()
    assert "schedule" in identity_queries[0].lower()
    assert "prepare" not in identity_queries[0].lower()
    low = output.lower()
    assert "identity" in low
    assert "resolve" in low
    assert "clarif" in low
    assert "packing" in low or "generic" in low
    assert "asked date" in low  # anchor on the asked date, not the most famous match
    assert "at most 5" not in output


def test_context_search_uses_configured_editorial_event_guides(monkeypatch):
    captured: dict[str, object] = {}

    def fake_web_search_tools(allowlist, source_tiers, news_tier):
        captured["allowlist"] = allowlist
        captured["source_tiers"] = source_tiers
        return [
            Tool("web_search", "", {}, lambda args, ctx: ""),
            Tool("recent_developments", "", {}, lambda args, ctx: ""),
        ]

    monkeypatch.setattr(
        "heynyc.core.tools.web_search.web_search_tools", fake_web_search_tools,
    )
    registry = Registry.discover(Path("heynyc/modules"))

    tools = events._context_tools(ToolContext(citations=CitationRegistry(), registry=registry))

    assert {"secretnyc.co", "nycforfree.co"} <= set(captured["allowlist"])
    assert [tool.name for tool in tools] == [
        "web_search", "recent_developments",
    ]


async def test_filtered_lane_citations_are_pruned(monkeypatch):
    """F057: a lane may register citations whose content the window filter then drops from
    the returned text. Those orphaned registrations must not survive the call, or the
    registry-wide faithfulness floor distrusts evidence the model never saw."""
    async def fake_ticketmaster(**kwargs):
        return []

    async def fake_parks(*args, **kwargs):
        return []

    async def quiet_editorial(ctx, window_start, window_end):
        return "Current editorial event guides unavailable for this lookup."

    async def web_handler(args, ctx):
        if "prefer" not in args:
            return "no identity rows"
        tomorrow = events.datetime.now(events.NYC_TZ) + events.timedelta(days=1)
        kept = ctx.citations.register(
            "https://www.nyc.gov/kept", snippet="kept row", title="Kept", kind="WEB",
        )
        orphan = ctx.citations.register(
            "https://www.nyc.gov/orphan", snippet="junk row", title="Orphan", kind="WEB",
        )
        return (
            f"[{kept}] Kept row dated {tomorrow.strftime('%B %-d, %Y')}"
            "\n\n"
            f"[{orphan}] Sports junk with no date"
        )

    monkeypatch.setattr(events, "ticketmaster_events", fake_ticketmaster)
    monkeypatch.setattr(events, "query_dataset", fake_parks)
    monkeypatch.setattr(events, "_editorial_context", quiet_editorial, raising=False)
    monkeypatch.setattr(events, "_context_tools", lambda ctx: (
        Tool("web_search", "", {}, web_handler),
    ))
    ctx = ToolContext(
        citations=CitationRegistry(), registry=Registry([]),
        query="what to prepare for tomorrows wc game",
    )

    output = await get_tools()[0].handler({"keyword": "world cup"}, ctx)

    kept_ids = {cid for cid, c in ctx.citations.mapping().items()
                if c["url"] == "https://www.nyc.gov/kept"}
    orphan_ids = {cid for cid, c in ctx.citations.mapping().items()
                  if c["url"] == "https://www.nyc.gov/orphan"}
    assert kept_ids, "windowed-in citation survives"
    assert not orphan_ids, "windowed-out citation is pruned"
    assert "Kept row" in output
