"""Offline tests for the Notify NYC advisories client + the advisories module tool.

Every HTTP call is mocked via httpx.MockTransport, routed by url path (`/rss/rss.xml` for the feed,
`/cap/<id>.xml` for each CAP alert) — no live Everbridge call. Covers: English-only filtering, CAP
field parsing, active-window filtering + severity sort, malformed-CAP tolerance, and the module
tool's grounding+citation / clean abstention. The shipped module load mirrors test_module_food_pantries.
"""
from __future__ import annotations

import asyncio
import json
from datetime import date, datetime, timezone

import httpx

from heynyc.core import config
from heynyc.core.citations import CitationRegistry
from heynyc.core.registry import Registry
from heynyc.core.tools import notify_nyc
from heynyc.core.tools.base import ToolContext
from heynyc.core.tools.notify_nyc import (
    RECENT_MESSAGES_URL,
    active_advisories,
    fetch_advisories,
    fetch_recent_advisories,
)
from heynyc.modules.advisories import tools as advisory_tools
from heynyc.modules.advisories.tools import get_tools

# --- canned feed fixtures --------------------------------------------------

FEED_BASE = "https://feeds.everbridge.net/feeds/453003085617722"


def _cap(identifier: str, *, severity: str, event: str, headline: str, expires: str,
         sent: str = "2026-07-02T12:00:00-04:00",
         area: str = "Bronx,Kings,New York,Queens,Staten Island",
         category: str = "Met", language: str = "en-US") -> str:
    # sent + expires are placed under <info> to match the verified live structure; the parser is
    # namespace- and layout-agnostic either way.
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<alert xmlns="urn:oasis:names:tc:emergency:cap:1.2">
  <identifier>{identifier}</identifier>
  <sender>notifynyc@cityhall.nyc.gov</sender>
  <status>Actual</status>
  <msgType>Alert</msgType>
  <scope>Public</scope>
  <info>
    <language>{language}</language>
    <category>{category}</category>
    <event>{event}</event>
    <urgency>Expected</urgency>
    <severity>{severity}</severity>
    <certainty>Likely</certainty>
    <sent>{sent}</sent>
    <expires>{expires}</expires>
    <headline>{headline}</headline>
    <area>
      <areaDesc>{area}</areaDesc>
    </area>
  </info>
</alert>"""


# One active (far-future expiry), one already expired.
CAP_ACTIVE = _cap("NYC-ACTIVE-1", severity="Severe", event="Extreme Heat",
                  headline="Heat Advisory in effect for NYC", expires="2099-07-02T19:45:28-04:00")
CAP_EXPIRED = _cap("NYC-EXPIRED-1", severity="Minor", event="Old Event",
                   headline="Expired advisory", expires="2020-01-01T00:00:00-05:00")


def _item(title: str, author: str, guid: str, cap_name: str, *, use_link: bool = False) -> str:
    cap_url = f"{FEED_BASE}/cap/{cap_name}"
    locator = (f"    <link>{cap_url}</link>\n" if use_link
               else f'    <enclosure url="{cap_url}" type="application/xml"/>\n')
    return (f"  <item>\n"
            f"    <title>{title}</title>\n"
            f"    <author>{author}</author>\n"
            f"    <guid>{guid}</guid>\n"
            f"{locator}"
            f"  </item>\n")


def _rss(*items: str) -> str:
    body = "".join(items)
    return (f"<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
            f"<rss version=\"2.0\"><channel>\n<title>Notify NYC</title>\n{body}</channel></rss>")


# Two English items (active + expired) + one non-English item that must be filtered out.
RSS_MAIN = _rss(
    _item("Heat Advisory (English)", "NYCEM [English]", "guid-active-en", "active.xml"),
    _item("Old Advisory (English)", "NYCEM [English]", "guid-expired-en", "expired.xml", use_link=True),
    _item("Aviso de calor (Spanish)", "NYCEM [Spanish]", "guid-active-es", "spanish.xml"),
)

CAPS_MAIN = {"active.xml": CAP_ACTIVE, "expired.xml": CAP_EXPIRED, "spanish.xml": CAP_ACTIVE}


def _client(rss: str, caps: dict[str, str], seen: list[str] | None = None) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if seen is not None:
            seen.append(path)
        if path.endswith("/rss/rss.xml"):
            return httpx.Response(200, text=rss, headers={"content-type": "application/rss+xml"})
        if "/cap/" in path:
            name = path.rsplit("/", 1)[-1]
            if name in caps:
                return httpx.Response(200, text=caps[name],
                                      headers={"content-type": "application/xml"})
        return httpx.Response(404)
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# --- client: fetch + parse -------------------------------------------------

async def test_fetch_advisories_keeps_english_and_parses():
    seen: list[str] = []
    client = _client(RSS_MAIN, CAPS_MAIN, seen=seen)
    advisories = (await fetch_advisories(client)).advisories
    await client.aclose()

    # Only the two English items were fetched + parsed; the Spanish CAP was never requested.
    assert len(advisories) == 2
    assert not any("spanish" in p for p in seen)

    by_id = {a.guid: a for a in advisories}
    active = by_id["NYC-ACTIVE-1"]
    assert active.headline == "Heat Advisory in effect for NYC"
    assert active.event == "Extreme Heat"
    assert active.severity == "Severe"
    assert active.category == "Met"
    assert active.urgency == "Expected"
    assert active.sent == "2026-07-02T12:00:00-04:00"
    assert active.expires == "2099-07-02T19:45:28-04:00"
    assert active.area_desc == "Bronx,Kings,New York,Queens,Staten Island"
    assert active.source_url == f"{FEED_BASE}/cap/active.xml"
    # The expired item resolved its CAP url from <link> (enclosure-fallback path).
    assert by_id["NYC-EXPIRED-1"].source_url == f"{FEED_BASE}/cap/expired.xml"


# The same alert (identifier NYC-ACTIVE-1) in English and Spanish — the CAP identifier is
# language-stable, which is what lets the Spanish variant overlay the English one per alert.
CAP_ACTIVE_ES = _cap("NYC-ACTIVE-1", severity="Severe", event="Calor Extremo",
                     headline="Aviso de calor en vigor para NYC",
                     expires="2099-07-02T19:45:28-04:00", language="es-US")

RSS_MULTILANG = _rss(
    _item("Heat Advisory (English)", "NYCEM [English]", "guid-active-en", "active.xml"),
    _item("Old Advisory (English)", "NYCEM [English]", "guid-expired-en", "expired.xml"),
    _item("Aviso de calor (Spanish)", "NYCEM [Spanish]", "guid-active-es", "active-es.xml"),
)
CAPS_MULTILANG = {"active.xml": CAP_ACTIVE, "expired.xml": CAP_EXPIRED, "active-es.xml": CAP_ACTIVE_ES}


async def test_fetch_advisories_surfaces_requested_language_variant():
    # Red-team/compliance 4a: a Spanish request returns the city's OFFICIAL Spanish translation of
    # the alert (not a paraphrase), keeping English as the per-alert fallback.
    seen: list[str] = []
    client = _client(RSS_MULTILANG, CAPS_MULTILANG, seen=seen)
    advisories = (await fetch_advisories(client, lang="Spanish")).advisories
    await client.aclose()

    by_id = {a.guid: a for a in advisories}
    assert by_id["NYC-ACTIVE-1"].headline == "Aviso de calor en vigor para NYC"  # Spanish won
    assert by_id["NYC-ACTIVE-1"].language == "es-US"
    # The expired alert has no Spanish variant → English fallback is still fetched + present.
    assert by_id["NYC-EXPIRED-1"].headline == "Expired advisory"
    assert any("active-es" in p for p in seen)  # the Spanish CAP WAS fetched this time


async def test_fetch_advisories_language_alias_resolves():
    # A code/alias ("es") resolves to the feed's "[Spanish]" language name.
    client = _client(RSS_MULTILANG, CAPS_MULTILANG)
    advisories = (await fetch_advisories(client, lang="es")).advisories
    await client.aclose()
    assert {a.guid: a for a in advisories}["NYC-ACTIVE-1"].headline == "Aviso de calor en vigor para NYC"


async def test_fetch_advisories_english_default_ignores_other_languages():
    # The English path is unchanged: no language requested → only English items fetched, Spanish skipped.
    seen: list[str] = []
    client = _client(RSS_MULTILANG, CAPS_MULTILANG, seen=seen)
    advisories = (await fetch_advisories(client)).advisories  # default English
    await client.aclose()
    assert {a.guid for a in advisories} == {"NYC-ACTIVE-1", "NYC-EXPIRED-1"}
    assert {a.guid: a for a in advisories}["NYC-ACTIVE-1"].headline == "Heat Advisory in effect for NYC"
    assert not any("active-es" in p for p in seen)  # Spanish CAP never requested by default


async def test_check_notify_nyc_tool_passes_language_through():
    # The module tool threads a `lang` arg to the feed and surfaces the official translation.

    citations = CitationRegistry()
    client = _client(RSS_MULTILANG, CAPS_MULTILANG)
    ctx = ToolContext(citations=citations, registry=Registry([]), http=client)
    out = await get_tools()[0].handler({"lang": "Spanish"}, ctx)
    await client.aclose()
    assert "Aviso de calor en vigor para NYC" in out
    assert "Heat Advisory in effect for NYC" not in out  # English active was replaced by Spanish


async def test_active_advisories_excludes_expired():
    now = datetime(2026, 7, 2, 18, 0, tzinfo=timezone.utc)
    client = _client(RSS_MAIN, CAPS_MAIN)
    active = (await active_advisories(client, now=now)).advisories
    await client.aclose()

    assert [a.guid for a in active] == ["NYC-ACTIVE-1"]  # expired one dropped


async def test_active_advisories_sorted_by_severity():
    # A dedicated fixture with two currently-active English advisories of differing severity.
    rss = _rss(
        _item("Severe (English)", "NYCEM [English]", "g-sev", "severe.xml"),
        _item("Extreme (English)", "NYCEM [English]", "g-ext", "extreme.xml"),
    )
    caps = {
        "severe.xml": _cap("NYC-SEV", severity="Severe", event="Heat",
                           headline="Severe alert", expires="2099-01-01T00:00:00-05:00"),
        "extreme.xml": _cap("NYC-EXT", severity="Extreme", event="Flash Flood",
                            headline="Extreme alert", expires="2099-01-01T00:00:00-05:00"),
    }
    now = datetime(2026, 7, 2, 18, 0, tzinfo=timezone.utc)
    client = _client(rss, caps)
    active = (await active_advisories(client, now=now)).advisories
    await client.aclose()

    assert [a.severity for a in active] == ["Extreme", "Severe"]  # most severe first


async def test_malformed_cap_is_skipped_not_fatal():
    rss = _rss(
        _item("Good (English)", "NYCEM [English]", "g-good", "good.xml"),
        _item("Bad (English)", "NYCEM [English]", "g-bad", "bad.xml"),
    )
    caps = {"good.xml": CAP_ACTIVE, "bad.xml": "<alert><info>oops truncated"}
    client = _client(rss, caps)
    advisories = (await fetch_advisories(client)).advisories
    await client.aclose()

    assert [a.guid for a in advisories] == ["NYC-ACTIVE-1"]  # malformed one silently dropped


async def test_fetch_advisories_returns_empty_on_rss_failure():
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    feed = await fetch_advisories(client)
    assert feed.advisories == []
    assert feed.confirmed is False  # a failed fetch is DEGRADED, not a confirmed all-clear
    await client.aclose()


# --- the module tool -------------------------------------------------------

async def test_check_notify_nyc_grounds_and_cites_active():
    citations = CitationRegistry()
    client = _client(RSS_MAIN, CAPS_MAIN)
    ctx = ToolContext(citations=citations, registry=Registry([]), http=client)
    out = await get_tools()[0].handler({}, ctx)
    await client.aclose()

    assert "Heat Advisory in effect for NYC" in out
    assert "Severe" in out
    assert "in effect until 2099-07-02T19:45:28-04:00" in out  # expiry stated as-of
    assert "{cite:S1}" in out                                  # grounded, cited
    assert "Expired advisory" not in out                       # expired never surfaced

    mapping = citations.mapping()
    assert mapping["S1"]["kind"] == "DATA"
    assert "everbridge.net" in mapping["S1"]["url"]             # resolvable CAP XML url
    assert mapping["S1"]["url"].endswith("/cap/active.xml")
    assert mapping["S1"]["valid_as_of"] == "2026-07-02T12:00:00-04:00"  # = sent, never fetch time
    assert mapping["S1"]["provenance"]["record_id"] == "NYC-ACTIVE-1"


async def test_check_notify_nyc_abstains_when_none_active():
    # Feed carries only an already-expired advisory → nothing active → clean abstention.
    rss = _rss(_item("Old (English)", "NYCEM [English]", "g-old", "expired.xml"))
    citations = CitationRegistry()
    client = _client(rss, {"expired.xml": CAP_EXPIRED})
    ctx = ToolContext(citations=citations, registry=Registry([]), http=client)
    out = await get_tools()[0].handler({"near": "Astoria, Queens"}, ctx)
    await client.aclose()

    low = out.lower()
    assert "no active" in low                     # says there are none
    assert "- " not in out                        # no fabricated advisory list
    assert len(citations) == 0                     # nothing fabricated/cited
    # No fabricated counts of "other" advisories beyond the feed.
    assert "notifynyc" in low or "311" in low     # routed to the official source


# --- SAFETY-CRITICAL fail-safe: a degraded feed must NEVER read as "all clear" ---
# The dangerous production bug: the Everbridge RSS returned HTTP 200 with an EMPTY body (zero items)
# mid-emergency, and the tool confidently reported "no active Notify NYC advisories". A feed that is
# unreachable, errored, empty, or unreadable is NOT a confirmed all-clear. It must say we COULD NOT
# confirm the advisories and route to the official live source + 311 (+ 911 for a life-threat).

async def test_check_notify_nyc_failsafe_on_empty_feed():
    # RSS reached (HTTP 200) but carries ZERO <item>s, the exact live failure. Must not claim clear.
    empty_rss = _rss()  # a well-formed RSS with no items at all
    citations = CitationRegistry()
    client = _client(empty_rss, {})
    ctx = ToolContext(citations=citations, registry=Registry([]), http=client)
    out = await get_tools()[0].handler({}, ctx)
    await client.aclose()

    low = out.lower()
    # It must NOT assert a confirmed all-clear.
    assert "no active notify nyc advisories" not in low
    assert "no advisories" not in low
    # It must say it could not confirm, and route to the official live source + 311 + 911.
    assert "could not confirm" in low
    assert "nyc.gov/notifynyc" in low
    assert "311" in low
    assert "911" in low
    assert len(citations) == 0  # nothing fabricated/cited


async def test_check_notify_nyc_failsafe_on_fetch_error():
    # RSS fetch fails outright (HTTP 500 / non-200). Same fail-safe: never a confident "no advisories".
    citations = CitationRegistry()
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    ctx = ToolContext(citations=citations, registry=Registry([]), http=client)
    out = await get_tools()[0].handler({}, ctx)
    await client.aclose()

    low = out.lower()
    assert "no active notify nyc advisories" not in low
    assert "no advisories" not in low
    assert "could not confirm" in low
    assert "nyc.gov/notifynyc" in low
    assert "311" in low
    assert "911" in low


async def test_active_advisories_flags_confirmed_vs_degraded():
    # The seam the handler relies on: a working feed with an all-expired set is a CONFIRMED all-clear
    # (confirmed=True, advisories=[]); an empty/unreachable feed is DEGRADED (confirmed=False).
    now = datetime(2026, 7, 2, 18, 0, tzinfo=timezone.utc)

    # Working feed, only an expired advisory → confirmed True, nothing active.
    reached = _client(_rss(_item("Old (English)", "NYCEM [English]", "g-old", "expired.xml")),
                      {"expired.xml": CAP_EXPIRED})
    feed = await active_advisories(reached, now=now)
    await reached.aclose()
    assert feed.confirmed is True
    assert feed.advisories == []

    # Empty feed (zero items) → degraded, could not confirm.
    empty = _client(_rss(), {})
    feed = await active_advisories(empty, now=now)
    await empty.aclose()
    assert feed.confirmed is False
    assert feed.advisories == []


# --- REAL-TIME source: the live Notify NYC /Home/RecentMessages endpoint -----
# Captured 2026-07-06 from https://a858-nycnotify.nyc.gov/notifynyc/Home/RecentMessages, the JSON
# the city's OWN Notify NYC portal renders. While the CAP/Everbridge feed sat empty mid-emergency,
# THIS endpoint carried the day's real alerts (Flood Advisory, Safe Overnight Locations: Flooding).
# Schema is thin: pubDate (MM/DD/YYYY HH:MM:SS, America/New_York), title, description (HTML-ish).
RECENT_MESSAGES_JSON = json.dumps([
    {"pubDate": "07/06/2026 00:33:21",
     "title": "Notify NYC - Mass Transit Restoration",
     "description": ("Notification issued 07-06-2026 at 12:33 AM.\n\nStaten Island Railway service "
                     "has resumed in both directions. Expect residual delays.\n\nTo view this "
                     "message in ASL, Español : "
                     "<a target='_new' href='http://on.nyc.gov/2qxH91g'>http://on.nyc.gov/2qxH91g</a>.")},
    {"pubDate": "07/05/2026 23:31:01",
     "title": "Notify NYC - Flood Advisory - 7/5 - 7/6 (NYC)",
     "description": ("Notification issued 07-05-2026 at 11:31 PM.\n\nA Flood Advisory is in effect "
                     "for NYC until 6:00 AM Monday. Avoid flooded roadways and never drive through "
                     "standing water.\n\nTo view this message in ASL : "
                     "<a target='_new' href='http://on.nyc.gov/flood'>http://on.nyc.gov/flood</a>.")},
    {"pubDate": "07/05/2026 22:13:35",
     "title": "Notify NYC - Safe Overnight Locations: Flooding (NYC)",
     "description": "Notification issued 07-05-2026 at 10:13 PM.\n\nSafe overnight locations are open due to flooding."},
])


def _recent_client(json_text: str) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.lower().endswith("/home/recentmessages"):
            return httpx.Response(200, text=json_text, headers={"content-type": "application/json"})
        return httpx.Response(404)
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _combo_client(rss: str, recent_json: str, caps: dict[str, str] | None = None) -> httpx.AsyncClient:
    """One client serving the CAP RSS, its CAP XMLs, AND the live RecentMessages endpoint."""
    caps = caps or {}
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/rss/rss.xml"):
            return httpx.Response(200, text=rss, headers={"content-type": "application/rss+xml"})
        if "/cap/" in path:
            name = path.rsplit("/", 1)[-1]
            if name in caps:
                return httpx.Response(200, text=caps[name], headers={"content-type": "application/xml"})
        if path.lower().endswith("/home/recentmessages"):
            return httpx.Response(200, text=recent_json, headers={"content-type": "application/json"})
        return httpx.Response(404)
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_fetch_recent_advisories_parses_live_format():
    client = _recent_client(RECENT_MESSAGES_JSON)
    feed = await fetch_recent_advisories(client)
    await client.aclose()

    assert feed.confirmed is True
    assert len(feed.notes) == 3
    # Newest first (07/06 00:33 leads 07/05 23:31 leads 07/05 22:13).
    assert feed.notes[0].title == "Notify NYC - Mass Transit Restoration"

    flood = next(n for n in feed.notes if "Flood Advisory" in n.title)
    assert flood.issued == "2026-07-05T23:31:01-04:00"   # pubDate parsed as ET → ISO 8601 w/ offset
    assert flood.issued_raw == "07/05/2026 23:31:01"     # raw kept for honest display
    assert "Flood Advisory is in effect" in flood.body
    assert "<a" not in flood.body and "href" not in flood.body  # HTML tags stripped from body
    assert "on.nyc.gov/flood" in flood.body               # the translation URL text is preserved
    assert flood.source_url == RECENT_MESSAGES_URL
    assert flood.guid                                     # a stable dedupe key exists


async def test_fetch_recent_advisories_failsafe_on_error():
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(503)))
    feed = await fetch_recent_advisories(client)
    await client.aclose()
    assert feed.confirmed is False
    assert feed.notes == []


async def test_fetch_recent_advisories_empty_array_is_degraded():
    # Reached but carrying nothing readable is NOT a confirmed all-clear.
    client = _recent_client("[]")
    feed = await fetch_recent_advisories(client)
    await client.aclose()
    assert feed.confirmed is False


async def test_current_awareness_reuses_a_recent_snapshot(monkeypatch):
    calls = 0

    async def loader():
        nonlocal calls
        calls += 1
        return notify_nyc.RecentFeed(confirmed=True, notes=[])

    monkeypatch.setattr(advisory_tools, "fetch_recent_advisories", loader)
    monkeypatch.setattr(advisory_tools, "_awareness_cache", None, raising=False)
    await advisory_tools.current_awareness()
    await advisory_tools.current_awareness()

    assert calls == 1


async def test_current_awareness_does_not_cache_a_degraded_fetch(monkeypatch):
    calls = 0

    async def loader():
        nonlocal calls
        calls += 1
        return notify_nyc.RecentFeed(confirmed=calls > 1, notes=[])

    monkeypatch.setattr(advisory_tools, "fetch_recent_advisories", loader)
    monkeypatch.setattr(advisory_tools, "_awareness_cache", None, raising=False)
    await advisory_tools.current_awareness()
    await advisory_tools.current_awareness()

    assert calls == 2


async def test_current_awareness_coalesces_concurrent_fetches(monkeypatch):
    calls = 0

    async def loader():
        nonlocal calls
        calls += 1
        await asyncio.sleep(0)
        return notify_nyc.RecentFeed(confirmed=True, notes=[])

    monkeypatch.setattr(advisory_tools, "fetch_recent_advisories", loader)
    monkeypatch.setattr(advisory_tools, "_awareness_cache", None, raising=False)
    await asyncio.gather(
        advisory_tools.current_awareness(),
        advisory_tools.current_awareness(),
    )

    assert calls == 1


async def test_incidental_check_with_nothing_active_returns_nothing_to_narrate():
    """F067: a preparation turn's forced advisories check came back empty and the model narrated
    the null result ("One important update: ... no active advisories") as if it were news. The
    agent tags its own forced checks `incidental`; an incidental check finding nothing returns
    an EMPTY result, so there is nothing to narrate. A resident who asked still gets NO_ACTIVE."""
    rss = _rss(_item("Old (English)", "NYCEM [English]", "g-old", "expired.xml"))
    client = _client(rss, {"expired.xml": CAP_EXPIRED})
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]), http=client)
    out = await get_tools()[0].handler({"incidental": True}, ctx)
    await client.aclose()
    assert out == ""


def test_recent_awareness_carries_notice_text_without_scope_parsing():
    """F061 (RULED 2026-07-18): no deterministic parsing of the feed's area formatting — today's
    five flood notices spelled areas as "BK/SI/MN/QN", "BK, QN", "parts of NYC", and "The Bronx",
    and any pattern keyed to that spelling breaks the day the feed changes it. The index carries
    each notice's full text and the MODEL judges scope and severity from meaning."""
    feed = notify_nyc.RecentFeed(confirmed=True, notes=[
        notify_nyc.RecentNote(
            title="Notify NYC - Flash Flood Warning",
            body="Flash Flood Warning for BK, QN til 3:30 PM. Move to higher ground.",
            issued="2026-07-16T13:25:40-04:00", issued_raw="07/16/2026 13:25:40",
            source_url=RECENT_MESSAGES_URL, guid="ffw",
        ),
    ])

    text = advisory_tools._recent_awareness(feed, date(2026, 7, 16))

    assert "Flash Flood Warning for BK, QN til 3:30 PM. Move to higher ground." in text
    assert "broad scope confirmed" not in text


def test_recent_awareness_carries_full_text_and_fail_open_safety_policy():
    """F061 (RULED 2026-07-18): the model, not a regex, decides what to surface — so the index
    carries each notice's full text, safety notices fail OPEN with their area stated even when
    no resident location is known, and only narrow low-stakes notices keep the overlap rule."""
    render = getattr(advisory_tools, "_recent_awareness", None)
    assert callable(render)
    feed = notify_nyc.RecentFeed(confirmed=True, notes=[
        notify_nyc.RecentNote(
            title="Notify NYC - Air Quality Health Advisory (NYC)",
            body="Air quality is unhealthy for everyone in all or part of NYC.",
            issued="2026-07-16T08:59:23-04:00", issued_raw="07/16/2026 08:59:23",
            source_url=RECENT_MESSAGES_URL, guid="aqi",
        ),
        notify_nyc.RecentNote(
            title="Notify NYC - FDNY Activity (BK)", body="full local body",
            issued="2026-07-16T13:25:40-04:00", issued_raw="07/16/2026 13:25:40",
            source_url=RECENT_MESSAGES_URL, guid="fdny",
        ),
        notify_nyc.RecentNote(
            title="Notify NYC - Old notice", body="old body",
            issued="2026-07-15T13:25:40-04:00", issued_raw="07/15/2026 13:25:40",
            source_url=RECENT_MESSAGES_URL, guid="old",
        ),
    ])

    text = render(feed, date(2026, 7, 16))

    assert "Air Quality Health Advisory" in text
    assert "FDNY Activity" in text
    assert "Old notice" in text
    assert "unhealthy for everyone in all or part of NYC" in text  # full text, model judges meaning
    assert "call `check_notify_nyc`" in text                         # cite discipline survives
    assert "immediate personal safety" in text                     # safety notices fail open
    assert "hasn't shared a location" in text
    assert "known location" in text                                # narrow class keeps overlap rule
    assert "already ended" in text                                 # expiry checked against now


def test_recent_awareness_keeps_seven_days_without_truncating_bodies():
    long_body = "A" * 450
    feed = notify_nyc.RecentFeed(confirmed=True, notes=[
        notify_nyc.RecentNote(
            title="Two day old notice", body=long_body,
            issued="2026-07-14T08:00:00-04:00", issued_raw="07/14/2026 08:00:00",
            source_url=RECENT_MESSAGES_URL, guid="recent-2d",
        ),
        notify_nyc.RecentNote(
            title="Eight day old notice", body="expired",
            issued="2026-07-08T08:00:00-04:00", issued_raw="07/08/2026 08:00:00",
            source_url=RECENT_MESSAGES_URL, guid="recent-8d",
        ),
    ])

    text = advisory_tools._recent_awareness(feed, date(2026, 7, 16))

    assert "Two day old notice" in text
    assert long_body in text
    assert "Eight day old notice" not in text


async def test_current_awareness_retains_recent_cache_when_refresh_degrades(monkeypatch):
    good = notify_nyc.RecentFeed(confirmed=True, notes=[
        notify_nyc.RecentNote(
            title="Cached notice", body="Exact cached message body",
            issued=datetime.now(advisory_tools.NYC_TZ).isoformat(),
            issued_raw="08/12/2026 01:00:00",
            source_url=RECENT_MESSAGES_URL, guid="cached-notice",
        )
    ])
    degraded = notify_nyc.RecentFeed(confirmed=False, notes=[])
    feeds = iter((good, degraded))

    async def fake_fetch():
        return next(feeds)

    monkeypatch.setattr(advisory_tools, "fetch_recent_advisories", fake_fetch)
    monkeypatch.setattr(advisory_tools, "_awareness_cache", None)
    first = await advisory_tools.current_awareness()
    cached_at, notes = advisory_tools._awareness_cache
    monkeypatch.setattr(
        advisory_tools, "_awareness_cache",
        (cached_at - advisory_tools._AWARENESS_TTL_S - 1, notes),
    )

    second = await advisory_tools.current_awareness()

    assert "Exact cached message body" in first
    assert "Exact cached message body" in second
    assert "refresh failed" in second.lower()


async def test_check_notify_nyc_falls_back_to_recent_when_cap_empty():
    # THE production scenario: the CAP/Everbridge feed is EMPTY, but Notify NYC IS carrying today's
    # flood alerts. The tool must surface them (grounded + cited), never fail safe or go silent.
    citations = CitationRegistry()
    client = _combo_client(rss=_rss(), recent_json=RECENT_MESSAGES_JSON)  # empty CAP + live recent
    ctx = ToolContext(citations=citations, registry=Registry([]), http=client)
    out = await get_tools()[0].handler({}, ctx)
    await client.aclose()

    low = out.lower()
    assert "flood advisory" in low                        # today's real alert surfaced
    assert "could not confirm" not in low                 # we DID confirm via the live source
    assert "no active notify nyc advisories" not in low   # not a false all-clear
    assert len(citations) >= 1                             # grounded + cited
    assert any("recentmessages" in c["url"].lower() for c in citations.mapping().values())
    flood = next(c for c in citations.mapping().values() if "Flood Advisory" in c["title"])
    assert "Avoid flooded roadways" in flood["snippet"]


async def test_recent_fallback_dedups_titles_already_delivered_this_conversation():
    delivered = frozenset(
        item["title"].casefold() for item in json.loads(RECENT_MESSAGES_JSON)
    )
    client = _combo_client(rss=_rss(), recent_json=RECENT_MESSAGES_JSON)
    ctx = ToolContext(
        citations=CitationRegistry(),
        registry=Registry([]),
        http=client,
        delivered_notify_titles=delivered,
    )

    out = await get_tools()[0].handler({}, ctx)
    await client.aclose()

    assert "Nothing new" in out
    assert "Do not re-brief" in out
    assert "Avoid flooded roadways" not in out


async def test_check_notify_nyc_prefers_cap_when_it_has_active():
    # When the CAP feed is working AND has an active advisory, it stays the source (structured,
    # with severity + expiry); the RecentMessages fallback is not needed.
    citations = CitationRegistry()
    client = _combo_client(rss=RSS_MAIN, recent_json=RECENT_MESSAGES_JSON, caps=CAPS_MAIN)
    # RSS_MAIN's active CAP expires 2099, so it's active for any real "now".
    ctx = ToolContext(citations=citations, registry=Registry([]), http=client)
    out = await get_tools()[0].handler({}, ctx)
    await client.aclose()
    assert "Heat Advisory in effect for NYC" in out       # the structured CAP advisory
    assert "in effect until 2099-07-02T19:45:28-04:00" in out


async def test_check_notify_nyc_combines_cap_and_recent_without_exact_duplicates():
    fireworks_title = "Notify NYC - Fireworks - 7/17 - Coney Island Beach (BK)"
    rss = _rss(_item(fireworks_title, "NYCEM [English]", "g-fireworks", "fireworks.xml"))
    cap = _cap(
        "NYC-FIREWORKS", severity="Moderate", event="Civil Emergency Message",
        headline=fireworks_title, expires="2099-07-17T22:00:00-04:00",
    )
    recent_json = json.dumps([
        {
            "pubDate": "07/16/2026 15:05:41", "title": fireworks_title,
            "description": "Fireworks near Coney Island Beach.",
        },
        {
            "pubDate": "07/16/2026 08:59:23",
            "title": "Notify NYC - Air Quality Health Advisory (AQI 151-200) - 7/16",
            "description": "Air quality is unhealthy for everyone in all or part of NYC.",
        },
    ])
    client = _combo_client(rss=rss, recent_json=recent_json, caps={"fireworks.xml": cap})
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]), http=client)

    out = await get_tools()[0].handler({}, ctx)
    await client.aclose()

    assert "Air Quality Health Advisory" in out
    assert out.count(fireworks_title) == 1


async def test_check_notify_nyc_reports_borough_notices_alongside_citywide():
    """F061 (RULED 2026-07-18): the tool never area-filters by parsing notice prose — today's
    borough-list flood warnings would have been filtered OUT of a "citywide" view. Every active
    notice returns, cited, and the model judges relevance from the full text."""
    recent_json = json.dumps([
        {
            "pubDate": "07/16/2026 08:59:23",
            "title": "Notify NYC - Air Quality Health Advisory (AQI 151-200) - 7/16",
            "description": "Air quality is unhealthy for everyone in all or part of NYC.",
        },
        {
            "pubDate": "07/16/2026 08:27:06",
            "title": "Notify NYC - Three Alarm Fire - Briggs Avenue (BX)",
            "description": "Emergency personnel are on scene in the Bronx.",
        },
    ])
    client = _combo_client(rss=_rss(), recent_json=recent_json)
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]), http=client)

    out = await get_tools()[0].handler({"citywide_only": True}, ctx)  # stale arg is ignored
    await client.aclose()

    assert "Air Quality Health Advisory" in out
    assert "Three Alarm Fire" in out
    schema = get_tools()[0].parameters["properties"]
    assert "citywide_only" not in schema


async def test_advisory_results_keep_non_overlapping_notices_out_of_plan_answers():
    client = _combo_client(rss=_rss(), recent_json=RECENT_MESSAGES_JSON)
    ctx = ToolContext(
        citations=CitationRegistry(),
        registry=Registry([]),
        http=client,
    )

    out = await get_tools()[0].handler({"near": "Flushing Meadows Corona Park"}, ctx)
    await client.aclose()

    assert "Do not enumerate notices that clearly do not overlap" in out
    assert "name the date and place you checked" in out
    assert "list these as-is" not in out


# --- the shipped module stays valid (mirrors test_module_food_pantries) ----

def test_advisories_module_loads_with_tool_and_eval():
    registry = Registry.discover(config.MODULES_DIR)
    module = next((m for m in registry.modules if m.name == "advisories"), None)
    assert module is not None
    assert module.category == "alerts"
    tool_names = {t.name for t in registry.load_module_tools()}
    assert "check_notify_nyc" in tool_names

    from heynyc.eval.cases import load_cases
    cases = [c for c in load_cases(registry) if c.module == "advisories"]
    assert cases, "advisories should ship eval cases"
    assert any(c.invariants.get("must_abstain_or_redirect") for c in cases)
    assert any(c.harm_category == "injection" for c in cases)


# --- F080 residual: repeat advisory calls in one conversation return a marker, not a re-brief ---

CAP_FLOOD = _cap("NYC-FLOOD-1", severity="Moderate", event="Flood Watch",
                 headline="Flood Watch for Queens", expires="2099-08-01T10:00:00-04:00")


async def test_check_notify_nyc_dedups_titles_already_delivered_this_conversation():
    """F080: luna re-fetched by choice and re-briefed a near-verbatim repeat. When every
    active advisory was already cited earlier in the conversation, the tool returns a
    compact already-shared marker instead of the payload, so there is nothing to re-brief."""
    citations = CitationRegistry()
    client = _client(RSS_MAIN, CAPS_MAIN)
    ctx = ToolContext(citations=citations, registry=Registry([]), http=client,
                      delivered_notify_titles=frozenset({"heat advisory in effect for nyc"}))
    out = await get_tools()[0].handler({}, ctx)
    await client.aclose()

    assert "unchanged" in out.lower()
    assert "do not re-brief" in out.lower()
    assert "full_text" in out                                # the explicit escape hatch is named
    assert "in effect until" not in out                      # full payload withheld
    # F083: the marker carries a PER-ITEM citation so a legitimate refer-back binds to the
    # right source; a citation-free marker pushed the model to pin remembered facts on
    # whatever fresh cite was at hand (observed live: waterbody claims cited to Belt Parkway).
    assert "{cite:S1}" in out
    assert citations.mapping()["S1"]["title"] == "Heat Advisory in effect for NYC"


async def test_check_notify_nyc_full_text_repeats_on_explicit_request():
    citations = CitationRegistry()
    client = _client(RSS_MAIN, CAPS_MAIN)
    ctx = ToolContext(citations=citations, registry=Registry([]), http=client,
                      delivered_notify_titles=frozenset({"heat advisory in effect for nyc"}))
    out = await get_tools()[0].handler({"full_text": True}, ctx)
    await client.aclose()

    assert "Heat Advisory in effect for NYC" in out
    assert "{cite:S1}" in out


async def test_check_notify_nyc_delivers_only_the_new_advisory():
    """A genuinely NEW advisory still arrives in full with its citation; the already-shared
    one shrinks to a do-not-re-brief mention."""
    rss = _rss(
        _item("Heat (English)", "NYCEM [English]", "g-heat", "active.xml"),
        _item("Flood (English)", "NYCEM [English]", "g-flood", "flood.xml"),
    )
    caps = {"active.xml": CAP_ACTIVE, "flood.xml": CAP_FLOOD}
    citations = CitationRegistry()
    client = _client(rss, caps)
    ctx = ToolContext(citations=citations, registry=Registry([]), http=client,
                      delivered_notify_titles=frozenset({"heat advisory in effect for nyc"}))
    out = await get_tools()[0].handler({}, ctx)
    await client.aclose()

    assert "Flood Watch for Queens" in out
    assert "{cite:S1}" in out                                # the new one is cited in full
    assert "2099-08-01T10:00:00-04:00" in out                # with its own payload
    assert "2099-07-02T19:45:28-04:00" not in out            # the old one's payload withheld
    assert "already shared" in out.lower()
    assert "Heat Advisory in effect for NYC" in out          # named, not re-briefed
    # F083: the still-active mention carries its own citation for correct refer-backs.
    assert "{cite:S2}" in out
    titles = {c["title"] for c in citations.mapping().values()}
    assert "Heat Advisory in effect for NYC" in titles


async def test_check_notify_nyc_empty_delivered_set_changes_nothing():
    citations = CitationRegistry()
    client = _client(RSS_MAIN, CAPS_MAIN)
    ctx = ToolContext(citations=citations, registry=Registry([]), http=client)
    out = await get_tools()[0].handler({}, ctx)
    await client.aclose()
    assert "Heat Advisory in effect for NYC" in out and "{cite:S1}" in out
