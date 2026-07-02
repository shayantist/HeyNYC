"""Offline tests for the Notify NYC advisories client + the advisories module tool.

Every HTTP call is mocked via httpx.MockTransport, routed by url path (`/rss/rss.xml` for the feed,
`/cap/<id>.xml` for each CAP alert) — no live Everbridge call. Covers: English-only filtering, CAP
field parsing, active-window filtering + severity sort, malformed-CAP tolerance, and the module
tool's grounding+citation / clean abstention. The shipped module load mirrors test_module_food_pantries.
"""
from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest

from heynyc.core import config
from heynyc.core.citations import CitationRegistry
from heynyc.core.registry import Registry
from heynyc.core.tools.base import ToolContext
from heynyc.core.tools.notify_nyc import (
    active_advisories,
    fetch_advisories,
)
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
    advisories = await fetch_advisories(client)
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


async def test_active_advisories_excludes_expired():
    now = datetime(2026, 7, 2, 18, 0, tzinfo=timezone.utc)
    client = _client(RSS_MAIN, CAPS_MAIN)
    active = await active_advisories(client, now=now)
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
    active = await active_advisories(client, now=now)
    await client.aclose()

    assert [a.severity for a in active] == ["Extreme", "Severe"]  # most severe first


async def test_malformed_cap_is_skipped_not_fatal():
    rss = _rss(
        _item("Good (English)", "NYCEM [English]", "g-good", "good.xml"),
        _item("Bad (English)", "NYCEM [English]", "g-bad", "bad.xml"),
    )
    caps = {"good.xml": CAP_ACTIVE, "bad.xml": "<alert><info>oops truncated"}
    client = _client(rss, caps)
    advisories = await fetch_advisories(client)
    await client.aclose()

    assert [a.guid for a in advisories] == ["NYC-ACTIVE-1"]  # malformed one silently dropped


async def test_fetch_advisories_returns_empty_on_rss_failure():
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    assert await fetch_advisories(client) == []
    await client.aclose()


# --- the module tool -------------------------------------------------------

async def test_nyc_advisories_grounds_and_cites_active():
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


async def test_nyc_advisories_abstains_when_none_active():
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


# --- the shipped module stays valid (mirrors test_module_food_pantries) ----

def test_advisories_module_loads_with_tool_and_eval():
    registry = Registry.discover(config.MODULES_DIR)
    module = next((m for m in registry.modules if m.name == "advisories"), None)
    assert module is not None
    assert module.category == "alerts"
    tool_names = {t.name for t in registry.load_module_tools()}
    assert "nyc_advisories" in tool_names

    from heynyc.eval.cases import load_cases
    cases = [c for c in load_cases(registry) if c.module == "advisories"]
    assert cases, "advisories should ship eval cases"
    assert any(c.invariants.get("must_abstain_or_redirect") for c in cases)
    assert any(c.harm_category == "injection" for c in cases)
