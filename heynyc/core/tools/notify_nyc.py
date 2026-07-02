"""Notify NYC advisories client — the city's emergency-alert feed (Everbridge / CAP).

Notify NYC (run by NYC Emergency Management) publishes emergency advisories — extreme heat,
air quality, boil-water notices, beach/pool closures, transit disruptions — as a public RSS feed
of CAP (Common Alerting Protocol 1.2) alerts on Everbridge. This adapter is the injectable seam
that mirrors `arcgis.query_feature_service`: pass an httpx client so tests stay fully offline.

The feed is two hops:
  1. RSS (`RSS_URL`, must follow redirects) — ~64 `<item>`s, one per (alert × language). Each item's
     `<author>` tags the language ("NYCEM [English]" / "NYCEM [Spanish]" / …); we keep only English.
     The CAP XML url is the item's `<enclosure url="...">` (fallback: `<link>` text).
  2. Each CAP XML (namespace `urn:oasis:names:tc:emergency:cap:1.2`) carries the structured alert:
     event/severity/urgency/category, `sent`/`expires` (ISO 8601 with tz offset), headline, and an
     `<area>` `areaDesc` — a comma-separated list of borough names that is OFTEN all five (a citywide
     default even for a local event), so it must NOT be over-trusted for geo filtering.

We parse namespace-AGNOSTICALLY (by local tag name) so a namespace-prefix change never breaks it,
and stay resilient: any network/parse error yields an empty list so the caller abstains rather than
fabricating — this feed never invents an advisory. Verified live 2026-07-02.
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
from xml.etree import ElementTree as ET

import httpx

# The Notify NYC Everbridge RSS feed. It 302-redirects, so callers must follow redirects.
RSS_URL = "https://feeds.everbridge.net/feeds/453003085617722/rss/rss.xml"

# CAP severity → sort rank (most severe first). Unknown / anything else sorts last.
_SEVERITY_RANK = {"Extreme": 0, "Severe": 1, "Moderate": 2, "Minor": 3}
_UNKNOWN_RANK = 4


@dataclass
class Advisory:
    headline: str
    event: str
    category: str
    severity: str
    urgency: str
    sent: str
    expires: str
    area_desc: str
    source_url: str
    guid: str


# A DTD/entity declaration in the prolog — the prerequisite for both the "billion laughs"
# entity-expansion attack and XXE. Legitimate RSS/CAP never uses one.
_DTD_RE = re.compile(r"<!\s*(?:DOCTYPE|ENTITY)\b", re.IGNORECASE)


def _safe_fromstring(xml_text: str) -> ET.Element:
    """Parse XML after refusing any DTD / entity declaration.

    This feed is external, and the stdlib parser is by default vulnerable to the "billion laughs"
    entity-expansion and XXE attacks — both of which require a DTD (`<!DOCTYPE … [ <!ENTITY …> ]>`).
    Rather than add a `defusedxml` dependency, we reject a DTD before parsing (a portable equivalent
    of `defusedxml`'s `forbid_dtd`). A stray external entity reference without a declaration is
    undefined and makes expat error out on its own — so the parser can never expand a bomb. Every
    caller fails safe (None / []) on the raised error.
    """
    if _DTD_RE.search(xml_text):
        raise ValueError("XML DTDs / entity declarations are not allowed")
    return ET.fromstring(xml_text)


def _local(tag: str) -> str:
    """Strip an XML namespace from a tag: '{urn:…:cap:1.2}info' → 'info' (bare tags pass through)."""
    return tag.rsplit("}", 1)[-1]


def _first_text(root: ET.Element, name: str) -> str:
    """The stripped text of the first element anywhere under `root` whose local tag is `name`, else ''.

    Namespace-agnostic and layout-tolerant: CAP places `sent` at the alert level and most fields
    under `<info>`, but each of these single-language CAP files has one `<info>`, so a first-match
    search returns the right value regardless of exact nesting.
    """
    for elem in root.iter():
        if _local(elem.tag) == name:
            return (elem.text or "").strip()
    return ""


def _direct_child_text(parent: ET.Element, name: str) -> str:
    """The stripped text of the first DIRECT child of `parent` with local tag `name`, else ''."""
    for child in parent:
        if _local(child.tag) == name:
            return (child.text or "").strip()
    return ""


def _parse_cap(xml_text: str, source_url: str) -> Optional[Advisory]:
    """Parse one CAP XML into an Advisory. Returns None on any parse failure — never raises.

    `source_url` is the resolvable CAP XML url (a DATA citation can re-fetch it). We require an
    `<info>` block and a non-empty `expires` (without it there's no active-window to reason about).
    The alert `<identifier>` is used as the dedupe guid.
    """
    try:
        root = _safe_fromstring(xml_text)
        info = next((e for e in root.iter() if _local(e.tag) == "info"), None)
        if info is None:
            return None
        expires = _first_text(root, "expires")
        if not expires:
            return None
        return Advisory(
            headline=_first_text(root, "headline"),
            event=_first_text(root, "event"),
            category=_first_text(root, "category"),
            severity=_first_text(root, "severity"),
            urgency=_first_text(root, "urgency"),
            sent=_first_text(root, "sent"),
            expires=expires,
            area_desc=_first_text(root, "areaDesc"),
            source_url=source_url,
            guid=_first_text(root, "identifier"),
        )
    except Exception:
        return None


def _english_cap_urls(rss_text: str) -> list[str]:
    """The CAP XML urls of the English items in the RSS feed (enclosure `url` attr, else `<link>`)."""
    try:
        root = _safe_fromstring(rss_text)
    except Exception:
        return []
    urls: list[str] = []
    for item in root.iter():
        if _local(item.tag) != "item":
            continue
        if "[English]" not in _direct_child_text(item, "author"):
            continue
        enclosure = next((c for c in item if _local(c.tag) == "enclosure"), None)
        url = (enclosure.get("url") if enclosure is not None else "") or _direct_child_text(item, "link")
        if url:
            urls.append(url)
    return urls


async def _fetch_cap(client: httpx.AsyncClient, url: str) -> Optional[Advisory]:
    """Fetch one CAP XML and parse it. Raises on HTTP failure (gather tolerates it)."""
    response = await client.get(url, follow_redirects=True)
    response.raise_for_status()
    return _parse_cap(response.text, url)


async def fetch_advisories(client: Optional[httpx.AsyncClient] = None) -> list[Advisory]:
    """Fetch the Notify NYC feed and return the English advisories (deduped).

    GETs the RSS (following redirects), keeps English items, fetches their CAP XMLs CONCURRENTLY
    (tolerating individual failures), parses each, and dedupes by guid (fallback headline+sent).
    Inject `client` to mock the HTTP calls offline, exactly like `arcgis.query_feature_service`.
    Any network/parse error yields `[]` so the caller abstains — this never crashes.
    """
    own_client = client is None
    client = client or httpx.AsyncClient(timeout=20.0)
    try:
        response = await client.get(RSS_URL, follow_redirects=True)
        response.raise_for_status()
        cap_urls = _english_cap_urls(response.text)
        results = await asyncio.gather(
            *(_fetch_cap(client, url) for url in cap_urls), return_exceptions=True
        )
    except Exception:
        return []
    finally:
        if own_client:
            await client.aclose()

    advisories: list[Advisory] = []
    seen: set[str] = set()
    for result in results:
        if not isinstance(result, Advisory):
            continue  # a failed/None CAP — tolerated, skipped
        key = result.guid or f"{result.headline}|{result.sent}"
        if key in seen:
            continue
        seen.add(key)
        advisories.append(result)
    return advisories


def _sent_key(advisory: Advisory) -> datetime:
    """Parse `sent` for sorting; unparseable → epoch (sorts oldest/last within a severity tier)."""
    try:
        return datetime.fromisoformat(advisory.sent)
    except (ValueError, TypeError):
        return datetime.min.replace(tzinfo=timezone.utc)


async def active_advisories(client: Optional[httpx.AsyncClient], now: datetime) -> list[Advisory]:
    """The advisories still in effect at `now` (a tz-aware datetime), most severe / most recent first.

    Keeps advisories whose `expires` (ISO 8601 WITH tz offset, parsed by `datetime.fromisoformat`)
    is strictly after `now`. Sorts by severity rank (Extreme→Severe→Moderate→Minor→Unknown) then
    `sent` descending. Resilient: a fetch failure yields `[]` (via `fetch_advisories`).
    """
    advisories = await fetch_advisories(client)
    active: list[Advisory] = []
    for advisory in advisories:
        try:
            expires = datetime.fromisoformat(advisory.expires)
        except (ValueError, TypeError):
            continue
        if expires.tzinfo is None or expires <= now:
            continue
        active.append(advisory)
    # Two-pass stable sort: sent-descending first, then severity rank ascending.
    active.sort(key=_sent_key, reverse=True)
    active.sort(key=lambda a: _SEVERITY_RANK.get(a.severity, _UNKNOWN_RANK))
    return active
