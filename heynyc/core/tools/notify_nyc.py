"""Notify NYC advisories client — the city's emergency-alert feed (Everbridge / CAP).

Notify NYC (run by NYC Emergency Management) publishes emergency advisories — extreme heat,
air quality, boil-water notices, beach/pool closures, transit disruptions — as a public RSS feed
of CAP (Common Alerting Protocol 1.2) alerts on Everbridge. This adapter is the injectable seam
that mirrors `arcgis.query_feature_service`: pass an httpx client so tests stay fully offline.

The feed is two hops:
  1. RSS (`RSS_URL`, must follow redirects) — ~64 `<item>`s, one per (alert × language). Each item's
     `<author>` tags the language ("NYCEM [English]" / "NYCEM [Spanish]" / …). We RETAIN every
     language variant the feed carries (~12 official languages besides English) so a caller can
     request an advisory in the user's language, falling back to the official English text when that
     language has no variant of a given alert. The CAP XML url is the item's `<enclosure url="...">`
     (fallback: `<link>` text). An official city translation beats an LLM paraphrase, so we surface it.
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
    language: str = ""  # CAP <language> code (e.g. "en-US", "es-US"); blank if the feed omits it


# The item <author> tags language in brackets: "NYCEM [English]", "NYCEM [Spanish]", … We key on
# that human-readable NAME (lowercased) because it's what the RSS actually carries per item.
_LANG_RE = re.compile(r"\[([^\]]+)\]")
DEFAULT_LANGUAGE = "english"  # the base + fallback language, and the default when none is requested

# Common ISO codes / native spellings → the feed's English language NAME (as it appears in [brackets]).
# The advisories manifest tells the agent to pass the language NAME, so this is a lenient safety net.
_LANG_ALIASES = {
    "en": "english", "en-us": "english",
    "es": "spanish", "es-us": "spanish", "español": "spanish", "espanol": "spanish",
    "zh": "chinese", "zh-cn": "chinese", "zh-hans": "chinese", "zh-hant": "chinese", "中文": "chinese",
    "ht": "haitian creole", "kreyòl": "haitian creole", "kreyol": "haitian creole",
    "ko": "korean", "한국어": "korean", "ru": "russian", "русский": "russian",
    "bn": "bengali", "বাংলা": "bengali", "ar": "arabic", "العربية": "arabic",
    "ur": "urdu", "اردو": "urdu", "fr": "french", "français": "french", "francais": "french",
    "pl": "polish", "yi": "yiddish", "it": "italian", "ja": "japanese", "pt": "portuguese",
}


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
            language=_first_text(root, "language"),
        )
    except Exception:
        return None


def _item_language(item: ET.Element) -> str:
    """The item's language NAME (lowercased) from its `<author>` '[…]' tag, else '' (e.g. 'english')."""
    m = _LANG_RE.search(_direct_child_text(item, "author"))
    return m.group(1).strip().lower() if m else ""


def _item_cap_url(item: ET.Element) -> str:
    """The item's CAP XML url: the `<enclosure url="...">` attr, else the `<link>` text."""
    enclosure = next((c for c in item if _local(c.tag) == "enclosure"), None)
    return (enclosure.get("url") if enclosure is not None else "") or _direct_child_text(item, "link")


def _cap_urls_by_language(rss_text: str) -> dict[str, list[str]]:
    """Map each language NAME (lowercased, from the item `<author>` tag) to its CAP XML urls.

    Retains ALL official-language variants the feed carries — English plus ~12 others — instead of
    discarding everything but English, so a caller can serve an advisory in the user's language.
    """
    try:
        root = _safe_fromstring(rss_text)
    except Exception:
        return {}
    by_lang: dict[str, list[str]] = {}
    for item in root.iter():
        if _local(item.tag) != "item":
            continue
        lang = _item_language(item)
        url = _item_cap_url(item)
        if lang and url:
            by_lang.setdefault(lang, []).append(url)
    return by_lang


def _english_cap_urls(rss_text: str) -> list[str]:
    """The CAP XML urls of the English items (compat shim over `_cap_urls_by_language`)."""
    return _cap_urls_by_language(rss_text).get(DEFAULT_LANGUAGE, [])


def _resolve_language(requested: Optional[str], available: list[str]) -> Optional[str]:
    """Match a requested language NAME/code to one the feed carries, else None. Lenient: exact,
    then alias-normalized, then substring either way (so 'Spanish', 'es', 'español' all resolve)."""
    key = (requested or "").strip().lower()
    if not key:
        return None
    key = _LANG_ALIASES.get(key, key)
    if key in available:
        return key
    for lang in available:
        if lang.startswith(key) or key in lang:
            return lang
    return None


async def _fetch_cap(client: httpx.AsyncClient, url: str) -> Optional[Advisory]:
    """Fetch one CAP XML and parse it. Raises on HTTP failure (gather tolerates it)."""
    response = await client.get(url, follow_redirects=True)
    response.raise_for_status()
    return _parse_cap(response.text, url)


def _advisory_key(advisory: Advisory) -> str:
    """The dedupe / alert-identity key: the CAP identifier (language-independent), else headline+sent."""
    return advisory.guid or f"{advisory.headline}|{advisory.sent}"


def _dedupe(results: list) -> list[Advisory]:
    """Keep the Advisory results (drop failed/None CAPs), deduped by alert key, order preserved."""
    advisories: list[Advisory] = []
    seen: set[str] = set()
    for result in results:
        if not isinstance(result, Advisory):
            continue  # a failed/None CAP — tolerated, skipped
        key = _advisory_key(result)
        if key in seen:
            continue
        seen.add(key)
        advisories.append(result)
    return advisories


async def fetch_advisories(
    client: Optional[httpx.AsyncClient] = None, lang: Optional[str] = None
) -> list[Advisory]:
    """Fetch the Notify NYC feed and return the advisories (deduped), in `lang` where available.

    GETs the RSS (following redirects), then fetches CAP XMLs CONCURRENTLY (tolerating individual
    failures), parses each, and dedupes by CAP identifier (the alert id, which is language-stable).
    The default (`lang=None` or English) fetches ONLY the English items — unchanged behavior. When a
    non-English language is requested AND the feed carries it, we ALSO fetch those variants and
    overlay them on the English base per alert: the requested language wins, English is the fallback
    for any alert with no variant in that language (an official city translation, not a paraphrase).
    Inject `client` to mock the HTTP calls offline. Any network/parse error yields `[]`.
    """
    own_client = client is None
    client = client or httpx.AsyncClient(timeout=20.0)
    try:
        response = await client.get(RSS_URL, follow_redirects=True)
        response.raise_for_status()
        by_lang = _cap_urls_by_language(response.text)
        english_urls = by_lang.get(DEFAULT_LANGUAGE, [])
        target = _resolve_language(lang, list(by_lang))
        target_urls = by_lang.get(target, []) if (target and target != DEFAULT_LANGUAGE) else []
        results = await asyncio.gather(
            *(_fetch_cap(client, url) for url in english_urls + target_urls),
            return_exceptions=True,
        )
    except Exception:
        return []
    finally:
        if own_client:
            await client.aclose()

    english_results = results[: len(english_urls)]
    target_results = results[len(english_urls):]
    advisories = _dedupe(english_results)
    if target_urls:  # overlay the requested-language variants onto the English base, per alert
        by_key = {_advisory_key(a): a for a in advisories}
        for translated in _dedupe(target_results):
            by_key[_advisory_key(translated)] = translated  # target wins; English stays the fallback
        advisories = list(by_key.values())
    return advisories


def _sent_key(advisory: Advisory) -> datetime:
    """Parse `sent` for sorting; unparseable → epoch (sorts oldest/last within a severity tier)."""
    try:
        return datetime.fromisoformat(advisory.sent)
    except (ValueError, TypeError):
        return datetime.min.replace(tzinfo=timezone.utc)


async def active_advisories(
    client: Optional[httpx.AsyncClient], now: datetime, lang: Optional[str] = None
) -> list[Advisory]:
    """The advisories still in effect at `now` (a tz-aware datetime), most severe / most recent first.

    Keeps advisories whose `expires` (ISO 8601 WITH tz offset, parsed by `datetime.fromisoformat`)
    is strictly after `now`. Sorts by severity rank (Extreme→Severe→Moderate→Minor→Unknown) then
    `sent` descending. `lang` (default English) surfaces the requested-language variant where the
    feed carries it, English fallback. Resilient: a fetch failure yields `[]` (via `fetch_advisories`).
    """
    advisories = await fetch_advisories(client, lang=lang)
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
