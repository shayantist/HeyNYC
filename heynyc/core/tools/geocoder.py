"""Pluggable forgiving geocoder (intersections / POIs / fuzzy) via geopy.

NYC GeoSearch (geo.py) stays the authoritative *address* path; this is the
swappable fallback for queries GeoSearch can't do. The provider is config-
selected (`HEYNYC_GEOCODER`) so the backend isn't hard-wired to any vendor or
city, geopy abstracts ~15 providers behind one interface and runs async via
`AioHTTPAdapter`.

Per Nominatim's usage policy the default provider sets a `user_agent` and is
dev/demo-grade (1 req/s, must self-host for production). Point `HEYNYC_GEOCODER`
at a keyed provider (mapbox) or a self-hosted Nominatim for real traffic; that
is a one-line config change because the seam is provider-agnostic.

Ref: https://geopy.readthedocs.io/en/stable/  (async + AsyncRateLimiter)
"""
from __future__ import annotations

import logging
from typing import Awaitable, Callable, Optional

from .. import config

logger = logging.getLogger("heynyc.geocoder")

# An async geocode callable: (text) -> Optional[geopy Location | GeoPoint]
GeocodeFn = Callable[[str], Awaitable[Optional[object]]]


def _confidence(provider: str, raw: dict) -> float:
    """Best-effort match confidence from a provider's raw payload.

    Note: providers differ. Pelias/Mapbox return a genuine match confidence;
    Nominatim's `importance` is a popularity score, NOT match confidence, so the
    geo-layer gate does not threshold on it (it gates on borough ambiguity)."""
    if not raw:
        return 0.0
    for key in ("confidence", "relevance"):  # pelias, mapbox
        if key in raw:
            try:
                return float(raw[key])
            except (TypeError, ValueError):
                return 0.0
    try:
        return float(raw.get("importance", 0.0))  # nominatim (popularity, informational)
    except (TypeError, ValueError):
        return 0.0


def _nyc_box():
    """NYC bounding box as geopy Points (SW, NE), keeps results inside the city.

    Without this, a global geocoder confidently returns the wrong city
    ("2920 Broadway" → Lorain, Ohio); biasing is essential, not optional."""
    from geopy.point import Point

    w, s, e, n = (float(x) for x in config.NYC_BBOX.split(","))
    lon, lat = (float(x) for x in config.NYC_PROXIMITY.split(","))
    return Point(s, w), Point(n, e), Point(lat, lon)  # sw, ne, proximity


def _build_geocode_fn(provider: str) -> GeocodeFn:
    """Construct an async geocode callable for the configured geopy provider."""
    from geopy.adapters import AioHTTPAdapter

    async def run(text: str):
        sw, ne, proximity = _nyc_box()
        # Public Nominatim is slow; the geopy default 1s timeout almost always
        # times out. Give it real headroom.
        if provider == "nominatim":
            from geopy.geocoders import Nominatim

            async with Nominatim(
                user_agent=config.HEYNYC_USER_AGENT, adapter_factory=AioHTTPAdapter, timeout=10
            ) as locator:
                return await locator.geocode(
                    text, country_codes="us", viewbox=[sw, ne], bounded=True, exactly_one=True
                )
        if provider == "mapbox":
            from geopy.geocoders import MapBox

            async with MapBox(
                api_key=config.MAPBOX_TOKEN, adapter_factory=AioHTTPAdapter, timeout=10
            ) as locator:
                return await locator.geocode(
                    text, proximity=proximity, bbox=[sw, ne], exactly_one=True
                )
        raise ValueError(f"unknown HEYNYC_GEOCODER '{provider}' (use nominatim | mapbox)")

    return run


async def forgiving_geocode(text: str, *, geocode_fn: Optional[GeocodeFn] = None):
    """Geocode via the configured provider → a GeoPoint, or None.

    `geocode_fn` is injectable for offline tests (no network)."""
    from .geo import GeoPoint  # lazy import to avoid a module cycle

    provider = config.HEYNYC_GEOCODER
    fn = geocode_fn or _build_geocode_fn(provider)
    try:
        loc = await fn(text)
    except Exception:
        logger.exception("geocoder '%s' failed for %r", provider, text)
        return None
    if loc is None:
        return None
    if isinstance(loc, GeoPoint):  # an injected test fake may return a GeoPoint directly
        return loc
    label = getattr(loc, "address", "") or ""
    # The NYC bbox clips a sliver of NJ, so a high-confidence neighbouring-state
    # match can sneak through ("Fordham Road" → Clifton, NJ). Reject non-NYC.
    if any(state in label.lower() for state in ("new jersey", "connecticut", "pennsylvania")):
        logger.info("geocoder '%s' returned a non-NYC match, rejecting: %s", provider, label)
        return None
    raw = getattr(loc, "raw", None) or {}
    return GeoPoint(
        lat=float(loc.latitude),
        lon=float(loc.longitude),
        label=label,
        confidence=_confidence(provider, raw),
        match_type=provider,
    )
