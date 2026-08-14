"""Pluggable general geocoder for places, streets, and intersections via geopy.

NYC GeoSearch (geo.py) stays limited to complete addresses that benefit from
authoritative PAD and BBL identity. The provider is config-selected
(`HEYNYC_GEOCODER`) so the backend is not hard-wired to any vendor or city.

Per Nominatim's usage policy the default provider sets a `user_agent` and is
dev/demo-grade (1 req/s, must self-host for production). Point `HEYNYC_GEOCODER`
at a keyed provider (mapbox) or a self-hosted Nominatim for real traffic; that
is a one-line config change because the seam is provider-agnostic.

Ref: https://geopy.readthedocs.io/en/stable/  (async + AsyncRateLimiter)
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from typing import Awaitable, Callable, Optional

from .. import config

logger = logging.getLogger("heynyc.geocoder")

# An async geocode callable: (text) -> Optional[geopy Location | GeoPoint]
GeocodeFn = Callable[[str], Awaitable[Optional[object]]]

_NOMINATIM_CACHE: OrderedDict[str, object] = OrderedDict()
_NOMINATIM_LOCK = asyncio.Lock()
_NOMINATIM_LAST_REQUEST = 0.0
_NOMINATIM_CACHE_SIZE = 512


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


async def _provider_geocode(
    provider: str, text: str, fn: GeocodeFn, *, apply_public_policy: bool = True
):
    """Call one provider, applying the public Nominatim policy in process."""
    if provider != "nominatim" or not apply_public_policy:
        return await fn(text)

    key = " ".join(text.casefold().split())
    cached = _NOMINATIM_CACHE.get(key)
    if cached is not None:
        _NOMINATIM_CACHE.move_to_end(key)
        return cached

    global _NOMINATIM_LAST_REQUEST
    async with _NOMINATIM_LOCK:
        cached = _NOMINATIM_CACHE.get(key)
        if cached is not None:
            _NOMINATIM_CACHE.move_to_end(key)
            return cached
        delay = 1.0 - (time.monotonic() - _NOMINATIM_LAST_REQUEST)
        if delay > 0:
            await asyncio.sleep(delay)
        try:
            location = await fn(text)
        finally:
            _NOMINATIM_LAST_REQUEST = time.monotonic()
        if location is not None:
            _NOMINATIM_CACHE[key] = location
            _NOMINATIM_CACHE.move_to_end(key)
            if len(_NOMINATIM_CACHE) > _NOMINATIM_CACHE_SIZE:
                _NOMINATIM_CACHE.popitem(last=False)
        return location


async def forgiving_geocode(text: str, *, geocode_fn: Optional[GeocodeFn] = None):
    """Geocode via the configured provider, returning a GeoPoint or None.

    `geocode_fn` is injectable for offline tests (no network)."""
    from .geo import GeoPoint  # lazy import to avoid a module cycle

    primary = config.HEYNYC_GEOCODER
    providers = [(primary, geocode_fn or _build_geocode_fn(primary), geocode_fn is None)]
    if geocode_fn is None and primary == "nominatim" and config.MAPBOX_TOKEN:
        providers.append(("mapbox", _build_geocode_fn("mapbox"), True))

    for provider, fn, apply_public_policy in providers:
        try:
            loc = await _provider_geocode(
                provider, text, fn, apply_public_policy=apply_public_policy
            )
        except Exception:
            logger.exception("geocoder '%s' failed for %r", provider, text)
            continue
        if loc is None:
            continue
        if isinstance(loc, GeoPoint):
            return loc
        label = getattr(loc, "address", "") or ""
        if any(state in label.lower() for state in ("new jersey", "connecticut", "pennsylvania")):
            logger.info("geocoder '%s' returned a non-NYC match, rejecting: %s", provider, label)
            continue
        raw = getattr(loc, "raw", None) or {}
        return GeoPoint(
            lat=float(loc.latitude),
            lon=float(loc.longitude),
            label=label,
            confidence=_confidence(provider, raw),
            match_type=provider,
        )
    return None
