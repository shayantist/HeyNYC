"""Geospatial grounding — geocoding, nearest-X, and distance.

The agent must NEVER emit a coordinate or distance from its own head. These
tools are the only authority: NYC GeoSearch for addresses, Socrata datasets for
place locations (ranked by Haversine), and OSRM for real travel distance.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx

from .. import config
from ..config import GEOSEARCH_BASE, OSRM_BASE
from ..citations import data_provenance
from .arcgis import feature_query_url, query_feature_service
from .base import Tool, ToolContext
from .datasets import dataset_url, normalize, query_dataset, row_url

_INTERSECTION_RE = re.compile(r"(?:\b(?:and|at)\b|&|/)", re.IGNORECASE)


def _looks_like_intersection(text: str) -> bool:
    """Heuristic: 'X and Y', 'X & Y', 'X/Y' — try the forgiving provider first
    for these (NYC GeoSearch can't do cross-streets)."""
    return bool(_INTERSECTION_RE.search(text)) and any(ch.isdigit() for ch in text)


EARTH_RADIUS_M = 6_371_000.0
METERS_PER_MILE = 1609.344

# NYC GeoSearch has no postalcode layer, so a bare ZIP like "10453" is parsed as
# a house number and returns a confidently-wrong street match. We resolve bare
# ZIPs from a bundled Census ZCTA-centroid table BEFORE GeoSearch ever sees them.
_ZIP_CENTROIDS: Optional[dict[str, tuple[float, float]]] = None
_ZCTA_PATH = Path(__file__).resolve().parent.parent / "data" / "zcta_centroids.tsv"


def _zip_centroid(zip5: str) -> Optional[tuple[float, float]]:
    """Centroid (lat, lon) for a 5-digit ZIP from the bundled Census ZCTA gazetteer.

    Lazily loads `heynyc/core/data/zcta_centroids.tsv` (zip<TAB>lat<TAB>lon, no
    header) into a module-level dict on first call. Returns None for an unknown
    ZIP. A missing/unreadable file degrades to an empty table — never crashes."""
    global _ZIP_CENTROIDS
    if _ZIP_CENTROIDS is None:
        table: dict[str, tuple[float, float]] = {}
        try:
            with _ZCTA_PATH.open(encoding="utf-8") as fh:
                for line in fh:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) != 3:
                        continue
                    z, lat, lon = parts
                    try:
                        table[z] = (float(lat), float(lon))
                    except ValueError:
                        continue
        except OSError:
            pass  # data file absent → empty table, bare ZIPs return None
        _ZIP_CENTROIDS = table
    return _ZIP_CENTROIDS.get(zip5)


def _in_nyc(lat: float, lon: float) -> bool:
    """True if (lat, lon) falls inside `config.NYC_BBOX` (w,s,e,n)."""
    w, s, e, n = (float(x) for x in config.NYC_BBOX.split(","))
    return w <= lon <= e and s <= lat <= n


@dataclass
class GeoPoint:
    lat: float
    lon: float
    label: str = ""
    confidence: float = 0.0
    match_type: str = ""
    # True when the result is too ambiguous/uncertain to answer for — the agent
    # should clarify (which borough? a street address?) rather than proceed.
    low_confidence: bool = False
    # NYC Borough-Block-Lot (10-char) from GeoSearch's PAD addendum — the building's
    # tax-lot key, needed for building-level datasets (HPD complaints/violations).
    # Only a specific street address carries one; ZIP/forgiving/POI matches leave it "".
    bbl: str = ""


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in meters."""
    rad = math.pi / 180.0
    dlat = (lat2 - lat1) * rad
    dlon = (lon2 - lon1) * rad
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1 * rad) * math.cos(lat2 * rad) * math.sin(dlon / 2) ** 2
    )
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def miles(meters: float) -> float:
    return meters / METERS_PER_MILE


def maps_link(lat: float, lon: float) -> str:
    """A Google Maps link to a coordinate. It's a deterministic URL transform of
    an already-grounded GeoPoint, so it carries no hallucination risk (no citation
    needed) — HeyNYC owns the grounded 'where', Maps owns navigation/ETA."""
    return f"https://www.google.com/maps/search/?api=1&query={lat:.5f},{lon:.5f}"


async def _geosearch_geocode(text: str, client: httpx.AsyncClient) -> Optional[GeoPoint]:
    """NYC GeoSearch (authoritative PAD data). Strict: no intersections/POIs."""
    response = await client.get(f"{GEOSEARCH_BASE}/search", params={"text": text, "size": 1})
    response.raise_for_status()
    features = (response.json() or {}).get("features") or []
    if not features:
        return None
    feature = features[0]
    lon, lat = feature["geometry"]["coordinates"]
    props = feature.get("properties", {})
    # Belt-and-suspenders: if GeoSearch parsed a 5-digit ZIP that appears in the
    # input as its "house number" (the BUG-1 misparse), reject rather than return
    # a confidently-wrong street. Narrow enough to never reject a real address.
    hn = props.get("housenumber")
    if isinstance(hn, str) and re.fullmatch(r"\d{5}", hn) and re.search(rf"\b{hn}\b", text):
        return None
    # GeoSearch (NYC PAD) attaches the building's Borough-Block-Lot under addendum.pad.bbl,
    # e.g. "1910 Monterey Ave Bronx" → "2030600032". Absent for non-addressed matches.
    addendum = props.get("addendum") or {}
    bbl = (addendum.get("pad") or {}).get("bbl") or ""
    return GeoPoint(
        lat=float(lat),
        lon=float(lon),
        label=props.get("label", ""),
        confidence=float(props.get("confidence", 0.0) or 0.0),
        match_type="geosearch",
        bbl=bbl,
    )


def _gate_low_confidence(point: GeoPoint) -> bool:
    """Flag a geocode too uncertain to answer for → the agent clarifies.

    Confidence-only: people overwhelmingly give an address, a place, or a
    neighborhood (and "near me" via the UI) — not bare cross-streets — and with
    NYC-biased Mapbox those resolve at high confidence (intersections included).
    So we don't special-case phrasing; we just gate a genuinely low provider
    score. Mapbox/Pelias return a real match confidence; GeoSearch (authoritative
    NYC PAD) and Nominatim popularity are not thresholded. The non-NYC reject in
    geocoder.py handles wrong-region matches separately.
    """
    if point.match_type in ("mapbox", "pelias"):
        return point.confidence < config.HEYNYC_GEOCODE_MIN_CONFIDENCE
    return False


async def geocode(text: str, *, client: Optional[httpx.AsyncClient] = None, forgiving=None) -> Optional[GeoPoint]:
    """Hybrid geocoder.

    Intersections/POI-ish inputs go to the forgiving provider first (GeoSearch
    can't do them); everything else tries GeoSearch first (free, NYC-authoritative)
    and falls back to the forgiving provider. The forgiving provider is a swappable
    geopy backend (see `geocoder.py`); `forgiving` is injectable for tests.
    Ambiguous intersection results are flagged `low_confidence` so the agent clarifies.
    """
    if forgiving is None:
        from .geocoder import forgiving_geocode
        forgiving = forgiving_geocode
    own = client is None
    client = client or httpx.AsyncClient(timeout=20.0)
    try:
        # ZIP guard: a bare ZIP-area query (a 5-digit token with no OTHER digit —
        # "10453", "10453 Bronx", "Bronx 10453") resolves from the bundled ZCTA
        # centroids, never GeoSearch (which has no postalcode layer and would
        # misparse it as a house number). A bare ZIP must never become an address.
        m = re.search(r"\b\d{5}\b", text)
        if m and not re.search(r"\d", text.replace(m.group(), "", 1)):
            centroid = _zip_centroid(m.group())
            if centroid is not None and _in_nyc(*centroid):
                lat, lon = centroid
                return GeoPoint(lat=lat, lon=lon, label=f"ZIP {m.group()} area",
                                confidence=1.0, match_type="zcta")
            return None  # unknown or non-NYC ZIP → don't fall through to GeoSearch
        if _looks_like_intersection(text):
            point = await forgiving(text) or await _geosearch_geocode(text, client)
        else:
            point = await _geosearch_geocode(text, client) or await forgiving(text)
        if point is not None and _gate_low_confidence(point):
            point.low_confidence = True
        return point
    finally:
        if own:
            await client.aclose()


async def travel_distance(
    origin: GeoPoint,
    dest: GeoPoint,
    *,
    mode: str = "driving",
    client: Optional[httpx.AsyncClient] = None,
) -> dict:
    """Real travel distance/time via OSRM; falls back to straight-line Haversine.

    The public OSRM server only serves the driving profile; self-host for
    walking/cycling. On any failure we degrade to Haversine with minutes=None.
    """
    profile = {"driving": "driving", "walking": "foot", "cycling": "bike"}.get(mode, "driving")
    coords = f"{origin.lon},{origin.lat};{dest.lon},{dest.lat}"
    own = client is None
    client = client or httpx.AsyncClient(timeout=20.0)
    try:
        response = await client.get(
            f"{OSRM_BASE}/route/v1/{profile}/{coords}", params={"overview": "false"}
        )
        response.raise_for_status()
        route = (response.json().get("routes") or [{}])[0]
        return {
            "meters": route["distance"],
            "minutes": round(route["duration"] / 60.0, 1),
            "mode": mode,
            "source": "osrm",
        }
    except Exception:
        straight = haversine_m(origin.lat, origin.lon, dest.lat, dest.lon)
        return {"meters": straight, "minutes": None, "mode": "straight-line", "source": "haversine"}
    finally:
        if own:
            await client.aclose()


# --- Tool handlers ---------------------------------------------------------

def _resolution_note(query: str, point: GeoPoint) -> str:
    """A transparency line so a wrong geocode is visible and correctable, not silent."""
    if point.match_type == "zcta":
        z = re.search(r"\d{5}", point.label)
        zip5 = z.group() if z else point.label
        return (f"(Resolved '{query}' to the center of ZIP {zip5}; "
                f"for a precise spot, give a street address.)")
    source = "NYC GeoSearch" if point.match_type == "geosearch" else "map search"
    note = f"(Resolved '{query}' to '{point.label}' via {source}."
    if _looks_like_intersection(query) and point.match_type == "geosearch":
        # Fell back to the strict geocoder for an intersection — least reliable case.
        note += " Intersections geocode imprecisely here — confirm with the user before relying on it.)"
    else:
        note += " If that's not the intended spot, ask for a street address.)"
    return note


def _clarify_message(query: str) -> str:
    """Returned when a location is too ambiguous to answer for — make the agent ask."""
    return (
        f"I couldn't reliably pin '{query}' to one place — it may match several spots in NYC "
        f"(e.g. a street that runs through multiple boroughs). Ask the user which borough it's "
        f"in, or for a specific street address, before giving any location-based answer. "
        f"Do NOT guess a borough."
    )


async def _geocode_handler(args: dict, ctx: ToolContext) -> str:
    point = await geocode(args["text"], client=ctx.http)
    if point is None:
        return f"Could not find '{args['text']}' in NYC. Ask for a more specific address."
    if point.low_confidence:
        return _clarify_message(args["text"])
    return (
        f"{point.label} → lat={point.lat:.5f}, lon={point.lon:.5f} "
        f"(confidence={point.confidence}). map: {maps_link(point.lat, point.lon)} "
        f"{_resolution_note(args['text'], point)}"
    )


def _place_citation(ctx, place, binding, *, origin_lat: float, origin_lon: float, dist_mi: float) -> str:
    """Register a row-addressed DATA citation: permalink URL + the row snapshot, content hash,
    field locator, and the distance derivation (so the eval floor can recompute it)."""
    if getattr(binding, "source", "socrata") == "arcgis":
        url = (feature_query_url(binding.url, place.record_id, id_field=binding.record_id_field)
               if place.record_id else binding.url)
        title = binding.title or "NYC ArcGIS finder"
    else:
        url = row_url(binding.id, place.record_id) if place.record_id else place.source_url
        title = f"NYC Open Data ({binding.id})"
    prov = data_provenance(
        place.raw,
        record_id=place.record_id,
        field_pointer="/",  # whole-row; field-level pointer is a later refinement
        derivation={"origin": [origin_lat, origin_lon], "point": [place.lat, place.lon],
                    "distance_mi": dist_mi},
    )
    return ctx.citations.register(
        url,
        snippet=f"{place.name} — {place.borough} (status: {place.status})",
        title=title,
        kind="DATA",
        valid_as_of=place.updated_at,
        provenance=prov,
    )


async def _nearest_handler(args: dict, ctx: ToolContext) -> str:
    category = args["category"]
    binding = ctx.registry.dataset_bindings().get(category)
    if binding is None:
        available = list(ctx.registry.dataset_bindings())
        return f"No dataset for category '{category}'. Available categories: {available}"

    origin = await geocode(args["near"], client=ctx.http)
    if origin is None:
        return f"Could not locate '{args['near']}'. Ask the user for a specific NYC address."
    if origin.low_confidence:
        return _clarify_message(args["near"])

    if binding.source == "arcgis":
        records = await query_feature_service(binding.url, where=binding.where or "1=1", client=ctx.http)
        url = binding.url
    else:
        records = await query_dataset(binding.id, where=binding.where, limit=2000, client=ctx.http)
        url = dataset_url(binding.id)
    places = normalize(records, binding.field_map, source_url=url, record_id_field=binding.record_id_field)
    if not places:
        return f"No '{category}' locations found in the dataset."

    k = int(args.get("k", 3))
    ordered = sorted(places, key=lambda p: haversine_m(origin.lat, origin.lon, p.lat, p.lon))
    # Datasets often have multiple rows per site (e.g. several features at one
    # playground); keep only the nearest occurrence of each named place.
    ranked: list = []
    seen: set[str] = set()
    for place in ordered:
        key = place.name.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        ranked.append(place)
        if len(ranked) >= k:
            break

    lines = [f"Origin: {origin.label} ({origin.lat:.5f},{origin.lon:.5f})", _resolution_note(args["near"], origin)]
    for place in ranked:
        dist_mi = miles(haversine_m(origin.lat, origin.lon, place.lat, place.lon))
        cite = _place_citation(ctx, place, binding,
                               origin_lat=origin.lat, origin_lon=origin.lon, dist_mi=dist_mi)
        where = place.address or place.borough or "NYC"
        phone = f" phone: {place.phone}" if place.phone else ""
        lines.append(
            f"- {place.name} ({where}) — {dist_mi:.2f} mi straight-line, "
            f"status={place.status or 'unknown'}{phone} {{cite:{cite}}} — directions: {maps_link(place.lat, place.lon)}"
        )
    return "\n".join(lines)


async def _distance_handler(args: dict, ctx: ToolContext) -> str:
    origin = await geocode(args["origin"], client=ctx.http)
    dest = await geocode(args["destination"], client=ctx.http)
    if origin is None or dest is None:
        missing = args["origin"] if origin is None else args["destination"]
        return f"Could not locate '{missing}' in NYC."
    if origin.low_confidence:
        return _clarify_message(args["origin"])
    if dest.low_confidence:
        return _clarify_message(args["destination"])
    result = await travel_distance(origin, dest, mode=args.get("mode", "driving"), client=ctx.http)
    dist_mi = miles(result["meters"])
    if result["minutes"] is not None:
        return f"{origin.label} → {dest.label}: {dist_mi:.2f} mi, ~{result['minutes']} min by {result['mode']} (OSRM)."
    return f"{origin.label} → {dest.label}: {dist_mi:.2f} mi straight-line (routing unavailable)."


def geo_tools() -> list[Tool]:
    return [
        Tool(
            name="geocode",
            description="Resolve an NYC address or place name to coordinates. Use before any location reasoning.",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string", "description": "An NYC address or place name."}},
                "required": ["text"],
            },
            open_world=True,  # external geocoders (GeoSearch/Mapbox)
            handler=_geocode_handler,
        ),
        Tool(
            name="nearest",
            description=(
                "Find the nearest NYC locations of a given category (e.g. cooling_center) to an "
                "address, ranked by distance. NEVER guess locations — always use this."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "category": {"type": "string", "description": "A known dataset category, e.g. 'cooling_center'."},
                    "near": {"type": "string", "description": "The NYC address or place to search near."},
                    "k": {"type": "integer", "description": "How many results (default 3).", "default": 3},
                },
                "required": ["category", "near"],
            },
            open_world=True,  # external dataset (Socrata) + geocoder
            handler=_nearest_handler,
        ),
        Tool(
            name="distance",
            description="Travel distance and time between two NYC places. NEVER estimate distances yourself.",
            parameters={
                "type": "object",
                "properties": {
                    "origin": {"type": "string"},
                    "destination": {"type": "string"},
                    "mode": {"type": "string", "enum": ["driving", "walking", "cycling"], "default": "driving"},
                },
                "required": ["origin", "destination"],
            },
            open_world=True,  # external routing (OSRM) + geocoder
            handler=_distance_handler,
        ),
    ]
