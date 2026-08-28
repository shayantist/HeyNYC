"""Geospatial grounding, geocoding, nearest-X, and distance.

The agent must NEVER emit a coordinate or distance from its own head. These
tools are the only authority: a general geocoder for places and streets, NYC
GeoSearch for complete addresses, source datasets for listed locations, and
OSRM for real travel distance.
"""
from __future__ import annotations

import math
import re
from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import httpx
from pydantic import Field

from .. import config
from ..citations import data_provenance
from ..config import GEOSEARCH_BASE, OSRM_BASE
from ..location import LocationRequest
from .arcgis import (
    feature_query_url,
)
from .arcgis import (
    query_feature_service_result as query_feature_service,
)
from .base import Tool, ToolContext, ToolFailure, ToolInput
from .datasets import (
    dataset_url,
    normalize,
    query_dataset,
    query_dataset_pages,
    row_url,
)

_INTERSECTION_RE = re.compile(r"(?:\band\b|\by\b|&|/)", re.IGNORECASE)
_NUMBERED_AT_RE = re.compile(r"\bat\b", re.IGNORECASE)
_STREET_SUFFIX_RE = re.compile(
    r"\b(?:street|st|calle|avenue|ave|avenida|road|rd|boulevard|blvd|lane|ln|drive|dr|parkway|pkwy|way)\b",
    re.IGNORECASE,
)
_INTERSECTION_SPLIT_RE = re.compile(r"(?:\band\b|\by\b|&|/|\bat\b)", re.IGNORECASE)
_DIRECTIONS = {
    "north": "n", "n": "n", "south": "s", "s": "s",
    "east": "e", "e": "e", "west": "w", "w": "w",
}
_STREET_SUFFIXES = {
    "street", "st", "calle", "avenue", "ave", "avenida", "road", "rd", "boulevard", "blvd",
    "lane", "ln", "drive", "dr", "parkway", "pkwy", "way",
}
_COORDINATE_RE = re.compile(
    r"^\s*(?P<lat>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*,\s*"
    r"(?P<lon>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*$"
)
_LOCATION_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_LOCATION_NEGATION_RE = re.compile(
    r"\b(?:no|not|never|without|don['’]?t|do\s+not)\b", re.IGNORECASE,
)
_LOCATION_STALE_RE = re.compile(r"\b(?:used to|formerly|previously)\b", re.IGNORECASE)
_LOCATION_TOKEN_ALIASES = {
    "st": "street", "ave": "avenue", "rd": "road", "blvd": "boulevard",
    "ln": "lane", "dr": "drive", "pkwy": "parkway", "centre": "center",
}


def _location_identity_token(token: str) -> str:
    token = re.sub(r"(?<=\d)(?:st|nd|rd|th)$", "", token)
    return _LOCATION_TOKEN_ALIASES.get(token, token)


def _looks_like_intersection(text: str) -> bool:
    """Heuristic: 'X and Y', 'X & Y', 'X/Y', try the forgiving provider first
    for these (NYC GeoSearch can't do cross-streets)."""
    match = _INTERSECTION_RE.search(text)
    if match:
        left, right = text[:match.start()], text[match.end():]
        if not left.strip() or not right.strip():
            return False
        if any(ch.isdigit() for ch in text):
            sides = [re.findall(r"[a-z0-9]+", side.lower()) for side in (left, right)]
            numbers = [int(value) for value in re.findall(r"\d+", text)]
            streetlike = bool(_STREET_SUFFIX_RE.search(text) or re.search(r"\b\d+(?:st|nd|rd|th)\b", text, re.IGNORECASE))
            return all(len(side) <= 4 for side in sides) and (streetlike or any(n >= 10 for n in numbers))
        return bool(_STREET_SUFFIX_RE.search(left) and _STREET_SUFFIX_RE.search(right))
    return bool(_NUMBERED_AT_RE.search(text)) and any(ch.isdigit() for ch in text)


def _intersection_identity_matches(query: str, label: str) -> bool:
    query_parts = _INTERSECTION_SPLIT_RE.split(query, maxsplit=1)
    label_parts = _INTERSECTION_SPLIT_RE.split(label, maxsplit=1)
    if len(query_parts) != 2 or len(label_parts) != 2:
        return False

    def tokens(value: str) -> set[str]:
        result = set()
        for token in re.findall(r"[a-z]+|\d+(?:st|nd|rd|th)?", value.lower()):
            token = re.sub(r"(?<=\d)(?:st|nd|rd|th)$", "", token)
            if token not in _STREET_SUFFIXES and token not in _DIRECTIONS:
                result.add(token)
        return result

    label_tokens = [tokens(part) for part in label_parts]
    if not all(any(tokens(part) & candidate for candidate in label_tokens) for part in query_parts):
        return False
    query_directions = {_DIRECTIONS[token] for token in re.findall(r"[a-z]+", query.lower()) if token in _DIRECTIONS}
    label_directions = {_DIRECTIONS[token] for token in re.findall(r"[a-z]+", label.lower()) if token in _DIRECTIONS}
    return not query_directions or query_directions.issubset(label_directions)


class NearestQuery(LocationRequest):
    """Validated constraints for the shared nearest-location lookup."""

    category: str = Field(description="A known dataset category, such as cooling_center.")
    near: str = Field(description="The NYC address or place to search near.")
    max_results: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Maximum locations requested by the resident. Set only when the resident explicitly "
            "asks for a number; otherwise omit it and the server returns three."
        ),
    )


class GeocodeInput(ToolInput):
    text: str = Field(description="NYC address or place")


class DistanceInput(ToolInput):
    origin: str = Field(description="Starting NYC place")
    destination: str = Field(description="Destination NYC place")
    mode: Literal["driving", "walking", "cycling", "transit"] = Field(
        default="driving",
        description="Travel mode",
    )


def resident_supplied_location(
    proposed: str,
    query: str,
    user_turns: tuple[str, ...],
    *,
    allow_prior: bool = False,
    semantic_location: str = "",
) -> str:
    """Return a resident-authored location span.

    Non-ASCII spans require semantic confirmation.
    """
    semantic_norm = semantic_location.strip().casefold()

    def semantically_confirmed(candidate: str, turn: str) -> bool:
        if candidate.isascii():
            return True
        candidate_norm = candidate.strip().casefold()
        return bool(
            semantic_norm
            and (
                candidate_norm in semantic_norm
                or semantic_norm in candidate_norm
            )
        )

    source_turns = [query]
    if allow_prior:
        source_turns.extend(reversed(user_turns))
    turns = list(dict.fromkeys(
        turn for turn in source_turns if turn
    ))
    if not turns:
        return ""
    parts = [part.strip() for part in proposed.split(",") if part.strip()]
    candidates = [", ".join(parts[:end]) for end in range(len(parts), 0, -1)]
    for turn in turns:
        confirmed_parts: list[tuple[str, int, int]] = []
        for part in parts:
            raw_start = turn.casefold().find(part.casefold())
            if raw_start < 0:
                continue
            raw_end = raw_start + len(part)
            clause_start = max(turn.rfind(mark, 0, raw_start) for mark in ".!?;,") + 1
            clause_ends = [turn.find(mark, raw_end) for mark in ".!?;,"]
            clause_end = min((end for end in clause_ends if end >= 0), default=len(turn))
            clause = turn[clause_start:clause_end]
            if _LOCATION_NEGATION_RE.search(clause) or _LOCATION_STALE_RE.search(clause):
                continue
            if semantically_confirmed(part, turn):
                confirmed_parts.append((turn[raw_start:raw_end], clause_start, clause_end))
        if len(confirmed_parts) >= 2 and len({item[1:] for item in confirmed_parts}) == 1:
            return ", ".join(item[0] for item in confirmed_parts)
        for candidate in candidates:
            raw_start = turn.casefold().find(candidate.casefold())
            if raw_start < 0:
                continue
            raw_end = raw_start + len(candidate)
            clause_start = max(turn.rfind(mark, 0, raw_start) for mark in ".!?;,") + 1
            clause_ends = [turn.find(mark, raw_end) for mark in ".!?;,"]
            clause_end = min((end for end in clause_ends if end >= 0), default=len(turn))
            clause = turn[clause_start:clause_end]
            if _LOCATION_NEGATION_RE.search(clause) or _LOCATION_STALE_RE.search(clause):
                return ""
            if not semantically_confirmed(candidate, turn):
                continue
            return turn[raw_start:raw_end]
        turn_matches = list(_LOCATION_TOKEN_RE.finditer(turn))
        turn_tokens = [
            _LOCATION_TOKEN_ALIASES.get(match.group().lower(), match.group().lower())
            for match in turn_matches
        ]
        for candidate in candidates:
            tokens = [
                _LOCATION_TOKEN_ALIASES.get(token.lower(), token.lower())
                for token in _LOCATION_TOKEN_RE.findall(candidate)
            ]
            if not tokens:
                continue
            for start in range(len(turn_tokens) - len(tokens) + 1):
                if tokens != turn_tokens[start:start + len(tokens)]:
                    continue
                raw_start = turn_matches[start].start()
                raw_end = turn_matches[start + len(tokens) - 1].end()
                clause_start = max(turn.rfind(mark, 0, raw_start) for mark in ".!?;,") + 1
                clause_ends = [turn.find(mark, raw_end) for mark in ".!?;,"]
                clause_end = min((end for end in clause_ends if end >= 0), default=len(turn))
                clause = turn[clause_start:clause_end]
                if _LOCATION_NEGATION_RE.search(clause) or _LOCATION_STALE_RE.search(clause):
                    return ""
                if not semantically_confirmed(candidate, turn):
                    continue
                resident_span = turn[raw_start:raw_end]
                return resident_span if resident_span.isascii() else candidate
        if (
            _looks_like_intersection(proposed)
            and _intersection_identity_matches(turn, proposed)
            and not _LOCATION_NEGATION_RE.search(turn)
            and not _LOCATION_STALE_RE.search(turn)
        ):
            proposed_borough = _detect_borough(proposed)
            if proposed_borough is None or proposed_borough == _detect_borough(turn):
                return proposed
    return ""


def current_resolved_location(proposed: str, ctx: ToolContext) -> Optional["GeoPoint"]:
    """Return the stored resolver result when the current turn did not replace it."""
    current = ctx.current_location
    if current is None:
        return None
    identities = {
        " ".join(value.casefold().split())
        for value in (current.label, current.resident_query)
        if value
    }
    supplied = resident_supplied_location(proposed, ctx.query, ())
    if supplied and " ".join(supplied.casefold().split()) not in identities:
        return None
    if not resident_supplied_location(current.resident_query, ctx.query, ()) and (
        _looks_like_numbered_address(ctx.query)
        or _looks_like_intersection(ctx.query)
        or _detect_borough(ctx.query) is not None
    ):
        return None
    return current


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
    ZIP. A missing/unreadable file degrades to an empty table, never crashes."""
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


# F079: GeoSearch has no neighborhood layer, so a bare famous-neighborhood name fell through
# to the fuzzy provider, which returned an arbitrary POI containing the words ("Upper West
# Side" → a Bronx playground at provider confidence 1.0). Neighborhood names resolve from the
# bundled NTA gazetteer (deterministic city data, rebuilt by scripts/build_nta_gazetteer.py)
# BEFORE any provider, the same shape as the ZCTA table above.
_NTA_GAZETTEER: Optional[dict[str, tuple[str, float, float]]] = None
_NTA_PATH = Path(__file__).resolve().parent.parent / "data" / "nta_neighborhoods.tsv"
_BOROUGH_WORD_RE = re.compile(r"\b(?:the\s+bronx|staten\s+island|bronx|brooklyn|manhattan|queens|nyc)\b")


def _nta_table() -> dict[str, tuple[str, float, float]]:
    """Lazily load `nta_neighborhoods.tsv` (key<TAB>borough<TAB>lat<TAB>lon<TAB>names).
    A missing/unreadable file degrades to an empty table, never crashes."""
    global _NTA_GAZETTEER
    if _NTA_GAZETTEER is None:
        table: dict[str, tuple[str, float, float]] = {}
        try:
            with _NTA_PATH.open(encoding="utf-8") as fh:
                for line in fh:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) != 5:
                        continue
                    key, borough, lat, lon, _names = parts
                    try:
                        table[key] = (borough, float(lat), float(lon))
                    except ValueError:
                        continue
        except OSError:
            pass  # data file absent → neighborhoods fall through to the providers
        _NTA_GAZETTEER = table
    return _NTA_GAZETTEER


def _normalize_area(text: str) -> str:
    """The gazetteer key form: casefolded, no apostrophes/commas/periods, no leading
    article, collapsed spaces. Mirrors normalize() in scripts/build_nta_gazetteer.py."""
    t = text.casefold().replace("'", "").replace("’", "")
    t = re.sub(r"[.,]", " ", t)
    t = re.sub(r"^\s*the\s+", "", t)
    return re.sub(r"\s+", " ", t).strip()


def _neighborhood_point(text: str) -> Optional["GeoPoint"]:
    """A deterministic neighborhood hit from the bundled NTA gazetteer, or None.

    Exact match on the normalized text first (so "Queens Village" survives), then with
    borough words stripped (so "Upper West Side, Manhattan" hits). A borough named in the
    query that contradicts the NTA's own borough is a miss, never an override."""
    normalized = _normalize_area(text)
    table = _nta_table()
    key = normalized
    hit = table.get(key)
    if hit is None:
        key = re.sub(r"\s+", " ", _BOROUGH_WORD_RE.sub(" ", normalized)).strip()
        hit = table.get(key) if key else None
    if hit is None:
        return None
    borough, lat, lon = hit
    named = _detect_borough(text)
    if named is not None and _BOROUGH_NAMES.get(named) != borough:
        return None
    return GeoPoint(
        lat=lat, lon=lon, label=f"{key.title()}, {borough}",
        confidence=1.0, match_type="nta",
    )


def nyc_neighborhood_borough(text: str) -> str | None:
    """Return the borough for an exact locality in the bundled NYC neighborhood gazetteer."""
    point = _neighborhood_point(text)
    return point.label.rsplit(", ", 1)[-1] if point is not None else None


def _in_nyc(lat: float, lon: float) -> bool:
    """True if (lat, lon) falls inside `config.NYC_BBOX` (w,s,e,n)."""
    w, s, e, n = (float(x) for x in config.NYC_BBOX.split(","))
    return w <= lon <= e and s <= lat <= n


def _in_rect(point: "GeoPoint", rect: tuple[float, float, float, float]) -> bool:
    """True when a point is inside a W,S,E,N rectangle."""
    w, s, e, n = rect
    return w <= point.lon <= e and s <= point.lat <= n


@dataclass
class GeoPoint:
    lat: float
    lon: float
    label: str = ""
    confidence: float = 0.0
    match_type: str = ""
    # True when the result is too ambiguous/uncertain to answer for, the agent
    # should clarify (which borough? a street address?) rather than proceed.
    low_confidence: bool = False
    # NYC Borough-Block-Lot (10-char) from GeoSearch's PAD addendum, the building's
    # tax-lot key, needed for building-level datasets (HPD complaints/violations).
    # Only a specific street address carries one; ZIP/forgiving/POI matches leave it "".
    bbl: str = ""
    resident_query: str = ""
    provider_id: str = ""
    provider_payload: dict = field(default_factory=dict)


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


def rank_nearby(
    origin: GeoPoint,
    locations: Iterable,
    *,
    key: Callable[[object], Hashable],
    limit: int | None = None,
) -> list[tuple[object, float]]:
    """Sort normalized locations by distance and keep one row per physical site."""
    ranked = sorted(
        (
            (location, haversine_m(origin.lat, origin.lon, location.lat, location.lon))
            for location in locations
        ),
        key=lambda item: item[1],
    )
    unique = []
    seen = set()
    for location, distance_m in ranked:
        identity = key(location)
        if identity in seen:
            continue
        seen.add(identity)
        unique.append((location, distance_m))
        if limit is not None and len(unique) >= limit:
            break
    return unique


def miles(meters: float) -> float:
    return meters / METERS_PER_MILE


def origin_precision(query: str, point: GeoPoint) -> str:
    """Classify whether a resolved origin is precise enough for ordinary distances."""
    if point.match_type == "coordinates":
        return "precise"
    if point.match_type == "geosearch" and point.bbl.strip():
        return "precise"
    if (
        point.match_type in ("geosearch", "mapbox", "nominatim")
        and not point.low_confidence
        and not _gate_low_confidence(point)
        and re.match(r"\s*\d+(?:-\d+)?[A-Za-z]?\b", query)
        and _STREET_SUFFIX_RE.search(query)
        and _fallback_landmark_identity_matches(query, point.label)
    ):
        return "precise"
    if (
        _looks_like_intersection(query)
        and not point.low_confidence
        and _intersection_identity_matches(query, point.label)
    ):
        return "precise"
    return "approximate"


def format_distance(
    query: str,
    origin: GeoPoint,
    distance_mi: float,
    *,
    unit: str = "mi",
    suffix: str = "straight-line",
    destination_query: str = "",
    destination: GeoPoint | None = None,
) -> str:
    """Format deterministic distance text with an in-line approximate-origin warning."""
    text = f"{distance_mi:.2f} {unit}"
    if suffix:
        text += suffix if suffix.startswith(",") else f" {suffix}"
    if (
        origin_precision(query, origin) != "precise"
        or (
            destination is not None
            and origin_precision(destination_query, destination) != "precise"
        )
    ):
        text += ", rough estimate from the resolved place point, not a street address"
    return text


def maps_link(lat: float, lon: float) -> str:
    """A Google Maps link to a coordinate. It's a deterministic URL transform of
    an already-grounded GeoPoint, so it carries no hallucination risk (no citation
    needed), HeyNYC owns the grounded 'where', Maps owns navigation/ETA."""
    return f"https://www.google.com/maps/search/?api=1&query={lat:.5f},{lon:.5f}"


def directions_link(
    origin: GeoPoint, destination: GeoPoint, *, mode: str | None = None
) -> str:
    """Open platform directions between two already-grounded points."""
    link = (
        "https://www.google.com/maps/dir/?api=1"
        f"&origin={origin.lat:.5f},{origin.lon:.5f}"
        f"&destination={destination.lat:.5f},{destination.lon:.5f}"
    )
    travel_mode = {
        "driving": "driving", "walking": "walking", "cycling": "bicycling", "transit": "transit",
    }.get(mode)
    return f"{link}&travelmode={travel_mode}" if travel_mode else link


# --- Borough-aware query biasing -----------------------------------------------------------------
# NYC GeoSearch does not infer a borough from a borough WORD sitting in a plain /search query, so a
# raw "125th Street Manhattan" resolves to a same-named "125 Street" in College Point, Queens. The
# fix is a QUERY change, not a geocoder swap: attach a hard boundary.rect for the named borough, and
# ALWAYS attach a citywide NYC rect as the floor so a plain no-borough address still stays inside the
# five boroughs. Verified live 2026-07-05 (docs/internal/strategy/2026-07-05-geocoder-upgrade.md). We do NOT
# use focus.point: it is only a soft re-rank, it did not fix this case in testing, and GeoSearch only
# documents it on /autocomplete, not /search.

# Borough bounding boxes as (min_lon, min_lat, max_lon, max_lat), the same W,S,E,N order as config.NYC_BBOX.
# These are the envelope values verified live on 2026-07-05 to disambiguate the borough class of error.
# TODO(prod): regenerate these envelopes from the official NYC DCP Borough Boundaries file so each rect matches its borough polygon exactly, instead of these hand-verified approximations.
_BOROUGH_RECT: dict[str, tuple[float, float, float, float]] = {
    "manhattan": (-74.0479, 40.6829, -73.9067, 40.8820),
    "bronx": (-73.9339, 40.7855, -73.7654, 40.9176),
    "brooklyn": (-74.0421, 40.5707, -73.8331, 40.7395),
    "queens": (-73.9626, 40.5416, -73.7004, 40.8007),
    "staten island": (-74.2591, 40.4774, -74.0492, 40.6518),
}

_BOROUGH_BOUNDARY_URL = (
    "https://services5.arcgis.com/GfwWNkhOj9bNBqoJ/ArcGIS/rest/services/"
    "v_NYC_Borough_Boundary/FeatureServer/0/query"
)
_BOROUGH_NAMES = {
    "manhattan": "Manhattan",
    "bronx": "Bronx",
    "brooklyn": "Brooklyn",
    "queens": "Queens",
    "staten island": "Staten Island",
}

# Borough detection patterns: a full name plus common aliases/abbreviations, each matched as a WHOLE
# token (\b...\b) so a short abbreviation never fires on a substring inside a street name (e.g. "si"
# inside "Business" or "Simone", "bk" inside a word). "the Bronx" is already covered by the plain
# "bronx" token. "NYC" and "New York" are intentionally absent, so a citywide query returns None and
# the citywide floor applies rather than a single borough.
_BOROUGH_PATTERNS: tuple[tuple[str, "re.Pattern[str]"], ...] = (
    ("staten island", re.compile(r"\bstaten\s+is(?:land)?\b|\bsi\b", re.IGNORECASE)),
    ("manhattan", re.compile(r"\bmanhattan\b|\bmn\b", re.IGNORECASE)),
    ("bronx", re.compile(r"\bbronx\b|\bbx\b", re.IGNORECASE)),
    ("brooklyn", re.compile(r"\bbrooklyn\b|\bbklyn\b|\bbk\b", re.IGNORECASE)),
    ("queens", re.compile(r"\bqueens\b|\bqn\b", re.IGNORECASE)),
)


def _detect_borough(text: str) -> Optional[str]:
    """Return the NYC borough key named in the query text, or None. When more than one borough word appears the LAST mention wins, because the borough is conventionally a suffix ("125th Street Manhattan"). Pure and offline: this only reads the string, it never calls a geocoder."""
    best_key: Optional[str] = None
    best_pos = -1
    for key, pattern in _BOROUGH_PATTERNS:
        pos = -1
        for match in pattern.finditer(text):
            pos = match.start()  # keep the last match position for this borough
        if pos > best_pos:
            best_pos = pos
            best_key = key
    return best_key


def _nyc_floor() -> tuple[float, float, float, float]:
    """The citywide NYC rect (min_lon, min_lat, max_lon, max_lat) parsed from config.NYC_BBOX (W,S,E,N)."""
    w, s, e, n = (float(x) for x in config.NYC_BBOX.split(","))
    return w, s, e, n


def _borough_rect(text: str) -> Optional[tuple[float, float, float, float]]:
    """The bounding box for a borough named in the query, or None when no borough is named (the floor then applies)."""
    key = _detect_borough(text)
    return _BOROUGH_RECT.get(key) if key else None


def _looks_like_named_area(text: str) -> bool:
    """A borough-qualified neighborhood or place name, not a numbered address."""
    return not re.search(r"\d", text) and _detect_borough(text) is not None


def _looks_like_numbered_address(text: str) -> bool:
    """A resident-supplied house number, not a numbered street such as 125th Street."""
    return bool(re.match(r"\s*(?:near\s+)?\d+(?:-\d+)?[a-z]?\s+", text, re.IGNORECASE))


async def _point_in_named_borough(
    point: "GeoPoint", borough: str, client: httpx.AsyncClient
) -> bool:
    """Fail-closed point containment against DCP's official borough polygons."""
    try:
        response = await client.get(
            _BOROUGH_BOUNDARY_URL,
            params={
                "where": f"BoroName='{_BOROUGH_NAMES[borough]}'",
                "geometry": f"{point.lon},{point.lat}",
                "geometryType": "esriGeometryPoint",
                "inSR": 4326,
                "spatialRel": "esriSpatialRelIntersects",
                "returnCountOnly": "true",
                "f": "json",
            },
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            return False
        return int(payload.get("count", 0)) > 0
    except (KeyError, TypeError, ValueError, httpx.HTTPError):
        return False


def _geosearch_params(text: str, rect: Optional[tuple[float, float, float, float]]) -> dict:
    """Build the GeoSearch /search params: the query text, size, and a hard boundary.rect. Uses the passed borough rect when present, otherwise the citywide NYC floor, so a result can never leave NYC."""
    w, s, e, n = rect or _nyc_floor()
    return {
        "text": text,
        "size": 1,
        "boundary.rect.min_lon": w,
        "boundary.rect.min_lat": s,
        "boundary.rect.max_lon": e,
        "boundary.rect.max_lat": n,
    }


def _fallback_landmark_identity_matches(text: str, label: str) -> bool:
    """Reject a fuzzy named-place substitution such as "Civic Plaza" → "Civic Yard"."""
    if _looks_like_intersection(text):
        return True
    ignored = {"a", "an", "the"}

    def address_identity(value: str) -> tuple[str, set[str]]:
        tokens = [
            _location_identity_token(token)
            for token in _LOCATION_TOKEN_RE.findall(value.casefold())
            if token not in ignored
        ]
        suffix = next(
            (index for index, token in enumerate(tokens) if token in _STREET_SUFFIXES),
            len(tokens),
        )
        number_match = re.match(r"\s*(\d+(?:-\d+)?[a-z]?)\b", value.casefold())
        number = number_match.group(1) if number_match else ""
        return number, set(tokens[:suffix + 1])

    if re.match(r"\s*\d+", text):
        query_number, query_tokens = address_identity(text)
        label_number, label_tokens = address_identity(label)
        if query_number and query_number != label_number:
            return False
        return query_tokens.issubset(label_tokens)

    query_tokens = {
        _location_identity_token(token)
        for token in _LOCATION_TOKEN_RE.findall(text.casefold())
        if token not in ignored
    }
    label_tokens = {
        _location_identity_token(token)
        for token in _LOCATION_TOKEN_RE.findall(label.casefold())
    }
    return not query_tokens or query_tokens.issubset(label_tokens)


async def _geosearch_geocode(
    text: str, client: httpx.AsyncClient, *, rect: Optional[tuple[float, float, float, float]] = None
) -> Optional[GeoPoint]:
    """NYC GeoSearch (authoritative PAD data). Strict: no intersections/POIs. A borough-aware boundary.rect (or the citywide floor when rect is None) hard-filters results to the right borough."""
    response = await client.get(f"{GEOSEARCH_BASE}/search", params=_geosearch_params(text, rect))
    response.raise_for_status()
    features = (response.json() or {}).get("features") or []
    if not features:
        return None
    feature = features[0]
    lon, lat = feature["geometry"]["coordinates"]
    props = feature.get("properties", {})
    if (
        props.get("match_type") == "fallback"
        and not _fallback_landmark_identity_matches(text, props.get("label", ""))
    ):
        return None
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
        low_confidence=bool(
            isinstance(hn, str)
            and hn.strip()
            and not re.search(rf"\b{re.escape(hn)}\b", text)
        ),
        bbl=bbl,
        resident_query=text,
        provider_id=str(feature.get("id") or props.get("id") or props.get("gid") or ""),
        provider_payload=feature,
    )


def _gate_low_confidence(point: GeoPoint) -> bool:
    """Flag a geocode too uncertain to answer for → the agent clarifies.

    Confidence-only: people overwhelmingly give an address, a place, or a
    neighborhood (and "near me" via the UI), not bare cross-streets, and with
    NYC-biased Mapbox those resolve at high confidence (intersections included).
    So we don't special-case phrasing; we just gate a genuinely low provider
    score. Mapbox/Pelias return a real match confidence; GeoSearch (authoritative
    NYC PAD) and Nominatim popularity are not thresholded. The non-NYC reject in
    geocoder.py handles wrong-region matches separately.
    """
    if point.match_type in ("mapbox", "pelias"):
        return point.confidence < config.HEYNYC_GEOCODE_MIN_CONFIDENCE
    return False


def _bare_name_expanded_to_street(query: str, label: str) -> bool:
    query_tokens = _LOCATION_TOKEN_RE.findall(query.casefold())
    first_label = label.split(",", 1)[0].strip()
    return (
        len(query_tokens) == 1
        and not _STREET_SUFFIX_RE.search(query)
        and bool(_STREET_SUFFIX_RE.search(first_label))
        and first_label.casefold() != query.strip().casefold()
    )


async def geocode(
    text: str, *, client: Optional[httpx.AsyncClient] = None, forgiving=None,
    borough_contains=None,
) -> Optional[GeoPoint]:
    """Hybrid geocoder.

    Free-form places, streets, and intersections go to the configured general
    geocoder first. Exact numbered NYC addresses use GeoSearch's authoritative PAD
    data first so callers can retain BBL identity. Either provider can fall back to
    the other. The general provider is swappable through `geocoder.py`, and
    `forgiving` is injectable for tests.
    Ambiguous intersection results are flagged `low_confidence` so the agent clarifies.
    """
    if text.strip().casefold() in {"here", "near me", "my location", "my current location", "current location"}:
        return None
    if forgiving is None:
        from .geocoder import forgiving_geocode
        forgiving = forgiving_geocode
    coordinate = _COORDINATE_RE.fullmatch(text)
    if coordinate:
        lat = float(coordinate.group("lat"))
        lon = float(coordinate.group("lon"))
        if not _in_nyc(lat, lon):
            return None
        return GeoPoint(
            lat=lat, lon=lon, label=f"{lat:.5f},{lon:.5f}", confidence=1.0,
            match_type="coordinates", resident_query=text,
        )
    borough_contains = borough_contains or _point_in_named_borough
    own = client is None
    client = client or httpx.AsyncClient(timeout=20.0)
    try:
        # ZIP guard: a bare ZIP-area query (a 5-digit token with no OTHER digit,
        # "10453", "10453 Bronx", "Bronx 10453") resolves from the bundled ZCTA
        # centroids, never GeoSearch (which has no postalcode layer and would
        # misparse it as a house number). A bare ZIP must never become an address.
        m = re.search(r"\b\d{5}\b", text)
        if m and not re.search(r"\d", text.replace(m.group(), "", 1)):
            centroid = _zip_centroid(m.group())
            if centroid is not None and _in_nyc(*centroid):
                lat, lon = centroid
                point = GeoPoint(lat=lat, lon=lon, label=f"ZIP {m.group()} area",
                                 confidence=1.0, match_type="zcta", resident_query=text)
                borough = _detect_borough(text)
                if borough is not None and not await borough_contains(point, borough, client):
                    return None
                return point
            return None  # unknown or non-NYC ZIP → don't fall through to GeoSearch
        # F079: neighborhoods resolve from the bundled NTA gazetteer, deterministic and
        # offline, before any provider can fuzzy-match an arbitrary POI.
        neighborhood = _neighborhood_point(text)
        if neighborhood is not None:
            neighborhood.resident_query = text
            return neighborhood
        # Borough-aware bias: a borough named in the query gives that borough's hard boundary.rect;
        # otherwise rect is None and _geosearch_geocode applies the citywide NYC floor. This is what
        # fixes "125th Street Manhattan" resolving to College Point, Queens.
        rect = _borough_rect(text)
        provider_text = (
            re.sub(r"\s+(?:and|at)\s+", " & ", text, count=1, flags=re.IGNORECASE)
            if _looks_like_intersection(text)
            else text
        )
        if _looks_like_numbered_address(text) and not _looks_like_intersection(text):
            point = await _geosearch_geocode(text, client, rect=rect) or await forgiving(text)
        else:
            point = await forgiving(provider_text)
            if point is not None and rect is not None and not _in_rect(point, rect):
                point = None
        if point is not None and rect is not None and not _in_rect(point, rect):
            point = None
        borough = _detect_borough(text)
        if point is not None and borough is not None:
            if not await borough_contains(point, borough, client):
                point = None
        if point is not None and _gate_low_confidence(point):
            # F064: a borough-qualified neighborhood name (not an intersection) resolves to a
            # usable centroid; don't push the resident for cross streets just because the provider
            # scores a neighborhood below the address-tuned floor. Intersection discipline (the
            # identity check below and the Mapbox confidence gate for intersections) is unchanged.
            if not (_looks_like_named_area(text) and not _looks_like_intersection(text)):
                point.low_confidence = True
        if point is not None and _bare_name_expanded_to_street(text, point.label):
            point.low_confidence = True
        if point is not None and _looks_like_intersection(text):
            if not _intersection_identity_matches(text, point.label):
                point.low_confidence = True
        if point is not None and not point.resident_query:
            point.resident_query = text
        if point is None and m and _looks_like_intersection(text.split(",", 1)[0]):
            centroid = _zip_centroid(m.group())
            if centroid is not None and _in_nyc(*centroid):
                lat, lon = centroid
                point = GeoPoint(
                    lat=lat,
                    lon=lon,
                    label=f"ZIP {m.group()} area",
                    confidence=1.0,
                    match_type="zcta",
                    resident_query=text,
                )
                if borough is not None and not await borough_contains(point, borough, client):
                    return None
        return point
    finally:
        if own:
            await client.aclose()


async def resolve_location(near: str, ctx: ToolContext) -> Optional[GeoPoint]:
    """Reuse a matching conversation location, or geocode a new spatial anchor."""
    current = current_resolved_location(near, ctx)
    if current is not None:
        return current
    return await strict_geocode(near, client=ctx.http)


async def strict_geocode(
    text: str,
    *,
    client: Optional[httpx.AsyncClient] = None,
) -> Optional[GeoPoint]:
    """Geocode while preserving an all-provider outage as an expected tool failure."""
    from .geocoder import forgiving_geocode

    async def provider(value: str):
        return await forgiving_geocode(value, raise_on_unavailable=True)

    return await geocode(text, client=client, forgiving=provider)


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
    if (
        mode != "driving"
        and OSRM_BASE.rstrip("/") == "https://router.project-osrm.org"
    ):
        return {
            "meters": haversine_m(origin.lat, origin.lon, dest.lat, dest.lon),
            "minutes": None,
            "mode": "straight-line",
            "source": "haversine",
        }
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

def _clarify_message(query: str) -> str:
    """Factual ambiguity metadata for a model-visible tool result."""
    return (
        f"Location resolution for '{query}' is ambiguous across multiple NYC places. "
        "Missing disambiguator: which borough or a specific street address."
    )


def resolved_location_citation(ctx: ToolContext, point: GeoPoint) -> str:
    """Register the resolver result separately from destination records."""
    snapshot = point.provider_payload or {
        "lat": point.lat,
        "lon": point.lon,
        "label": point.label,
        "match_type": point.match_type,
    }
    return ctx.citations.register(
        maps_link(point.lat, point.lon),
        snippet=f"{point.resident_query or point.label} resolved to {point.label}",
        title="Resolved NYC location",
        kind="DATA",
        provenance=data_provenance(
            snapshot,
            record_id=point.provider_id or f"{point.lat:.5f},{point.lon:.5f}",
            field_pointer="/",
            derivation={
                "point": [point.lat, point.lon],
                "origin_query": point.resident_query,
                "origin_label": point.label,
            },
        ),
    )


async def _geocode_handler(
    args: GeocodeInput,
    ctx: ToolContext,
) -> GeoPoint | ToolFailure | None:
    from .geocoder import GeocoderUnavailable, forgiving_geocode

    async def visible_failure_geocode(text: str):
        return await forgiving_geocode(text, raise_on_unavailable=True)

    try:
        point = await geocode(
            args["text"],
            client=ctx.http,
            forgiving=visible_failure_geocode,
        )
    except GeocoderUnavailable:
        return ToolFailure(
            status="unavailable",
            reason="Every configured geocoder failed.",
            retryable=True,
        )
    if args.get("as_current_location"):
        ctx.current_location = point if point is not None and not point.low_confidence else None
    return point


def _place_citation(
    ctx,
    place,
    binding,
    *,
    origin_query: str,
    origin: GeoPoint,
    dist_mi: float,
) -> str:
    """Register a row-addressed DATA citation: permalink URL + the row snapshot, content hash,
    field locator, and the distance derivation (so the eval floor can recompute it)."""
    if getattr(binding, "source", "socrata") == "arcgis":
        url = (feature_query_url(binding.url, place.record_id, id_field=binding.record_id_field)
               if place.record_id else binding.url)
        title = binding.title or "NYC ArcGIS finder"
    else:
        url = row_url(binding.id, place.record_id) if place.record_id else place.source_url
        title = f"NYC Open Data ({binding.id})"
    derivation = {
        "origin": [origin.lat, origin.lon],
        "origin_query": origin_query,
        "origin_label": origin.label,
        "point": [place.lat, place.lon],
        "distance_mi": dist_mi,
    }
    limitations = getattr(binding, "limitations", "")
    if limitations:
        derivation["limitations"] = limitations
    prov = data_provenance(
        place.raw,
        record_id=place.record_id,
        field_pointer="/",  # whole-row; field-level pointer is a later refinement
        derivation=derivation,
    )
    return ctx.citations.register(
        url,
        snippet=f"{place.name}, {place.borough} (status: {place.status})",
        title=title,
        kind="DATA",
        valid_as_of=place.updated_at,
        provenance=prov,
    )


async def _nearest_handler(args: NearestQuery, ctx: ToolContext) -> str:
    query = NearestQuery.model_validate(args)
    category = query.category
    binding = ctx.registry.dataset_bindings().get(category)
    if binding is None:
        available = list(ctx.registry.dataset_bindings())
        return f"No dataset for category '{category}'. Available categories: {available}"
    if query.near.strip().casefold() in {"new york", "new york city", "nyc", "the city"}:
        return "Ask the user for a NYC neighborhood, address, or landmark before ranking nearby sites."

    origin = await resolve_location(query.near, ctx)
    if origin is None:
        return f"Could not locate '{query.near}'. Ask the user for a specific NYC address."
    if origin.low_confidence:
        return _clarify_message(query.near)

    if binding.source == "arcgis":
        result = await query_feature_service(
            binding.url,
            where=binding.where or "1=1",
            client=ctx.http,
        )
        records = result if isinstance(result, list) else result.records
        url = binding.url
        complete = True if isinstance(result, list) else result.complete
    else:
        result = await query_dataset_pages(
            binding.id,
            where=binding.where,
            client=ctx.http,
            _query=query_dataset,
        )
        records = result.records
        complete = result.complete
        url = dataset_url(binding.id)
    places = normalize(records, binding.field_map, source_url=url, record_id_field=binding.record_id_field)
    if not places:
        return f"No '{category}' locations found in the dataset."

    max_results = query.max_results or 3
    ranked = rank_nearby(
        origin,
        places,
        key=lambda place: place.name.strip().casefold(),
        limit=max_results,
    )

    lines = [f"Origin: {origin.label} ({origin.lat:.5f},{origin.lon:.5f})"]
    origin_cite = ""
    for place, distance_m in ranked:
        dist_mi = miles(distance_m)
        cite = _place_citation(
            ctx,
            place,
            binding,
            origin_query=query.near,
            origin=origin,
            dist_mi=dist_mi,
        )
        origin_cite = origin_cite or cite
        where = place.address or place.borough or "NYC"
        phone = f" phone: {place.phone}" if place.phone else ""
        updated = f" record updated={place.updated_at[:10]}" if place.updated_at else ""
        website = f", official info: {place.website}" if place.website else ""
        hours = f", hours: {place.hours}" if place.hours else ""
        lines.append(
            f"- {place.name} ({where}), {format_distance(query.near, origin, dist_mi)}, "
            f"status={place.status or 'unknown'}{phone}{updated} {{cite:{cite}}}, "
            f"directions: {maps_link(place.lat, place.lon)}{website}{hours}"
        )
    if origin_cite:
        lines[0] += f" {{cite:{origin_cite}}}"
    if binding.limitations:
        lines.append(f"Source limit: {binding.limitations}")
    if not complete:
        lines.append("The city dataset returned only a partial page set, so closer sites may exist.")
    return "\n".join(lines)


async def _distance_handler(args: DistanceInput, ctx: ToolContext) -> str:
    origin = await strict_geocode(args["origin"], client=ctx.http)
    dest = await strict_geocode(args["destination"], client=ctx.http)
    if origin is None or dest is None:
        missing = args["origin"] if origin is None else args["destination"]
        return f"Could not locate '{missing}' in NYC."
    if origin.low_confidence:
        return _clarify_message(args["origin"])
    if dest.low_confidence:
        return _clarify_message(args["destination"])
    result = await travel_distance(origin, dest, mode=args.get("mode", "driving"), client=ctx.http)
    dist_mi = miles(result["meters"])
    directions = directions_link(origin, dest, mode=args.get("mode"))
    snapshot = {
        "origin": {"label": origin.label, "latitude": origin.lat, "longitude": origin.lon},
        "destination": {"label": dest.label, "latitude": dest.lat, "longitude": dest.lon},
        "route": result,
    }
    citation = ctx.citations.register(
        directions,
        snippet=f"{origin.label} to {dest.label}, {dist_mi:.2f} mi",
        title="Resolved route inputs",
        kind="DATA",
        provenance=data_provenance(
            snapshot,
            record_id=directions,
            field_pointer="/",
            derivation={
                "origin": [origin.lat, origin.lon],
                "point": [dest.lat, dest.lon],
                "distance_mi": dist_mi,
            },
        ),
    )
    if result["minutes"] is not None:
        distance = format_distance(
            args["origin"], origin, dist_mi,
            suffix=f", ~{result['minutes']} min by {result['mode']} (OSRM)",
            destination_query=args["destination"], destination=dest,
        )
        return f"{origin.label} → {dest.label}: {distance}. Directions: {directions} {{cite:{citation}}}"
    return (
        f"{origin.label} → {dest.label}: "
        f"{format_distance(args['origin'], origin, dist_mi, destination_query=args['destination'], destination=dest)} "
        f"(routing unavailable). Directions: {directions} {{cite:{citation}}}"
    )


def geo_tools() -> list[Tool]:
    return [
        Tool(
            name="geocode",
            description="Resolve an NYC address or place name to a typed location.",
            input_type=GeocodeInput,
            open_world=True,  # external geocoders (GeoSearch/Mapbox)
            handler=_geocode_handler,
            return_type=GeoPoint | ToolFailure | None,
        ),
        Tool(
            name="nearest",
            description=(
                "Find the nearest NYC locations of a given category (e.g. cooling_center) to an "
                "address, ranked by distance. NEVER guess locations, always use this."
            ),
            input_type=NearestQuery,
            open_world=True,  # external dataset (Socrata) + geocoder
            handler=_nearest_handler,
        ),
        Tool(
            name="distance",
            description=(
                "Travel distance and time between two NYC places and return a grounded Directions "
                "link. NEVER estimate distances yourself."
            ),
            input_type=DistanceInput,
            open_world=True,  # external routing (OSRM) + geocoder
            handler=_distance_handler,
        ),
    ]
