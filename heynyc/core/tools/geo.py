"""Geospatial grounding, geocoding, nearest-X, and distance.

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
from ..citations import data_provenance
from ..config import GEOSEARCH_BASE, OSRM_BASE
from .arcgis import feature_query_url, query_feature_service
from .base import Tool, ToolContext
from .datasets import dataset_url, normalize, query_dataset, row_url

_INTERSECTION_RE = re.compile(r"(?:\band\b|&|/)", re.IGNORECASE)
_NUMBERED_AT_RE = re.compile(r"\bat\b", re.IGNORECASE)
_STREET_SUFFIX_RE = re.compile(
    r"\b(?:street|st|avenue|ave|road|rd|boulevard|blvd|lane|ln|drive|dr|parkway|pkwy|way)\b",
    re.IGNORECASE,
)
_INTERSECTION_SPLIT_RE = re.compile(r"(?:\band\b|&|/|\bat\b)", re.IGNORECASE)
_DIRECTIONS = {
    "north": "n", "n": "n", "south": "s", "s": "s",
    "east": "e", "e": "e", "west": "w", "w": "w",
}
_STREET_SUFFIXES = {
    "street", "st", "avenue", "ave", "road", "rd", "boulevard", "blvd",
    "lane", "ln", "drive", "dr", "parkway", "pkwy", "way",
}
_COORDINATE_RE = re.compile(
    r"^\s*(?P<lat>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*,\s*"
    r"(?P<lon>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*$"
)
_COUNT_VALUE = r"one|two|three|four|five|six|seven|eight|nine|ten|\d+"
_VERB_COUNT_RE = re.compile(
    rf"\b(?:show|give|list|find|return|need|want|have)\s+(?:me\s+)?(?P<count>{_COUNT_VALUE})\b",
    re.IGNORECASE,
)
_NOUN_COUNT_RE = re.compile(
    rf"\b(?P<count>{_COUNT_VALUE})\s+"
    r"(?:(?:nearest|nearby|cooling|water|drinking|public|food|indoor|activated|free)\s+)?"
    r"(?:results?|places?|options?|locations?|centers?|stations?|fountains?|restrooms?|pantries?|clinics?|events?)\b",
    re.IGNORECASE,
)
_MORE_RESULTS_RE = re.compile(r"\b(?:all|more|additional|another)\b", re.IGNORECASE)
_COUNT_VALUES = {
    word: number
    for number, word in enumerate(
        ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")
    )
}
_LOCATION_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_LOCATION_NEGATION_RE = re.compile(
    r"\b(?:no|not|never|without|don['’]?t|do\s+not)\b", re.IGNORECASE,
)
_LOCATION_STALE_RE = re.compile(r"\b(?:used to|formerly|previously)\b", re.IGNORECASE)
_LOCATION_TOKEN_ALIASES = {
    "st": "street", "ave": "avenue", "rd": "road", "blvd": "boulevard",
    "ln": "lane", "dr": "drive", "pkwy": "parkway",
}


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


def _requested_result_limit(value: object, query: str, *, default: int = 3) -> int:
    """Trust model counts only when the resident asked for a count or more results."""
    try:
        requested = min(max(int(value), 1), 10)
    except (TypeError, ValueError):
        requested = default
    if not query or _MORE_RESULTS_RE.search(query):
        return requested
    match = _VERB_COUNT_RE.search(query) or _NOUN_COUNT_RE.search(query)
    if match:
        raw_count = match.group("count").casefold()
        count = _COUNT_VALUES.get(raw_count, int(raw_count) if raw_count.isdigit() else default)
        return min(max(count, 1), 10)
    return default


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
                return turn[raw_start:raw_end]
    return ""


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
    needed), HeyNYC owns the grounded 'where', Maps owns navigation/ETA."""
    return f"https://www.google.com/maps/search/?api=1&query={lat:.5f},{lon:.5f}"


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


async def geocode(
    text: str, *, client: Optional[httpx.AsyncClient] = None, forgiving=None,
    borough_contains=None,
) -> Optional[GeoPoint]:
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
    coordinate = _COORDINATE_RE.fullmatch(text)
    if coordinate:
        lat = float(coordinate.group("lat"))
        lon = float(coordinate.group("lon"))
        if not _in_nyc(lat, lon):
            return None
        return GeoPoint(
            lat=lat, lon=lon, label=f"{lat:.5f},{lon:.5f}", confidence=1.0,
            match_type="coordinates",
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
                                 confidence=1.0, match_type="zcta")
                borough = _detect_borough(text)
                if borough is not None and not await borough_contains(point, borough, client):
                    return None
                return point
            return None  # unknown or non-NYC ZIP → don't fall through to GeoSearch
        # F079: neighborhoods resolve from the bundled NTA gazetteer, deterministic and
        # offline, before any provider can fuzzy-match an arbitrary POI.
        neighborhood = _neighborhood_point(text)
        if neighborhood is not None:
            return neighborhood
        # Borough-aware bias: a borough named in the query gives that borough's hard boundary.rect;
        # otherwise rect is None and _geosearch_geocode applies the citywide NYC floor. This is what
        # fixes "125th Street Manhattan" resolving to College Point, Queens.
        rect = _borough_rect(text)
        if _looks_like_intersection(text) or _looks_like_named_area(text):
            point = await forgiving(text)
            if point is not None and rect is not None and not _in_rect(point, rect):
                point = None
            point = point or await _geosearch_geocode(text, client, rect=rect)
        else:
            point = await _geosearch_geocode(text, client, rect=rect) or await forgiving(text)
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
        if point is not None and _looks_like_intersection(text):
            if not _intersection_identity_matches(text, point.label):
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
        # Fell back to the strict geocoder for an intersection, least reliable case.
        note += " Intersections geocode imprecisely here, confirm with the user before relying on it.)"
    else:
        note += " If that's not the intended spot, ask for a street address.)"
    return note


def _clarify_message(query: str) -> str:
    """Returned when a location is too ambiguous to answer for, make the agent ask."""
    return (
        f"I couldn't reliably pin '{query}' to one place, it may match several spots in NYC "
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
    derivation = {
        "origin": [origin_lat, origin_lon],
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

    k = _requested_result_limit(args.get("k", 3), ctx.query)
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
        updated = f" record updated={place.updated_at[:10]}" if place.updated_at else ""
        website = f", official info: {place.website}" if place.website else ""
        hours = f", hours: {place.hours}" if place.hours else ""
        lines.append(
            f"- {place.name} ({where}), {dist_mi:.2f} mi straight-line, "
            f"status={place.status or 'unknown'}{phone}{updated} {{cite:{cite}}}, "
            f"directions: {maps_link(place.lat, place.lon)}{website}{hours}"
        )
    if binding.limitations:
        lines.append(f"Source limit: {binding.limitations}")
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
                "address, ranked by distance. NEVER guess locations, always use this."
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
