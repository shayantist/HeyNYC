"""Build the bundled NYC neighborhood gazetteer from the city's 2020 NTA dataset (F079).

Fetches the 262 Neighborhood Tabulation Areas from NYC Open Data (9nt8-h7nd, verified live
2026-07-22), computes each polygon's bounding-box midpoint as its centroid, and writes
`heynyc/core/data/nta_neighborhoods.tsv` (normalized-key<TAB>boroname<TAB>lat<TAB>lon<TAB>ntaname).
Deterministic city DATA consulted before any fuzzy geocoder, the same shape as the bundled
ZCTA centroids. Keys are pre-normalized (casefold, no apostrophes, collapsed spaces) and cover:

- every full NTA name (ntatype 0, residential neighborhoods only)
- each unique part of a compound name ("Upper West Side-Lincoln Square" also keys
  "upper west side" and "lincoln square"); repeated single-word parts are skipped because they
  are ambiguous ("Bedford" could refer to more than one NTA)
- a curated alias table for the short forms residents actually text (uws, fidi, bed-stuy...)

A key claimed by more than one NTA gets the average of the member centroids (they share the
name because they are adjacent splits of one colloquial neighborhood).

Usage: uv run python scripts/build_nta_gazetteer.py
"""
from __future__ import annotations

import json
import re
import sys
import urllib.request
from pathlib import Path

DATASET = "https://data.cityofnewyork.us/resource/9nt8-h7nd.json?$limit=500"
OUT = Path(__file__).resolve().parent.parent / "heynyc" / "core" / "data" / "nta_neighborhoods.tsv"

# Short forms residents text, mapped to the normalized full name they mean. Verified against
# the fetched ntaname list at build time: the build fails loudly if a target stops matching.
ALIASES = {
    "uws": "upper west side",
    "ues": "upper east side",
    "fidi": "financial district",
    "les": "lower east side",
    "bed-stuy": "bedford-stuyvesant",
    "bedstuy": "bedford-stuyvesant",
    "the village": "greenwich village",
    "wburg": "williamsburg",
    "greenpoint": "greenpoint",
}


def normalize(name: str) -> str:
    text = name.casefold().replace("'", "").replace("’", "")
    text = re.sub(r"[.,]", " ", text)
    text = re.sub(r"^the\s+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def bbox_mid(geom: dict) -> tuple[float, float]:
    lons: list[float] = []
    lats: list[float] = []
    for polygon in geom["coordinates"]:
        for ring in polygon:
            for lon, lat in ring:
                lons.append(lon)
                lats.append(lat)
    # ponytail: bbox midpoint, not a true area centroid; fine for centering a search
    # radius on a neighborhood, upgrade to shoelace centroid if placement ever matters
    return (min(lats) + max(lats)) / 2, (min(lons) + max(lons)) / 2


def main() -> int:
    with urllib.request.urlopen(DATASET, timeout=60) as resp:
        rows = json.load(resp)
    hoods = [r for r in rows if r.get("ntatype") == "0" and r.get("the_geom")]
    print(f"fetched {len(rows)} NTAs, {len(hoods)} residential (ntatype 0)")

    # key -> list of (borough, lat, lon, ntaname)
    entries: dict[str, list[tuple[str, float, float, str]]] = {}
    single_parts: dict[str, list[tuple[dict, float, float]]] = {}

    def add(key: str, row: dict, lat: float, lon: float) -> None:
        entries.setdefault(key, []).append((row["boroname"], lat, lon, row["ntaname"]))

    for row in hoods:
        lat, lon = bbox_mid(row["the_geom"])
        full = normalize(row["ntaname"])
        add(full, row, lat, lon)
        for part in row["ntaname"].split("-"):
            key = normalize(part)
            if key == full:
                continue
            if len(key.split()) >= 2:
                add(key, row, lat, lon)
            elif len(key) > 2:
                single_parts.setdefault(key, []).append((row, lat, lon))

    for key, matches in single_parts.items():
        if len(matches) == 1:
            row, lat, lon = matches[0]
            add(key, row, lat, lon)

    # Aliases resolve against the keys built above; fail loudly if a target vanished.
    for alias, target in ALIASES.items():
        tkey = normalize(target)
        matches = entries.get(tkey) or [
            e for k, es in entries.items() for e in es if tkey in k
        ]
        if not matches:
            print(f"ALIAS TARGET NOT FOUND: {alias} -> {target}", file=sys.stderr)
            return 1
        entries.setdefault(normalize(alias), []).extend(matches)

    lines = []
    for key in sorted(entries):
        members = entries[key]
        boroughs = {b for b, _, _, _ in members}
        if len(boroughs) > 1:
            # Same name in two boroughs is genuine ambiguity; leave it to the clarify path.
            print(f"skipping cross-borough key: {key} ({sorted(boroughs)})")
            continue
        lat = sum(m[1] for m in members) / len(members)
        lon = sum(m[2] for m in members) / len(members)
        names = sorted({m[3] for m in members})
        lines.append(f"{key}\t{members[0][0]}\t{lat:.6f}\t{lon:.6f}\t{'; '.join(names)}")

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(lines)} keys -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
