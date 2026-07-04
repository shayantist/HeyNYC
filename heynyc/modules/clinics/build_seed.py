"""BUILD-TIME seed generator for the NYC Care / H+H Gotham clinic seed.

Run once (offline is fine except it needs the live geocoder) to (re)generate
`data/nyc_care_sites.tsv`. NOT imported at runtime — the tool reads the TSV.

Source of the site list: https://www.nychealthandhospitals.org/locations/ (fetched
2026-07-04). 11 H+H acute-care hospitals + 29 Gotham Health community health centers.
Every field below was transcribed from that page — never completed from memory; a
field the page didn't show is left blank. Each address is geocoded via the shared
NYC geocoder (GeoSearch) at build time and the resulting lat/lon is written to the
TSV, so the runtime tool never geocodes a seed row.

    uv run python -m heynyc.modules.clinics.build_seed

Any row that fails to geocode or lands outside the NYC bounding box is printed as a
FLAG and written with blank coordinates (the runtime loader drops coordinate-less rows).
"""
from __future__ import annotations

import asyncio
import csv
from pathlib import Path

import httpx

from heynyc.core.tools.geo import _in_nyc, geocode

SOURCE_URL = "https://www.nychealthandhospitals.org/locations/"
SEED_PATH = Path(__file__).resolve().parent / "data" / "nyc_care_sites.tsv"

# (name, hh_type, street, borough, zip, phone) — phone is the site's own line from the
# page; blank when the page shows only the central appointment line (1-844-692-4692).
HOSPITALS = [
    ("NYC Health + Hospitals/Bellevue", "Hospital", "462 First Avenue", "Manhattan", "10016", "212-562-1000"),
    ("NYC Health + Hospitals/Elmhurst", "Hospital", "79-01 Broadway", "Queens", "11373", "718-334-4000"),
    ("NYC Health + Hospitals/Harlem", "Hospital", "506 Lenox Avenue", "Manhattan", "10037", "212-939-1000"),
    ("NYC Health + Hospitals/Jacobi", "Hospital", "1400 Pelham Parkway South", "Bronx", "10461", "718-918-5000"),
    ("NYC Health + Hospitals/Kings County", "Hospital", "451 Clarkson Avenue", "Brooklyn", "11203", "718-245-3131"),
    ("NYC Health + Hospitals/Lincoln", "Hospital", "234 East 149th Street", "Bronx", "10451", "718-579-5000"),
    ("NYC Health + Hospitals/Metropolitan", "Hospital", "1901 First Avenue", "Manhattan", "10029", "212-423-6262"),
    ("NYC Health + Hospitals/North Central Bronx", "Hospital", "3424 Kossuth Avenue", "Bronx", "10467", "718-519-5000"),
    ("NYC Health + Hospitals/Queens", "Hospital", "82-68 164th Street", "Queens", "11432", "718-883-3000"),
    ("NYC Health + Hospitals/South Brooklyn Health (Ruth Bader Ginsburg Hospital)", "Hospital",
     "2601 Ocean Parkway", "Brooklyn", "11235", ""),
    ("NYC Health + Hospitals/Woodhull", "Hospital", "760 Broadway", "Brooklyn", "11206", "718-963-8000"),
]

GOTHAM = [
    ("NYC Health + Hospitals/Gotham Health, Bedford", "485 Throop Avenue", "Brooklyn", "11221"),
    ("NYC Health + Hospitals/Gotham Health, Belvis", "545 East 142nd Street", "Bronx", "10454"),
    ("NYC Health + Hospitals/Gotham Health, Broadway", "815 Broadway", "Brooklyn", "11206"),
    ("NYC Health + Hospitals/Gotham Health, Brownsville", "259 Bristol Street", "Brooklyn", "11212"),
    ("NYC Health + Hospitals/Gotham Health, Bushwick", "335 Central Avenue", "Brooklyn", "11221"),
    ("NYC Health + Hospitals/Gotham Health, Crown Heights", "1218 Prospect Place", "Brooklyn", "11213"),
    ("NYC Health + Hospitals/Gotham Health, Cumberland", "100 North Portland Avenue", "Brooklyn", "11205"),
    ("NYC Health + Hospitals/Gotham Health, Dyckman", "175 Nagle Avenue", "Manhattan", "10034"),
    ("NYC Health + Hospitals/Gotham Health, East New York", "2094 Pitkin Avenue", "Brooklyn", "11207"),
    ("NYC Health + Hospitals/Gotham Health, Gouverneur", "227 Madison Street", "Manhattan", "10002"),
    ("NYC Health + Hospitals/Gotham Health, Greenpoint", "875 Manhattan Avenue", "Brooklyn", "11222"),
    ("NYC Health + Hospitals/Gotham Health, Gun Hill", "1012 East Gun Hill Road", "Bronx", "10469"),
    ("NYC Health + Hospitals/Gotham Health, Jackson Heights", "34-33 Junction Boulevard", "Queens", "11372"),
    ("NYC Health + Hospitals/Gotham Health, Jonathan Williams", "333 Roebling Street", "Brooklyn", "11211"),
    ("NYC Health + Hospitals/Gotham Health, Judson", "34 Spring Street", "Manhattan", "10012"),
    ("NYC Health + Hospitals/Gotham Health, Lefrak", "59-17 Junction Boulevard", "Queens", "11368"),
    ("NYC Health + Hospitals/Gotham Health, Morrisania", "1225 Gerard Avenue", "Bronx", "10452"),
    ("NYC Health + Hospitals/Gotham Health, Parsons", "90-37 Parsons Boulevard", "Queens", "11432"),
    ("NYC Health + Hospitals/Gotham Health, Ridgewood", "769 Onderdonk Avenue", "Queens", "11385"),
    ("NYC Health + Hospitals/Gotham Health, Roberto Clemente Center", "540 East 13th Street", "Manhattan", "10009"),
    ("NYC Health + Hospitals/Gotham Health, Roosevelt", "37-50 72nd Street", "Queens", "11372"),
    ("NYC Health + Hospitals/Gotham Health, South Queens", "114-02 Guy R. Brewer Boulevard", "Queens", "11434"),
    ("NYC Health + Hospitals/Gotham Health, Springfield Gardens", "134-64 Springfield Boulevard", "Queens", "11413"),
    ("NYC Health + Hospitals/Gotham Health, St. Nicholas", "281 West 127th Street", "Manhattan", "10027"),
    ("NYC Health + Hospitals/Gotham Health, Sydenham", "264 West 118th Street", "Manhattan", "10026"),
    ("NYC Health + Hospitals/Gotham Health, Tremont", "1920 Webster Avenue", "Bronx", "10457"),
    ("NYC Health + Hospitals/Gotham Health, Vanderbilt", "165 Vanderbilt Avenue", "Staten Island", "10304"),
    ("NYC Health + Hospitals/Gotham Health, Williamsburg", "279 Graham Avenue", "Brooklyn", "11211"),
    ("NYC Health + Hospitals/Gotham Health, Woodside", "50-53 Newtown Road", "Queens", "11377"),
]

COLUMNS = ["name", "hh_type", "street", "borough", "zip", "phone", "url", "lat", "lon"]


def _sites() -> list[dict]:
    rows: list[dict] = []
    for name, hh_type, street, borough, zip_cd, phone in HOSPITALS:
        rows.append({"name": name, "hh_type": hh_type, "street": street, "borough": borough,
                     "zip": zip_cd, "phone": phone, "url": ""})
    for name, street, borough, zip_cd in GOTHAM:
        rows.append({"name": name, "hh_type": "Gotham Health", "street": street, "borough": borough,
                     "zip": zip_cd, "phone": "", "url": ""})
    return rows


async def main() -> None:
    rows = _sites()
    print(f"{len(rows)} sites ({len(HOSPITALS)} hospitals + {len(GOTHAM)} Gotham Health) — geocoding...")
    flags: list[str] = []
    async with httpx.AsyncClient(timeout=20.0) as client:
        for row in rows:
            query = f"{row['street']}, {row['borough']}, NY {row['zip']}"
            point = await geocode(query, client=client)
            if point is None:
                row["lat"], row["lon"] = "", ""
                flags.append(f"GEOCODE FAILED: {row['name']} — {query!r}")
            elif not _in_nyc(point.lat, point.lon):
                row["lat"], row["lon"] = "", ""
                flags.append(f"OUTSIDE NYC: {row['name']} — {query!r} -> {point.lat},{point.lon}")
            else:
                row["lat"], row["lon"] = f"{point.lat:.6f}", f"{point.lon:.6f}"

    SEED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SEED_PATH.open("w", encoding="utf-8", newline="") as fh:
        fh.write(f"# NYC Care / NYC Health + Hospitals sites — source: {SOURCE_URL} (fetched 2026-07-04)\n")
        fh.write("# Built by heynyc/modules/clinics/build_seed.py; lat/lon geocoded via NYC GeoSearch.\n")
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    geocoded = sum(1 for r in rows if r["lat"])
    print(f"Wrote {SEED_PATH} — {geocoded}/{len(rows)} geocoded inside NYC.")
    if flags:
        print("\nFLAGS:")
        for flag in flags:
            print(f"  ! {flag}")
    else:
        print("No geocode failures — all sites resolved inside NYC.")


if __name__ == "__main__":
    asyncio.run(main())
