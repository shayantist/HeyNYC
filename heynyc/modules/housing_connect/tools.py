"""housing_connect module tool: `find_housing_connect_lotteries`, currently-open NYC
affordable-housing lotteries, plus a deep-link handoff to the login-gated portal.

Grounded in one NYC Open Data (Socrata) dataset:

  - Advertised Lotteries on Housing Connect by Lottery (vy5i-a666), one row per
    lottery, filtered to the currently-open slice:
    `lottery_status='Active' AND lottery_end_date >= <today>`.

For each open lottery it reports the name, borough, unit count + bedroom mix, the
income bands (AMI) the units span, any set-aside preferences (mobility / vision-
hearing / NYCHA / municipal-employee / community-board / senior), and the
application deadline, each a resolvable, filtered DATA citation (re-fetch by
lottery_id -> verify). This dataset is HPD's advertised-lotteries REPORTING feed,
not a live view of the portal, so results are phrased "as of the city's last data
refresh" and the user is always handed off to the portal to confirm and apply.

There is NO per-lottery URL in the data and the application itself is behind a
login, so the handoff goes to the portal HOME (housingconnect.nyc.gov/PublicWeb/):
the finder never fabricates a listing URL and never submits anything on the user's
behalf (human-in-the-loop). Empty results are stated plainly ("no open lotteries as
of the last refresh"), never spun into a fabricated listing, and still deep-link the
portal so the user can watch for new lotteries.
"""
from __future__ import annotations

from datetime import datetime
from urllib.parse import quote
from zoneinfo import ZoneInfo

import httpx
from pydantic import Field

from heynyc.core.citations import data_provenance
from heynyc.core.tools.base import Tool, ToolContext, ToolInput
from heynyc.core.tools.datasets import dataset_url, query_dataset_pages

DATASET_ID = "vy5i-a666"  # Advertised Lotteries on Housing Connect by Lottery
PORTAL = "https://housingconnect.nyc.gov/PublicWeb/"

# Borough is a 2-letter code in the data ('BX'); map to a full name for display and
# reverse-map common user phrasings to the code for the optional borough filter.
_BOROUGH_NAME = {
    "BK": "Brooklyn", "BX": "Bronx", "MN": "Manhattan", "QN": "Queens",
    "SI": "Staten Island", "Multiple": "multiple boroughs",
}
_BOROUGH_CODE = {
    "bk": "BK", "brooklyn": "BK", "kings": "BK",
    "bx": "BX", "bronx": "BX", "the bronx": "BX",
    "mn": "MN", "manhattan": "MN",
    "qn": "QN", "queens": "QN",
    "si": "SI", "staten island": "SI", "staten": "SI",
}

# AMI band columns -> label (each value is a count of units at that band).
_AMI_BANDS = (
    ("applied_income_ami_ext_low", "extremely low"),
    ("applied_income_ami_very_low", "very low"),
    ("applied_income_ami_low", "low"),
    ("applied_income_ami_moderate", "moderate"),
    ("applied_income_ami_middle", "middle"),
    ("applied_income_ami_above", "above"),
)
# Unit-distribution columns -> label (each value is a count of units of that size).
_UNIT_TYPES = (
    ("unit_distribution_studio", "studio"),
    ("unit_distribution_1bed", "1BR"),
    ("unit_distribution_2bed", "2BR"),
    ("unit_distribution_3bed", "3BR"),
    ("unit_distribution_4bed", "4BR"),
)
# Set-aside percent columns -> label (each value is a percent of units set aside).
_SET_ASIDES = (
    ("lottery_mobility_percent", "mobility disability"),
    ("lottery_vision_hearing_percent", "vision/hearing disability"),
    ("lottery_nycha_percent", "NYCHA residents"),
    ("lottery_municipal_employee_percent", "municipal employees"),
    ("lottery_community_board_percent", "community-board residents"),
    ("lottery_62_percent", "seniors 62+"),
)


class HousingConnectInput(ToolInput):
    borough: str | None = Field(default=None, description="NYC borough")
    max_results: int | None = Field(
        default=None,
        ge=1,
        description="Resident-requested listing count",
    )


def _today() -> str:
    """Today's date in NYC ('YYYY-MM-DD'). The injectable seam for "open now": tests
    monkeypatch this to a fixed date so the currently-open slice is deterministic and
    no live clock leaks into the unit tests."""
    return datetime.now(ZoneInfo("America/New_York")).date().isoformat()


def _iso_date(value) -> str:
    """A Socrata datetime ('2026-07-13T00:00:00.000') -> 'YYYY-MM-DD'; '' if blank/odd."""
    text = str(value or "").strip()
    return text[:10] if len(text) >= 10 else ""


def _as_int(value) -> int:
    """A Socrata numeric string ('39') -> int; 0 for blank/missing (Socrata omits null columns)."""
    text = str(value or "").strip()
    try:
        return int(float(text)) if text else 0
    except ValueError:
        return 0


def _filtered_url(where: str) -> str:
    """A resolvable, filtered Socrata permalink: /resource/vy5i-a666.json?$where=<encoded>.
    Re-fetching it returns exactly the row(s) this citation is grounded in (auditable)."""
    return f"{dataset_url(DATASET_ID)}?$where={quote(where)}"


def _borough_code(raw: str) -> str | None:
    """Map a user's borough phrasing ('Bronx', 'bx', 'the bronx') to the data's 2-letter
    code, or None if it isn't a recognizable borough (then we don't filter)."""
    return _BOROUGH_CODE.get(raw.strip().lower())


def _labels_with_count(row: dict, columns) -> list[str]:
    """['very low', 'low', ...] for the AMI columns present with a positive count."""
    return [label for col, label in columns if _as_int(row.get(col)) > 0]


def _mix_with_count(row: dict, columns) -> list[str]:
    """['2 studio', '1 1BR', ...] for unit-distribution/set-aside columns with a positive value."""
    return [f"{_as_int(row.get(col))} {label}" for col, label in columns if _as_int(row.get(col)) > 0]


def _handoff(no_listing: bool = False) -> str:
    lead = (
        "New lotteries post regularly, so it's worth checking back and setting up alerts on the portal"
        if no_listing else
        "These come from HPD's advertised-lotteries reporting feed, not a live view of the portal, so "
        "confirm each deadline and apply on Housing Connect itself"
    )
    return (
        f"{lead}: {PORTAL} . I can't submit an application for you; you create an account, log in, and "
        "apply there before the deadline. There's no direct link to an individual listing in the data, "
        "so search the listing name "
        "on the portal. Don't invent lotteries, deadlines, income limits, unit counts, or set-asides "
        "beyond what's cited above."
    )


async def _handler(args: HousingConnectInput, ctx: ToolContext) -> str:
    borough_arg = (args.get("borough") or "").strip()
    max_results = int(args.get("max_results") or 5)
    today = _today()
    where = f"lottery_status='Active' AND lottery_end_date >= '{today}'"

    boro_code = _borough_code(borough_arg) if borough_arg else None
    if boro_code:
        where += f" AND borough='{boro_code}'"
    boro_note = f" in {_BOROUGH_NAME.get(boro_code, borough_arg)}" if boro_code else ""

    try:
        result = await query_dataset_pages(
            DATASET_ID,
            where=where,
            order="lottery_end_date",
            client=ctx.http,
        )
    except httpx.HTTPError:
        return (
            f"I couldn't reach the city's Housing Connect data right now, so I can't list open "
            f"lotteries. Don't guess what's open. The user can check and apply directly on the portal: "
            f"{PORTAL} , or call 311."
        )

    rows = result.records
    if not rows:
        # State the absence plainly and STILL hand off; ground the "none open" claim in the
        # (empty) filtered query so it is auditable, and never fabricate a listing.
        prov = data_provenance({"open_lotteries": 0, "as_of": today, "borough": boro_code or "all"},
                               record_id="open-slice", field_pointer="/")
        cite = ctx.citations.register(
            _filtered_url(where),
            snippet=f"0 open Housing Connect lotteries{boro_note} as of {today}",
            title="Advertised Lotteries on Housing Connect (vy5i-a666)",
            kind="DATA", valid_as_of=today, provenance=prov,
        )
        return (
            f"I don't see any affordable-housing lotteries open{boro_note} on NYC Housing Connect as of "
            f"the city's last data refresh {{cite:{cite}}}. {_handoff(no_listing=True)}"
        )

    lines = [
        f"Affordable-housing lotteries currently open on NYC Housing Connect{boro_note} "
        f"(as of the city's last data refresh):"
    ]
    shown_rows = rows[:max_results]
    if len(shown_rows) < len(rows):
        lines.append(
            f"Showing {len(shown_rows)} of {len(rows)} matching lotteries. Ask for more to see "
            "the rest."
        )
    if not result.complete:
        lines.append(
            f"The city dataset returned only a partial page set after {result.pages_fetched} "
            "page(s), so more matching lotteries may exist."
        )
    for row in shown_rows:
        lottery_id = str(row.get("lottery_id") or "").strip()
        name = str(row.get("lottery_name") or "a listing").strip()
        boro = _BOROUGH_NAME.get(str(row.get("borough") or "").strip(),
                                 str(row.get("borough") or "").strip())
        units = _as_int(row.get("unit_count"))
        deadline = _iso_date(row.get("lottery_end_date"))
        mix = _mix_with_count(row, _UNIT_TYPES)
        ami = _labels_with_count(row, _AMI_BANDS)
        set_asides = _mix_with_count(row, _SET_ASIDES)
        as_of = _iso_date(row.get(":updated_at"))

        normalized = {
            "lottery_id": lottery_id, "lottery_name": name, "borough": str(row.get("borough") or ""),
            "unit_count": units, "lottery_end_date": deadline,
            "unit_mix": mix, "income_bands_ami": ami, "set_asides": set_asides,
        }
        cite = ctx.citations.register(
            _filtered_url(f"lottery_id='{lottery_id}'"),
            snippet=f"{name} ({boro}): {units} units, apply by {deadline}",
            title="Advertised Lotteries on Housing Connect (vy5i-a666)",
            kind="DATA", valid_as_of=as_of,
            provenance=data_provenance(
                row,
                record_id=lottery_id,
                field_pointer="/",
                derivation=normalized,
            ),
        )

        detail = f"- {name}, {boro}. {units} unit(s)"
        if mix:
            detail += f" ({', '.join(mix)})"
        if ami:
            detail += f". Income bands: {', '.join(ami)} (% of AMI)"
        if deadline:
            detail += f". Deadline to apply: {deadline}"
        if set_asides:
            detail += f". Set-asides: {'; '.join(set_asides)}"
        lines.append(detail + f" {{cite:{cite}}}")

    lines.append(_handoff())
    return "\n".join(lines)


def get_tools() -> list[Tool]:
    return [
        Tool(
            name="find_housing_connect_lotteries",
            description=(
                "List NYC affordable-housing lotteries currently OPEN on Housing Connect (the "
                "affordable-housing lottery), grounded in NYC Open Data (dataset vy5i-a666) and cited. "
                "Returns each open lottery's name, borough, unit count + bedroom mix, income bands (AMI), "
                "set-aside preferences, and the application DEADLINE, then hands off to the portal for the "
                "user to create an account / log in / apply themselves (the apply step is login-gated, so "
                "this never submits anything). Pass optional `borough` (Bronx / Brooklyn / Manhattan / "
                "Queens / Staten Island) to narrow; omit for citywide. It is HPD's advertised-lotteries "
                "reporting feed ('as of the last refresh'), NOT a real-time portal mirror, and there is no "
                "per-listing URL, so it deep-links the portal home. If nothing is open it says so and still "
                "links the portal; it never invents a lottery, deadline, or income limit. Use for 'what "
                "affordable housing lotteries are open, how do I apply for an apartment / the housing "
                "lottery / Housing Connect'."
            ),
            input_type=HousingConnectInput,
            handler=_handler,
            open_world=True,  # hits the live Socrata Housing Connect dataset
        ),
    ]
