"""benefits module tool: NYC Benefits & Programs navigator, grounded in live Socrata data.

Fetches the keyless Benefits & Programs catalog (kvhd-5fmu, ~97 programs) and ranks it
against the user's need with the project's hybrid retriever (the same `Embedder` +
`InMemoryVectorStore` `index_search` uses). Returns grounded program records — each a DATA
citation carrying the program's per-row `updated_at` as `valid_as_of`. The tool never
asserts personalized eligibility; "do I qualify" defers to the official screener / 311.
"""
from __future__ import annotations

import html
import re
from datetime import date

import httpx

from heynyc.core import config
from heynyc.core.citations import api_provenance, data_provenance
from heynyc.core.freshness import staleness_caveat
from heynyc.core.tools.base import Tool, ToolContext
from heynyc.core.tools.datasets import query_dataset, row_url
from heynyc.modules.benefits import screening

_TAG_RE = re.compile(r"<[^>]+>")


def _clean(value) -> str:
    """Normalize a Socrata field: None / literal 'NULL' / blanks → ''; strip any HTML markup.

    Real rows carry HTML in prose fields and the literal string 'NULL' in empty url/text
    fields — both would otherwise leak into grounding or get cited as a fake link."""
    if value is None:
        return ""
    text = str(value).strip()
    if text.upper() == "NULL":
        return ""
    if "<" in text and ">" in text:
        text = re.sub(r"<li[^>]*>", " • ", text, flags=re.IGNORECASE)
        text = _TAG_RE.sub(" ", text)
        text = html.unescape(text)
        text = re.sub(r"\s+", " ", text).strip()
    return text

DATASET_ID = "kvhd-5fmu"
FRESHNESS_DAYS = 365  # §12 staleness guard: benefits eligibility rules re-check annually
SOURCE_URL = (
    "https://data.cityofnewyork.us/Social-Services/"
    "NYC-Benefits-Platform-Benefits-and-Programs-Datase/kvhd-5fmu"
)
CATEGORIES = [
    "Cash & expenses", "Child Care", "City ID Card", "Education", "Enrichment",
    "Family Services", "Food", "Health", "Housing", "Work",
]
OFFICIAL = "the official screener at https://access.nyc.gov or call 311"

# Fields whose text we match the user's need against (cleaned of HTML, then embedded).
_SEARCH_FIELDS = (
    "program_name", "plain_language_program_name", "program_acronym", "brief_excerpt",
    "program_description", "program_category", "population_served",
)


def _doc_text(record: dict) -> str:
    """The searchable blob for a program — what it IS, HTML-stripped."""
    return " ".join(filter(None, (_clean(record.get(f)) for f in _SEARCH_FIELDS)))


def _retrieve(catalog: list[dict], query: str, limit: int, embedder) -> list[dict]:
    """Rank the catalog against the query with the project's hybrid retriever.

    Reuses the index stack (`Embedder` + `InMemoryVectorStore`, the same hybrid
    semantic+keyword scoring `index_search` uses) rather than a bespoke ranker.
    Socrata's server-side `$q` is conjunctive — a verbose query like "food stamps
    SNAP WIC" returns ZERO (caught by the live eval) — so we fetch the small catalog
    and rank locally. Hybrid retrieval over a small structured catalog is the
    documented best practice (lexical catches acronyms like SNAP/IDNYC; embeddings
    catch meaning like "can't afford food" → SNAP). The embedded store is cached
    (`index.cache`) so the catalog is embedded once per (content, model), not per call."""
    from heynyc.core.index import default_embedder
    from heynyc.core.index.cache import embedded_store
    from heynyc.core.index.store import IndexDoc

    if not catalog:
        return []
    embedder = embedder or default_embedder()
    docs = [
        IndexDoc(id=str(i), text=_doc_text(r), title=_clean(r.get("program_name")))
        for i, r in enumerate(catalog)
    ]
    store = embedded_store(docs, embedder)  # embeds once per (content, model), then reuses
    # RRF ranks; it has no absolute relevance floor (every doc gets a positive rank score), so
    # we return the top-k candidates and let the agent judge relevance / abstain (agent-as-judge).
    hits = store.search(embedder.embed([query])[0], query, k=limit)
    return [catalog[int(doc.id)] for doc, _score in hits]


def _as_of(record: dict) -> str:
    """Per-program 'as of' date — the row's updated_at, truncated to YYYY-MM-DD."""
    return _clean(record.get("updated_at"))[:10]


def _apply_url(record: dict) -> str:
    return _clean(record.get("url_of_online_application")) or _clean(
        record.get("url_of_pdf_application_forms")
    )


def _block(record: dict, cite: str, as_of: str, today: str) -> str:
    name = _clean(record.get("program_name"))
    parts = [f"- {name} ({_clean(record.get('program_category'))}) {{cite:{cite}}}"]
    plain = _clean(record.get("plain_language_program_name"))
    if plain:
        parts.append(f"  What it is: {plain}")
    elig = _clean(record.get("plain_language_eligibility"))
    if elig:
        parts.append(f"  Who may roughly qualify (general guidance, NOT a determination): {elig}")
    heads = _clean(record.get("heads_up"))
    if heads:
        parts.append(f"  Heads up: {heads}")
    how = _clean(record.get("how_to_apply_summary"))
    if how:
        parts.append(f"  How to apply: {how}")
    docs = _clean(record.get("required_documents_summary"))
    if docs:
        parts.append(f"  Documents: {docs}")
    url = _apply_url(record)
    if url:
        parts.append(f"  Apply: {url}")
    parts.append(f"  As of: {as_of or 'unknown'}")
    caveat = staleness_caveat(as_of, today, FRESHNESS_DAYS)
    if caveat:
        parts.append(f"  {caveat}")
    return "\n".join(parts)


async def _handler(args: dict, ctx: ToolContext) -> str:
    query = (args.get("query") or "").strip()
    category = (args.get("category") or "").strip()
    limit = int(args.get("limit") or 8)

    # Allowlist the category before it reaches the SoQL $where clause. The JSON-schema enum
    # is advisory only; this is the actual SoQL-injection guard.
    if category and category not in CATEGORIES:
        return (
            f"ERROR: unknown benefits category '{category}'. Choose exactly one of: "
            f"{', '.join(CATEGORIES)} — or omit it and rely on the query."
        )
    where = f"program_category='{category}'" if category else None
    try:
        # Fetch the catalog (optionally category-scoped), then rank in-memory — `$q` is
        # conjunctive and drops verbose queries (see _rank). 200 covers all ~97 rows.
        catalog = await query_dataset(DATASET_ID, where=where, limit=200, client=ctx.http)
    except httpx.HTTPError:
        return (
            "ERROR: couldn't reach the NYC Benefits & Programs dataset right now. "
            f"Don't guess — tell the user to try {OFFICIAL}."
        )

    records = _retrieve(catalog, query, limit, ctx.embedder)
    if not records:
        return (
            f"No NYC benefit programs in the dataset matched '{query}'. "
            f"Don't fabricate one — suggest {OFFICIAL}."
        )

    today = date.today().isoformat()
    blocks = []
    for record in records:
        as_of = _as_of(record)
        name = _clean(record.get("program_name"))
        cite = ctx.citations.register(
            _apply_url(record) or SOURCE_URL,
            snippet=f"{name} — {_clean(record.get('plain_language_program_name'))}",
            title=name or "NYC benefit program",
            kind="DATA",
            valid_as_of=as_of,
        )
        blocks.append(_block(record, cite, as_of, today))

    header = (
        "NYC benefit programs from the city's Benefits & Programs dataset. Eligibility text is "
        "general guidance with an 'as of' date, NOT a personalized determination — route any "
        f"'do I qualify' to {OFFICIAL}:\n"
    )
    return header + "\n".join(blocks)


_ESTIMATE = ("an estimate from NYC's official screener (ACCESS NYC), not a determination; "
             "you'll need to apply to find out for sure")
_ACCESS_NYC = "https://access.nyc.gov/eligibility/"


async def _screen_handler(args: dict, ctx: ToolContext) -> str:
    household = dict(args.get("household") or {})
    persons = list(args.get("persons") or [])
    interested = args.get("interested_programs") or None
    base, user, pw = config.screening_creds()
    if not (user and pw):
        return ("ERROR: eligibility screening isn't configured. Use benefits_search and tell the "
                f"user to check {OFFICIAL}.")
    try:
        screening.assert_pii_free(household, persons)
    except ValueError as exc:
        return f"ERROR: {exc} Collect only age, household type, and income — never names/DOB/address."

    own = ctx.http is None
    client = ctx.http or httpx.AsyncClient(timeout=30.0)
    try:
        token = await screening.get_token(client, base, user, pw)
        try:
            result = await screening.screen(client, base, token, household, persons, interested)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 401:            # token went stale → re-auth once
                screening.clear_token(base)
                token = await screening.get_token(client, base, user, pw)
                result = await screening.screen(client, base, token, household, persons, interested)
            elif exc.response.status_code == 504:          # screening timeout — NEVER a false negative
                return ("ERROR: the screener is busy right now — don't tell the user they're ineligible; "
                        "ask them to try again in a moment.")
            else:
                raise
        if (result.get("type") or "").upper() == "FAILURE":
            errs = "; ".join(e.get("message", "") for e in result.get("errors", []))
            return f"ERROR: the screener rejected the inputs ({errs}). Re-ask for the missing field."
        eligible = result.get("eligiblePrograms") or []
        catalog = await query_dataset(DATASET_ID, limit=200, client=client)
    except httpx.HTTPError:
        return f"ERROR: couldn't reach the screener right now. Don't guess — point the user to {OFFICIAL}."
    finally:
        if own:
            await client.aclose()

    today = date.today().isoformat()
    by_code: dict[str, dict] = {}
    for row in catalog:
        code = _clean(row.get("program_code"))
        # prefer the English row if the dataset carries language variants
        if code and (code not in by_code or _clean(row.get("language")).lower() == "english"):
            by_code[code] = row

    verdict = ctx.citations.register(
        _ACCESS_NYC,
        snippet=f"NYC Benefits Screening API — likely-eligible estimate ({len(eligible)} program(s))",
        title="NYC Benefits Screening (ACCESS NYC)",
        kind="DATA", valid_as_of=today,
        provenance=api_provenance(
            endpoint=f"POST {base}/eligibilityPrograms",
            request_summary=screening.request_summary(household, persons),
            response={"eligiblePrograms": eligible},
            field_pointer="/eligiblePrograms", as_of=today),
    )
    if not eligible:
        return ("Based on what you shared, the screener didn't return any likely-eligible programs "
                f"{{cite:{verdict}}}. That is NOT a determination of ineligibility — encourage the user "
                f"to apply or check {OFFICIAL}; more detail may surface more programs.")

    lines = [f"Based on what you shared, you're likely eligible for these — {_ESTIMATE} {{cite:{verdict}}}:"]
    for prog in eligible:
        code, name = prog.get("code", ""), prog.get("name", "")
        row = by_code.get(code)
        if row:
            rid = _clean(row.get(":id"))
            cite = ctx.citations.register(
                row_url(DATASET_ID, rid) if rid else SOURCE_URL,
                snippet=f"{_clean(row.get('program_name')) or name} — likely eligible (program_code {code})",
                title=_clean(row.get("program_name")) or name, kind="DATA", valid_as_of=_as_of(row),
                provenance=data_provenance(row, record_id=rid, field_pointer="/"))
            url = _apply_url(row)
            lines.append(f"- {name} ({_clean(row.get('program_category'))}) {{cite:{cite}}}"
                         + (f" — apply: {url}" if url else ""))
        else:  # screenable but not in our catalog cache — fall back to the API's name, cite the verdict
            lines.append(f"- {name} {{cite:{verdict}}}")
    lines.append("A program not listed here doesn't mean you're ineligible. "
                 "Want help applying to any of these?")
    return "\n".join(lines)


def screen_eligibility_tool() -> Tool:
    return Tool(
        name="screen_eligibility",
        description=(
            "Estimate which NYC benefit programs a household is LIKELY eligible for, via the city's "
            "official Benefits Screening API (the ACCESS NYC rules engine). Pass a PII-FREE profile "
            "gathered from the user: household flags + a list of persons (age + householdMemberType "
            "required; optional income/flags). NEVER pass names, DOB, SSN, or address. Returns a "
            "likely-eligible estimate (NOT a determination); a program's absence is never proof of "
            "ineligibility."),
        parameters={
            "type": "object",
            "properties": {
                "household": {"type": "object", "description":
                    "Household-level flags: livingRenting, livingRentalType (enum), livingOwner, "
                    "livingShelter, cashOnHand (number). PII-free."},
                "persons": {"type": "array", "description":
                    "1-8 people; at least one householdMemberType='HeadOfHousehold'.",
                    "items": {"type": "object", "properties": {
                        "age": {"type": "integer"},
                        "householdMemberType": {"type": "string"},
                        "incomes": {"type": "array", "items": {"type": "object", "properties": {
                            "amount": {"type": "string"}, "type": {"type": "string"},
                            "frequency": {"type": "string"}}}},
                        "student": {"type": "boolean"}, "pregnant": {"type": "boolean"},
                        "disabled": {"type": "boolean"}, "veteran": {"type": "boolean"},
                        "unemployed": {"type": "boolean"}},
                        "required": ["age", "householdMemberType"]}},
                "interested_programs": {"type": "array", "items": {"type": "string"},
                    "description": "Optional program-code filter."},
            },
            "required": ["persons"],
        },
        handler=_screen_handler,
        open_world=True,
    )


def get_tools() -> list[Tool]:
    tools = [
        Tool(
            name="benefits_search",
            description=(
                "Search NYC benefit/assistance programs (SNAP, Fair Fares, HEAP, SCRIE/DRIE, "
                "IDNYC, WIC, Cash Assistance, child care, etc.) in the city's live Benefits & "
                "Programs dataset. Pass `query` as the user's need in plain words; optionally "
                "filter by `category`. Returns grounded program info, rough eligibility, apply "
                "steps, and an 'as of' date. Use for benefit/eligibility questions; it does NOT "
                "make personalized eligibility determinations (route those to the official screener)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The user's need in plain words (e.g. 'help paying rent', 'food stamps', 'child care costs').",
                    },
                    "category": {
                        "type": "string",
                        "enum": CATEGORIES,
                        "description": "Optional category filter.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max programs to return (default 8).",
                    },
                },
                "required": ["query"],
            },
            handler=_handler,
            open_world=True,  # hits the live Socrata dataset
        )
    ]
    _, user, pw = config.screening_creds()
    if user and pw:  # the screener only appears when its API creds are configured
        tools.append(screen_eligibility_tool())
    return tools
