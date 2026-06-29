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

from heynyc.core.freshness import staleness_caveat
from heynyc.core.tools.base import Tool, ToolContext
from heynyc.core.tools.datasets import query_dataset

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


def get_tools() -> list[Tool]:
    return [
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
