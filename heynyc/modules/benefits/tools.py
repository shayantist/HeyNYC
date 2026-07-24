"""benefits module tool: NYC Benefits & Programs navigator, grounded in live Socrata data.

Fetches the keyless Benefits & Programs catalog (kvhd-5fmu, ~97 programs) and ranks it
against the user's need with the project's hybrid retriever (the same `Embedder` +
`InMemoryVectorStore` `index_search` uses). Returns grounded program records, each a DATA
citation carrying the program's per-row `updated_at` as `valid_as_of`. The tool never
asserts personalized eligibility; "do I qualify" defers to the official screener / 311.
"""
from __future__ import annotations

import html
import os
import re
import tempfile
import uuid
from datetime import date
from pathlib import Path

import httpx

from heynyc.core import config
from heynyc.core.citations import api_provenance, data_provenance
from heynyc.core.freshness import staleness_caveat
from heynyc.core.tools.base import Tool, ToolContext
from heynyc.core.tools.datasets import query_dataset, row_url
from heynyc.modules.benefits import application as appmod
from heynyc.modules.benefits import screening

_TAG_RE = re.compile(r"<[^>]+>")
_HREF_RE = re.compile(r'href="([^"]+)"', re.IGNORECASE)


def _clean(value) -> str:
    """Normalize a Socrata field: None / literal 'NULL' / blanks → ''; strip any HTML markup.

    Real rows carry HTML in prose fields and the literal string 'NULL' in empty url/text
    fields, both would otherwise leak into grounding or get cited as a fake link."""
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


def _first_href(value) -> str:
    """Pull the first real URL out of an HTML field. The dataset buries per-program deep links
    inside <a href="..."> in prose fields (get_help_online, how_to_apply_summary, ...); _clean()
    strips the markup, so we extract the href here, otherwise the real link is lost and the model
    is tempted to invent one."""
    if not value:
        return ""
    m = _HREF_RE.search(str(value))
    if not m:
        return ""
    url = html.unescape(m.group(1)).strip()
    if url.startswith(("mailto:", "tel:", "#")):
        return ""
    if url.startswith("//"):
        url = "https:" + url
    elif not url.startswith("http") and "." in url.split("/")[0]:
        url = "https://" + url          # scheme-less domain like "nyc.gov/taxprep"
    return url if url.startswith("http") else ""

# The catalog can carry the same program in several official languages (a `language` column). We
# surface the row matching the user's language when asked, English by default / fallback, an
# official city translation beats an LLM paraphrase. ISO codes / native spellings normalize to the
# English language NAME the dataset uses.
DEFAULT_LANG = "english"
_LANG_ALIASES = {
    "en": "english", "es": "spanish", "español": "spanish", "espanol": "spanish",
    "zh": "chinese", "zh-cn": "chinese", "ht": "haitian creole", "ko": "korean", "ru": "russian",
    "bn": "bengali", "ar": "arabic", "ur": "urdu", "fr": "french", "français": "french",
    "pl": "polish", "yi": "yiddish", "it": "italian",
}


def _norm_lang(lang) -> str:
    """A requested language hint → the English language NAME the dataset uses (default English)."""
    key = (lang or "").strip().lower()
    return _LANG_ALIASES.get(key, key) or DEFAULT_LANG


def _row_language(record: dict) -> str:
    return _clean(record.get("language")).lower()


def _program_key(record: dict) -> str:
    """Per-program identity for collapsing language variants: program_code, else program_name."""
    return _clean(record.get("program_code")) or _clean(record.get("program_name")).lower()


def _prefer_language(catalog: list[dict], lang) -> list[dict]:
    """Collapse language-variant rows to ONE row per program, preferring the requested language
    (default English), English fallback, then first-seen. A dataset with no `language` column is
    unaffected, each program appears once, so this is a no-op. Order preserved."""
    target = _norm_lang(lang)

    def rank(row: dict) -> int:
        rl = _row_language(row)
        if rl == target:
            return 3
        if not rl:          # untagged row → acceptable default (English-equivalent)
            return 1
        return 2 if rl == DEFAULT_LANG else 0

    best: dict[str, dict] = {}
    order: list[str] = []
    for i, row in enumerate(catalog):
        key = _program_key(row) or f"__row_{i}"
        if key not in best:
            order.append(key)
            best[key] = row
        elif rank(row) > rank(best[key]):
            best[key] = row
    return [best[k] for k in order]


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
    """The searchable blob for a program, what it IS, HTML-stripped."""
    return " ".join(filter(None, (_clean(record.get(f)) for f in _SEARCH_FIELDS)))


def _retrieve(catalog: list[dict], query: str, limit: int, embedder) -> list[dict]:
    """Rank the catalog against the query with the project's hybrid retriever.

    Reuses the index stack (`Embedder` + `InMemoryVectorStore`, the same hybrid
    semantic+keyword scoring `index_search` uses) rather than a bespoke ranker.
    Socrata's server-side `$q` is conjunctive, a verbose query like "food stamps
    SNAP WIC" returns ZERO (caught by the live eval), so we fetch the small catalog
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
    """Per-program 'as of' date, the row's updated_at, truncated to YYYY-MM-DD."""
    return _clean(record.get("updated_at"))[:10]


def _apply_url(record: dict) -> str:
    return _clean(record.get("url_of_online_application")) or _clean(
        record.get("url_of_pdf_application_forms")
    )


_HELP_FIELDS = ("get_help_online", "get_help_summary", "how_to_apply_summary",
                "how_to_apply_or_enroll_online")


def _help_url(record: dict) -> str:
    """A real 'learn more / get help' deep link for the program, the href buried in its help /
    how-to-apply prose, or its office-locations map. Empty if the row carries none."""
    for f in _HELP_FIELDS:
        u = _first_href(record.get(f))
        if u:
            return u
    return _clean(record.get("office_locations_url"))


def _best_url(record: dict) -> str:
    """The most specific REAL url for a program's citation: the apply url, else a help/how-to deep
    link, else the dataset landing page as a last resort, never the model's invention."""
    return _apply_url(record) or _help_url(record) or SOURCE_URL


def _citation_snippet(record: dict) -> str:
    """Put the facts most likely to be cited where the evaluator and footer can see them."""
    fields = (
        "program_name",
        "plain_language_program_name",
        "how_to_apply_summary",
        "get_help_summary",
        "get_help_online",
        "plain_language_eligibility",
        "heads_up",
    )
    return " | ".join(filter(None, (_clean(record.get(field)) for field in fields)))[:1000]


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
    else:
        help_url = _help_url(record)
        if help_url:
            parts.append(f"  Learn more / get help: {help_url}")
    parts.append(f"  As of: {as_of or 'unknown'}")
    caveat = staleness_caveat(as_of, today, FRESHNESS_DAYS)
    if caveat:
        parts.append(f"  {caveat}")
    return "\n".join(parts)


async def _handler(args: dict, ctx: ToolContext) -> str:
    query = (args.get("query") or "").strip()
    category = (args.get("category") or "").strip()
    lang = (args.get("lang") or "").strip() or None
    limit = int(args.get("limit") or 8)

    # Allowlist the category before it reaches the SoQL $where clause. The JSON-schema enum
    # is advisory only; this is the actual SoQL-injection guard.
    if category and category not in CATEGORIES:
        return (
            f"ERROR: unknown benefits category '{category}'. Choose exactly one of: "
            f"{', '.join(CATEGORIES)}, or omit it and rely on the query."
        )
    where = f"program_category='{category}'" if category else None
    try:
        # Fetch the catalog (optionally category-scoped), then rank in-memory, `$q` is
        # conjunctive and drops verbose queries (see _rank). 200 covers all ~97 rows.
        catalog = await query_dataset(DATASET_ID, where=where, limit=200, client=ctx.http)
    except httpx.HTTPError:
        return (
            "ERROR: couldn't reach the NYC Benefits & Programs dataset right now. "
            f"Don't guess, tell the user to try {OFFICIAL}."
        )

    # Collapse any language-variant rows to one per program, preferring the user's language
    # (English by default / fallback) so the official translation surfaces, then rank.
    catalog = _prefer_language(catalog, lang)
    records = _retrieve(catalog, query, limit, ctx.embedder)
    if not records:
        return (
            f"No NYC benefit programs in the dataset matched '{query}'. "
            f"Don't fabricate one, suggest {OFFICIAL}."
        )

    today = date.today().isoformat()
    blocks = []
    for record in records:
        as_of = _as_of(record)
        name = _clean(record.get("program_name"))
        record_id = _clean(record.get(":id")) or _clean(record.get("program_code")) or _program_key(record)
        cite = ctx.citations.register(
            _best_url(record),
            snippet=_citation_snippet(record),
            title=name or "NYC benefit program",
            kind="DATA",
            valid_as_of=as_of,
            provenance=data_provenance(record, record_id=record_id, field_pointer="/"),
        )
        blocks.append(_block(record, cite, as_of, today))

    header = (
        "NYC benefit programs from the city's Benefits & Programs dataset. Eligibility text is "
        "general guidance with an 'as of' date, NOT a personalized determination. An 'as of' "
        "date records the dataset update; it does not prove a rule or limit is current today. "
        "Confirm exact current amounts and rules on the official program page. Route any "
        f"'do I qualify' to {OFFICIAL}:\n"
    )
    return header + "\n".join(blocks)


_ESTIMATE = ("an estimate from NYC's official screener (ACCESS NYC), not a determination; "
             "you'll need to apply to find out for sure")
_ACCESS_NYC = "https://access.nyc.gov/eligibility/"


async def _screen_handler(args: dict, ctx: ToolContext) -> str:
    try:
        screening.validate_arguments(args)
        household = dict(args.get("household") or {})
        persons = list(args.get("persons") or [])
        interested = args.get("interested_programs") or None
        lang = (args.get("lang") or "").strip() or None
        goal = (args.get("goal") or "").strip()
        show_all = bool(args.get("show_all"))
        screening.assert_pii_free(household, persons)
    except ValueError as exc:
        return f"ERROR: {exc} Collect only age, household type, and income, never names/DOB/address."
    base, user, pw = config.screening_creds()
    if not (user and pw):
        return ("ERROR: eligibility screening isn't configured. Use benefits_search and tell the "
                f"user to check {OFFICIAL}.")

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
            elif exc.response.status_code == 400:
                try:
                    payload = exc.response.json()
                except ValueError:
                    payload = {}
                raw_errors = payload.get("errors", []) if isinstance(payload, dict) else []
                if not isinstance(raw_errors, list):
                    raw_errors = []
                errors = "; ".join(
                    item.get("message", "") for item in raw_errors
                    if isinstance(item, dict) and item.get("message")
                )
                detail = errors or "the profile did not match the City's request contract"
                return f"ERROR: the screener rejected the inputs ({detail}). Re-ask only for that field."
            elif exc.response.status_code == 504:          # screening timeout, NEVER a false negative
                return ("ERROR: the screener is busy right now, don't tell the user they're ineligible; "
                        "ask them to try again in a moment.")
            else:
                raise
        if (result.get("type") or "").upper() == "FAILURE":
            errs = "; ".join(e.get("message", "") for e in result.get("errors", []))
            return f"ERROR: the screener rejected the inputs ({errs}). Re-ask for the missing field."
        eligible = result.get("eligiblePrograms") or []
        try:
            catalog = await query_dataset(DATASET_ID, limit=200, client=client)
            if not isinstance(catalog, list) or not all(isinstance(row, dict) for row in catalog):
                catalog = []
        except (httpx.HTTPError, ValueError):
            # Catalog details enrich a successful screening verdict, but are not the verdict.
            # Keep the API names and provenance if Socrata is temporarily unavailable.
            catalog = []
    except ValueError as exc:
        return f"ERROR: {exc}. Rebuild the profile using only the documented screening fields."
    except httpx.HTTPError:
        return f"ERROR: couldn't reach the screener right now. Don't guess, point the user to {OFFICIAL}."
    finally:
        if own:
            await client.aclose()

    today = date.today().isoformat()
    # Collapse language-variant rows to one per program, preferring the user's language (English by
    # default / fallback), then index by program_code for the verdict lookup below.
    by_code: dict[str, dict] = {}
    for row in _prefer_language(catalog, lang):
        code = _clean(row.get("program_code"))
        if code and code not in by_code:
            by_code[code] = row

    verdict = ctx.citations.register(
        _ACCESS_NYC,
        snippet=f"NYC Benefits Screening API, likely-eligible estimate ({len(eligible)} program(s))",
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
                f"{{cite:{verdict}}}. That is NOT a determination of ineligibility, encourage the user "
                f"to apply or check {OFFICIAL}; more detail may surface more programs.")

    if len(eligible) > 3 and not show_all:
        if not goal:
            return (
                f"The official screener returned {len(eligible)} likely matches {_ESTIMATE} "
                f"{{cite:{verdict}}}. Which need matters most right now? Tell me in your own words, "
                "then reply `/screen` again. Reply `/screen all` if you want every match."
            )
        eligible_rows = [by_code[program.get("code", "")] for program in eligible
                         if program.get("code", "") in by_code]
        ranked_rows = _retrieve(eligible_rows, goal, 3, ctx.embedder)
        programs_by_code = {program.get("code", ""): program for program in eligible}
        displayed = [programs_by_code[_clean(row.get("program_code"))] for row in ranked_rows]
        if not displayed:
            return (
                f"The official screener returned {len(eligible)} likely matches {_ESTIMATE} "
                f"{{cite:{verdict}}}, but I couldn't reliably match them to '{goal}'. Tell me a "
                "different need, then reply `/screen` again, or reply `/screen all`."
            )
    else:
        displayed = eligible

    match_word = "match" if len(eligible) == 1 else "matches"
    lines = [f"Based on what you shared, you're likely eligible for {len(eligible)} {match_word}, "
             f"{_ESTIMATE} {{cite:{verdict}}}:"]
    if len(displayed) < len(eligible):
        lines.append("This is a phone-friendly shortlist, not an official ranking.")
    for prog in displayed:
        code, name = prog.get("code", ""), prog.get("name", "")
        row = by_code.get(code)
        if row:
            rid = _clean(row.get(":id"))
            cite = ctx.citations.register(
                row_url(DATASET_ID, rid) if rid else SOURCE_URL,
                snippet=f"{_clean(row.get('program_name')) or name}, likely eligible (program_code {code})",
                title=_clean(row.get("program_name")) or name, kind="DATA", valid_as_of=_as_of(row),
                provenance=data_provenance(row, record_id=rid, field_pointer="/"))
            url = _apply_url(row)
            lines.append(f"- {name} ({_clean(row.get('program_category'))}) {{cite:{cite}}}"
                         + (f", apply: {url}" if url else ""))
        else:  # screenable but not in our catalog cache, fall back to the API's name, cite the verdict
            lines.append(f"- {name} {{cite:{verdict}}}")
    remaining = len(eligible) - len(displayed)
    if remaining:
        lines.append(f"There {'is' if remaining == 1 else 'are'} {remaining} other "
                     f"match{'es' if remaining != 1 else ''}. Reply `/screen all` to see them.")
    lines.append("A program not listed here doesn't mean you're ineligible. Want help applying?")
    return "\n".join(lines)


def screen_eligibility_tool() -> Tool:
    return Tool(
        name="screen_eligibility",
        description=(
            "Estimate which NYC benefit programs a household is LIKELY eligible for, via the city's "
            "official Benefits Screening API (the ACCESS NYC rules engine). Pass a PII-FREE profile "
            "gathered from the user: household flags + a list of persons (age + householdMemberType "
            "required; optional income/flags). Pass goal only when the resident stated a need and "
            "show_all only when they explicitly requested every match. NEVER pass names, DOB, SSN, "
            "or address. Returns a "
            "likely-eligible estimate (NOT a determination); a program's absence is never proof of "
            "ineligibility."),
        parameters=screening.request_schema(),
        handler=_screen_handler,
        open_world=True,
        resident_fact_scope=("/household", "/persons"),
    )


def _forms_enabled() -> bool:
    return os.getenv("HEYNYC_FORMS", "").lower() in ("1", "true", "yes")


async def _prepare_application_handler(args: dict, ctx: ToolContext) -> str:
    """Prepare a draft LDSS-4826 from CONFIRMED, user-provided answers. Two steps:
    confirmed=false → return the field-level review + two-tier attestation (no PDF produced);
    confirmed=true → render the PDF and hand it back. Fills only provided slots, never invents,
    never logs PII, and degrades if the official form has drifted (integrity guard)."""
    incoming = dict(args.get("slots") or {})
    confirmed = bool(args.get("confirmed"))
    if getattr(ctx, "drafts", None) is not None:
        if confirmed:
            if incoming:
                return ("NEED_REVIEW: confirmed fields cannot change. Save the edits with "
                        "confirmed=false, show the new review, then ask for confirmation again.")
            raw = ctx.drafts.load("snap")
        else:
            raw = ctx.drafts.merge("snap", incoming)
    else:
        raw = incoming
    clean, missing, errors = appmod.validate_slots(raw)
    if errors:
        return "NEED_FIX: " + "; ".join(errors) + ", re-ask the user only for these; never guess."
    if missing:
        labels = ", ".join(s.label for s in appmod.SLOTS if s.key in missing)
        return (f"NEED_MORE: still need {labels}. Ask the user for these in plain language, "
                f"do not fill them in yourself.")
    if not appmod.verify_template_integrity():
        url = appmod.template_provenance().get("source_url", "otda.ny.gov")
        return (f"CANNOT_FILL: the official form may have changed, don't auto-fill it. Send the "
                f"user the blank form at {url} and offer to walk them through it instead.")
    if not confirmed:                                  # the meaningful-attestation gate
        return "REVIEW: " + appmod.review_request(clean)
    try:
        pdf = appmod.fill_application(clean)           # bytes; values never logged
    except appmod.FormDriftError:
        url = appmod.template_provenance().get("source_url", "otda.ny.gov")
        return f"CANNOT_FILL: the form's layout changed, don't auto-fill. Blank form: {url}."
    out_dir = Path(ctx.output_dir) if getattr(ctx, "output_dir", None) else Path(tempfile.mkdtemp())
    out_path = out_dir / f"snap-ldss4826-{uuid.uuid4().hex[:8]}.pdf"
    out_path.write_bytes(pdf)
    # Completion: the filled PDF exists, so the accumulated PII draft has served its purpose. Clear it
    # now (retention / data-minimization, security finding F1) instead of waiting for the TTL sweep.
    if getattr(ctx, "drafts", None) is not None:
        ctx.drafts.clear("snap")
    # The PDF is delivered out-of-band by the channel from the request's artifacts dir; we do NOT
    # put the filesystem path in the text the model sees (defense-in-depth against a path leak).
    return appmod.application_summary(clean, missing) + "\n(Your filled draft is attached as a document.)"


def prepare_application_tool() -> Tool:
    props = {s.key: {"type": "string",
                     "description": s.label + (", read back for the user to re-confirm"
                                               if s.high_stakes else "")}
             for s in appmod.SLOTS}
    return Tool(
        name="prepare_snap_application",
        description=(
            "Prepare a DRAFT of the official NYS SNAP application (LDSS-4826) for the user to print, "
            "sign, and mail THEMSELVES, it is never submitted for them and is not a determination. "
            "You are a scribe: only transcribe answers the USER gave; never invent, infer, or coach a "
            "value, and never decide eligibility. Two steps: (1) call with confirmed=false to get a "
            "field-level review + the attestation, show it to the user, and have them re-confirm the "
            "high-stakes fields (name, DOB, SSN, income) in their own words; (2) only AFTER they "
            "confirm, call again with confirmed=true to get the draft. If the tool returns NEED_MORE / "
            "NEED_FIX / CANNOT_FILL, follow it. Never promise approval."),
        parameters={
            "type": "object",
            "properties": {
                "slots": {
                    "type": "object",
                    "description": ("Answers the user gave, keyed by field name. PII only, never "
                                    "logged or sent anywhere but the user's own draft. With a "
                                    "persisted reviewed draft, pass an empty object when confirmed=true."),
                    "properties": props,
                },
                "confirmed": {
                    "type": "boolean",
                    "description": ("False (or omit) to get the review first; true ONLY after the "
                                    "user has reviewed and confirmed their answers."),
                },
            },
            "required": ["slots"],
        },
        handler=_prepare_application_handler,
        read_only=False, idempotent=False,    # writes a file
        requires_approval=True,                # side-effecting: produces a signable artifact
        open_world=False,
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
                    "lang": {
                        "type": "string",
                        "description": "Optional language NAME (e.g. 'Spanish'), pass the language "
                        "the user is writing in. When the dataset carries an official translation of "
                        "a program, the matching-language row is returned; English by default and as "
                        "the fallback. Program names, apply links, and 'as of' dates stay as given.",
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
    if _forms_enabled():  # the form-fill scribe appears only when HEYNYC_FORMS is set
        tools.append(prepare_application_tool())
    return tools
