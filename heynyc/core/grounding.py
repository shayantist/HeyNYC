"""Deterministic cited-claim grounding, shared by the eval gate (Part C) and the runtime guard.

For every ``{cite:Sn}`` marker in an answer, verify the SALIENT STRUCTURED facts in that sentence
(law/section numbers via addresses & unit numbers, phone numbers, dollar amounts, verbatim
unit-bearing numbers, street addresses, proper-noun names) actually occur in the source that marker
points at, or in the user's own query. Pure in-memory string/token matching: DATA snapshots and
DOC/WEB snippets were captured at query time, so this is a string match, never a re-fetch (near-free,
no latency, no LLM, no API).

This module is the single implementation used by BOTH:
  • the eval harness (``eval/checks.py::check_cited_claim_grounding`` is a thin wrapper), and
  • the RUNTIME guard (``core/agent.py`` runs it on the final answer before it reaches the user).

THE #1 RULE: never false-fail a grounded answer. So we (a) extract only HIGH-SIGNAL, specific tokens;
(b) normalize aggressively before matching; (c) count the user's own QUERY as a legitimate source (the
agent restating the location the user gave, an origin address, a neighborhood, is not a
hallucination); (d) match leniently (digit-runs, substring, allow-one-word-miss) so we err toward a
false PASS; and (e) only BLOCK on a fact whose absence is CONCLUSIVE, see the severity model below.
Derived values (distances, counts) are deliberately NOT extracted, check_data_grounding re-derives
distances; counts are the model's own tally across rows.

Two things determine severity, tuned against real module-eval traces (validation in test_checks.py):
  • TOKEN KIND, a verbatim structured fact (phone / dollar amount / street address / unit number) is
    copied literally from a source; a multi-word PROPER-NOUN name drifts too much to trust (acronyms
    vs the spelled-out program, a correct neighborhood the row omits, plural/typo, aggregated DOC
    snippets), so a name mismatch is only ever SOFT (informational).
  • SOURCE COMPLETENESS, a DATA `snapshot` / API `response` is the WHOLE captured source, so a fact's
    absence is conclusive; a DOC/WEB snippet or a label-only catalog row is a TRUNCATED excerpt, so the
    fact may live in the un-captured remainder → absence proves nothing.
A mismatch BLOCKS only when the token is a structured fact AND every cited source is a complete
capture. Everything else is SOFT. Validation showed this yields ZERO blocking false-fails across all
modules while a fabricated phone/address/amount cited to a DATA snapshot still hard-fails.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Optional

# Toggle blocking off here if a future module surfaces a hard false-fail. (Kept as the shared source of
# truth; eval/checks.py re-exports this name so its tests still import it from there.)
_CITED_CLAIM_GROUNDING_BLOCKING = True

_CITE_REF_RE = re.compile(r"\{cite:(S\d+)\}")
_WS_RE = re.compile(r"\s+")
# A 10-digit US phone in any punctuation, optional leading country code. Bounded so it can't grab a
# fragment of a longer digit run.
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?1[\s.\-]?)?\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}(?!\d)")
_MONEY_RE = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?")
_NUMBER_VALUE_RE = re.compile(r"(?<![\w])\d[\d,]*(?:\.\d+)?(?![\w])")
# A number carrying a VERBATIM unit (temperature / percent), a fact quoted from source text, not a
# computed/rounded value. Distances/times/counts are intentionally excluded (reformatting → false-fail).
_UNIT_NUM_RE = re.compile(r"(\d{1,4})\s*(?:°\s*[fc]?|degrees?\b|%|percent\b)", re.IGNORECASE)
_STREET_TYPES = {
    "street", "st", "avenue", "ave", "av", "boulevard", "blvd", "road", "rd", "place", "pl",
    "drive", "dr", "lane", "ln", "court", "ct", "parkway", "pkwy", "plaza", "terrace", "ter",
    "way", "concourse", "broadway", "expressway", "turnpike", "square", "highway", "hwy",
}
# Compass directions in addresses, the source abbreviates ("E"/"W"), the agent often spells them out
# ("East"/"West"), so they must never be a required match word.
_DIRECTIONS = {"north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest"}
# A street address: a house number, then street-name words that must each be Capitalized or a
# numbered/ordinal street ("107th"), then a street type. Requiring capitalized middle words stops the
# pattern from greedily eating a lowercase run like "9 miles from 2920 Broadway" (a distance + origin)
# as a single fake address. The street type is the only case-insensitive part (scoped inline flag).
_ADDRESS_RE = re.compile(
    r"\b(\d{1,5})\s+"
    r"((?:(?:[A-Z][A-Za-z.'&\-]*|\d+(?:st|nd|rd|th))\s+){0,4}?)"
    r"(?i:(street|st|avenue|ave|av|boulevard|blvd|road|rd|place|pl|drive|dr|lane|ln|court|ct|"
    r"parkway|pkwy|plaza|terrace|ter|way|concourse|broadway|expressway|turnpike|highway|hwy))"
    r"\b"
)
_CAP = r"[A-Z][A-Za-z&.'’\-]*"
_CONNECTOR = r"(?:of|the|and|at|for|to|on|de|la|el|&)"
# 2+ consecutive Capitalized words (small connectors allowed between), e.g. "New York Common Pantry",
# "Right to Counsel", "Sedgwick Library".
_PROPER_NOUN_RE = re.compile(rf"{_CAP}(?:\s+(?:{_CONNECTOR}\s+)?{_CAP})+")
# Generic proper-noun words that are NOT source-specific facts: a claim mentioning "New York City",
# "Google Maps", or a borough asserts nothing the cited row must contain. Stripping these is the main
# guard against false-failing on civic/geographic boilerplate the agent adds for readability.
_GENERIC_PN_WORDS = {
    "new", "nueva", "york", "city", "ciudad", "nyc", "manhattan", "brooklyn", "queens", "bronx", "staten", "island",
    "united", "states", "america", "usa", "google", "maps", "map", "borough", "the", "and", "for",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july", "august", "september",
    "october", "november", "december", "today", "tonight", "tomorrow",
    # day/month ABBREVIATIONS, an answer's "Fri Jul 3" is a date, not a source-specific fact.
    "mon", "tue", "tues", "wed", "weds", "thu", "thur", "thurs", "fri", "sat", "sun",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
}
# Common English words that are routinely Capitalized at a sentence/clause start (imperatives the agent
# opens with, plus function words) but are NOT source-specific facts. Stripping them from a proper
# noun's significant words prevents the #1 failure mode, a leading verb like "Reach" or "Under"
# getting swept into "Reach Sedgwick Library" and then not found in the row. Stripping only ever makes
# the check MORE lenient; a real place/program name always keeps a distinctive word.
_COMMON_WORDS = {
    "reach", "call", "try", "visit", "contact", "head", "stop", "check", "see", "ask", "apply",
    "file", "bring", "use", "find", "get", "got", "note", "consider", "look", "cool", "stay",
    "dial", "text", "email", "walk", "drive", "take", "come", "meet", "register", "enroll", "sign",
    "show", "tell", "give", "need", "want", "seek", "under", "over", "near", "from", "with", "into",
    "onto", "about", "after", "before", "during", "without", "within", "this", "that", "these",
    "those", "there", "here", "where", "when", "what", "which", "who", "your", "you", "they",
    "their", "them", "our", "her", "his", "its", "one", "two", "some", "any", "all", "most", "more",
    "each", "both", "based", "located", "open", "closed", "yes", "also", "please", "thanks",
    "thank", "hello", "hey", "otherwise", "then", "still", "just", "only", "even", "not", "make",
    "sure", "right", "left", "help", "free", "real", "good", "best", "great", "next", "last",
    # gerund/verb forms the agent opens headings and clauses with ("Applying for SNAP", "Getting …")
    "applying", "getting", "finding", "calling", "visiting", "bringing", "filing", "looking",
    "planning", "paying", "filling", "choosing", "picking", "booking", "scheduling", "checking",
    "here's", "heres",
}


def _norm(s) -> str:
    """Lowercase, punctuation→space, whitespace collapsed, the shared match space."""
    return _WS_RE.sub(" ", re.sub(r"[^\w\s]", " ", str(s).lower())).strip()


def _digits(s) -> str:
    return re.sub(r"\D", "", str(s))


def _stringify(obj) -> str:
    """Flatten a snapshot (dict/list/scalar) into one searchable string of keys + values."""
    if isinstance(obj, dict):
        return " ".join(f"{k} {_stringify(v)}" for k, v in obj.items())
    if isinstance(obj, (list, tuple)):
        return " ".join(_stringify(v) for v in obj)
    return str(obj)


def _citation_blob(c: dict) -> Optional[str]:
    """The captured source content for a citation: the DATA `snapshot` (or, for an auditable API
    exchange, the captured `response`, never the redacted request_summary), plus snippet + title
    (DOC/WEB carry only snippet + title). None if the source has none, then it cannot be verified."""
    parts: list[str] = []
    prov = c.get("provenance") or {}
    for key in ("snapshot", "response"):
        if prov.get(key):
            parts.append(_stringify(prov[key]))
    if c.get("snippet"):
        parts.append(str(c["snippet"]))
    if c.get("title"):
        parts.append(str(c["title"]))
    return " ".join(parts) if parts else None


def _split_claims(text: str) -> list[str]:
    """Sentence/line/list-item segments. Over-splitting is SAFE here: a segment with no {cite:Sn}
    marker is never inspected, and fewer tokens per claim means fewer chances to false-fail."""
    parts = re.split(r"[\n\r]+|(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p and p.strip()]


def _cited_sentences(text: str) -> list[tuple[str, list[str]]]:
    """Tier-2's unit: each sentence that carries a citation, paired with the {cite:Sn} ids it cites,
    with the markers stripped from the text. A segment that is ONLY citation markers annotates the
    sentence BEFORE it, agents routinely trail '{cite:Sn}' after the closing period, which
    _split_claims separates into its own segment, so it is merged back onto that sentence rather than
    checked as an empty claim. (Tier-1 is unaffected: it keeps its per-segment loop.)"""
    out: list[tuple[str, list[str]]] = []
    for seg in _split_claims(text):
        cited = _CITE_REF_RE.findall(seg)
        bare = _WS_RE.sub(" ", _CITE_REF_RE.sub(" ", seg)).strip()
        if cited and not bare and out:  # a marker-only segment: attach its ids to the prior sentence
            prev_text, prev_cited = out[-1]
            out[-1] = (prev_text, prev_cited + [c for c in cited if c not in prev_cited])
        elif bare:  # a real sentence (with or without its own marker); a later trailing marker may join
            out.append((bare, cited))
    return [(t, c) for t, c in out if c]


def _salient_tokens(claim: str) -> list[dict]:
    """High-signal, specific facts only. Each token carries how to match it (digit-run / phrase /
    all-significant-words). Common words, lone lowercase words, and the cite marker are never tokens."""
    tokens: list[dict] = []
    for m in _PHONE_RE.finditer(claim):
        d = _digits(m.group())
        if len(d) == 11 and d.startswith("1"):
            d = d[1:]
        if len(d) == 10:
            tokens.append({"kind": "phone", "text": m.group().strip(), "digits": d})
    for m in _MONEY_RE.finditer(claim):
        d = _digits(m.group())
        if len(d) >= 2:  # skip "$5", too small, a coincidental match is likely
            tokens.append({"kind": "money", "text": m.group().strip(), "digits": d})
    for m in _UNIT_NUM_RE.finditer(claim):
        tokens.append({"kind": "unit_number", "text": m.group().strip(), "digits": m.group(1)})
    for m in _ADDRESS_RE.finditer(claim):
        full = m.group().strip()
        # Numeric parts (house number AND numbered streets like "107th") match as digit-runs, so
        # "221 W 107th St" grounds against a source that stores "221 W 107 St". Ordinal suffixes,
        # compass directions, and the street type are dropped, they drift between source and answer.
        nums = re.findall(r"\d+", full)
        raw = re.findall(r"[A-Za-z]+", f"{m.group(2)} {m.group(3)}")
        words = [w.lower() for w in raw
                 if len(w) >= 3 and w.lower() not in _STREET_TYPES and w.lower() not in _DIRECTIONS]
        tokens.append({"kind": "address", "text": full, "nums": nums, "words": words, "phrase": _norm(full)})
    for m in _PROPER_NOUN_RE.finditer(claim):
        phrase = m.group().strip()
        sig = [w for w in _norm(phrase).split()
               if len(w) >= 3 and w not in _GENERIC_PN_WORDS and w not in _COMMON_WORDS
               and w not in _STREET_TYPES]  # a street type (Ave/Blvd) is abbreviated inconsistently
        if not sig:  # nothing source-specific left after stripping boilerplate → not a fact to verify
            continue
        tokens.append({"kind": "proper_noun", "text": phrase, "words": sig, "phrase": _norm(phrase)})
    return tokens


def _word_in(w: str, blob_norm: str) -> bool:
    """Substring presence, tolerant of a plural→singular drift (answer "Performances" vs source
    "Performance")."""
    return w in blob_norm or (len(w) > 4 and w.endswith("s") and w[:-1] in blob_norm)


def _mostly_present(words: list[str], blob_norm: str) -> bool:
    """A multi-word name is grounded if (nearly) all its significant words appear. We allow ONE miss
    (for names of 2+ words) so lexical drift doesn't false-fail, a normalized source typo (the
    FoodHelp row stores "…FOOD PANTY", the agent writes "…Food Pantry"), an added neighborhood word
    ("Far Rockaway" where the row's name is "Rockaway SNAP Center"). A genuine fabrication misses
    most/all words, so it still fails."""
    if not words:
        return True
    matched = sum(1 for w in words if _word_in(w, blob_norm))
    return matched == 1 if len(words) == 1 else matched >= len(words) - 1


def _money_value(text: str) -> Optional[Decimal]:
    try:
        return Decimal(text.replace("$", "").replace(",", "").strip())
    except InvalidOperation:
        return None


def _token_matches(tok: dict, blob_norm: str, blob_digits: str, blob: str) -> bool:
    """Lenient by design (favor a false PASS): digit-run substring for numbers, phrase-or-mostly-words
    substring for addresses/proper-nouns."""
    kind = tok["kind"]
    if kind == "money":
        value = _money_value(tok["text"])
        return value is not None and value in {
            parsed
            for match in _NUMBER_VALUE_RE.finditer(blob)
            if (parsed := _money_value(match.group())) is not None
        }
    if kind in ("phone", "unit_number"):
        return (not tok["digits"]) or tok["digits"] in blob_digits
    if kind == "address":
        if any(nd not in blob_digits for nd in tok["nums"]):
            return False
        return bool(tok["phrase"] and tok["phrase"] in blob_norm) or _mostly_present(tok["words"], blob_norm)
    if kind == "proper_noun":
        return bool(tok["phrase"] and tok["phrase"] in blob_norm) or _mostly_present(tok["words"], blob_norm)
    return True


def _locate(tok: dict, cid: str, c: dict) -> str:
    """Best-effort 'cited exactly here' pointer: the snapshot field (JSON-pointer-ish), or which of
    snippet/title carried the fact. Falls back to the citation id when we can't pin a field."""
    use_digits = tok["kind"] in ("phone", "money", "unit_number", "address")
    if tok["kind"] == "address":
        needle = tok["nums"][0] if tok["nums"] else ""
    else:
        needle = tok["digits"] if use_digits else tok.get("phrase", "")
    snapshot = (c.get("provenance") or {}).get("snapshot")
    if snapshot and needle:
        for k, v in snapshot.items():
            hay = _digits(_stringify(v)) if use_digits else _norm(_stringify(v))
            if needle in hay:
                return f"{cid}#/{k}"
    for field_name in ("snippet", "title"):
        val = c.get(field_name)
        if not val or not needle:
            continue
        hay = _digits(val) if use_digits else _norm(val)
        if needle in hay:
            return f"{cid}#{field_name}"
    return f"{cid}#source"


@dataclass
class Mismatch:
    """A cited fact absent from its source. Carries enough for the runtime guard to (a) tell the model
    exactly what to fix (Tier 3 feedback) and (b) strip the offending sentence (Tier 4 abstain)."""
    claim: str          # the sentence/segment the fact appeared in (a substring of the answer)
    cited: list[str]    # the {cite:Sn} ids that segment cited
    kind: str           # phone / money / unit_number / address / proper_noun
    text: str           # the offending token text as written in the answer
    message: str        # human-readable "S1: phone '(212) 555-0100' not in cited source"


@dataclass
class NLIMismatch:
    """Tier-2: a cited PROSE sentence a faithfulness/NLI checker judged UNSUPPORTED by the source it
    cites. Distinct from Mismatch (a Tier-1 structured-token absence): this is a per-sentence entailment
    miss, the class Tier-1 is structurally silent on (a fabricated statute cited to a soft source)."""
    claim: str          # the sentence checked, with {cite:Sn} markers stripped
    cited: list[str]    # the {cite:Sn} ids that segment cited
    score: float        # the backend's support probability, 0..1 (below threshold → recorded)
    reason: str = ""    # optional backend note (the prompted backend fills this)


@dataclass
class GroundingResult:
    """The verdict over one answer. `blocking` is True only when a HARD (verbatim-structured) fact is
    absent from an all-complete-capture claim, the ONLY condition the runtime guard acts on."""
    passed: bool
    detail: str
    blocking: bool
    checked: int
    locations: list = field(default_factory=list)
    hard_failures: list = field(default_factory=list)  # list[Mismatch], blocking, drives retry/strip
    soft_failures: list = field(default_factory=list)   # list[Mismatch], informational only
    # Tier-2 (opt-in): populated only when a checker is passed to check_grounding. Empty / zero on the
    # default (nli=None) path, so today's every caller sees a byte-identical result.
    nli_failures: list = field(default_factory=list)    # list[NLIMismatch], per-sentence entailment misses
    nli_checked: int = 0                                 # how many cited sentences the NLI checker ran on


def check_grounding(
    text: str,
    citations: dict,
    query: str = "",
    *,
    nli=None,
    nli_blocking: bool = False,
    nli_threshold: float = 0.5,
) -> Optional[GroundingResult]:
    """Verify every {cite:Sn}'d salient fact against its source (or the user's query).

    Returns a GroundingResult, or None when there is nothing to verify (no citation markers, or no
    salient facts sit next to any citation). Pure, in-memory, deterministic, no LLM, no network.

    Tier-2 (opt-in, off by default): pass ``nli`` (an object with ``check(claim, source) -> NLIVerdict``,
    see core.nli) to also run a per-sentence faithfulness check on every cited sentence, scoring it
    against the same source text Tier-1 assembled. A sentence whose support score is below
    ``nli_threshold`` is recorded in ``nli_failures``. When ``nli is None`` (today's every caller) this
    function behaves BYTE-IDENTICALLY to before, the Tier-2 fields stay empty and nothing else changes.
    ``nli_blocking`` (used only when the follow-on flips Tier-2 on live) lets an NLI failure also drive
    ``blocking``; by default blocking stays governed solely by Tier-1's hard_failures."""
    text = text or ""
    if "{cite:" not in text:
        return None
    query_norm = _norm(query) if query else ""
    query_digits = _digits(query) if query else ""
    hard_failures: list[Mismatch] = []
    soft_failures: list[Mismatch] = []
    nli_failures: list[NLIMismatch] = []
    locations: list[dict] = []
    checked = 0
    nli_checked = 0
    for claim in _split_claims(text):
        cited = _CITE_REF_RE.findall(claim)
        if not cited:
            continue
        # Classify each cited source. COMPLETE = we captured the whole source (a DATA snapshot or an
        # API response), so a fact's ABSENCE is conclusive. EXCERPT = only a truncated snippet/title
        # (DOC, WEB, or a label-only catalog row), the fact may live in the un-captured remainder of
        # the page, so absence there is NOT proof of fabrication. EMPTY = nothing captured. We only
        # BLOCK when every cited source is COMPLETE; excerpt/empty mismatches are informational.
        blobs: dict[str, str] = {}
        complete = excerpt = 0
        for cid in cited:
            c = citations.get(cid)
            blob = _citation_blob(c) if c else None
            if blob is None:
                continue  # empty capture
            blobs[cid] = blob
            prov = c.get("provenance") or {}
            if prov.get("snapshot") or prov.get("response"):
                complete += 1
            else:
                excerpt += 1
        if not blobs and not query_norm:
            continue  # nothing to verify against
        combined_norm = _norm(" ".join(blobs.values())) if blobs else ""
        combined_digits = _digits(" ".join(blobs.values())) if blobs else ""
        all_complete = complete >= 1 and excerpt == 0 and complete == len(cited)
        for tok in _salient_tokens(_CITE_REF_RE.sub(" ", claim)):
            checked += 1
            # 1) grounded in one specific cited source → record exactly where.
            hit = None
            for cid, blob in blobs.items():
                if _token_matches(tok, _norm(blob), _digits(blob), blob):
                    hit = _locate(tok, cid, citations[cid])
                    break
            # 2) grounded across the union of the claim's cited sources (phrase split across sources).
            combined = " ".join(blobs.values())
            if hit is None and blobs and _token_matches(tok, combined_norm, combined_digits, combined):
                hit = f"{'+'.join(blobs)}#source"
            # 3) a legitimate restatement of the user's own query (origin address, neighborhood, …).
            if hit is None and _token_matches(tok, query_norm, query_digits, query):
                hit = "user-query"
            if hit is not None:
                locations.append({"token": tok["text"], "kind": tok["kind"], "where": hit})
                continue
            # 4) absent everywhere. All cited sources empty → can't verify, don't fail. Otherwise it's a
            #    catch: HARD (blocks) only for a verbatim structured fact whose sources are ALL complete
            #    captures; a proper-noun mismatch, or anything cited to an excerpt/label, stays SOFT.
            if not blobs:
                continue
            msg = f"{'/'.join(cited)}: {tok['kind']} '{tok['text']}' not in cited source"
            mismatch = Mismatch(claim=claim, cited=cited, kind=tok["kind"], text=tok["text"], message=msg)
            hard = all_complete and tok["kind"] != "proper_noun"
            (hard_failures if hard else soft_failures).append(mismatch)
    # Tier-2 (opt-in): per-sentence faithfulness/NLI. A SEPARATE pass so Tier-1 above is byte-for-byte
    # untouched. Fully gated on `nli`, skipped entirely when nli is None. For each cited sentence, run
    # the checker against the SAME captured source text Tier-1 assembles (snapshot / snippet / title).
    # It earns its keep on the excerpt-cited prose Tier-1 deliberately passes; complete-source claims
    # get a cheap second look too.
    if nli is not None:
        for claim_text, cited in _cited_sentences(text):
            blobs = {cid: _citation_blob(citations[cid]) for cid in cited
                     if citations.get(cid) and _citation_blob(citations[cid]) is not None}
            if not blobs:
                continue  # every cited source is an empty capture, nothing to check against
            verdict = nli.check(claim_text, " ".join(blobs.values()))
            nli_checked += 1
            if verdict.score < nli_threshold:
                nli_failures.append(NLIMismatch(
                    claim=claim_text, cited=cited, score=verdict.score, reason=verdict.reason))
    if checked == 0 and nli_checked == 0:
        return None  # nothing to verify (no salient facts, and no cited sentence for Tier-2 either)
    failures = hard_failures + soft_failures
    # Only a HARD (verbatim-fact) Tier-1 mismatch blocks; a proper-noun-only mismatch is informational.
    # Tier-2 blocks only when explicitly enabled (nli_blocking), the live default leaves blocking to
    # Tier-1, so turning a checker on for telemetry never changes what ships.
    blocking = (_CITED_CLAIM_GROUNDING_BLOCKING and bool(hard_failures)) or (nli_blocking and bool(nli_failures))
    detail_parts = [m.message for m in failures]
    detail_parts += [
        f"{'/'.join(f.cited)}: NLI unsupported (score {f.score:.2f})"
        + (f": {f.reason}" if f.reason else "")
        for f in nli_failures
    ]
    detail = "; ".join(detail_parts)
    return GroundingResult(
        passed=not failures and not nli_failures, detail=detail, blocking=blocking, checked=checked,
        locations=locations, hard_failures=hard_failures, soft_failures=soft_failures,
        nli_failures=nli_failures, nli_checked=nli_checked,
    )
