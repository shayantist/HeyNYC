#!/usr/bin/env python3
"""Derive the public `docs/testing/` records from the gitignored internal docs.

The internal `docs/internal/eval/` sources (failure DB, red-team write-ups,
methodology) are kept local. This generator emits a redacted, tracked, public
subset into `docs/testing/` by DETERMINISTIC rules so the public repo can link to
real evidence without shipping internal paths, spend, owner rulings, or unreviewed
adversarial transcripts. It is a GENERATOR, not a hand fork: never edit the
artifacts by hand; edit the source and rerun. A drift guard
(`tests/test_export_testing_docs.py`) fails CI if the tracked files diverge.

Run:  python scripts/export_testing_docs.py         (writes docs/testing/)
      python scripts/export_testing_docs.py --check  (exit 1 on drift, writes nothing)

Determinism: no wall-clock. The header's "Source dated" is parsed from the source
doc's own date line and the hash is over the source bytes, so a re-run on any day
reproduces byte-identical output as long as the source is unchanged.
"""
from __future__ import annotations

import hashlib
import importlib.util
import re
import sys
from pathlib import Path


def _load_unwrap():
    """Import the shared `unwrap_text` from `scripts/unwrap_docs.py` without polluting
    sys.path, so it resolves both on a direct run and when this file is loaded by
    importlib in the drift test. Returns None (identity fallback) if the helper is
    not present yet."""
    path = Path(__file__).resolve().parent / "unwrap_docs.py"
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location("unwrap_docs", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, "unwrap_text", None)


_unwrap_text = _load_unwrap()

# --------------------------------------------------------------------------- #
# Which internal source maps to which public artifact, and how it is exported.
# The link-rewrite table below (internal doc basename -> public artifact) also
# collapses the two red-team sources onto the single merged `red-team.md`.
# --------------------------------------------------------------------------- #
# source basename (under docs/internal/eval/) -> public artifact filename
_ARTIFACT_NAMES = {
    "failure-db.md": "failure-db.md",
    "red-team-v1.md": "red-team.md",
    "red-team-v2-methodology.md": "red-team.md",
    "benchmark-methodology.md": "benchmarks.md",
}

_SCRIPT_REL = "scripts/export_testing_docs.py"
_INTERNAL_EVAL = "docs/internal/eval"

# One-sentence bridge between the two words a stranger meets first: the public
# umbrella (docs/testing/) and the code-level term of art (heynyc/eval/).
_EVAL_BRIDGE = (
    "These are HeyNYC's public test records; the eval harness that produces the "
    "gate results lives in `heynyc/eval/`."
)

# Phone-shaped strings that ARE documented public hotlines and must survive redaction.
_HOTLINE_ALLOWLIST = {
    "8003540365",   # ActionNYC (free immigration legal help)
    "18003540365",
    "8886929355",   # NYC Well
    "18886929355",
    "8002738255",   # 988 Suicide & Crisis Lifeline legacy line
    "18002738255",
}

# Crisis / self-harm markers: a withheld section carrying any of these is flagged so
# the owner hand-picks it. Crisis transcripts NEVER auto-publish.
_CRISIS_MARKERS = ("self-harm", "suicid", "988", "kill myself", "end my life",
                   "overdose", "aspirin", "chest pain")

_REVIEW = "REVIEW-REQUIRED"


# --------------------------------------------------------------------------- #
# Redaction rules (each independently unit-tested, positive + negative)
# --------------------------------------------------------------------------- #
_PATH_RE = re.compile(r"(?:\.data|/private/tmp|/tmp)/[A-Za-z0-9._/\-]+")


def redact_internal_paths(text: str) -> str:
    """Rule 1: strip .data/, /tmp/, and session paths (they name nothing public)."""
    return _PATH_RE.sub("[internal path]", text)


_MONEY_RE = re.compile(r"\$\d[\d,]*(?:\.\d+)?[BMK]?")


def redact_money(text: str) -> str:
    """Rule 2: strip dollar amounts (API spend and contested figures alike)."""
    return _MONEY_RE.sub("[amount]", text)


_PHONE_RE = re.compile(r"\b(?:1[-.\s])?\d{3}[-.\s]\d{3}[-.\s]\d{4}\b")


def redact_phones(text: str) -> str:
    """Rule 3: strip phone-shaped strings except documented public hotlines."""

    def _sub(m: re.Match[str]) -> str:
        digits = re.sub(r"\D", "", m.group(0))
        return m.group(0) if digits in _HOTLINE_ALLOWLIST else "[phone]"

    return _PHONE_RE.sub(_sub, text)


_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def rewrite_internal_doc_links(text: str) -> str:
    """Rule 4: rewrite/drop internal docs/ references.

    - external (http) links: untouched.
    - shipped repo paths (tests/, heynyc/): kept, rebased for docs/testing/ depth.
    - internal doc with a public artifact: rewritten to that artifact (sibling).
    - any other internal doc/data ref: hyperlink dropped, link text retained.
    """

    def _sub(m: re.Match[str]) -> str:
        label, url = m.group(1), m.group(2)
        if url.startswith(("http://", "https://", "#", "mailto:")):
            return m.group(0)
        base = url.split("#", 1)[0]
        # Shipped, public repo files: rebase to reach the repo root from docs/testing/.
        if "/tests/" in base or base.startswith("../../tests/"):
            return f"[{label}]({_rebase(url, 'tests/')})"
        if "/heynyc/" in base or base.startswith("../../heynyc/"):
            return f"[{label}]({_rebase(url, 'heynyc/')})"
        # Internal doc that has a public artifact: point at the sibling artifact.
        for src_name, art_name in _ARTIFACT_NAMES.items():
            if base.endswith(src_name):
                return f"[{label}]({art_name}{_frag(url)})"
        # Everything else internal (ROADMAP, strategy, specs, .data, other eval docs): drop link.
        return label

    return _LINK_RE.sub(_sub, text)


def _rebase(url: str, marker: str) -> str:
    """Rewrite a doc-relative link to a shipped file into a docs/testing/-relative one.

    docs/testing/ sits two levels below the repo root, so a shipped file is reached
    with `../../<tail>` regardless of the depth the internal source wrote it at."""
    frag = _frag(url)
    base = url.split("#", 1)[0]
    tail = base[base.index(marker):]  # e.g. "tests/test_agent.py"
    return f"../../{tail}{frag}"


def _frag(url: str) -> str:
    return f"#{url.split('#', 1)[1]}" if "#" in url else ""


# Matches the governance marker with the date in either position, e.g.
# "RULED (2026-07-18):", "(RULED 2026-07-18):", "PENDING OWNER (2026-07-12):", bare "RULED".
_GOV_RE = re.compile(
    r"\(?\s*\b(?:RULED|PENDING OWNER)\b\s*\(?\s*(?:\d{4}-\d{2}-\d{2})?\s*\)?:?"
)


def redact_governance(text: str) -> str:
    """Rule 5: strip owner-decision governance markers (RULED / PENDING OWNER).

    The marker and its date are replaced with a neutral tag; the surrounding
    technical description of the fix stays, since that is legitimately public.
    """
    return _GOV_RE.sub("[internal decision]", text)


def normalize_em_dashes(text: str) -> str:
    """HeyNYC rule 1 bans em dashes in every doc. The internal sources predate/ignore
    it; the public export must comply, so em dashes become spaced hyphens. Deterministic,
    so the drift guard is unaffected."""
    return text.replace("—", "-")


def redact_all(text: str) -> str:
    """Compose the redactions. Link rewrite runs first so path/money passes never
    corrupt a URL that is about to be kept or de-linked."""
    text = rewrite_internal_doc_links(text)
    text = redact_internal_paths(text)
    text = redact_money(text)
    text = redact_phones(text)
    text = redact_governance(text)
    text = normalize_em_dashes(text)
    return text


# --------------------------------------------------------------------------- #
# Generated-file header (deterministic: source hash + source-declared date)
# --------------------------------------------------------------------------- #
_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")


def source_hash(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def source_date(raw: str) -> str:
    m = _DATE_RE.search(raw)
    return m.group(1) if m else "undated"


def header(sources: list[tuple[str, str]]) -> str:
    """Do-not-edit banner. `sources` is a list of (source_rel, raw_text); the merged
    red-team artifact carries two, so each source's SHA-256 and declared date is listed."""
    noun = "source" if len(sources) == 1 else "sources"
    names = ", ".join(f"`{rel}`" for rel, _ in sources)
    lines = [
        "> **Generated file. Do not edit by hand.**",
        f"> Generated by `{_SCRIPT_REL}` from the internal {noun} {names}.",
        f"> {_EVAL_BRIDGE}",
        "> Edit the source and regenerate; direct edits are overwritten and fail the "
        "drift guard.",
    ]
    for rel, raw in sources:
        lines.append(f"> Source `{rel}` SHA-256: `{source_hash(raw)}`")
        lines.append(f"> Source `{rel}` dated: {source_date(raw)}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Exporter: failure register (grouped by the nine-category taxonomy)
# --------------------------------------------------------------------------- #
_ROW_RE = re.compile(r"^\|\s*F\d+\b")
_SPLIT_RE = re.compile(r"\s*\|\|\s*(?=F\d+\s*\|)")  # recover rows glued by `|| Fnnn |`

# The project's nine-category failure taxonomy, in its canonical (ROADMAP) order.
# Rows tagged outside this set are NOT recategorized: they group under their own tag
# after the nine, which surfaces them for the tag audit rather than hiding them.
_CATEGORY_TAXONOMY = (
    "citation-integrity",
    "retrieval-identity",
    "operations",
    "channels-transport",
    "scope-gating",
    "conversation-memory",
    "multilingual-equity",
    "location-usefulness",
    "emergency-safety",
)
_TAG_RE = re.compile(r"\*\*([a-z][a-z-]*)\*\*")


def _category_tag(category_cell: str) -> str:
    """The taxonomy tag is the first bold lowercase-hyphen token of the category cell."""
    m = _TAG_RE.search(category_cell)
    return m.group(1) if m else "(uncategorized)"


def _failure_rows(raw: str) -> list[tuple[int, str, str, str, str]]:
    rows: list[tuple[int, str, str, str, str]] = []
    for line in raw.splitlines():
        if not _ROW_RE.match(line):
            continue
        for i, part in enumerate(_SPLIT_RE.split(line)):
            if i > 0:
                part = "| " + part  # restore the leading pipe the split consumed
            cells = [c.strip() for c in part.strip().strip("|").split("|")]
            if len(cells) < 3 or not cells[0].startswith("F"):
                continue
            fid = cells[0]
            observed = cells[1]
            category = cells[2]
            fix_status = " · ".join(c for c in cells[3:] if c)
            num = int(re.sub(r"\D", "", fid) or 0)
            rows.append((num, fid, observed, category, fix_status))
    rows.sort(key=lambda r: r[0])  # deterministic F-number order (source is out of order)
    return rows


_COLUMNS = (
    "| ID | Observed failure | Category and class | Fix and status |\n"
    "| --- | --- | --- | --- |\n"
)


def export_failure_register(raw: str, source_rel: str) -> str:
    rows = _failure_rows(raw)
    groups: dict[str, list[tuple[int, str, str, str, str]]] = {}
    for row in rows:
        groups.setdefault(_category_tag(row[3]), []).append(row)
    # Canonical nine first, then any off-taxonomy tags (alpha) so no row is dropped.
    ordered = [c for c in _CATEGORY_TAXONOMY if c in groups]
    ordered += sorted(c for c in groups if c not in _CATEGORY_TAXONOMY)

    intro = (
        "# HeyNYC public failure register\n\n"
        "Every resident-facing failure found in testing, its category, and the pinned "
        "test or eval case that now guards against it. This is the public, redacted view "
        "of the project's operational risk register (internal decision records, spend "
        "figures, and internal paths removed). The named tests and eval cases are runnable "
        "public evidence: they ship in this repository.\n\n"
        f"**Total: {len(rows)} failures across {len(ordered)} categories.**\n\n"
        "Families (cross-row root-cause clusters, named as the rows name them): "
        "**over-denial** (F069, F071, F075), where a plausibly-NYC need is denied at the "
        "scope gate; **conversation-repetition** (F062, F078, F080), where a follow-up "
        "re-briefs settled context instead of answering the delta; and **event-identity** "
        "(F046, F053, F058), where an event turn never resolves which event is meant "
        "before answering.\n"
    )

    sections = []
    for cat in ordered:
        grp = groups[cat]
        body = "".join(
            f"| {fid} | {redact_all(observed)} | {redact_all(category)} | {redact_all(fix_status)} |\n"
            for _num, fid, observed, category, fix_status in grp
        )
        sections.append(f"## {cat} ({len(grp)})\n\n{_COLUMNS}{body}")

    return f"{header([(source_rel, raw)])}\n{intro}\n" + "\n".join(sections)


# --------------------------------------------------------------------------- #
# Exporter: methodology docs (near-verbatim minus the same redactions)
# --------------------------------------------------------------------------- #
def export_methodology(raw: str, source_rel: str) -> str:
    return f"{header([(source_rel, raw)])}\n{redact_all(raw)}"


# --------------------------------------------------------------------------- #
# Red-team: per-category counts + outcomes only; verbatim -> placeholder
# --------------------------------------------------------------------------- #
_WITHHOLD_HEADING = ("verbatim", "full query set", "per-item", "per item", "query set")


def _sections(raw: str) -> list[tuple[str, str]]:
    """Split into (h2_heading, body) pairs; the pre-H2 preamble is heading ''."""
    parts = re.split(r"(?m)^(## .+)$", raw)
    out = [("", parts[0])]
    for i in range(1, len(parts), 2):
        out.append((parts[i], parts[i + 1] if i + 1 < len(parts) else ""))
    return out


def _withhold(heading: str, body: str) -> bool:
    """Withhold a section that is verbatim content: its heading names the query set,
    or its body carries a fenced block (where the red-team quotes the adversarial
    prompts and model responses verbatim). Counts and outcomes carry neither."""
    if any(k in heading.lower() for k in _WITHHOLD_HEADING):
        return True
    return "```" in body


def _placeholder(heading: str, crisis: bool) -> str:
    note = (
        "This section of the internal red-team contains verbatim adversarial prompts "
        "and/or model responses. The public export publishes per-category counts and "
        "outcomes only; verbatim exchanges are withheld pending owner review."
    )
    if crisis:
        note += (
            " This section also contains crisis / self-harm content, which NEVER "
            "auto-publishes. The owner hand-picks any such exchange before release."
        )
    tag = f"{_REVIEW}: CRISIS / SELF-HARM" if crisis else _REVIEW
    return f"{heading}\n\n> **[{tag}]** {note}\n"


_INLINE_QUOTE_RE = re.compile(r"\"[^\"]{0,400}\"")
_DOSE_RE = re.compile(r"\b\d+\s?m[gl]\b", re.IGNORECASE)


def redact_crisis_quotes(text: str) -> str:
    """A kept section describes outcomes, but the source sometimes reproduces a short
    verbatim snippet inline. Any inline quote that carries crisis / self-harm content
    or a medication dose is withheld: that content NEVER auto-publishes, the owner
    hand-picks it. The surrounding outcome prose stays."""

    def _sub(m: re.Match[str]) -> str:
        q = m.group(0)
        low = q.lower()
        if _DOSE_RE.search(q) or any(mk in low for mk in _CRISIS_MARKERS):
            return '"[crisis / self-harm content withheld for owner review]"'
        return q

    return _INLINE_QUOTE_RE.sub(_sub, text)


def _red_team_results_body(raw: str) -> str:
    """The v1 summary: per-category counts and outcomes kept, verbatim sections
    replaced by REVIEW-REQUIRED placeholders. No header (the merged file owns it)."""
    out: list[str] = []
    for heading, body in _sections(raw):
        if heading and _withhold(heading, body):
            crisis = any(m in body.lower() for m in _CRISIS_MARKERS)
            out.append(_placeholder(heading, crisis))
        else:
            chunk = f"{heading}\n{body}" if heading else body
            out.append(redact_crisis_quotes(redact_all(chunk)).rstrip("\n"))
    return "\n".join(out).rstrip("\n")


def _nest_body(body: str) -> str:
    """Drop a source body's own H1 title and push every remaining ATX heading down one
    level, so the source's sections nest under the merged file's H2 section headers.
    Fenced code blocks (e.g. shell examples whose comments start with `#`) are skipped."""
    out: list[str] = []
    dropped_title = False
    in_fence = False
    for line in body.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            out.append(line)
            continue
        if not in_fence:
            if not dropped_title and line.startswith("# "):
                dropped_title = True
                continue
            if line.startswith("#"):
                line = "#" + line
        out.append(line)
    return "\n".join(out).strip("\n")


_REDTEAM_STATUS = (
    "The current adversarial suite is 205 cases, but its full run has not yet been "
    "executed (owner-gated live budget). The per-category results below are from the "
    "completed 137-query v1 run plus the standing per-change adversarial matrices, not "
    "a full 205-case rerun."
)


def export_red_team_merged(v2_raw: str, v1_raw: str, v2_rel: str, v1_rel: str) -> str:
    """One public red-team record from both internal sources: the v2 methodology
    ('How we red-team') and the v1 results ('Results to date'), each nested under an H2,
    with the v1 REVIEW-REQUIRED placeholders preserved and both source hashes in the header."""
    hdr = header([(v2_rel, v2_raw), (v1_rel, v1_raw)])
    method = _nest_body(redact_all(v2_raw))
    results = _nest_body(_red_team_results_body(v1_raw))
    return (
        f"{hdr}\n"
        "# HeyNYC red-team\n\n"
        "## How we red-team\n\n"
        f"{method}\n\n"
        "## Results to date\n\n"
        f"{_REDTEAM_STATUS}\n\n"
        f"{results}\n"
    )


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
# single-source: internal source basename -> (public artifact, exporter)
_SINGLE_EXPORTERS = {
    "failure-db.md": ("failure-db.md", export_failure_register),
    "benchmark-methodology.md": ("benchmarks.md", export_methodology),
}


def docs_available(project_root: Path) -> bool:
    return (project_root / "docs" / "internal" / "eval" / "failure-db.md").exists()


def generate_all(internal_root: Path) -> dict[str, str]:
    """Map public artifact filename -> content. Empty if the internal docs are absent
    (a public clone), which is what lets the drift guard skip cleanly. `internal_root`
    is the docs/internal/ tree; sources live under its eval/ subdir."""
    eval_dir = internal_root / "eval"
    artifacts: dict[str, str] = {}
    for src_name, (art_name, exporter) in _SINGLE_EXPORTERS.items():
        src = eval_dir / src_name
        if not src.exists():
            continue
        raw = src.read_text()
        artifacts[art_name] = exporter(raw, f"{_INTERNAL_EVAL}/{src_name}")
    # Merged red-team: both sources required, both hashes in the header.
    v2, v1 = eval_dir / "red-team-v2-methodology.md", eval_dir / "red-team-v1.md"
    if v2.exists() and v1.exists():
        artifacts["red-team.md"] = export_red_team_merged(
            v2.read_text(), v1.read_text(),
            f"{_INTERNAL_EVAL}/red-team-v2-methodology.md",
            f"{_INTERNAL_EVAL}/red-team-v1.md",
        )
    # Final step: unwrap hard-wrapped prose to full lines so regeneration stays unwrapped
    # and the drift test compares like against like. The shared helper preserves fences,
    # tables, lists, headers, and blockquotes (incl. this file's header banner).
    if _unwrap_text is not None:
        artifacts = {name: _unwrap_text(content) for name, content in artifacts.items()}
    # TODO(unwrap): if scripts/unwrap_docs.py is ever removed, artifacts ship hard-wrapped
    # until it returns; wire the shared `unwrap_text` back in here.
    return artifacts


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    check_only = "--check" in argv
    root = Path(__file__).resolve().parents[1]
    if not docs_available(root):
        print("internal docs/ tree absent; nothing to export (public clone).")
        return 0
    out_dir = root / "docs" / "testing"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts = generate_all(root / "docs" / "internal")
    drift = False
    for name, content in artifacts.items():
        target = out_dir / name
        current = target.read_text() if target.exists() else None
        if current == content:
            continue
        drift = True
        if check_only:
            print(f"DRIFT: {name} is out of date; run {_SCRIPT_REL}")
        else:
            target.write_text(content)
            print(f"wrote docs/testing/{name}")
    if check_only:
        return 1 if drift else 0
    if not drift:
        print("docs/testing/ already up to date.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
