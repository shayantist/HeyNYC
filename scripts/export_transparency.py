#!/usr/bin/env python3
"""Derive the public `transparency/` artifacts from the gitignored internal docs.

The internal `docs/eval/` sources (failure DB, red-team write-ups, methodology) are
kept local. This generator emits a redacted, tracked, public subset into
`transparency/` by DETERMINISTIC rules so the public repo can link to real
evidence without shipping internal paths, spend, owner rulings, or unreviewed
adversarial transcripts. It is a GENERATOR, not a hand fork: never edit the
artifacts by hand; edit the source and rerun. A drift guard
(`tests/test_transparency_export.py`) fails CI if the tracked files diverge.

Run:  python scripts/export_transparency.py         (writes transparency/)
      python scripts/export_transparency.py --check  (exit 1 on drift, writes nothing)

Determinism: no wall-clock. The header's "Source dated" is parsed from the source
doc's own date line and the hash is over the source bytes, so a re-run on any day
reproduces byte-identical output as long as the source is unchanged.
"""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# Which internal source maps to which public artifact, and how it is exported.
# The map is also the docs/ -> transparency/ link-rewrite table.
# --------------------------------------------------------------------------- #
# source basename (under docs/eval/) -> public artifact filename
_ARTIFACT_NAMES = {
    "failure-db.md": "failure-register.md",
    "red-team-v1.md": "red-team-summary.md",
    "benchmark-methodology.md": "benchmark-methodology.md",
    "red-team-v2-methodology.md": "red-team-v2-methodology.md",
}

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
    - shipped repo paths (tests/, heynyc/): kept, rebased for transparency/ depth.
    - internal doc with a public artifact: rewritten to that artifact (sibling).
    - any other internal doc/data ref: hyperlink dropped, link text retained.
    """

    def _sub(m: re.Match[str]) -> str:
        label, url = m.group(1), m.group(2)
        if url.startswith(("http://", "https://", "#", "mailto:")):
            return m.group(0)
        base = url.split("#", 1)[0]
        # Shipped, public repo files: rebase ../../ (from docs/eval/) to ../ (from transparency/).
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
    """Rewrite a doc-relative link to a shipped file into a transparency-relative one."""
    frag = _frag(url)
    base = url.split("#", 1)[0]
    tail = base[base.index(marker):]  # e.g. "tests/test_agent.py"
    return f"../{tail}{frag}"


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


def header(source_rel: str, raw: str) -> str:
    return (
        "> **Generated file. Do not edit by hand.**\n"
        f"> Generated by `scripts/export_transparency.py` from the internal source "
        f"`{source_rel}`.\n"
        "> Edit the source and regenerate; direct edits are overwritten and fail the "
        "drift guard.\n"
        f"> Source content SHA-256: `{source_hash(raw)}`\n"
        f"> Source dated: {source_date(raw)}\n"
    )


# --------------------------------------------------------------------------- #
# Exporter: failure register (redacted 4-column table)
# --------------------------------------------------------------------------- #
_ROW_RE = re.compile(r"^\|\s*F\d+\b")
_SPLIT_RE = re.compile(r"\s*\|\|\s*(?=F\d+\s*\|)")  # recover rows glued by `|| Fnnn |`


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


def export_failure_register(raw: str, source_rel: str) -> str:
    intro = (
        "# HeyNYC public failure register\n\n"
        "Every resident-facing failure found in testing, its category, and the pinned "
        "test or eval case that now guards against it. This is the public, redacted view "
        "of the project's operational risk register (internal decision records, spend "
        "figures, and internal paths removed). The named tests and eval cases are runnable "
        "public evidence: they ship in this repository.\n\n"
        "| ID | Observed failure | Category and class | Fix and status |\n"
        "| --- | --- | --- | --- |\n"
    )
    body = "".join(
        f"| {fid} | {redact_all(observed)} | {redact_all(category)} | {redact_all(fix_status)} |\n"
        for _num, fid, observed, category, fix_status in _failure_rows(raw)
    )
    return f"{header(source_rel, raw)}\n{intro}{body}"


# --------------------------------------------------------------------------- #
# Exporter: methodology docs (near-verbatim minus the same redactions)
# --------------------------------------------------------------------------- #
def export_methodology(raw: str, source_rel: str) -> str:
    return f"{header(source_rel, raw)}\n{redact_all(raw)}"


# --------------------------------------------------------------------------- #
# Exporter: red-team (per-category counts + outcomes only; verbatim -> placeholder)
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
        "and/or model responses. The public transparency export publishes per-category "
        "counts and outcomes only; verbatim exchanges are withheld pending owner review."
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


def export_red_team(raw: str, source_rel: str) -> str:
    out = [header(source_rel, raw)]
    for heading, body in _sections(raw):
        if heading and _withhold(heading, body):
            crisis = any(m in body.lower() for m in _CRISIS_MARKERS)
            out.append(_placeholder(heading, crisis))
        else:
            chunk = f"{heading}\n{body}" if heading else body
            out.append(redact_crisis_quotes(redact_all(chunk)).rstrip("\n"))
    return "\n".join(out).rstrip("\n") + "\n"


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
_EXPORTERS = {
    "failure-db.md": export_failure_register,
    "benchmark-methodology.md": export_methodology,
    "red-team-v2-methodology.md": export_methodology,
    "red-team-v1.md": export_red_team,
}


def docs_available(project_root: Path) -> bool:
    return (project_root / "docs" / "eval" / "failure-db.md").exists()


def generate_all(docs_root: Path) -> dict[str, str]:
    """Map public artifact filename -> content. Empty if the internal docs are absent
    (a public clone), which is what lets the drift guard skip cleanly."""
    artifacts: dict[str, str] = {}
    for src_name, exporter in _EXPORTERS.items():
        src = docs_root / "eval" / src_name
        if not src.exists():
            continue
        raw = src.read_text()
        source_rel = f"docs/eval/{src_name}"
        artifacts[_ARTIFACT_NAMES[src_name]] = exporter(raw, source_rel)
    return artifacts


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    check_only = "--check" in argv
    root = Path(__file__).resolve().parents[1]
    if not docs_available(root):
        print("internal docs/ tree absent; nothing to export (public clone).")
        return 0
    out_dir = root / "transparency"
    out_dir.mkdir(exist_ok=True)
    artifacts = generate_all(root / "docs")
    drift = False
    for name, content in artifacts.items():
        target = out_dir / name
        current = target.read_text() if target.exists() else None
        if current == content:
            continue
        drift = True
        if check_only:
            print(f"DRIFT: {name} is out of date; run scripts/export_transparency.py")
        else:
            target.write_text(content)
            print(f"wrote transparency/{name}")
    if check_only:
        return 1 if drift else 0
    if not drift:
        print("transparency/ already up to date.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
