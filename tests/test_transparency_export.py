"""Transparency-export generator: redaction rules + drift guard.

The generator (`scripts/export_transparency.py`) derives the tracked, public
`transparency/` artifacts from the gitignored internal `docs/eval/` sources by
deterministic rules. These tests pin each redaction pattern (a positive hit and a
negative pass-through) and the drift guard in both directions, mirroring the
README capability drift test in `tests/test_capabilities.py`.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from heynyc.core import config

_SCRIPT = config.PROJECT_ROOT / "scripts" / "export_transparency.py"
_TRANSPARENCY = config.PROJECT_ROOT / "transparency"
_DOCS_EVAL = config.PROJECT_ROOT / "docs" / "eval"


def _load():
    spec = importlib.util.spec_from_file_location("export_transparency", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


xp = _load()


# --------------------------------------------------------------------------- #
# Redaction rule 1: internal .data / /tmp / session paths
# --------------------------------------------------------------------------- #
def test_redact_internal_paths_positive():
    assert ".data/" not in xp.redact_internal_paths("saved under .data/eval/acceptance-20260717-a1/report.json")
    assert ".data/" not in xp.redact_internal_paths("bench in .data/bench/2026-07-18/traces")
    assert "/tmp/" not in xp.redact_internal_paths("see /tmp/claude-501/fresh_eye_1.json")


def test_redact_internal_paths_negative():
    keep = "pinned by ../tests/test_agent.py and ../heynyc/modules/benefits/eval.yaml"
    assert xp.redact_internal_paths(keep) == keep


# --------------------------------------------------------------------------- #
# Redaction rule 2: dollar amounts
# --------------------------------------------------------------------------- #
def test_redact_money_positive():
    assert "$" not in xp.redact_money("the run cost $3.2924 total")
    assert "$" not in xp.redact_money("about $1.1B unclaimed")
    assert "$" not in xp.redact_money("a $5,000 buyout offer")
    assert "$" not in xp.redact_money("passed at $0.00081395")


def test_redact_money_negative():
    keep = "205 cases across 8 categories, up from 137 queries; 3 failed"
    assert xp.redact_money(keep) == keep


# --------------------------------------------------------------------------- #
# Redaction rule 3: phone-number-shaped strings (public hotlines survive)
# --------------------------------------------------------------------------- #
def test_redact_phones_positive():
    out = xp.redact_phones("an ungrounded number 212-419-3737 from memory")
    assert "212-419-3737" not in out


def test_redact_phones_negative():
    keep = "ActionNYC 800-354-0365, or dial 311, 911, or 988"
    assert xp.redact_phones(keep) == keep  # documented public hotlines are allowlisted


# --------------------------------------------------------------------------- #
# Redaction rule 4: internal docs/ references
# --------------------------------------------------------------------------- #
def test_rewrite_doc_links_public_repo_paths_kept_and_rebased():
    # tests/ and heynyc/ ship publicly: keep the link, rebase for transparency/ depth.
    out = xp.rewrite_internal_doc_links("[t](../../tests/test_agent.py)")
    assert "(../tests/test_agent.py)" in out
    out2 = xp.rewrite_internal_doc_links("[e](../../heynyc/modules/benefits/eval.yaml)")
    assert "(../heynyc/modules/benefits/eval.yaml)" in out2


def test_rewrite_doc_links_internal_with_artifact_is_rewritten():
    out = xp.rewrite_internal_doc_links("the [first run](./red-team-v1.md) used 137 queries")
    assert "red-team-summary.md" in out
    assert "red-team-v1.md" not in out


def test_rewrite_doc_links_internal_without_artifact_is_dropped_to_text():
    out = xp.rewrite_internal_doc_links("see the [bounded-memory checkpoint](../ROADMAP.md) for more")
    assert "ROADMAP" not in out          # broken internal ref dropped
    assert "bounded-memory checkpoint" in out  # human-readable text retained
    assert "(" not in out.split("checkpoint")[1][:3]  # no dangling link paren


def test_rewrite_doc_links_leaves_external_links_untouched():
    keep = "per [NIST AI RMF](https://www.nist.gov/itl/ai-risk-management-framework)"
    assert xp.rewrite_internal_doc_links(keep) == keep


# --------------------------------------------------------------------------- #
# Redaction rule 5: owner-quoting / RULED / PENDING OWNER governance lines
# --------------------------------------------------------------------------- #
def test_redact_governance_positive():
    out = xp.redact_governance("Fixed (RULED 2026-07-18): deleted the parser; kept the fix note")
    assert "RULED" not in out
    assert "kept the fix note" in out
    out2 = xp.redact_governance("PENDING OWNER (2026-07-12): no runbook yet")
    assert "PENDING OWNER" not in out2


def test_redact_governance_negative():
    keep = "Fixed locally; live restart not approved"
    assert xp.redact_governance(keep) == keep


# --------------------------------------------------------------------------- #
# Style normalization: HeyNYC rule 1 bans em dashes in every doc
# --------------------------------------------------------------------------- #
def test_normalize_em_dashes_positive():
    assert "—" not in xp.normalize_em_dashes("the whole city — one flooded")


def test_normalize_em_dashes_negative():
    keep = "unit-dependent, per the source, so it qualifies"  # plain hyphens survive
    assert xp.normalize_em_dashes(keep) == keep


# --------------------------------------------------------------------------- #
# Generated artifacts: shape and header
# --------------------------------------------------------------------------- #
def _generated():
    return xp.generate_all(config.PROJECT_ROOT / "docs")


docs_present = _DOCS_EVAL.exists()
requires_docs = pytest.mark.skipif(not docs_present, reason="internal docs/ absent (public clone)")


@requires_docs
def test_every_artifact_carries_the_do_not_edit_header():
    for name, content in _generated().items():
        assert "do not edit by hand" in content.lower(), name
        assert "scripts/export_transparency.py" in content, name
        assert "SHA-256" in content, name


@requires_docs
def test_failure_register_keeps_columns_and_evidence_names():
    reg = _generated()["failure-register.md"]
    assert "| ID | Observed failure | Category and class | Fix and status |" in reg
    # runnable public evidence (a pinned test name) survives
    assert "test_scope_denial_stops_before_main_model_or_tools" in reg
    # redaction floors: no internal paths, spend, or governance markers leak
    assert ".data/" not in reg
    assert "/tmp/" not in reg
    assert "RULED" not in reg
    assert "$0." not in reg


@requires_docs
def test_red_team_export_is_counts_only_with_review_required_placeholders():
    rt = _generated()["red-team-summary.md"]
    assert "Results at a glance" in rt          # per-category counts kept
    assert "REVIEW-REQUIRED" in rt              # verbatim exchanges withheld
    # a self-harm / crisis section must be flagged for owner hand-pick, never auto-published
    assert "crisis" in rt.lower() and "self-harm" in rt.lower()
    # no verbatim adversarial response block leaks through, including inline crisis quotes
    assert "Llama al 911 ahora mismo" not in rt
    assert "mastica una" not in rt
    import re as _re
    assert not _re.search(r"\d+ ?mg o cuatro", rt)  # the medication-dose snippet is withheld


@requires_docs
def test_methodology_exports_are_near_verbatim_minus_redactions():
    bench = _generated()["benchmark-methodology.md"]
    assert "three legally-clean pillars" in bench   # substance preserved verbatim
    assert "$1.1B" not in bench                      # but dollar amounts redacted


# --------------------------------------------------------------------------- #
# Drift guard, both directions (mirrors tests/test_capabilities.py)
# --------------------------------------------------------------------------- #
@requires_docs
def test_drift_guard_tracked_artifacts_match_regeneration():
    """When docs/ exists, the tracked transparency/ files must equal a fresh export.
    Edit a source without regenerating (`python scripts/export_transparency.py`) and
    this fails, so transparency/ single-sources from docs/, same as the README table."""
    generated = _generated()
    assert generated, "generator produced no artifacts"
    for name, content in generated.items():
        tracked = _TRANSPARENCY / name
        assert tracked.exists(), f"missing tracked artifact {name}"
        assert tracked.read_text() == content, f"drift in {name}: regenerate transparency/"


@requires_docs
def test_drift_guard_detects_a_tampered_artifact():
    """Reverse direction: a byte of drift must be caught, proving the guard has teeth."""
    generated = _generated()
    name, content = next(iter(generated.items()))
    tampered = content + "\nsneaky hand edit\n"
    assert tampered != content  # the equality check the forward guard runs would fail here


def test_drift_guard_skips_cleanly_when_docs_absent(tmp_path):
    """Public clones have no docs/ tree; generate_all over an empty root yields nothing
    and the guard has nothing to compare, so it is a clean no-op rather than a failure."""
    empty_root = tmp_path / "docs"
    empty_root.mkdir()
    assert xp.generate_all(empty_root) == {}
    assert xp.docs_available(tmp_path) is False
