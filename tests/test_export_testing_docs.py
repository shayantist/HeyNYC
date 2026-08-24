"""Public test-records generator: redaction rules + grouping + drift guard.

The generator (`scripts/export_testing_docs.py`) derives the tracked, public
`docs/testing/` records from the gitignored internal `docs/internal/eval/` sources by
deterministic rules. These tests pin each redaction pattern (a positive hit and a
negative pass-through), the failure-register grouping, and the drift guard in both
directions, mirroring the README capability drift test in `tests/test_capabilities.py`.
"""
from __future__ import annotations

import importlib.util
import re

import pytest

from heynyc.core import config

_SCRIPT = config.PROJECT_ROOT / "scripts" / "export_testing_docs.py"
_TESTING = config.PROJECT_ROOT / "docs" / "testing"
_INTERNAL = config.PROJECT_ROOT / "docs" / "internal"
_INTERNAL_EVAL = _INTERNAL / "eval"


def _load():
    spec = importlib.util.spec_from_file_location("export_testing_docs", _SCRIPT)
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
    assert ".agents" not in xp.redact_internal_paths(
        "reviewed in .agents/tmp/f120-f123-final-review.final.md"
    )
    assert "/tmp/" not in xp.redact_internal_paths("see /tmp/claude-501/fresh_eye_1.json")


def test_redact_internal_paths_negative():
    keep = "pinned by ../../tests/test_agent.py and ../../heynyc/modules/benefits/eval.yaml"
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
    # tests/ and heynyc/ ship publicly: keep the link, rebase two levels for docs/testing/ depth.
    out = xp.rewrite_internal_doc_links("[t](../../tests/test_agent.py)")
    assert "(../../tests/test_agent.py)" in out
    out2 = xp.rewrite_internal_doc_links("[e](../../heynyc/modules/benefits/eval.yaml)")
    assert "(../../heynyc/modules/benefits/eval.yaml)" in out2


def test_rewrite_doc_links_internal_with_artifact_is_rewritten():
    out = xp.rewrite_internal_doc_links("the [first run](./red-team-v1.md) used 137 queries")
    assert "red-team.md" in out
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
# Failure-register grouping: the taxonomy tag is the first bold token
# --------------------------------------------------------------------------- #
def test_category_tag_reads_the_first_bold_token():
    assert xp._category_tag("**scope-gating** · over-denial on register") == "scope-gating"
    assert xp._category_tag("**scope-gating**-adjacent wrong remedy") == "scope-gating"
    assert xp._category_tag("**resident-outcome / retry-exhaustion** · lost answer") == "resident-outcome"


def test_failure_rows_excludes_trace_matrix_labels():
    raw = """\
| F188 | Canonical failure | **operations** · detail | evidence | OPEN |
| F188 Notify NYC cost | WI1 | NI1 |
"""
    assert [row[1] for row in xp._failure_rows(raw)] == ["F188"]


# --------------------------------------------------------------------------- #
# Generated artifacts: shape and header
# --------------------------------------------------------------------------- #
def _generated():
    return xp.generate_all(_INTERNAL)


docs_present = _INTERNAL_EVAL.exists()
requires_docs = pytest.mark.skipif(not docs_present, reason="internal docs/ absent (public clone)")


@requires_docs
def test_every_artifact_carries_the_do_not_edit_header():
    for name, content in _generated().items():
        assert "do not edit by hand" in content.lower(), name
        assert "scripts/export_testing_docs.py" in content, name
        assert "SHA-256" in content, name
        # the one-sentence eval/testing bridge rides in every header
        assert "the eval harness that produces the gate results lives in" in content, name


@requires_docs
def test_failure_register_is_grouped_by_taxonomy_with_total_and_families():
    reg = _generated()["failure-db.md"]
    # column header appears once per section but at least once
    assert "| ID | Observed failure | Category and class | Fix and status |" in reg
    # total line up top
    assert re.search(r"\*\*Total: \d+ failures across \d+ categories\.\*\*", reg)
    # families paragraph names the three clusters exactly as the rows name them
    assert "over-denial" in reg and "F069" in reg and "F071" in reg and "F075" in reg
    assert "conversation-repetition" in reg and "F078" in reg and "F080" in reg
    assert "event-identity" in reg and "F046" in reg and "F053" in reg and "F058" in reg
    # every one of the nine taxonomy categories heads a section
    for cat in xp._CATEGORY_TAXONOMY:
        assert f"## {cat} (" in reg, cat
    # no row is dropped in grouping and none is duplicated
    ids = re.findall(r"^\| (F\d+) \|", reg, re.M)
    assert len(ids) == len(set(ids))
    assert "**Total: 281 failures" in reg
    assert "## (uncategorized)" not in reg
    assert "Failure IDs are append-only" in reg
    assert "`FIXED LOCALLY` means" in reg
    assert "F001" in ids and "F080" in ids
    # runnable public evidence (a pinned test name) survives
    assert "test_scope_denial_stops_before_main_model_or_tools" in reg
    # redaction floors: no internal paths, spend, or governance markers leak
    assert ".data/" not in reg
    assert "/tmp/" not in reg
    assert "RULED" not in reg
    assert "$0." not in reg


@requires_docs
def test_failure_register_surfaces_off_taxonomy_tags_as_their_own_sections():
    # The tag audit: rows tagged outside the nine are NOT recategorized; they group
    # under their own tag AFTER the nine, which is how the export surfaces them.
    reg = _generated()["failure-db.md"]
    assert "## awareness-policy (" in reg
    assert "## conversation-continuity (" in reg
    last_taxonomy = reg.index("## emergency-safety (")
    assert reg.index("## awareness-policy (") > last_taxonomy
    assert reg.index("## conversation-continuity (") > last_taxonomy


@requires_docs
def test_red_team_merges_both_sources_with_two_hashes():
    rt = _generated()["red-team.md"]
    # both internal sources are named and hashed in the one header
    assert "docs/internal/eval/red-team-v1.md" in rt
    assert "docs/internal/eval/red-team-v2-methodology.md" in rt
    assert rt.count("SHA-256") >= 2
    # the two ruled sections
    assert "## How we red-team" in rt
    assert "## Results to date" in rt
    # the honest status line about the un-run 205-case suite
    assert "205 cases" in rt and "not yet been executed" in rt


@requires_docs
def test_red_team_export_is_counts_only_with_review_required_placeholders():
    rt = _generated()["red-team.md"]
    assert "Results at a glance" in rt          # per-category counts kept
    assert "REVIEW-REQUIRED" in rt              # verbatim exchanges withheld
    # a self-harm / crisis section must be flagged for owner hand-pick, never auto-published
    assert "crisis" in rt.lower() and "self-harm" in rt.lower()
    # no verbatim adversarial response block leaks through, including inline crisis quotes
    assert "Llama al 911 ahora mismo" not in rt
    assert "mastica una" not in rt
    assert not re.search(r"\d+ ?mg o cuatro", rt)  # the medication-dose snippet is withheld


@requires_docs
def test_methodology_exports_are_near_verbatim_minus_redactions():
    bench = _generated()["benchmarks.md"]
    assert "three legally-clean pillars" in bench   # substance preserved verbatim
    assert "$1.1B" not in bench                      # but dollar amounts redacted


# --------------------------------------------------------------------------- #
# Drift guard, both directions (mirrors tests/test_capabilities.py)
# --------------------------------------------------------------------------- #
@requires_docs
def test_drift_guard_tracked_artifacts_match_regeneration():
    """When docs/internal/ exists, the tracked docs/testing/ files must equal a fresh
    export. Edit a source without regenerating (`python scripts/export_testing_docs.py`)
    and this fails, so docs/testing/ single-sources from docs/internal/, same as the
    README capability table."""
    generated = _generated()
    assert generated, "generator produced no artifacts"
    for name, content in generated.items():
        tracked = _TESTING / name
        assert tracked.exists(), f"missing tracked artifact {name}"
        assert tracked.read_text() == content, f"drift in {name}: regenerate docs/testing/"


@requires_docs
def test_drift_guard_detects_a_tampered_artifact():
    """Reverse direction: a byte of drift must be caught, proving the guard has teeth."""
    generated = _generated()
    name, content = next(iter(generated.items()))
    tampered = content + "\nsneaky hand edit\n"
    assert tampered != content  # the equality check the forward guard runs would fail here


def test_drift_guard_skips_cleanly_when_docs_absent(tmp_path):
    """Public clones have no docs/internal/ tree; generate_all over an empty internal
    root yields nothing and the guard has nothing to compare, so it is a clean no-op
    rather than a failure."""
    empty_internal = tmp_path / "internal"
    empty_internal.mkdir()
    assert xp.generate_all(empty_internal) == {}
    assert xp.docs_available(tmp_path) is False
