"""Offline tests for the LDSS-4826 application module (no network, no LLM)."""
from __future__ import annotations

from heynyc.modules.benefits.application import SLOTS, validate_slots


def test_validate_accepts_good_input_and_reports_no_missing():
    raw = {"legal_name": "Ana Diaz", "residence_street": "1 Main St",
           "residence_city": "Bronx", "residence_zip": "10453",
           "monthly_income": 1500}
    clean, missing, errors = validate_slots(raw)
    assert errors == []
    assert clean["legal_name"] == "Ana Diaz"
    assert clean["monthly_income"] == 1500.0          # money coerced to float
    assert missing == []                               # dob/ssn/phone are optional


def test_validate_flags_missing_required_and_never_invents():
    clean, missing, errors = validate_slots({"legal_name": "Ana Diaz"})
    assert "residence_street" in missing and "residence_zip" in missing
    # the validator NEVER fabricates a value for a field it wasn't given
    assert "residence_street" not in clean


def test_validate_rejects_unknown_keys_and_bad_types():
    clean, missing, errors = validate_slots(
        {"legal_name": "Ana", "residence_street": "1 Main St",
         "residence_city": "Bronx", "residence_zip": "10453",
         "evil_key": "x", "monthly_income": "lots"})
    assert any("evil_key" in e for e in errors)        # unknown key flagged
    assert any("monthly_income" in e for e in errors)  # bad number flagged
    assert "evil_key" not in clean                     # never silently accepted


def test_validate_enforces_iso_date():
    clean, missing, errors = validate_slots(
        {"legal_name": "A", "residence_street": "1", "residence_city": "B",
         "residence_zip": "10453", "dob": "04/02/1990"})
    assert any("dob" in e for e in errors)             # must be YYYY-MM-DD
    assert "dob" not in clean


def test_required_set_matches_form_minimum():
    # the form's stated minimum to file is name + address; nothing else is required
    required = {s.key for s in SLOTS if s.required}
    assert required == {"legal_name", "residence_street", "residence_city", "residence_zip"}


def test_summary_lists_values_missing_and_disclaimer():
    from heynyc.modules.benefits.application import DISCLAIMER, application_summary
    s = application_summary(
        {"legal_name": "Ana Diaz", "monthly_income": 1500.0}, ["residence_street"])
    assert "Legal name: Ana Diaz" in s
    assert "$1,500.00" in s                       # money formatted
    assert "Home street address" in s            # surfaced as still-needed
    assert DISCLAIMER in s                        # disclaimer always present


def test_summary_is_stable_for_rerender():
    # editing = re-render from the same clean dict → identical output (deterministic)
    from heynyc.modules.benefits.application import application_summary
    clean = {"legal_name": "Ana Diaz", "residence_street": "1 Main St",
             "residence_city": "Bronx", "residence_zip": "10453"}
    assert application_summary(clean, []) == application_summary(dict(clean), [])


def test_summary_carries_provenance_stamp():
    from heynyc.modules.benefits.application import application_summary
    s = application_summary({"legal_name": "Ana Diaz"}, ["residence_street"])
    assert "Rev. 12/23" in s and "otda.ny.gov" in s     # form revision surfaced to the user


def test_template_provenance_and_integrity():
    from heynyc.modules.benefits.application import (
        provenance_stamp,
        template_provenance,
        verify_template_integrity,
    )
    meta = template_provenance()
    assert meta["revision"] == "Rev. 12/23" and meta["num_pages"] == 12
    assert verify_template_integrity() is True          # vendored PDF matches recorded sha256
    assert "Rev. 12/23" in provenance_stamp()


def test_integrity_fails_on_hash_mismatch():
    # a swapped/updated template (different hash) is caught → caller must degrade, not fill
    from heynyc.modules.benefits.application import TEMPLATE, verify_template_integrity
    assert verify_template_integrity(TEMPLATE, {"sha256": "deadbeef"}) is False


# --- PDF overlay fill (Tasks 1 + 3) ----------------------------------------

def test_fill_returns_a_real_pdf_with_values_actually_placed():
    import io

    from pypdf import PdfReader

    from heynyc.modules.benefits.application import fill_application

    clean = {"legal_name": "Ana Diaz", "residence_street": "1 Main St",
             "residence_city": "Bronx", "residence_zip": "10453"}
    pdf = fill_application(clean)
    assert pdf[:4] == b"%PDF" and len(pdf) > 1000
    text = PdfReader(io.BytesIO(pdf)).pages[2].extract_text()   # the data page (index 2)
    assert "Ana Diaz" in text and "10453" in text    # the values are really drawn onto the page


def test_fill_raises_form_drift_when_anchor_missing():
    import pytest

    from heynyc.modules.benefits.application import FormDriftError, fill_application
    bad = {"mode": "overlay-anchor", "fields": {
        "legal_name": {"page": 2, "anchor": "NoSuchLabelXYZ", "dx": 6, "dy": 0, "size": 9}}}
    with pytest.raises(FormDriftError):
        fill_application({"legal_name": "Ana Diaz"}, fmap=bad)


def test_demo_fields_clear_the_form_underlines():
    from heynyc.modules.benefits.application import load_map

    fields = load_map()["fields"]
    for key in ("legal_name", "residence_street", "residence_city", "residence_zip"):
        assert fields[key]["dy"] >= 4


def test_high_stakes_fields_are_the_consequential_ones():
    from heynyc.modules.benefits.application import SLOTS
    hs = {s.key for s in SLOTS if s.high_stakes}
    assert hs == {"legal_name", "dob", "ssn", "monthly_income"}


def test_review_request_flags_high_stakes_and_carries_two_tier_attestation():
    from heynyc.modules.benefits.application import (
        APPLICANT_ATTESTATION,
        SCRIBE_CERT,
        review_request,
    )
    clean = {"legal_name": "Ana Diaz", "monthly_income": 1500.0, "phone": "555-1212"}
    r = review_request(clean)
    assert "Ana Diaz" in r and "$1,500.00" in r
    # only high-stakes fields (name, income) get the double-check prompt; phone does not
    assert r.count("double-check") == 2
    assert SCRIBE_CERT in r and APPLICANT_ATTESTATION in r
    assert "answers ," not in r
    assert "what to change" in r          # invites edits, not a yes/no rubber-stamp
