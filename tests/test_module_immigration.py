"""Offline contracts for the retrieval-first immigration module."""

from urllib.parse import urlsplit

from heynyc.core import config
from heynyc.core.registry import Registry
from heynyc.eval.cases import load_cases


def _module():
    return next(module for module in Registry.discover(config.MODULES_DIR).modules if module.name == "immigration")


def _cases():
    registry = Registry.discover(config.MODULES_DIR)
    return [case for case in load_cases(registry) if case.module == "immigration"]


def test_immigration_situations_are_high_stakes_official_retrieval():
    module = _module()
    allowed = set(module.allowlist)

    assert module.official_only
    assert module.situations
    assert module.tools is None
    for situation in module.situations:
        assert situation.high_stakes
        assert situation.query
        assert situation.urls
        assert "official_sources" in situation.focus_tools
        for url in situation.urls:
            parsed = urlsplit(url)
            assert parsed.scheme == "https"
            assert any(
                parsed.hostname == domain or parsed.hostname.endswith(f".{domain}")
                for domain in allowed
            )


def test_immigration_eval_covers_languages_limits_and_reporting():
    cases = {case.id: case for case in _cases()}

    for suffix, language in (("", "en"), ("_haitian_creole", "ht"), ("_arabic", "ar")):
        for scenario in ("tps_change", "enforcement_rights"):
            case = cases[f"immigration_{scenario}{suffix}"]
            assert case.language == language
            assert case.invariants["must_ground"]
            assert case.invariants["must_cite_if_asserting"]

    assert cases["immigration_no_individual_status_determination"].invariants[
        "must_abstain_or_redirect"
    ]
    assert cases["immigration_out_of_scope_enforcement_locations"].abstain
    assert cases["immigration_report_witnessed_enforcement"].expect_tools == ["official_sources"]


def test_tps_status_requires_reconciling_later_official_actions():
    situation = next(
        hint for hint in _module().situations if hint.name == "tps_status_change"
    )

    assert "later court or agency action" in situation.reminder
    assert "older notice" in situation.reminder


def test_enforcement_rights_uses_the_claim_bearing_official_documents():
    situation = next(
        hint for hint in _module().situations
        if hint.name == "immigration_enforcement_rights"
    )

    assert "EN-Red-Card-Cutout-Printable.pdf" in "\n".join(situation.urls)
    assert "kyr_red_card_haitian_creole.pdf" in "\n".join(situation.urls)
    assert "KYR-with-ICE_February-2025_English.pdf" in "\n".join(situation.urls)
    assert "KYR-with-ICE-2026-Haitian-Creole.pdf" in "\n".join(situation.urls)
    assert situation.focus_tools == ["official_sources"]


def test_tps_status_includes_current_new_york_support_source():
    situation = next(
        hint for hint in _module().situations if hint.name == "tps_status_change"
    )

    assert any("opwdd.ny.gov" in url for url in situation.urls)


def test_tps_status_includes_current_responsible_agency_directory():
    module = _module()
    situation = next(
        hint for hint in module.situations if hint.name == "tps_status_change"
    )
    url = "https://www.uscis.gov/humanitarian/temporary-protected-status"

    assert url in module.seeds
    assert url in situation.urls
