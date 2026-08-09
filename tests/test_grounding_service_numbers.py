from heynyc.core.grounding import check_grounding


def _citation(snippet: str) -> dict:
    return {
        "S1": {
            "title": "Official service guidance",
            "snippet": snippet,
            "provenance": {"evidence_grade": "authoritative"},
        }
    }


def test_uncited_civic_service_number_is_blocked():
    verdict = check_grounding(
        "Call 311 for help. {cite:S1}",
        _citation("Apply online for help."),
    )

    assert verdict is not None
    assert verdict.blocking
    assert [failure.kind for failure in verdict.hard_failures] == ["service_number"]


def test_civic_service_number_passes_when_the_source_supports_it():
    verdict = check_grounding(
        "Call 311 for help. {cite:S1}",
        _citation("Call 311 for help with this service."),
    )

    assert verdict is not None
    assert not verdict.blocking


def test_civic_service_number_does_not_match_inside_a_full_phone_number():
    verdict = check_grounding(
        "Call 311 for help. {cite:S1}",
        _citation("Call 212-555-0311 for this office."),
    )

    assert verdict is not None
    assert verdict.blocking
