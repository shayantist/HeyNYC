from heynyc.core.grounding import check_grounding


def _data_citation(name: str, comments: str) -> dict:
    return {
        "kind": "DATA",
        "title": "NYC Open Data",
        "snippet": name,
        "provenance": {
            "evidence_grade": "authoritative",
            "snapshot": {"facility_name": name, "comments": comments},
        },
    }


def test_f178_rejects_a_dropped_restricted_schedule_window_in_spanish():
    citations = {
        "S1": _data_citation(
            "Hunts Point SNAP Center",
            "Work Requirements (8:30am to 9:00am only); Monday-Friday 8:30am to 5:00pm",
        ),
        "S2": _data_citation(
            "East End SNAP Center",
            "Monday-Friday 8:30am to 5:00pm",
        ),
    }

    verdict = check_grounding(
        "Ambos centros atienden de lunes a viernes, de 8:30 a. m. a 5:00 p. m. "
        "{cite:S1} {cite:S2}",
        citations,
    )

    assert verdict is not None
    assert verdict.blocking
    schedule_failure = next(
        failure
        for failure in verdict.hard_failures
        if failure.kind == "schedule_qualifier"
    )
    assert "{cite:" not in schedule_failure.claim


def test_f178_accepts_the_restricted_schedule_window_in_spanish():
    citations = {
        "S1": _data_citation(
            "Hunts Point SNAP Center",
            "Work Requirements (8:30am to 9:00am only); Monday-Friday 8:30am to 5:00pm",
        )
    }

    verdict = check_grounding(
        "Hunts Point atiende requisitos laborales de 8:30 a. m. a 9:00 a. m.; el horario "
        "habitual es de 8:30 a. m. a 5:00 p. m. {cite:S1}",
        citations,
    )

    assert verdict is not None
    assert not verdict.blocking


def test_ordinary_schedule_has_no_extra_qualifier_requirement():
    citations = {
        "S1": _data_citation(
            "East End SNAP Center",
            "Monday-Friday 8:30am to 5:00pm",
        )
    }

    verdict = check_grounding(
        "East End atiende de 8:30 a. m. a 5:00 p. m. {cite:S1}",
        citations,
    )

    assert verdict is not None
    assert not verdict.blocking
