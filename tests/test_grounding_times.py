from datetime import date

from heynyc.core.grounding import check_grounding


def _citations() -> dict:
    return {
        "S2": {
            "kind": "DATA",
            "provenance": {
                "snapshot": {
                    "fp_tue_open1": "10:00 AM",
                    "fp_tue_close1": "06:00 PM",
                }
            },
        },
        "S3": {
            "kind": "DATA",
            "provenance": {
                "snapshot": {"lookup_at": "2026-08-11T03:40:46-04:00"}
            },
        },
    }


def test_clock_time_is_not_semantically_parsed() -> None:
    result = check_grounding(
        "Ahora son las 3:40 a. m. del martes. {cite:S2}",
        _citations(),
        current_date=date(2026, 8, 11),
    )

    assert result is None


def test_f200_current_time_is_not_parsed_from_resident_language() -> None:
    result = check_grounding(
        "Ahora son las 3:40 a. m. del martes. {cite:S3}",
        _citations(),
        current_date=date(2026, 8, 11),
    )

    assert result is None


def test_f200_reformatted_schedule_times_are_not_semantically_parsed() -> None:
    result = check_grounding(
        "El horario es de 10:00 a. m. a 6:00 p. m. {cite:S2}",
        _citations(),
        current_date=date(2026, 8, 11),
    )

    assert result is None


def test_f233_only_in_an_unrelated_snapshot_field_does_not_restrict_hours() -> None:
    citations = {
        "S1": {
            "kind": "DATA",
            "provenance": {
                "snapshot": {
                    "Facility_name": "Senior Center - OLDER ADULTS ONLY",
                    "Tuesday": "9a-4p",
                    "cc_tue_open1": "09:00 AM",
                    "cc_tue_close1": "04:00 PM",
                }
            },
        }
    }

    result = check_grounding(
        "The center is scheduled open until 4 p.m. {cite:S1}",
        citations,
        current_date=date(2026, 8, 11),
    )

    assert result is None
