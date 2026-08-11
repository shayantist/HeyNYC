from datetime import date

from heynyc.core.citations import data_provenance
from heynyc.core.grounding import check_grounding


def _lookup_citation() -> dict:
    snapshot = {"lookup_at": "2026-08-11T03:15:48-04:00"}
    return {
        "kind": "DATA",
        "title": "NYC FoodHelp availability lookup",
        "snippet": "Current lookup",
        "provenance": data_provenance(
            snapshot,
            record_id="availability-summary",
            field_pointer="/",
        ),
    }


def test_weekday_is_not_semantically_parsed() -> None:
    verdict = check_grounding(
        "Ahora es martes. El horario de jueves no sirve hoy. {cite:S1}",
        {"S1": _lookup_citation()},
        current_date=date(2026, 8, 11),
    )

    assert verdict is None


def test_f196_current_spanish_weekday_is_not_semantically_parsed() -> None:
    verdict = check_grounding(
        "Hoy es martes. {cite:S1}",
        {"S1": _lookup_citation()},
        current_date=date(2026, 8, 11),
    )

    assert verdict is None
