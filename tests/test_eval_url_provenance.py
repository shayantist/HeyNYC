from heynyc.eval.cases import EvalCase
from heynyc.eval.checks import run_checks
from heynyc.eval.runner import CaseResult
from heynyc.core.citations import data_provenance


def _result(text: str, citations: dict) -> CaseResult:
    return CaseResult(
        case=EvalCase(id="url-provenance", module="global", query="Help"),
        text=text,
        citations=citations,
    )


async def test_eval_rejects_a_direct_url_not_returned_by_a_tool() -> None:
    result = _result(
        "Official plan: https://nyc.gov/housing-plan-for-a",
        {"S1": {"url": "https://nyc.gov/housing-plan-for-a-n", "kind": "WEB"}},
    )

    checks = await run_checks(result)

    provenance = next(check for check in checks if check.name == "url_provenance")
    assert not provenance.passed
    assert "https://nyc.gov/housing-plan-for-a" in provenance.detail


async def test_eval_accepts_a_registered_direct_url_without_checking_the_network() -> None:
    calls = 0

    async def checker(_url: str) -> int:
        nonlocal calls
        calls += 1
        return 404

    result = _result(
        "Official plan: https://nyc.gov/housing-plan-for-a-n",
        {"S1": {"url": "https://nyc.gov/housing-plan-for-a-n", "kind": "WEB"}},
    )

    checks = await run_checks(result, link_checker=checker)

    provenance = next(check for check in checks if check.name == "url_provenance")
    assert provenance.passed
    assert calls == 0


async def test_eval_stops_a_markdown_url_before_bengali_case_suffix() -> None:
    result = _result(
        "[ACCESS HRA](https://a069-access.nyc.gov/ACCESSNYC/application.do)-তে যান।",
        {
            "S1": {
                "url": "https://a069-access.nyc.gov/ACCESSNYC/application.do",
                "kind": "WEB",
            }
        },
    )

    checks = await run_checks(result)

    provenance = next(check for check in checks if check.name == "url_provenance")
    assert provenance.passed


async def test_eval_accepts_an_exact_url_returned_by_a_tool() -> None:
    result = _result("Directions: https://finder.nyc.gov/foodhelp", {})
    result.messages = [
        {"role": "tool", "content": "Use https://finder.nyc.gov/foodhelp for current listings."}
    ]

    checks = await run_checks(result)

    provenance = next(check for check in checks if check.name == "url_provenance")
    assert provenance.passed


async def test_eval_accepts_a_map_url_derived_from_cited_data_coordinates() -> None:
    citations = {
        "S1": {
            "url": "https://data.cityofnewyork.us/resource/example/row.json",
            "kind": "DATA",
            "provenance": data_provenance(
                {"name": "Bryant Park"},
                record_id="row",
                field_pointer="/",
                derivation={"point": [40.75404, -73.98270]},
            ),
        }
    }
    result = _result(
        "[Map](https://www.google.com/maps/search/?api=1&query=40.75404,-73.98270) {cite:S1}",
        citations,
    )

    checks = await run_checks(result)

    provenance = next(check for check in checks if check.name == "url_provenance")
    assert provenance.passed


async def test_eval_rejects_a_map_url_with_coordinates_not_in_cited_data() -> None:
    citations = {
        "S1": {
            "url": "https://data.cityofnewyork.us/resource/example/row.json",
            "kind": "DATA",
            "provenance": data_provenance(
                {"name": "Bryant Park"},
                record_id="row",
                field_pointer="/",
                derivation={"point": [40.75404, -73.98270]},
            ),
        }
    }
    result = _result(
        "[Map](https://www.google.com/maps/search/?api=1&query=40.70000,-73.90000) {cite:S1}",
        citations,
    )

    checks = await run_checks(result)

    provenance = next(check for check in checks if check.name == "url_provenance")
    assert not provenance.passed


async def test_eval_ignores_malformed_map_coordinates_without_crashing() -> None:
    result = _result(
        "[Map](https://www.google.com/maps/search/?api=1&query=40.75404,-73.98270) {cite:S1}",
        {
            "S1": {
                "kind": "DATA",
                "provenance": {"derivation": {"point": ["not-a-latitude", "not-a-longitude"]}},
            }
        },
    )

    checks = await run_checks(result)

    provenance = next(check for check in checks if check.name == "url_provenance")
    assert not provenance.passed


async def test_eval_accepts_a_map_url_derived_from_cited_snapshot_coordinates() -> None:
    result = _result(
        "[Map](https://www.google.com/maps/search/?api=1&query=40.75404,-73.98270) {cite:S1}",
        {
            "S1": {
                "kind": "DATA",
                "provenance": {"snapshot": {"latitude": 40.75404, "longitude": -73.98270}},
            }
        },
    )

    checks = await run_checks(result)

    provenance = next(check for check in checks if check.name == "url_provenance")
    assert provenance.passed


async def test_eval_does_not_check_an_unused_internal_citation() -> None:
    calls = 0

    async def checker(_url: str) -> int:
        nonlocal calls
        calls += 1
        return 404

    checks = await run_checks(
        _result(
            "I hit a temporary problem before I could verify an answer.",
            {"S1": {"url": "https://dead.example/unused", "kind": "WEB"}},
        ),
        link_checker=checker,
    )

    assert all(check.name != "link_liveness" for check in checks)
    assert calls == 0
