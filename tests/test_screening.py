import json as _json

import httpx
import pytest

from heynyc.modules.benefits import screening
from heynyc.modules.benefits.screening import assert_pii_free, request_summary, screen


def _client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_get_token_fetches_and_caches():
    screening.clear_token("https://sb")
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        assert req.url.path == "/authToken"
        return httpx.Response(200, json={"type": "SUCCESS", "token": "tok-123"})

    c = _client(handler)
    t1 = await screening.get_token(c, "https://sb", "u", "p")
    t2 = await screening.get_token(c, "https://sb", "u", "p")  # cached
    await c.aclose()
    assert t1 == t2 == "tok-123"
    assert calls["n"] == 1  # second call hit the cache


def test_pii_guard_passes_clean_profile():
    assert_pii_free({"livingRenting": True}, [{"age": 32, "householdMemberType": "HeadOfHousehold"}])


def test_pii_guard_rejects_name_key_and_ssn_value():
    with pytest.raises(ValueError):
        assert_pii_free({}, [{"age": 30, "name": "Jane Doe"}])
    with pytest.raises(ValueError):
        assert_pii_free({"note": "123-45-6789"}, [{"age": 30}])


def test_pii_guard_recurses_into_nested_income_value():
    # A phone number buried in a nested persons[].incomes[].* field must be caught, and the
    # error must name the offending field path so the handler can report where it slipped in.
    with pytest.raises(ValueError, match=r"persons\[0\]\.incomes\[0\]"):
        assert_pii_free(
            {},
            [{"age": 30, "householdMemberType": "HeadOfHousehold",
              "incomes": [{"amount": "1200", "type": "call 555-123-4567", "frequency": "Monthly"}]}],
        )


def test_pii_guard_recurses_into_nested_key_name():
    # A PII key name nested below the top level must still be rejected.
    with pytest.raises(ValueError, match=r"ssn"):
        assert_pii_free({}, [{"age": 30, "incomes": [{"amount": "1200", "ssn": "x"}]}])


@pytest.mark.parametrize("value", [
    "555-123-4567",              # phone (dashed)
    "(212) 555-0199",            # phone (parenthesized)
    "jane.doe@example.com",      # email
    "123 Main Street",           # street address
    "A123456789",                # immigration A-number
    "1234567890123456789",       # EBT card (19 digits)
    "123-45-6789",               # SSN
    "01/15/1990",                # DOB
])
def test_pii_guard_catches_each_identifier_class(value):
    with pytest.raises(ValueError):
        assert_pii_free({}, [{"age": 30, "householdMemberType": "HeadOfHousehold", "note": value}])


def test_pii_guard_allows_legit_numeric_fields():
    # income amounts, household size / counts, ages, a bare zip are legitimate and MUST pass,
    # including string-typed values nested in incomes.
    assert_pii_free(
        {"livingRenting": True, "householdSize": 3, "cashOnHand": 500, "zip": "10001"},
        [
            {"age": 45, "householdMemberType": "HeadOfHousehold",
             "incomes": [{"amount": "1200", "type": "Wages", "frequency": "Monthly"},
                         {"amount": "200000", "type": "SelfEmployment", "frequency": "Annual"}]},
            {"age": 7, "householdMemberType": "Child"},
        ],
    )


async def test_screen_posts_body_and_returns_programs():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["path"] = req.url.path
        seen["auth"] = req.headers.get("Authorization")
        seen["body"] = _json.loads(req.content)
        return httpx.Response(200, json={"type": "SUCCESS",
                                         "eligiblePrograms": [{"code": "S2R007", "name": "SNAP"}]})

    c = _client(handler)
    out = await screen(c, "https://sb", "tok", {"livingRenting": True},
                       [{"age": 32, "householdMemberType": "HeadOfHousehold"}])
    await c.aclose()
    assert out["eligiblePrograms"][0]["code"] == "S2R007"
    assert seen["path"] == "/eligibilityPrograms"
    assert seen["auth"] == "tok"                       # raw token per docs (verify Bearer on sandbox)
    assert seen["body"][0]["withholdPayload"] is True
    assert seen["body"][0]["person"][0]["age"] == 32


def test_request_summary_is_redacted():
    s = request_summary({"livingRenting": True},
                        [{"age": 32, "incomes": [{"amount": "1200", "type": "Wages", "frequency": "Monthly"}]}])
    assert s == {"persons": 1, "household_keys": ["livingRenting"], "has_income": True}
    assert "1200" not in str(s)                        # no raw amounts in provenance
