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
