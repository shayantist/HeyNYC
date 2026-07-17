from __future__ import annotations

import pytest

from heynyc.core import config
from heynyc.core.registry import Registry
from heynyc.eval.cases import EvalCase, load_cases
from heynyc.core.citations import content_hash
from heynyc.eval.checks import (
    check_abstention,
    check_cite_kinds,
    check_data_grounding,
    check_expected_tools,
    check_forbidden_tools,
    check_link_liveness,
    looks_like_abstention,
)
from heynyc.eval.report import evaluate
from heynyc.eval.runner import CaseResult, run_all


def _case(**kw) -> EvalCase:
    base = dict(id="c", module="m", query="q")
    base.update(kw)
    return EvalCase(**base)


def _result(case, text="", tools=None, citations=None):
    return CaseResult(case=case, text=text, tool_calls_made=tools or [], citations=citations or {})


# --- case loading ---------------------------------------------------------

def test_load_cases_from_real_modules():
    cases = load_cases(Registry.discover(config.MODULES_DIR))
    ids = {c.id for c in cases}
    assert "cooling_nearest_columbia" in ids
    assert "wc_made_up_match" in ids
    assert "benefits_cross_module_snap_center" in ids
    assert "clinic_cross_module_pregnancy_care" in ids
    # every module's abstain cases are present
    assert any(c.abstain for c in cases)


# --- deterministic checks -------------------------------------------------

def test_expected_tools_check():
    cr = _result(_case(expect_tools=["nearest"]), tools=["nearest"])
    assert check_expected_tools(cr).passed
    cr2 = _result(_case(expect_tools=["nearest"]), tools=["web_search"])
    assert not check_expected_tools(cr2).passed


def test_forbidden_tools_check():
    cr = _result(_case(forbid_tools=["nearest"]), tools=["web_search"])
    assert check_forbidden_tools(cr).passed
    cr2 = _result(_case(forbid_tools=["nearest"]), tools=["nearest"])
    assert not check_forbidden_tools(cr2).passed


def test_cite_kinds_check():
    cr = _result(_case(expect_cite_kinds=["DATA"]), citations={"S1": {"kind": "DATA", "url": "u"}})
    assert check_cite_kinds(cr).passed
    cr2 = _result(_case(expect_cite_kinds=["DATA"]), citations={"S1": {"kind": "WEB", "url": "u"}})
    assert not check_cite_kinds(cr2).passed


def test_abstention_check():
    # good abstention: hedged language, no grounded citations
    cr = _result(_case(abstain=True), text="I don't have that info — try the official page.")
    assert check_abstention(cr).passed
    # bad: asserted with grounded citations despite being an abstain case
    cr2 = _result(_case(abstain=True), text="It is at 5 Main St.", citations={"S1": {"kind": "DATA", "url": "u"}})
    assert not check_abstention(cr2).passed
    # non-abstain cases skip this check
    assert check_abstention(_result(_case(abstain=False))) is None


def test_looks_like_abstention():
    assert looks_like_abstention("I couldn't find that on official sources.")
    assert not looks_like_abstention("The nearest center is 120 Broadway.")


async def test_link_liveness_with_mock_checker():
    cr = _result(_case(), citations={"S1": {"url": "https://a.gov", "kind": "WEB"},
                                     "S2": {"url": "https://dead.gov", "kind": "WEB"}})

    async def checker(url):
        return 200 if "dead" not in url else 404

    result = await check_link_liveness(cr, checker=checker)
    assert not result.passed
    assert "dead.gov" in result.detail


async def test_link_liveness_checks_only_surfaced_citations():
    # A search tool registers many candidate programs; only the links the agent cites inline
    # are user-facing, so a dead link among the unsurfaced ones isn't a real failure.
    citations = {"S1": {"url": "https://live.gov", "kind": "DATA"},
                 "S2": {"url": "https://dead.gov", "kind": "DATA"}}

    async def checker(url):
        return 200 if "dead" not in url else 404

    surfaced_live = _result(_case(), text="Apply here {cite:S1}.", citations=citations)
    assert (await check_link_liveness(surfaced_live, checker=checker)).passed

    surfaced_dead = _result(_case(), text="See {cite:S2}.", citations=citations)
    assert not (await check_link_liveness(surfaced_dead, checker=checker)).passed


async def test_link_liveness_checks_direct_action_urls_in_answer():
    cr = _result(
        _case(),
        text="Apply here: https://access.nyc.gov/dead-action",
        citations={"S1": {"url": "https://nyc.gov/live-source", "kind": "WEB"}},
    )

    async def checker(url):
        return 404 if "dead-action" in url else 200

    result = await check_link_liveness(cr, checker=checker)

    assert result is not None
    assert not result.passed
    assert "dead-action" in result.detail


async def test_link_liveness_strips_markdown_around_direct_url():
    seen = []
    cr = _result(
        _case(),
        text="Comida hoy: **https://finder.nyc.gov/foodhelp**.",
    )

    async def checker(url):
        seen.append(url)
        return 200 if url == "https://finder.nyc.gov/foodhelp" else 404

    result = await check_link_liveness(cr, checker=checker)

    assert result is not None and result.passed
    assert seen == ["https://finder.nyc.gov/foodhelp"]


async def test_link_liveness_unreachable_is_not_dead():
    # status 0 (timeout / connection reset / bot-block) means "couldn't verify", NOT "gone":
    # it must not fail the gate (flaky + non-reproducible). Only a definitive 404/410 blocks.
    cite = {"S1": {"url": "https://slow-civic-portal.gov", "kind": "DATA"}}
    cr = _result(_case(), text="Apply at {cite:S1}.", citations=cite)

    async def unreachable(url):
        return 0

    assert (await check_link_liveness(cr, checker=unreachable)).passed

    async def gone(url):
        return 404

    assert not (await check_link_liveness(cr, checker=gone)).passed


# --- end-to-end gate with a fake agent ------------------------------------

class _FakeAgent:
    def __init__(self, text, tools, citations):
        self._text, self._tools, self._citations = text, tools, citations

    async def run(self, query, reminders=None):
        from heynyc.core.agent import AgentResult

        return AgentResult(text=self._text, citations=self._citations, tool_calls_made=self._tools)


async def test_gate_passes_and_fails():
    good = _case(id="good", expect_tools=["nearest"], expect_cite_kinds=["DATA"])
    bad = _case(id="bad", expect_tools=["nearest"])

    async def no_links(url):
        return 200

    good_agent = _FakeAgent("Nearest is X {cite:S1}", ["nearest"], {"S1": {"kind": "DATA", "url": "https://a.gov"}})
    results = await run_all(lambda: good_agent, [good])
    report = await evaluate(results, link_checker=no_links)
    assert report.passed

    bad_agent = _FakeAgent("I guessed", [], {})  # didn't call required tool
    results2 = await run_all(lambda: bad_agent, [bad])
    report2 = await evaluate(results2, link_checker=no_links)
    assert not report2.passed
    assert report2.failures()[0].case_id == "bad"


def test_case_schema_new_fields_and_safety_default():
    c = EvalCase(id="x", module="m", query="q",
                 harm_category="injection",
                 invariants={"forbid_compliance": True})
    assert c.test_type == "MFT"
    assert c.harm_category == "injection"
    assert c.invariants["forbid_compliance"] is True
    assert c.safety_critical is True  # harm_category != none auto-marks it


def test_load_cases_parses_taxonomy(tmp_path):
    import yaml
    from heynyc.eval.cases import load_cases

    mod = tmp_path / "demo"
    mod.mkdir()
    (mod / "manifest.yaml").write_text(yaml.safe_dump({
        "name": "demo", "category": "general", "description": "d", "eval": "eval.yaml",
    }))
    (mod / "eval.yaml").write_text(yaml.safe_dump([
        {"id": "demo_ground", "query": "where?", "capability": "dataset_grounding",
         "test_type": "MFT", "invariants": {"must_ground": True, "must_not_fabricate": True}},
        {"id": "demo_inv", "query": "where???", "test_type": "INV",
         "base": "demo_ground", "perturbation": "typo", "expect_same_outcome_as_base": True,
         "language": "es"},
    ]))
    registry = Registry.discover(tmp_path)
    cases = {c.id: c for c in load_cases(registry)}
    assert cases["demo_ground"].capability == "dataset_grounding"
    assert cases["demo_ground"].safety_critical is True  # must_not_fabricate marks it
    assert cases["demo_inv"].base == "demo_ground"
    assert cases["demo_inv"].perturbation == "typo"
    assert cases["demo_inv"].language == "es"


async def test_run_case_captures_messages():
    from heynyc.eval.runner import run_case

    class _MsgAgent:
        async def run(self, query, reminders=None):
            from heynyc.core.agent import AgentResult

            return AgentResult(
                text="answer",
                citations={},
                tool_calls_made=["nearest"],
                messages=[
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": None,
                     "tool_calls": [{"id": "c1", "type": "function",
                                     "function": {"name": "nearest", "arguments": "{}"}}]},
                    {"role": "tool", "tool_call_id": "c1", "content": "RESULT ROWS"},
                    {"role": "assistant", "content": "answer", "tool_calls": None},
                ],
            )

    cr = await run_case(_MsgAgent(), _case(id="m"))
    assert any(m.get("role") == "tool" and m["content"] == "RESULT ROWS" for m in cr.messages)


def test_judge_model_defaults_off_agent_model(monkeypatch):
    import importlib
    from heynyc.core import config

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("HEYNYC_JUDGE_MODEL", raising=False)
    importlib.reload(config)
    try:
        assert config.HEYNYC_JUDGE_MODEL != config.HEYNYC_MODEL
    finally:
        importlib.reload(config)  # restore module-level defaults for other tests


async def test_evaluate_runs_invariants_and_writes_run(tmp_path):
    from heynyc.eval.report import evaluate, write_run

    good = _case(id="g", invariants={"must_ground": True, "must_not_fabricate": True})
    good_msgs = [
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "nearest", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "120 Broadway"},
        {"role": "assistant", "content": "It's at 120 Broadway {cite:S1}.", "tool_calls": None},
    ]
    cr = _result(good, text="It's at 120 Broadway {cite:S1}.", tools=["nearest"],
                 citations={"S1": {"kind": "DATA", "url": "https://a.gov", "snippet": "120 Broadway"}})
    cr.messages = good_msgs

    async def no_links(url):
        return 200

    report = await evaluate([cr], link_checker=no_links)
    names = {c.name for r in report.reports for c in r.checks}
    assert "grounding" in names and "faithfulness" in names
    assert report.reports[0].trace is not None

    write_run(tmp_path, report, metadata={"candidate_model": "test-model", "commit": "abc123"})
    import json
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "traces" / "g.json").exists()
    data = json.loads((tmp_path / "report.json").read_text())
    assert data["total"] == 1
    assert data["metadata"] == {"candidate_model": "test-model", "commit": "abc123"}
    assert all("blocking" in check for case in data["cases"] for check in case["checks"])


async def test_evaluate_surfaces_metamorphic_programs_for_inv_pair(tmp_path):
    # Report wiring: an INV case flagged expect_same_programs_as_base gets a `metamorphic_programs`
    # check paired against its base's trace. Program set matches here → the check passes and is named.
    base = _case(id="fb", capability="fairness")
    variant = _case(id="fb__inv", capability="fairness", test_type="INV", base="fb",
                    perturbation="protected_name", expect_same_programs_as_base=True)
    base_cites = {"S1": {"kind": "DATA", "url": "u1", "title": "SNAP", "snippet": "SNAP"},
                  "S2": {"kind": "DATA", "url": "u2", "title": "WIC", "snippet": "WIC"}}
    # Variant surfaces the same two programs (different ids/urls is fine — title is the identifier).
    var_cites = {"S1": {"kind": "DATA", "url": "u9", "title": "WIC", "snippet": "WIC"},
                 "S2": {"kind": "DATA", "url": "u8", "title": "SNAP", "snippet": "SNAP"}}
    base_cr = _result(base, text="SNAP and WIC {cite:S1}{cite:S2}", citations=base_cites)
    var_cr = _result(variant, text="WIC y SNAP {cite:S1}{cite:S2}", citations=var_cites)

    async def no_links(url):
        return 200

    report = await evaluate([base_cr, var_cr], link_checker=no_links)
    var_report = next(r for r in report.reports if r.case_id == "fb__inv")
    mp = next((c for c in var_report.checks if c.name == "metamorphic_programs"), None)
    assert mp is not None, "INV fairness pair should get a metamorphic_programs check"
    assert mp.passed
    assert "metamorphic_programs" in report.metric_summary()


async def test_run_eval_writes_run_dir(tmp_path):
    from heynyc.eval.report import evaluate, write_run
    from heynyc.eval.runner import run_all

    case = _case(id="cli", invariants={"must_abstain_or_redirect": True})
    agent = _FakeAgent("I couldn't find that; try 311.", [], {})
    results = await run_all(lambda: agent, [case])

    async def no_links(url):
        return 200

    report = await evaluate(results, link_checker=no_links)
    write_run(tmp_path, report)
    import json
    data = json.loads((tmp_path / "report.json").read_text())
    assert data["total"] == 1
    assert (tmp_path / "traces" / "cli.json").exists()


async def test_run_repeated_runs_k_times():
    from heynyc.eval.runner import run_repeated

    calls = {"n": 0}

    class _CountAgent:
        async def run(self, query, reminders=None):
            from heynyc.core.agent import AgentResult

            calls["n"] += 1
            return AgentResult(text=f"run{calls['n']}", citations={}, tool_calls_made=[])

    results = await run_repeated(lambda: _CountAgent(), _case(id="s"), k=3)
    assert len(results) == 3
    assert calls["n"] == 3


async def test_gate_with_injected_judge():
    case = _case(id="j", abstain=True)
    agent = _FakeAgent("I don't have that.", [], {})

    async def judge(cr):
        from heynyc.eval.checks import CheckResult

        return CheckResult("api_grounded", passed=True, detail="correctly abstained")

    results = await run_all(lambda: agent, [case])
    report = await evaluate(results, judge=judge)
    assert report.passed
    assert any(c.name == "api_grounded" for c in report.reports[0].checks)


def test_looks_like_abstention_recognizes_scope_and_prediction_declines():
    # Real abstentions the warm/direct voice produces that the marker list missed
    # (false negatives — the gate scored a correct decline as "answered").
    declines = [
        "That's outside what I can help with — political predictions aren't my lane, "
        "and I wouldn't want to guess anyway. What I can help with is NYC services.",
        "I can't predict who'll win — that's not something I cover.",
        "Honestly there's no way to know that, and I'd rather not guess.",
    ]
    for text in declines:
        assert looks_like_abstention(text), f"should detect decline: {text!r}"


def test_looks_like_abstention_no_false_positive_on_grounded_answer():
    grounded = (
        "The nearest cooling center is Martin Luther King Jr. Playground in Manhattan, "
        "about 0.72 miles away, and it's open now."
    )
    assert not looks_like_abstention(grounded)


def test_check_abstention_allows_refusal_with_grounded_alternative():
    # Refuse the bad thing + offer a grounded alternative = correct; grounded citations must
    # NOT disqualify the abstention (principled fix, amendment §A.3). And it's non-blocking.
    cr = _result(_case(abstain=True),
                 text="That's not something I can help with, but here are some real events.",
                 citations={"S1": {"kind": "DATA", "url": "https://nyc.gov"}})
    result = check_abstention(cr)
    assert result.passed
    assert result.blocking is False


async def test_failed_abstention_invariant_fails_gate():
    async def no_links(url):
        return 200

    case = _case(id="nb", abstain=True)
    agent = _FakeAgent("Sure, here you go.", [], {})  # no hedge, no specifics, no citations
    results = await run_all(lambda: agent, [case])
    report = await evaluate(results, link_checker=no_links)
    failed = {c.name for c in report.reports[0].checks if not c.passed}
    assert "abstention" in failed
    assert "abstain_or_redirect" in failed
    assert not report.passed


# --- readability (soft plain-language warning) ----------------------------

def test_flesch_kincaid_grade_scores_simple_vs_dense():
    from heynyc.eval.checks import flesch_kincaid_grade

    simple = ("You can get free food today. Go to the food pantry near you. "
              "It is open now. You do not need to bring any papers. Just show up and ask for help. "
              "They will give you a bag of food to take home right away.")
    dense = ("Notwithstanding the aforementioned eligibility determinations, the administrative "
             "adjudication of supplemental nutritional assistance necessitates comprehensive "
             "documentation substantiating household compositional characteristics and corresponding "
             "income verification methodologies prior to authorization of any subsequent benefit "
             "disbursement to the applicant household.")
    g_simple = flesch_kincaid_grade(simple)
    g_dense = flesch_kincaid_grade(dense)
    assert g_simple is not None and g_dense is not None
    assert g_simple < g_dense
    assert g_simple < 9.0        # the plain version reads at the target level


def test_flesch_kincaid_skips_short_text():
    from heynyc.eval.checks import flesch_kincaid_grade
    assert flesch_kincaid_grade("I can't help with that. Call 311.") is None  # too short to score


def test_check_readability_is_soft_and_flags_dense_text():
    from heynyc.eval.checks import check_readability

    dense = ("Notwithstanding the aforementioned eligibility determinations, the administrative "
             "adjudication of supplemental nutritional assistance necessitates comprehensive "
             "documentation substantiating household compositional characteristics and corresponding "
             "income verification methodologies prior to authorization of any subsequent benefit "
             "disbursement to the applicant household.")
    res = check_readability(_result(_case(), text=dense))
    assert res is not None
    assert not res.passed          # flagged as too dense
    assert res.blocking is False   # ...but SOFT — never gates the run


def test_check_readability_skips_non_english_text():
    from heynyc.eval.checks import check_readability

    spanish = ("La determinación administrativa de elegibilidad requiere documentación detallada "
               "sobre la composición del hogar y la verificación de ingresos antes de autorizar "
               "beneficios adicionales para la familia solicitante. " * 3)
    assert check_readability(_result(_case(language="es"), text=spanish)) is None


async def test_readability_warning_does_not_fail_gate():
    # A dense but otherwise-valid answer: the readability check warns but must not block the gate.
    dense = ("Notwithstanding the aforementioned determinations, the administrative adjudication of "
             "supplemental assistance necessitates comprehensive documentation substantiating the "
             "compositional characteristics and corresponding income verification methodologies of "
             "the applicant household prior to any subsequent benefit disbursement henceforth.")
    agent = _FakeAgent(dense, [], {})

    async def no_links(url):
        return 200

    results = await run_all(lambda: agent, [_case(id="rd")])
    report = await evaluate(results, link_checker=no_links)
    flagged = {c.name for c in report.reports[0].checks if not c.passed}
    assert "readability" in flagged
    assert report.passed


# --- data grounding (Part C: row-addressed DATA citations) ----------------

def _data_citation(distance_mi, snapshot=None, hash_override=None):
    snap = snapshot if snapshot is not None else {":id": "row-9", "status": "Activated"}
    return {
        "url": "https://data.cityofnewyork.us/resource/h2bn-gu9k/row-9.json",
        "kind": "DATA", "snippet": "Marconi Park",
        "provenance": {
            "record_id": "row-9", "field_pointer": "/",
            "content_hash": hash_override or content_hash(snap), "snapshot": snap,
            "derivation": {"origin": [40.75, -73.87], "point": [40.74, -73.88], "distance_mi": distance_mi},
        },
    }


def _true_mi():
    from heynyc.core.tools.geo import haversine_m, miles
    return round(miles(haversine_m(40.75, -73.87, 40.74, -73.88)), 2)


def test_data_grounding_passes_when_distance_recomputes():
    res = check_data_grounding(_result(_case(), citations={"S1": _data_citation(_true_mi())}))
    assert res.passed and res.blocking


def test_data_grounding_fails_on_wrong_distance():
    cr = _result(_case(), citations={"S1": _data_citation(5.0)})  # fabricated distance
    assert not check_data_grounding(cr).passed


def test_data_grounding_fails_on_tampered_snapshot():
    cr = _result(_case(), citations={"S1": _data_citation(_true_mi(), hash_override="deadbeef")})
    assert not check_data_grounding(cr).passed


def test_data_grounding_ignores_non_data_citations():
    cr = _result(_case(), citations={"S1": {"url": "https://nyc.gov", "kind": "DOC", "snippet": "x"}})
    assert check_data_grounding(cr) is None
