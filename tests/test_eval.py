from __future__ import annotations

from heynyc.core import config
from heynyc.core.citations import content_hash
from heynyc.core.registry import Registry
from heynyc.eval.cases import EvalCase, load_cases
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
    assert "cross_module_family_events_and_cooling_preserve_evidence" in ids
    assert "convo_spanish_snap_screen_and_food_tonight" in ids
    # every module's abstain cases are present
    assert any(c.abstain for c in cases)


def test_eval_run_metadata_persists_repeat_outcomes():
    from types import SimpleNamespace

    from heynyc.__main__ import _eval_run_metadata

    result = SimpleNamespace(
        case=SimpleNamespace(id="repeat-me"),
        usage={},
    )
    repeat_summary = {
        "k": 3,
        "eligible_case_count": 1,
        "reliable_case_count": 1,
        "cases": [{"case_id": "repeat-me", "passed": [True, True, True], "reliable": True}],
    }

    metadata = _eval_run_metadata("model", [result], repeat_summary=repeat_summary)

    assert metadata["repeat"] == repeat_summary


def test_explicit_eval_cases_are_repeat_targets_even_when_not_safety_critical():
    from heynyc.__main__ import _repeat_eval_cases

    ordinary = _case(id="ordinary")
    safety = _case(id="safety", harm_category="misinformation")

    assert _repeat_eval_cases([ordinary, safety], ["ordinary"]) == [ordinary]
    assert _repeat_eval_cases([ordinary, safety], []) == [safety]


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


async def test_link_liveness_strips_sentence_punctuation_after_markdown_url():
    seen = []
    cr = _result(
        _case(),
        text=(
            "সরকারি পেজ: [ACCESS NYC SNAP]"
            "(https://access.nyc.gov/programs/supplemental-nutrition-assistance-program-snap/)।"
        ),
    )

    async def checker(url):
        seen.append(url)
        return 200 if url.endswith("supplemental-nutrition-assistance-program-snap/") else 404

    result = await check_link_liveness(cr, checker=checker)

    assert result is not None and result.passed
    assert seen == [
        "https://access.nyc.gov/programs/supplemental-nutrition-assistance-program-snap/"
    ]


async def test_link_liveness_keeps_unicode_closing_character_in_iri_path():
    seen = []
    url = "https://example.gov/কেন্দ্র】"
    cr = _result(_case(), text=f"Official page: {url}")

    async def checker(candidate):
        seen.append(candidate)
        return 200 if candidate == url else 404

    result = await check_link_liveness(cr, checker=checker)

    assert result is not None and result.passed
    assert seen == [url]


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
         "test_type": "MFT", "invariants": {"must_ground": True, "must_not_fabricate": True},
         "utility_criterion": "Return a useful grounded place"},
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
    assert cases["demo_inv"].utility_criterion == "Return a useful grounded place"


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


async def test_pydantic_eval_agent_confirms_fact_reviews_but_not_actions():
    from heynyc.core.agent import AgentResult
    from heynyc.eval.runner import PydanticEvalAgent

    class Conversation:
        def __init__(self, tool_name, fact_confirmation_names=()):
            self.runtime = self
            self.tool_name = tool_name
            self.fact_confirmation_names = set(fact_confirmation_names)
            self.pending_approvals = {}
            self.resumed = False

        def is_fact_confirmation(self, tool_name):
            return tool_name in self.fact_confirmation_names

        async def send(self, message, **kwargs):
            self.pending_approvals = {
                "call-1": {"tool_name": self.tool_name, "args": {"value": 1}}
            }
            return AgentResult(
                text="",
                citations={},
                status="approval_required",
                tool_calls_made=[self.tool_name],
                usage={"input_tokens": 2, "cost_usd": 0.1},
            )

        def dump_state(self):
            return self

        def conversation_from_state(self, state):
            return state

        async def resume_approvals(self, approvals):
            self.resumed = True
            return AgentResult(
                text="done",
                citations={},
                status="success",
                tool_calls_made=["screen_eligibility"],
                usage={"input_tokens": 3, "cost_usd": 0.2},
            )

    class Runtime:
        def __init__(self, tool_name, fact_confirmation_names=()):
            self._conversation = Conversation(tool_name, fact_confirmation_names)

        def conversation(self):
            return self._conversation

    facts_runtime = Runtime(
        "confirm_screen_eligibility_facts",
        {"confirm_screen_eligibility_facts"},
    )
    facts = await PydanticEvalAgent(facts_runtime).run("screen me")
    assert facts.text == "done"
    assert facts_runtime._conversation.resumed
    assert facts.usage["input_tokens"] == 5

    action_runtime = Runtime("submit_application")
    action = await PydanticEvalAgent(action_runtime).run("submit it")
    assert action.status == "approval_required"
    assert not action_runtime._conversation.resumed

    disguised_action_runtime = Runtime("confirm_submit_facts")
    disguised_action = await PydanticEvalAgent(disguised_action_runtime).run("submit it")
    assert disguised_action.status == "approval_required"
    assert not disguised_action_runtime._conversation.resumed


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


async def test_eval_repeat_reuses_initial_run_and_writes_every_trace(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    import heynyc.__main__ as cli
    import heynyc.eval as eval_pkg
    import heynyc.eval.bench as bench

    case = _case(id="repeat_saved")
    initial = _result(case, text="initial")
    initial.usage = {
        "cost_usd": 0.01,
        "input_tokens": 10,
        "output_tokens": 1,
        "latency_ms": 100,
        "n_model_calls": 2,
        "n_tool_calls": 1,
    }
    extras = [_result(case, text=f"extra-{index}") for index in range(2)]
    for result in extras:
        result.usage = {
            "cost_usd": 0.01,
            "input_tokens": 10,
            "output_tokens": 1,
            "latency_ms": 100,
            "n_model_calls": 2,
            "n_tool_calls": 1,
        }

    async def fake_run_all(factory, cases, reminders=None):
        return [initial]

    async def fake_run_repeated(factory, target, k, reminders=None):
        assert target is case
        assert k == 2
        return extras

    monkeypatch.setattr(
        cli.Registry,
        "discover",
        lambda *args, **kwargs: SimpleNamespace(modules={"m": object()}),
    )
    monkeypatch.setattr(cli, "_load_retriever", lambda required=False: None)
    monkeypatch.setattr(eval_pkg, "load_cases", lambda registry: [case])
    monkeypatch.setattr(eval_pkg, "run_all", fake_run_all)
    monkeypatch.setattr(eval_pkg, "run_repeated", fake_run_repeated)
    monkeypatch.setattr(bench, "build_eval_agent", lambda *args, **kwargs: object())
    writes = []
    real_write_run = eval_pkg.write_run

    def tracked_write_run(directory, *args, **kwargs):
        writes.append(directory)
        return real_write_run(directory, *args, **kwargs)

    monkeypatch.setattr(eval_pkg, "write_run", tracked_write_run)

    import pytest

    with pytest.raises(SystemExit, match="0"):
        await cli._cmd_eval(
            use_api_judge=False,
            repeat=3,
            out=str(tmp_path),
            case_ids=[case.id],
        )

    repeat_root = tmp_path / "repeats" / case.id
    saved = [
        json.loads((repeat_root / f"run-{index:02d}" / "traces" / f"{case.id}.json").read_text())
        for index in range(1, 4)
    ]
    assert [item["final_text"] for item in saved] == ["initial", "extra-0", "extra-1"]
    metadata = json.loads((tmp_path / "report.json").read_text())["metadata"]
    assert metadata["input_tokens"] == 30
    assert metadata["output_tokens"] == 3
    assert metadata["candidate_cost_usd"] == 0.03
    assert metadata["latency_ms"] == 300
    assert metadata["n_model_calls"] == 6
    assert metadata["n_tool_calls"] == 3
    assert writes[-1] == tmp_path


async def test_eval_repeat_failure_blocks_the_run_and_report(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    import heynyc.__main__ as cli
    import heynyc.eval as eval_pkg
    import heynyc.eval.bench as bench

    case = _case(id="repeat_failure", expect_tools=["needed"])
    initial = _result(case, text="initial", tools=["needed"])
    failed_repeat = _result(case, text="missed tool")

    async def fake_run_all(factory, cases, reminders=None):
        return [initial]

    async def fake_run_repeated(factory, target, k, reminders=None):
        return [failed_repeat]

    monkeypatch.setattr(
        cli.Registry,
        "discover",
        lambda *args, **kwargs: SimpleNamespace(modules={"m": object()}),
    )
    monkeypatch.setattr(cli, "_load_retriever", lambda required=False: None)
    monkeypatch.setattr(eval_pkg, "load_cases", lambda registry: [case])
    monkeypatch.setattr(eval_pkg, "run_all", fake_run_all)
    monkeypatch.setattr(eval_pkg, "run_repeated", fake_run_repeated)
    monkeypatch.setattr(bench, "build_eval_agent", lambda *args, **kwargs: object())

    import pytest

    with pytest.raises(SystemExit) as exc:
        await cli._cmd_eval(
            use_api_judge=False,
            repeat=2,
            out=str(tmp_path),
            case_ids=[case.id],
        )

    assert exc.value.code == 1
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["passed"] is False
    assert report["metadata"]["repeat"]["reliable_case_count"] == 0
    assert "(FAIL)" in (tmp_path / "report.txt").read_text()


def test_repeat_count_must_be_positive():
    import argparse

    import pytest

    from heynyc.__main__ import _positive_int

    assert _positive_int("3") == 3
    with pytest.raises(argparse.ArgumentTypeError):
        _positive_int("0")
    with pytest.raises(argparse.ArgumentTypeError):
        _positive_int("-1")


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


async def test_utility_case_requires_qualitative_review_before_promotion(tmp_path):
    from heynyc.eval.report import write_run

    case = _case(id="utility", utility_criterion="Give a useful next step")
    report = await evaluate([_result(case, text="A mechanically valid answer")])

    assert report.passed
    assert report.mechanical_passed_count == 1
    assert report.qualitative_pending_count == 1
    assert not report.promotion_ready
    assert "MECHANICAL PASS, QUALITATIVE REVIEW REQUIRED" in report.render()

    write_run(tmp_path, report)
    import json

    data = json.loads((tmp_path / "report.json").read_text())
    assert data["mechanical_passed"] is True
    assert data["qualitative_review_required"] is True
    assert data["qualitative_pending_count"] == 1
    assert data["promotion_ready"] is False
    assert data["cases"][0]["promotion_ready"] is False


async def test_utility_case_promotes_only_after_passing_qualitative_review():
    case = _case(id="utility", utility_criterion="Give a useful next step")

    async def judge(cr):
        from heynyc.eval.checks import CheckResult

        return CheckResult("resident_outcome", passed=True, detail="grounded and useful")

    report = await evaluate(
        [_result(case, text="A useful grounded answer")],
        judge=judge,
    )

    assert report.qualitative_pending_count == 0
    assert report.promotion_ready
    assert "(PASS)" in report.render()


async def test_failed_qualitative_review_does_not_erase_mechanical_pass():
    case = _case(id="utility", utility_criterion="Give a useful next step")

    async def judge(cr):
        from heynyc.eval.checks import CheckResult

        return CheckResult("resident_outcome", passed=False, detail="not useful")

    report = await evaluate([_result(case, text="A valid but unhelpful answer")], judge=judge)

    assert report.mechanical_passed_count == 1
    assert report.passed_count == 0
    assert not report.promotion_ready
    assert "(FAIL)" in report.render()


def test_mixed_failed_and_pending_reviews_render_as_fail():
    from heynyc.eval.checks import CheckResult
    from heynyc.eval.report import CaseReport, GateReport

    review = CheckResult("redteam_safe", passed=False, detail="unsafe")
    failed = CaseReport(
        "failed",
        "redteam",
        [CheckResult("mechanical", passed=True), review],
        qualitative_review_required=True,
        qualitative_review=review,
    )
    pending = CaseReport(
        "pending",
        "redteam",
        [CheckResult("mechanical", passed=True)],
        qualitative_review_required=True,
    )

    rendered = GateReport([failed, pending]).render()

    assert "(FAIL)" in rendered
    assert "MECHANICAL PASS, QUALITATIVE REVIEW REQUIRED" not in rendered


async def test_resident_outcome_named_check_is_mechanical_without_utility_criterion():
    from heynyc.eval.checks import CheckResult
    from heynyc.eval.report import CaseReport

    report = CaseReport(
        case_id="ordinary",
        module="m",
        checks=[CheckResult("resident_outcome", passed=False)],
    )

    assert not report.mechanical_passed
    assert not report.promotion_ready


async def test_nonblocking_judge_cannot_satisfy_required_qualitative_review():
    from heynyc.eval.checks import CheckResult

    case = _case(id="utility", utility_criterion="Give a useful next step")

    async def judge(cr):
        return CheckResult(
            "resident_outcome",
            passed=True,
            detail="advisory only",
            blocking=False,
        )

    report = await evaluate([_result(case, text="A valid answer")], judge=judge)

    assert not report.promotion_ready


async def test_case_without_utility_criterion_keeps_existing_pass_behavior():
    report = await evaluate([_result(_case(id="ordinary"), text="A valid answer")])

    assert report.passed
    assert report.qualitative_pending_count == 0
    assert report.promotion_ready
    assert "(PASS)" in report.render()


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


def test_load_cases_parses_tags_and_global_file(tmp_path):
    import yaml as _yaml

    from heynyc.eval.cases import load_cases

    mod = tmp_path / "demo"
    mod.mkdir()
    (mod / "manifest.yaml").write_text(_yaml.safe_dump({
        "name": "demo", "category": "general", "description": "d", "eval": "eval.yaml",
    }))
    (mod / "eval.yaml").write_text(_yaml.safe_dump([
        {"id": "demo_tagged", "query": "where?", "tags": ["F046", "retrieval-identity"]},
    ]))
    global_file = tmp_path / "global.yaml"
    global_file.write_text(_yaml.safe_dump([
        {"id": "global_cross_module", "query": "events and food together?", "tags": ["F051"]},
    ]))
    registry = Registry.discover(tmp_path)

    cases = {c.id: c for c in load_cases(registry, global_path=global_file)}

    assert cases["demo_tagged"].tags == ["F046", "retrieval-identity"]
    assert cases["global_cross_module"].module == "global"
    # Absent global file stays harmless.
    assert "global_cross_module" not in {
        c.id for c in load_cases(registry, global_path=tmp_path / "missing.yaml")
    }


def test_select_cases_by_tag_id_and_deterministic_sample():
    from heynyc.eval.cases import select_cases

    cases = [
        EvalCase(id=f"case_{i}", module="m", query="q", tags=(["F046"] if i % 2 else []))
        for i in range(10)
    ]

    tagged = select_cases(cases, tags=["F046"])
    assert [c.id for c in tagged] == [f"case_{i}" for i in range(10) if i % 2]

    picked = select_cases(cases, case_ids=["case_3", "case_4"])
    assert [c.id for c in picked] == ["case_3", "case_4"]

    import pytest as _pytest
    with _pytest.raises(SystemExit):
        select_cases(cases, case_ids=["nope"])

    sampled_a = select_cases(cases, sample=3, seed=7)
    sampled_b = select_cases(cases, sample=3, seed=7)
    sampled_c = select_cases(cases, sample=3, seed=8)
    assert [c.id for c in sampled_a] == [c.id for c in sampled_b]
    assert len(sampled_a) == 3
    assert [c.id for c in sampled_a] != [c.id for c in sampled_c]


async def test_bare_eval_run_requires_all_flag(capsys, monkeypatch):
    """Cost guard: a bare `heynyc eval` must not silently launch the full paid gate."""
    import heynyc.eval as eval_pkg
    from heynyc.__main__ import _cmd_eval

    def boom(*args, **kwargs):
        raise AssertionError("run must not start without --all")

    monkeypatch.setattr(eval_pkg, "run_all", boom)

    await _cmd_eval(use_api_judge=False)

    out = capsys.readouterr().out
    assert "--all" in out
    assert "220" in out or "case" in out.lower()


def test_multi_turn_case_schema_derives_final_query(tmp_path):
    import yaml as _yaml

    from heynyc.eval.cases import load_cases

    mod = tmp_path / "demo"
    mod.mkdir()
    (mod / "manifest.yaml").write_text(_yaml.safe_dump({
        "name": "demo", "category": "general", "description": "d", "eval": "eval.yaml",
    }))
    (mod / "eval.yaml").write_text(_yaml.safe_dump([
        {"id": "convo_case", "turns": ["first question", "and a follow-up?"]},
        {"id": "single_case", "query": "one shot"},
    ]))
    registry = Registry.discover(tmp_path)

    cases = {c.id: c for c in load_cases(registry)}

    assert cases["convo_case"].turns == ["first question", "and a follow-up?"]
    assert cases["convo_case"].query == "and a follow-up?"  # checks apply to the final turn
    assert cases["single_case"].turns == ["one shot"]  # single-turn stays the degenerate case


async def test_runner_plays_multi_turn_cases_through_a_conversation():
    """The follow-up turn must see the first turn as real history (the F052 contract),
    exercised through the same Conversation wrapper the surfaces use."""
    from heynyc.core.agent import Agent
    from heynyc.core.registry import Registry as _Registry
    from heynyc.eval.runner import run_case

    seen_histories = []

    async def complete(messages, tool_schemas):
        seen_histories.append([str(m.get("content") or "") for m in messages])
        return {"role": "assistant", "content": f"answer {len(seen_histories)}", "tool_calls": None}

    agent = Agent(_Registry([]), tools={}, complete_fn=complete)
    case = EvalCase(
        id="convo", module="global", query="and the follow-up?",
        turns=["first question", "and the follow-up?"],
    )

    result = await run_case(agent, case)

    assert result.text == "answer 2"
    final_call = seen_histories[-1]
    assert any("first question" in content for content in final_call)
    assert any("answer 1" in content for content in final_call)  # prior assistant turn visible
