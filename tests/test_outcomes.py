from __future__ import annotations

from pathlib import Path

from heynyc.core import outcomes


class _Result:
    """Minimal AgentResult stand-in: only the fields the milestone helpers read."""

    def __init__(self, tool_calls_made, citations):
        self.tool_calls_made = tool_calls_made
        self.citations = citations


def _screen_result(program_count: int) -> _Result:
    # Mirror the screen handler: a verdict citation always, plus one DATA citation per
    # eligible program whose snippet carries `program_code` (the signal that programs
    # were shown). PII-free (program names only).
    cites = {"S1": {"kind": "DATA", "title": "NYC Benefits Screening (ACCESS NYC)",
                    "snippet": f"likely-eligible estimate ({program_count} program(s))"}}
    for i in range(program_count):
        cites[f"S{i + 2}"] = {"kind": "DATA", "title": f"Program {i}",
                              "snippet": f"Program {i}, likely eligible (program_code P{i})"}
    return _Result([outcomes.SCREEN_TOOL], cites)


def test_screener_eligible_true_when_programs_shown():
    assert outcomes.screener_eligible(_screen_result(3)) is True


def test_screener_eligible_false_when_no_programs():
    assert outcomes.screener_eligible(_screen_result(0)) is False


def test_screener_eligible_false_when_screen_tool_not_fired():
    # Citations present but the screen tool never fired that turn -> not a screening.
    r = _Result([], _screen_result(2).citations)
    assert outcomes.screener_eligible(r) is False


def test_milestones_from_result():
    m = outcomes.milestones_from_result(_screen_result(2), produced_artifact=True)
    assert m == {"eligible_shown": True, "form_ready": True}
    m2 = outcomes.milestones_from_result(_Result([], {}), produced_artifact=False)
    assert m2 == {"eligible_shown": False, "form_ready": False}


def test_record_milestone_skips_when_no_outcome(tmp_path: Path):
    path = tmp_path / "outcomes.jsonl"
    assert outcomes.record_milestone(
        path, user_key="u1", eligible_shown=False, form_ready=False) is None
    assert not path.exists()


def test_record_milestone_is_pii_free(tmp_path: Path):
    path = tmp_path / "outcomes.jsonl"
    rec = outcomes.record_milestone(
        path, user_key="deadbeefcafef00d", eligible_shown=True, form_ready=False)
    # only the salted key + two booleans + a timestamp; nothing that could carry a
    # phone number, name, or form answer.
    assert set(rec) == {"ts", "user_key", "eligible_shown", "form_ready"}
    assert rec["user_key"] == "deadbeefcafef00d"
    assert outcomes.load(path) == [rec]


def test_funnel_counts_and_dropoff():
    telemetry_records = [
        {"tool_names": ["benefits_search"]},           # a plain answer
        {"tool_names": ["screen_access_nyc_eligibility"]},        # screened, no eligible
        {"tool_names": ["screen_access_nyc_eligibility"]},        # screened, eligible shown
        {"tool_names": ["prepare_snap_application"]},  # apply started (review)
        {"tool_names": ["prepare_snap_application"]},  # apply started -> form ready
    ]
    outcome_records = [
        {"eligible_shown": True, "form_ready": False},
        {"eligible_shown": False, "form_ready": True},
    ]
    f = outcomes.funnel(telemetry_records, outcome_records)
    c = f["counts"]
    assert c["turns"] == 5
    assert c["screened"] == 2
    assert c["eligible_shown"] == 1
    assert c["apply_started"] == 2
    assert c["form_ready"] == 1
    assert f["dropoff"]["screened"] == {"from": "turns", "lost": 3, "rate": 3 / 5}
    assert f["dropoff"]["form_ready"]["lost"] == 1  # apply_started 2 -> form_ready 1


def test_funnel_empty_no_zero_division():
    f = outcomes.funnel([], [])
    assert f["counts"] == {"turns": 0, "screened": 0, "eligible_shown": 0,
                           "apply_started": 0, "form_ready": 0}
    assert f["dropoff"]["screened"]["rate"] == 0.0


def test_funnel_roundtrips_through_the_sidecar(tmp_path: Path):
    opath = outcomes.default_path(tmp_path)
    outcomes.record_milestone(opath, user_key="u1", eligible_shown=True, form_ready=True)
    telem = [{"tool_names": ["screen_access_nyc_eligibility", "prepare_snap_application"]}]
    f = outcomes.funnel(telem, outcomes.load(opath))
    assert f["counts"]["eligible_shown"] == 1 and f["counts"]["form_ready"] == 1


def test_outcomes_report_renders(tmp_path: Path, capsys):
    from heynyc.__main__ import _render_outcomes
    from heynyc.core import telemetry

    tpath = telemetry.default_path(tmp_path)
    telemetry.record_turn(tpath, session_id="u1", model="m",
                          usage={"input_tokens": 1, "output_tokens": 1},
                          n_tool_calls=1, tool_names=["screen_access_nyc_eligibility"], status="success")
    outcomes.record_milestone(outcomes.default_path(tmp_path),
                              user_key="u1", eligible_shown=True, form_ready=False)
    _render_outcomes(tpath, outcomes.default_path(tmp_path))
    out = capsys.readouterr().out
    assert "funnel" in out.lower()
