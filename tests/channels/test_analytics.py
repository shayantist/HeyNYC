import json
from dataclasses import dataclass, field

from heynyc.channels import analytics
from heynyc.core import telemetry


@dataclass
class FakeResult:
    text: str
    citations: dict = field(default_factory=dict)
    status: str = "success"
    tool_calls_made: list = field(default_factory=list)
    usage: dict = field(default_factory=lambda: {"input_tokens": 5, "output_tokens": 9, "latency_ms": 12.0})
    diagnostics: dict = field(default_factory=dict)


def test_record_turn_merges_extra(tmp_path):
    path = tmp_path / "t.jsonl"
    telemetry.record_turn(
        path, session_id="k", model="m", usage={"input_tokens": 1, "output_tokens": 2},
        n_tool_calls=0, tool_names=[], status="success", extra={"channel": "whatsapp_meta"},
    )
    rec = json.loads(path.read_text().splitlines()[0])
    assert rec["channel"] == "whatsapp_meta" and rec["session_id"] == "k"


def test_record_interaction_leaves_semantic_outcome_unclassified(tmp_path):
    path = tmp_path / "t.jsonl"
    res = FakeResult("Programs: SNAP, HEAP {cite:S1}.",
                     {"S1": {"url": "https://nyc.gov", "title": "T", "kind": "DATA"}})
    rec = analytics.record_interaction(
        telemetry_path=path, model="m", user_key="k", channel="whatsapp_meta", result=res)
    assert rec["channel"] == "whatsapp_meta"
    assert rec["outcome"] == "unclassified"
    assert rec["n_citations"] == 1


def test_record_interaction_counts_only_doc_citations_used_in_the_answer(tmp_path):
    path = tmp_path / "t.jsonl"
    res = FakeResult(
        "Use the current guide {cite:S1}.",
        {
            "S1": {"url": "https://nyc.gov/one", "title": "One", "kind": "DOC"},
            "S2": {"url": "https://nyc.gov/two", "title": "Two", "kind": "DOC"},
        },
        tool_calls_made=["index_search", "web_search"],
    )

    rec = analytics.record_interaction(
        telemetry_path=path, model="m", user_key="k", channel="whatsapp_meta", result=res
    )

    assert rec["used_doc_citations"] == 1


def test_record_interaction_keeps_pii_free_failure_diagnostics(tmp_path):
    path = tmp_path / "t.jsonl"
    res = FakeResult(
        "I couldn't verify an answer.",
        status="error",
        usage={
            "retry_kinds": ["output"],
            "stalled_model_requests": 0,
        },
        diagnostics={
            "failure_type": "UnexpectedModelBehavior",
            "validation_rejections": [
                {"stage": "structured_grounding", "private_detail": "do not persist"},
            ],
        },
    )

    rec = analytics.record_interaction(
        telemetry_path=path,
        model="m",
        user_key="k",
        channel="whatsapp_twilio",
        result=res,
    )

    assert rec["failure_type"] == "UnexpectedModelBehavior"
    assert rec["validation_rejection_stages"] == ["structured_grounding"]
    assert rec["validation_rejection_count"] == 1
    assert rec["retry_kinds"] == ["output"]
    assert "private_detail" not in json.dumps(rec)


def test_feedback_log_appends(tmp_path):
    path = tmp_path / "fb.jsonl"
    analytics.feedback_log(path, {"user_key": "k", "flag": "wrong"})
    analytics.feedback_log(path, {"user_key": "k", "flag": "report"})
    records = analytics.load_feedback(path)
    assert len(records) == 2 and records[0]["flag"] == "wrong"
