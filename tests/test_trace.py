import json
from pathlib import Path

from heynyc.eval.cases import EvalCase
from heynyc.eval.runner import CaseResult
from heynyc.eval.trace import build_trace, classify_outcome


def _cr(**kw):
    base = dict(case=EvalCase(id="c", module="m", query="where is X?"),
                text="", tool_calls_made=[], citations={}, messages=[])
    base.update(kw)
    return CaseResult(**base)


def test_build_trace_extracts_tool_and_retriever_spans():
    cr = _cr(
        text="It's at 120 Broadway {cite:S1}.",
        citations={"S1": {"id": "S1", "url": "https://a.gov", "title": "", "snippet": "120 Broadway", "kind": "DATA"}},
        messages=[
            {"role": "user", "content": "where is X?"},
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": "nearest", "arguments": '{"address":"X"}'}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "120 Broadway, 0.2mi"},
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "c2", "type": "function",
                             "function": {"name": "index_search", "arguments": '{"q":"cooling"}'}}]},
            {"role": "tool", "tool_call_id": "c2", "content": "doc text"},
            {"role": "assistant", "content": "It's at 120 Broadway {cite:S1}.", "tool_calls": None},
        ],
    )
    trace = build_trace(cr)
    kinds = [(s.kind, s.name) for s in trace.spans if s.kind != "llm"]
    assert ("tool", "nearest") in kinds
    assert ("retriever", "index_search") in kinds
    nearest = next(s for s in trace.spans if s.name == "nearest")
    assert nearest.input == {"address": "X"}
    assert nearest.output == "120 Broadway, 0.2mi"
    assert trace.outcome == "answered"


def test_serialized_spans_use_openinference_attributes():
    cr = _cr(
        text="120 Broadway",
        messages=[
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": "nearest", "arguments": '{"address":"X"}'}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "120 Broadway"},
            {"role": "assistant", "content": "120 Broadway", "tool_calls": None},
        ],
    )
    spans = build_trace(cr).to_dict()["spans"]
    by_kind = {s["attributes"]["openinference.span.kind"]: s["attributes"] for s in spans}
    assert "TOOL" in by_kind and "LLM" in by_kind
    tool = by_kind["TOOL"]
    assert tool["tool.name"] == "nearest"
    assert tool["tool_call.function.name"] == "nearest"
    assert json.loads(tool["tool_call.function.arguments"]) == {"address": "X"}
    assert tool["output.value"] == "120 Broadway"
    llm = by_kind["LLM"]
    assert llm["llm.output_messages.0.message.role"] == "assistant"
    assert llm["llm.output_messages.0.message.content"] == "120 Broadway"


def test_retriever_span_uses_document_attributes():
    cr = _cr(
        text="ok",
        messages=[
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "r1", "type": "function",
                             "function": {"name": "index_search", "arguments": '{"q":"x"}'}}]},
            {"role": "tool", "tool_call_id": "r1", "content": "retrieved passage"},
            {"role": "assistant", "content": "ok", "tool_calls": None},
        ],
    )
    spans = build_trace(cr).to_dict()["spans"]
    retr = next(s["attributes"] for s in spans
                if s["attributes"]["openinference.span.kind"] == "RETRIEVER")
    assert retr["retrieval.documents.0.document.content"] == "retrieved passage"


def test_classify_outcome():
    assert classify_outcome("", "error") == "error"
    assert classify_outcome("I couldn't find that on official sources.", "success") == "abstained"
    assert classify_outcome("I help with NYC services and events.", "success") == "redirected"
    assert classify_outcome("The nearest center is 120 Broadway.", "success") == "answered"


def test_trace_write_roundtrip(tmp_path: Path):
    cr = _cr(text="hi", messages=[{"role": "assistant", "content": "hi", "tool_calls": None}])
    trace = build_trace(cr)
    path = trace.write(tmp_path)
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["case_id"] == "c"
    assert data["outcome"] in {"answered", "abstained", "redirected", "error"}
    assert isinstance(data["spans"], list)


def test_trace_preserves_expected_response_language():
    cr = _cr(
        case=EvalCase(id="es", module="m", query="¿Dónde?", language="es"),
        text="Aquí.",
    )

    trace = build_trace(cr)

    assert trace.language == "es"
    assert trace.to_dict()["language"] == "es"
