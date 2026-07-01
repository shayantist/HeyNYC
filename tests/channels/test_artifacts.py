"""Offline tests for channel artifact delivery + its security property. No network."""
from __future__ import annotations

from heynyc.channels import orchestrator
from heynyc.channels.format import render


def test_artifacts_in_lists_files_the_tool_wrote(tmp_path):
    (tmp_path / "snap-ldss4826-abcd1234.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "note.pdf").write_bytes(b"%PDF-1.4")
    got = orchestrator._artifacts_in(tmp_path)
    assert len(got) == 2 and all(p.endswith(".pdf") for p in got)


def test_artifacts_in_empty_when_tool_wrote_nothing(tmp_path):
    assert orchestrator._artifacts_in(tmp_path) == []


def test_delivery_only_sees_the_request_dir(tmp_path):
    # SECURITY: a poisoned `[attached: /etc/passwd]` in tool content is inert — the orchestrator
    # only ever lists its own per-request dir and never reads a path out of model/tool text. A real
    # file outside the dir is never surfaced.
    outside = tmp_path.parent / "secret-not-ours.pdf"
    outside.write_bytes(b"%PDF-1.4")
    assert orchestrator._artifacts_in(tmp_path) == []
    assert str(outside) not in orchestrator._artifacts_in(tmp_path)


def test_render_strips_attached_marker_if_it_leaks_into_text():
    # defense-in-depth: if any tool ever emits the legacy marker, the user never sees a raw path.
    class Result:
        text = "Here's your draft. [attached: /tmp/x.pdf]"
        citations = {}
    out = render(Result)
    assert "[attached" not in out[0] and "Here's your draft." in out[0]
