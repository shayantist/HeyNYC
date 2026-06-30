from dataclasses import dataclass, field
from heynyc.channels.format import render, WA_LIMIT


@dataclass
class FakeResult:
    text: str
    citations: dict = field(default_factory=dict)


def test_strips_markers_and_appends_sources():
    r = FakeResult(
        "Cooling centers are open {cite:S1}. Bring ID {cite:S2}.",
        {"S1": {"url": "https://nyc.gov/cool", "title": "Cooling"},
         "S2": {"url": "https://nyc.gov/id", "title": ""}},
    )
    out = render(r)
    assert len(out) == 1
    body = out[0]
    assert "{cite:" not in body
    assert "Cooling centers are open. Bring ID." in body
    assert "Sources:" in body
    assert "https://nyc.gov/cool" in body and "https://nyc.gov/id" in body


def test_no_citations_no_footer():
    out = render(FakeResult("Just a plain reply."))
    assert out == ["Just a plain reply."]


def test_splits_long_text_on_paragraph_boundaries_footer_last():
    para = "x" * 3000
    r = FakeResult(f"{para}\n\n{para}", {"S1": {"url": "https://nyc.gov", "title": "T"}})
    out = render(r)
    assert len(out) >= 2
    assert all(len(c) <= WA_LIMIT for c in out)
    assert "Sources:" in out[-1] and "Sources:" not in out[0]
