"""The `about_heynyc` tool serves HeyNYC's OWN shipped docs (privacy notice + README FAQ) as
citable DOC sources, so self-description answers quote the running code's real files and can't
drift from deployed behavior."""
from __future__ import annotations

import subprocess

from heynyc.core.citations import CitationRegistry
from heynyc.core.registry import Registry
from heynyc.core.tools import build_toolbox
from heynyc.core.tools.about import _faq_section, about_tools
from heynyc.core.tools.base import ToolContext


def _ctx():
    return ToolContext(citations=CitationRegistry(), registry=Registry([]))


def _tool():
    return about_tools()[0]


async def test_serves_current_file_content_not_memory(tmp_path, monkeypatch):
    """The proof it can't drift: edit the shipped docs, the tool reflects the edit verbatim,
    because it reads them at call time rather than reciting the model's memory."""
    import heynyc.core.tools.about as about

    privacy = tmp_path / "PRIVACY.md"
    readme = tmp_path / "README.md"
    privacy.write_text("SENTINEL: we keep an encrypted transcript for 30 days.")
    readme.write_text("# HeyNYC\n\n## FAQ\n\nSENTINEL-FAQ: text NEW to start fresh.\n\n## Next\n")
    monkeypatch.setattr(about, "_PRIVACY", privacy)
    monkeypatch.setattr(about, "_README", readme)

    ctx = _ctx()
    out = await _tool().handler({}, ctx)

    assert "SENTINEL: we keep an encrypted transcript for 30 days." in out
    assert "SENTINEL-FAQ: text NEW to start fresh." in out
    assert "## Next" not in out                       # FAQ section only, stops at the next heading
    # Registered as DOC sources, like other document citations, and cited inline for grounding.
    kinds = {c["kind"] for c in ctx.citations.mapping().values()}
    assert kinds == {"DOC"}
    assert len(ctx.citations) == 2
    assert "{cite:S1}" in out and "{cite:S2}" in out


async def test_reads_the_real_shipped_privacy_and_faq(tmp_path):
    """Wiring check: with no monkeypatch it reads the actual repo files, so the answer is
    grounded in what really shipped."""
    out = await _tool().handler({}, _ctx())
    assert "encrypted conversation transcript" in out   # a real phrase from PRIVACY.md
    assert "REPORT" in out                               # a real phrase from the README FAQ


def test_faq_section_extracts_only_the_faq_block():
    md = "# Title\n\nintro\n\n## FAQ\n\nq and a here\n\n## Other\n\nnot this\n"
    assert _faq_section(md) == "q and a here"
    assert _faq_section("no faq heading here") == ""


def test_about_tool_is_registered_in_the_toolbox():
    tools = build_toolbox(Registry([]))
    assert "about_heynyc" in tools
    assert tools["about_heynyc"].read_only is True


# F167: a resident asking HeyNYC to file a form for her was told no, and the refusal cited
# `README.md#faq` and `PRIVACY.md`. Correct sourcing, but she sees an unclickable path
async def test_citations_are_immutable_links_to_shipped_revision():
    ctx = _ctx()
    await _tool().handler({}, ctx)

    revision = subprocess.check_output(
        ("git", "rev-parse", "HEAD"),
        text=True,
    ).strip().lower()
    base = f"https://github.com/shayantist/HeyNYC/blob/{revision}"
    urls = {citation["url"] for citation in ctx.citations.mapping().values()}
    assert urls == {f"{base}/PRIVACY.md", f"{base}/README.md#faq"}


async def test_about_tool_fails_closed_without_an_immutable_doc_locator(monkeypatch):
    import heynyc.core.tools.about as about

    monkeypatch.setattr(about, "_DOCS_BASE", "")
    ctx = _ctx()

    out = await _tool().handler({}, ctx)

    assert "could not be linked" in out
    assert len(ctx.citations) == 0
