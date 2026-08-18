"""Offline tests for the workers module's `get_worker_rights_guidance` tool.

Static-but-official facts (NY Labor Law section 196-d + the NYS DOL tips FAQ) are baked in and
returned WITH a DOC citation - no network. Mirrors the housing guidance tests: a grounding tool
call, a DOC citation whose snippet is backed by the returned fact, typed topic validation, and
that the shipped module loads with its tool + eval cases.
"""
from __future__ import annotations

import re

from heynyc.core import config
from heynyc.core.citations import CitationRegistry
from heynyc.core.registry import Registry
from heynyc.core.tools.base import ToolContext
from heynyc.modules.workers.tools import _GUIDANCE, get_tools

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tool():
    return next(t for t in get_tools() if t.name == "get_worker_rights_guidance")


async def _run(topic: str):
    citations = CitationRegistry()
    ctx = ToolContext(citations=citations, registry=Registry([]))
    out = await _tool().handler({"topic": topic}, ctx)
    return out, citations


async def test_tips_grounds_section_196d_body_and_cites():
    out, citations = await _run("tips")
    # the section 196-d body language is present verbatim, not paraphrased
    assert "New York Labor Law section 196-d" in out
    assert ("no employer or their agent may demand, accept, or keep any part of a tip or gratuity"
            in out)
    assert "file a wage claim with the New York State Department of Labor" in out
    # three DOC citations: the statute (nysenate), enforcement FAQ (dol.ny.gov), and AG guidance
    mapping = citations.mapping()
    assert len(mapping) == 3
    assert {c["kind"] for c in mapping.values()} == {"DOC"}
    urls = {c["url"] for c in mapping.values()}
    assert "https://www.nysenate.gov/legislation/laws/LAB/196-D" in urls
    assert any("dol.ny.gov" in u for u in urls)
    assert any("ag.ny.gov" in u for u in urls)
    assert "{cite:S1}" in out and "{cite:S2}" in out and "{cite:S3}" in out


async def test_tips_does_not_interpret_free_text_tool_arguments():
    out, citations = await _run("my boss is keeping our tips")

    assert "section 196-d" not in out
    assert len(citations) == 0


async def test_tips_grounds_immigrant_worker_protections_and_cites():
    out, citations = await _run("tips")
    assert "no matter your immigration status" in out
    assert "does not report workers or witnesses to immigration authorities" in out
    mapping = citations.mapping()
    assert any(c["kind"] == "DOC" and "ag.ny.gov" in c["url"] for c in mapping.values())


async def test_unknown_topic_abstains_without_citation():
    out, citations = await _run("overtime pay dispute")
    assert len(citations) == 0             # nothing grounded → nothing cited
    assert "{cite:" not in out
    assert "dol.ny.gov" in out or "311" in out   # routes the user onward


def test_each_fact_snippet_is_a_token_subset_of_its_body():
    """The eval faithfulness contract: every _Fact.snippet token appears in its body's tokens, so a
    DOC citation can never outrun the fact it cites."""
    for _intro, facts in _GUIDANCE.values():
        for fact in facts:
            body_tokens = set(_TOKEN_RE.findall(fact.body.lower()))
            snippet_tokens = [t for t in _TOKEN_RE.findall(fact.snippet.lower()) if len(t) > 1]
            missing = [t for t in snippet_tokens if t not in body_tokens]
            assert not missing, f"snippet tokens not backed by body: {missing}"


def test_workers_module_loads_with_tool_and_eval():
    registry = Registry.discover(config.MODULES_DIR)
    module = next((m for m in registry.modules if m.name == "workers"), None)
    assert module is not None
    assert module.category == "workers"
    tool_names = {t.name for t in registry.load_module_tools()}
    assert "get_worker_rights_guidance" in tool_names

    from heynyc.eval.cases import load_cases
    cases = [c for c in load_cases(registry) if c.module == "workers"]
    assert cases, "workers should ship eval cases"
    by_id = {case.id: case for case in cases}
    assert by_id["workers_tip_theft"].expect_tools == ["get_worker_rights_guidance"]
    assert by_id["workers_f154_withheld_wages_immigration_threat"].expect_tools == [
        "web_fetch"
    ]


def test_worker_guidance_schema_exposes_only_the_supported_topic():
    topic = _tool().parameters["properties"]["topic"]

    assert topic["enum"] == ["tips"]
    assert "free text" not in topic["description"].lower()
