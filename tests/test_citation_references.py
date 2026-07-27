from types import SimpleNamespace

import pytest
from pydantic_ai import ModelRetry

from heynyc.core.citations import CitationRegistry
from heynyc.core.pydantic_runtime.runtime import PydanticRuntimeAdapter
from heynyc.core.tools.base import ToolContext
from heynyc.eval.cases import EvalCase
from heynyc.eval.checks import run_checks
from heynyc.eval.runner import CaseResult


async def test_pydantic_plain_output_rejects_unknown_citation_id():
    runtime = object.__new__(PydanticRuntimeAdapter)
    runtime._semantic_verifier = None
    context = SimpleNamespace(
        deps=ToolContext(citations=CitationRegistry(), registry=None)
    )

    with pytest.raises(ModelRetry, match="citation IDs returned by tools"):
        await runtime._validate_grounding(
            context,
            "I resolved the landmark approximately. {cite:S4}",
        )


async def test_pydantic_plain_output_accepts_registered_citation_id():
    citations = CitationRegistry()
    citation_id = citations.register(
        "https://www.nyc.gov/example",
        title="NYC example",
        snippet="This option is listed.",
        kind="DOC",
    )
    runtime = object.__new__(PydanticRuntimeAdapter)
    runtime._semantic_verifier = None
    context = SimpleNamespace(
        deps=ToolContext(citations=citations, registry=None)
    )

    output = f"This option is listed. {{cite:{citation_id}}}"

    assert await runtime._validate_grounding(context, output) == output


async def test_pydantic_plain_output_rejects_discovery_citation_id():
    citations = CitationRegistry()
    citation_id = citations.register(
        "https://www.nyc.gov/example",
        title="Search result",
        snippet="A truncated discovery snippet",
        kind="WEB",
        provenance={"evidence_grade": "discovery"},
    )
    runtime = object.__new__(PydanticRuntimeAdapter)
    runtime._semantic_verifier = None
    context = SimpleNamespace(
        deps=ToolContext(citations=citations, registry=None)
    )

    with pytest.raises(ModelRetry, match="authoritative source"):
        await runtime._validate_grounding(
            context,
            f"An unsupported completion of the snippet. {{cite:{citation_id}}}",
        )


async def test_eval_rejects_unknown_citation_id():
    result = CaseResult(
        case=EvalCase(id="unknown-citation", module="test", query="Where is it?"),
        text="I resolved the landmark approximately. {cite:S4}",
        citations={"S1": {"kind": "DATA"}},
    )

    checks = await run_checks(result, link_checker=lambda _: None)
    check = next(
        (candidate for candidate in checks if candidate.name == "citation_references"),
        None,
    )

    assert check is not None
    assert check.passed is False
    assert check.blocking is True
    assert check.detail == "unknown citation ids: S4"


async def test_eval_rejects_discovery_citation_id():
    result = CaseResult(
        case=EvalCase(id="discovery-citation", module="test", query="Is that confirmed?"),
        text="The search snippet confirms it. {cite:S1}",
        citations={
            "S1": {
                "kind": "WEB",
                "provenance": {"evidence_grade": "discovery"},
            },
        },
    )

    checks = await run_checks(result, link_checker=lambda _: None)
    check = next(
        candidate for candidate in checks
        if candidate.name == "citation_references"
    )

    assert check.passed is False
    assert check.blocking is True
    assert check.detail == "discovery-only citation ids: S1"


async def test_eval_rejects_malformed_citation_id():
    result = CaseResult(
        case=EvalCase(id="malformed-citation", module="test", query="Where is it?"),
        text="I resolved the landmark approximately. {cite:FOO}",
        citations={"S1": {"kind": "DATA"}},
    )

    checks = await run_checks(result, link_checker=lambda _: None)
    check = next(
        candidate for candidate in checks if candidate.name == "citation_references"
    )

    assert check.passed is False
    assert check.detail == "unknown citation ids: FOO"


async def test_eval_accepts_registered_citation_id():
    result = CaseResult(
        case=EvalCase(id="known-citation", module="test", query="Where is it?"),
        text="This option is listed. {cite:S1}",
        citations={"S1": {"kind": "DOC"}},
    )

    checks = await run_checks(result, link_checker=lambda _: None)
    check = next(
        candidate for candidate in checks if candidate.name == "citation_references"
    )

    assert check.passed is True
