"""Optional LLM-as-judge for groundedness / citation-supports-claim.

Deterministic checks catch most issues; this adds the harder judgment: does every
factual claim in the answer actually trace to a cited source (or, for abstain cases,
did it correctly decline)? Costs an LLM call per case, so it's opt-in. The completion
function is injectable for testing.
"""
from __future__ import annotations

import json
from typing import Awaitable, Callable, Optional

from .checks import CheckResult
from .runner import CaseResult

JudgeComplete = Callable[[str], Awaitable[str]]

_RUBRIC = """\
You are a strict grader for a NYC civic assistant whose #1 rule is to never make up facts.

User question:
{query}

Assistant answer:
{answer}

Cited sources (id → url → snippet):
{citations}

This case {abstain_clause}.

Grade with JSON only: {{"grounded": true|false, "reason": "<one sentence>"}}.
- If the answer states specific facts (locations, distances, hours, dates, eligibility, prices),
  each must be supported by a cited source. Unsupported specifics → grounded=false.
- If this case should abstain, grounded=true only if the assistant declined / said it wasn't
  sure / pointed to an official source instead of asserting unverified facts.
"""


DEFAULT_JUDGE_MODEL = "openai/gpt-4o-mini"  # engine default; the app injects via make_llm_judge()


async def _litellm_complete(prompt: str, model: str) -> str:
    import litellm

    resp = await litellm.acompletion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return resp.choices[0].message.content or ""


def _format_citations(citations: dict) -> str:
    if not citations:
        return "(none)"
    return "\n".join(f"{cid} → {c.get('url','')} → {c.get('snippet','')[:160]}" for cid, c in citations.items())


def make_llm_judge(model: str = DEFAULT_JUDGE_MODEL):
    """Build a judge callable bound to `model` (the app injects its configured judge model)."""
    async def _judge(cr: CaseResult, complete_fn: Optional[JudgeComplete] = None) -> CheckResult:
        return await _run_judge(cr, complete_fn, model)
    return _judge


async def llm_judge(cr: CaseResult, complete_fn: Optional[JudgeComplete] = None) -> CheckResult:
    """Back-compat default judge (uses DEFAULT_JUDGE_MODEL)."""
    return await _run_judge(cr, complete_fn, DEFAULT_JUDGE_MODEL)


async def _run_judge(cr: CaseResult, complete_fn: Optional[JudgeComplete], model: str) -> CheckResult:
    if cr.error:
        return CheckResult("llm_grounded", passed=False, detail=f"agent error: {cr.error}")
    complete = complete_fn or (lambda p: _litellm_complete(p, model))
    prompt = _RUBRIC.format(
        query=cr.case.query,
        answer=cr.text or "(empty)",
        citations=_format_citations(cr.citations),
        abstain_clause="SHOULD abstain (decline / say it doesn't know)" if cr.case.abstain
        else "should answer if it can ground the answer",
    )
    raw = await complete(prompt)
    try:
        start, end = raw.find("{"), raw.rfind("}")
        verdict = json.loads(raw[start : end + 1])
    except Exception:
        return CheckResult("llm_grounded", passed=False, detail=f"unparseable judge output: {raw[:120]}")
    return CheckResult(
        "llm_grounded",
        passed=bool(verdict.get("grounded")),
        detail=verdict.get("reason", ""),
    )
