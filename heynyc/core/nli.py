"""Tier-2 faithfulness / NLI checker, per-sentence textual entailment against a cited chunk.

Tier-1 (core.grounding) catches the STRUCTURED facts it can parse (phones, dollars, addresses,
quotes) and is deliberately silent on fabricated PROSE cited to a soft (WEB/DOC) source, a made-up
statute like "Ley Local 56 de 2021" reads as a proper noun, and proper-noun mismatches are always
soft. Tier-2 closes that gap: given a claim sentence and the text of the chunk it cites, it decides
"does the source SUPPORT this sentence?" This is closed-book entailment, so it needs no world
knowledge and no current-events cutoff (see the design spec, 2026-07-09-tier2-nli-checker-design.md).

One tiny interface (NLIChecker), three backends, matching the codebase's inject-the-model pattern so
tests stay offline and the real backend is swappable:
  • MockNLI     , deterministic, no network, no model; the backend every unit test uses.
  • MiniCheckNLI, lazy-loads MiniCheck-Flan-T5-Large (~770M) via the `minicheck` package. ALL heavy
                   imports (torch / transformers / minicheck) live INSIDE the class, so importing this
                   module drags in nothing and the base install stays light (like the [whatsapp] extra).
  • PromptedNLI , reuses the litellm seam with a tight, JSON-only entailment prompt; pointable at any
                   litellm model, including a self-hosted ollama/<model>. A general LLM is a WEAKER
                   faithfulness judge than a dedicated checker, so this is a prototype stand-in, not the
                   production Tier-2.

This module is off by default: nothing wires it into the live agent loop yet (that is the follow-on).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Protocol, Union

_WS_RE = re.compile(r"\s+")
# The default support threshold: below this, the caller records an NLI failure. Conservative /
# high-recall-of-unsupported per the spec; the real calibration on a labeled set is a follow-on.
DEFAULT_SUPPORT_THRESHOLD = 0.5


@dataclass
class NLIVerdict:
    supported: bool     # is the claim entailed by the source? (backend-native boolean)
    score: float        # 0..1 support probability (backend-native; the grounding hook thresholds this)
    backend: str        # "mock" | "minicheck" | "prompted", for provenance / telemetry
    reason: str = ""    # optional short note (PromptedNLI fills this; mock / minicheck may not)


class NLIChecker(Protocol):
    def check(self, claim: str, source: str) -> NLIVerdict: ...


def _norm_claim(claim: str) -> str:
    """The mapping key space for MockNLI: lowercase, whitespace collapsed."""
    return _WS_RE.sub(" ", str(claim).strip().lower())


# A MockNLI rule is either a callable (claim, source) -> bool|float, or a mapping from a normalized
# claim to a bool|float.
MockRule = Union[Callable[[str, str], Union[bool, float]], Mapping[str, Union[bool, float]]]


class MockNLI:
    """Deterministic checker for tests: no network, no model. Drive the Tier-2 branch exactly by
    passing either a callable ``(claim, source) -> bool|float`` or a mapping keyed by the NORMALIZED
    claim (lowercased, whitespace-collapsed). A bool becomes score 1.0 / 0.0; a float is used as-is and
    thresholded to set ``supported``."""

    backend = "mock"

    def __init__(self, rule: MockRule, *, threshold: float = DEFAULT_SUPPORT_THRESHOLD, default: bool = True):
        self._rule = rule
        self._threshold = threshold
        self._default = default

    def check(self, claim: str, source: str) -> NLIVerdict:
        if callable(self._rule):
            out: Union[bool, float] = self._rule(claim, source)
        else:
            out = self._rule.get(_norm_claim(claim), self._default)
        if isinstance(out, bool):
            score = 1.0 if out else 0.0
            supported = out
        else:
            score = float(out)
            supported = score >= self._threshold
        return NLIVerdict(supported=supported, score=score, backend=self.backend)


class MiniCheckNLI:
    """The headline backend: a purpose-built EMNLP-2024 faithfulness checker, self-hosted, zero data
    egress. Lazy-loads MiniCheck-Flan-T5-Large (~770M) on first ``check()``; swap ``model_name`` to
    ``'Bespoke-MiniCheck-7B'`` for the stronger / heavier model. Heavy imports live inside the methods
    so importing this module never pulls torch / transformers / minicheck."""

    backend = "minicheck"

    def __init__(self, model_name: str = "flan-t5-large", cache_dir: Optional[str] = None):
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._scorer = None  # built on first check()

    def _ensure_scorer(self):
        if self._scorer is None:
            # Heavy: pulls torch / transformers / minicheck. Import INSIDE the method so module import
            # stays light and the offline suite never touches the model. See the [nli] extra.
            from minicheck.minicheck import MiniCheck

            self._scorer = MiniCheck(model_name=self._model_name, cache_dir=self._cache_dir)
        return self._scorer

    def check(self, claim: str, source: str) -> NLIVerdict:
        scorer = self._ensure_scorer()
        # MiniCheck.score(docs, claims) -> (pred_labels, max_support_probs, used_chunks, per_chunk_probs)
        # per the package's own docstring; we score one (doc, claim) pair. (Verified against the
        # installed minicheck source, 2026-07-11.)
        pred_labels, support_probs, _chunks, _per_chunk = scorer.score(docs=[source], claims=[claim])
        score = float(support_probs[0])
        supported = bool(pred_labels[0])
        return NLIVerdict(supported=supported, score=score, backend=self.backend)


# The system instruction: JSON-only, and explicitly "judge SUPPORT, not truth" so a stale / small
# model cannot flag a correct-but-unstated claim as a fabrication for the wrong reason.
_PROMPT = (
    "You are a strict faithfulness checker for a civic assistant. Decide ONLY whether the SOURCE text "
    "supports the CLAIM. Judge SUPPORT, not real-world truth: if the SOURCE does not state or clearly "
    "entail the claim, it is unsupported, even if the claim happens to be true in the world. Pay "
    "attention to specific details (a named law or statute number, a dollar amount, a date): if the "
    "claim asserts a detail the SOURCE never states, it is NOT supported. Reply with ONLY a JSON "
    'object and nothing else: {"supported": true or false, "reason": "<one short sentence>"}.'
)

# A single JSON object anywhere in the response body (some models wrap it in prose despite the ask).
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)

# Injected completion fn: (model, messages) -> an OpenAI-shaped response with .choices[0].message.content.
CompletionFn = Callable[[str, list], object]


def _parse_verdict(content: str) -> tuple[bool, str]:
    """Parse the model's JSON verdict. FAIL SAFE: an unparseable / malformed response is read as
    UNSUPPORTED (not a false all-clear), so a garbled judge over-flags rather than waves a lie through."""
    m = _JSON_OBJ_RE.search(content or "")
    if m is None:
        return False, f"unparseable NLI response: {(content or '')[:120]}"
    try:
        data = json.loads(m.group(0))
    except (ValueError, TypeError):
        return False, f"unparseable NLI response: {(content or '')[:120]}"
    if not isinstance(data, dict) or "supported" not in data:
        return False, f"NLI response missing 'supported': {(content or '')[:120]}"
    return bool(data.get("supported")), str(data.get("reason", ""))


class PromptedNLI:
    """The always-runnable fallback: a general LLM behind a tight, JSON-only entailment prompt via the
    existing litellm seam. Point ``model`` at ``ollama/<model>`` for a self-hosted, free check. A
    general model is a weaker faithfulness judge than MiniCheck, so this is a prototype stand-in. Inject
    ``completion_fn`` in tests to run with no network."""

    backend = "prompted"

    def __init__(
        self,
        model: str = "ollama/qwen3.5:9b",
        *,
        completion_fn: Optional[CompletionFn] = None,
        api_base: Optional[str] = None,
    ):
        self._model = model
        self._api_base = api_base
        self._completion_fn = completion_fn

    def _build_messages(self, claim: str, source: str) -> list[dict]:
        user = f"SOURCE:\n{source}\n\nCLAIM:\n{claim}"
        return [{"role": "system", "content": _PROMPT}, {"role": "user", "content": user}]

    def _complete(self, messages: list[dict]):
        if self._completion_fn is not None:
            return self._completion_fn(self._model, messages)
        from litellm import completion  # lazy: keep litellm import out of module load

        kwargs: dict = {"model": self._model, "messages": messages, "temperature": 0.0}
        if self._api_base:
            kwargs["api_base"] = self._api_base
        if "ollama" in self._model:
            kwargs["format"] = "json"  # litellm's ollama JSON mode
        return completion(**kwargs)

    def check(self, claim: str, source: str) -> NLIVerdict:
        resp = self._complete(self._build_messages(claim, source))
        content = resp.choices[0].message.content or ""
        supported, reason = _parse_verdict(content)
        return NLIVerdict(
            supported=supported, score=1.0 if supported else 0.0, backend=self.backend, reason=reason
        )
