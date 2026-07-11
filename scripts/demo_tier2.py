#!/usr/bin/env python
"""Watch the Tier-2 NLI faithfulness checker catch a prose fabrication Tier-1 is silent on.

    uv run python scripts/demo_tier2.py                                   # SAFE default: MockNLI, loads NO model
    uv run --extra nli python scripts/demo_tier2.py --backend minicheck   # real catch: local MiniCheck-Flan-T5
    uv run python scripts/demo_tier2.py --backend auto                    # try MiniCheck, then a local Ollama model (HEAVY)

Two fixtures, both reconstructed from verified real materials (see the design spec,
docs/superpowers/specs/2026-07-09-tier2-nli-checker-design.md):

  1. THE CATCH. A Spanish sentence that conflates a SUPPORTED clause ("restaurants must accept cash")
     with an INVENTED statute ("Ley Local 56 de 2021"), cited to the real DCWP cashless page (which
     supports the cash rule but never mentions any such law). Tier-1 passes it silently (the statute
     reads as a proper noun cited to a WEB excerpt, which is soft by design). Tier-2 flags it.
  2. THE NO-FALSE-POSITIVE. The clean answer names the law DESCRIPTIVELY ("the Cashless Ban Law") and
     cites the same page. Tier-2 passes it — it kills the lie without nuking a correct answer.

Backend selection is EXPLICIT and defaults to MockNLI (--backend mock) so a plain run never loads a
model. Pass --backend minicheck / ollama / auto for a real-model catch; those load a local model and
are heavy. Only MiniCheck / Ollama are real-model catches; a Mock run proves the MECHANISM only.
"""
from __future__ import annotations

import re

from heynyc.core.grounding import check_grounding
from heynyc.core.nli import MiniCheckNLI, MockNLI, PromptedNLI

# Real captured text of the DCWP cashless page, fetched 2026-07-11 from
# https://www.nyc.gov/site/dca/consumers/Prohibition-of-Cashless-Establishments.page — SUPPORTS
# "businesses must accept cash", contains NO "Local Law 56" and no "2021".
_DCWP_TEXT = (
    "Prohibition of Cashless Establishments. NYC businesses must accept cash unless they have a "
    "machine to convert cash to a prepaid card. They cannot charge more for paying in cash. You can "
    "file a complaint about a retail or food store, including a food cart, in New York City that "
    "refuses cash payments. Cash means U.S. currency and coins. Your store may refuse cash payments "
    "for telephone, mail, or internet-based transactions, unless the transaction takes place in the "
    "store. Read Guidance for Food Stores and Retail Establishments Regarding Telephone and "
    "Internet-based Transaction Exceptions to the Cashless Establishment Ban."
)
_CITATIONS = {
    "S3": {
        "url": "https://www.nyc.gov/site/dca/consumers/Prohibition-of-Cashless-Establishments.page",
        "kind": "WEB",  # a WEB excerpt: no snapshot, so Tier-1 treats it as incomplete and never blocks
        "title": "Prohibition of Cashless Establishments - DCWP",
        "snippet": _DCWP_TEXT,
    }
}

_FIXTURES = [
    ("THE CATCH        (fabricated statute)",
     "Bajo la Ley Local 56 de 2021, los restaurantes deben aceptar efectivo. {cite:S3}"),
    ("THE CLEAN ANSWER (law named descriptively)",
     "Under the Cashless Ban Law, food establishments must accept cash. {cite:S3}"),
]


def _statute_rule(claim: str, source: str) -> bool:
    """Mock-only fallback rule: a claim asserting a specific 'Ley Local NN' statute is unsupported
    unless that exact string is in the source. Used ONLY when no real model is available."""
    m = re.search(r"ley local \d+", claim.lower())
    return not (m and m.group(0) not in source.lower())


def _select_backend(mode: str):
    """Return (checker, label, is_real_model) for the requested mode. Default 'mock' loads NO model.
    'minicheck' / 'ollama' load a local model (heavy); 'auto' tries MiniCheck, then Ollama, then Mock."""
    if mode in ("minicheck", "auto"):
        try:
            nli = MiniCheckNLI(model_name="flan-t5-large")
            nli.check("The sky is blue.", "The sky is blue today.")  # force load now, surface any failure
            return nli, "MiniCheck-Flan-T5-Large (self-hosted, free; the production Tier-2 target)", True
        except Exception as exc:  # download / load / import failure
            print(f"[MiniCheck unavailable: {type(exc).__name__}: {exc}]")
            if mode == "minicheck":
                raise SystemExit("MiniCheck requested but unavailable; run `uv sync --extra nli` first.")
    if mode in ("ollama", "auto"):
        try:
            nli = PromptedNLI(model="ollama/qwen3.5:9b")
            nli.check("The sky is blue.", "The sky is blue today.")  # force a call, surface any failure
            return nli, "PromptedNLI via Ollama qwen3.5:9b (self-hosted, free; prototype stand-in)", True
        except Exception as exc:
            print(f"[Ollama unavailable: {type(exc).__name__}: {exc}]")
            if mode == "ollama":
                raise SystemExit("Ollama requested but unavailable.")
    return MockNLI(_statute_rule), "MockNLI (MECHANISM ONLY - real-model catch PENDING)", False


def _tier1_line(res) -> str:
    if res is None:
        return "PASS (silent): nothing structured to verify next to the citation"
    verdict = "BLOCKS" if res.blocking else "PASS (non-blocking)"
    note = res.detail or "no failures"
    return f"{verdict}: {note}"


def _tier2_line(res) -> str:
    if res is None:
        return "did not run (no cited sentence)"
    if not res.nli_failures:
        return f"PASS: {res.nli_checked} cited sentence(s) checked, all supported"
    parts = []
    for f in res.nli_failures:
        reason = f" - {f.reason}" if f.reason else ""
        parts.append(f"UNSUPPORTED (score {f.score:.2f}) [{'/'.join(f.cited)}]{reason}")
    return f"FLAGGED: {'; '.join(parts)}"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Tier-2 NLI demo")
    parser.add_argument("--backend", choices=["mock", "minicheck", "ollama", "auto"], default="mock",
                        help="mock (default, loads no model) | minicheck | ollama | auto (real backends load a local model)")
    args = parser.parse_args()
    print("=" * 88)
    print("Tier-2 NLI faithfulness checker - catching a prose fabrication Tier-1 is silent on")
    print("=" * 88)
    nli, label, is_real = _select_backend(args.backend)
    print(f"\nBackend: {label}")
    if not is_real:
        print("WARNING: no local model available; this run proves the MECHANISM, not a real-model catch.")
    print(f"Cited source S3: DCWP cashless page (WEB excerpt)\n  {_DCWP_TEXT[:96]}...\n")

    for label_fx, answer in _FIXTURES:
        t1 = check_grounding(answer, _CITATIONS)                 # Tier-1 only (today's live path)
        t2 = check_grounding(answer, _CITATIONS, nli=nli)        # Tier-1 + Tier-2
        print("-" * 88)
        print(label_fx)
        print(f"  answer : {answer}")
        print(f"  Tier-1 : {_tier1_line(t1)}")
        print(f"  Tier-2 : {_tier2_line(t2)}")
    print("-" * 88)


if __name__ == "__main__":
    main()
