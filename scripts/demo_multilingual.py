#!/usr/bin/env python
"""Watch the translate-at-edge pipeline freeze a citation and catch a mutated one.

    uv run python scripts/demo_multilingual.py                 # SAFE default: MockTranslator, loads NO model

This runs entirely on a deterministic MockTranslator: no language model, no API spend, no network. It
proves the MECHANISM (masking + the entity round-trip gate), not a real translation. The real backends
(LangDetectDetector for the router, PromptedTranslator via litellm/Ollama for the machine-translation
fallback) are lazy/injected and deliberately NOT exercised here.

Three fixtures, all on the same verified English answer about NYC's source-of-income protection
(Local Law 34 of 2020 / Administrative Code 20-840):

  1. THE FREEZE.  A faithful mock translation. Masking freezes "Local Law 34 of 2020" into a placeholder,
     the translator rewrites only the prose around it, and unmask restores the citation VERBATIM. The
     entity gate back-translates and every entity round-trips -> the Spanish answer ships.
  2. THE CATCH.   A mock translator that mutates the statute on the way back to English (Law 34 -> Law 56,
     the exact "Ley Local 56 de 2021" failure this pipeline exists to stop). The gate catches it and
     FALLS BACK to the English answer plus a pointer to the official page. The lie never ships.
  3. THE OFFICIAL LOOKUP. When the city has already published a human Spanish translation, the pipeline
     serves it OUTRIGHT and skips machine translation entirely.

See the design spec: docs/superpowers/specs/2026-07-05-multilingual-translate-at-edge-design.md.
"""
from __future__ import annotations

from heynyc.core.multilingual import (
    Glossary,
    MockLanguageDetector,
    MockTranslator,
    MultilingualPipeline,
    mask_entities,
    unmask,
)

# A verified English answer (the English path is where every model is most faithful). Real facts:
# Local Law 34 of 2020 / Administrative Code section 20-840 is NYC's source-of-income discrimination law.
_ANSWER_EN = (
    "Under Local Law 34 of 2020, a landlord cannot refuse your Section 8 voucher. {cite:S1} "
    "It is enforced under Administrative Code 20-840, with civil penalties up to $2,500. {cite:S2}"
)

# A Spanish-speaking user's query. The MockLanguageDetector routes anything with "ayuda" to Spanish.
_QUERY_ES = "Necesito ayuda: mi casero no acepta mi vale de vivienda"
_OFFICIAL_URL = "https://www.nyc.gov/site/cchr/law/source-of-income.page"

_RULE = "-" * 92


def _mutate_on_backtranslate(text: str, target_lang: str) -> str:
    """A translator that mutates the statute number on the way BACK to English -- the fluent-but-wrong
    citation a fluency score would wave through, and exactly what the entity gate is built to catch."""
    if target_lang == "en":
        return text.replace("Local Law 34 of 2020", "Local Law 56 of 2021")
    return f"[{target_lang}] {text}"


def _show_masking() -> None:
    masked, entities = mask_entities(_ANSWER_EN)
    print("MASKING (copy-don't-translate): the translator never sees the citations/numbers\n")
    print(f"  English answer : {_ANSWER_EN}")
    print(f"  Masked prose   : {masked}")
    print("  Frozen entities:")
    for i, e in enumerate(entities):
        print(f"    [[{i}]] = {e.kind:<10} {e.text!r}")
    restored = unmask(MockTranslator().translate(masked, "es"), entities)
    verbatim = "Local Law 34 of 2020" in restored
    print(f"\n  After translate + unmask, 'Local Law 34 of 2020' present verbatim: {verbatim}")
    print(f"  Restored       : {restored}")


def _show_pipeline() -> None:
    detector = MockLanguageDetector(lambda t: "es" if "ayuda" in t.lower() else "en")

    # 1. THE FREEZE -- faithful translation ships.
    faithful = MultilingualPipeline(MockTranslator(), detector=detector)
    r1 = faithful.process(_ANSWER_EN, query=_QUERY_ES)

    # 2. THE CATCH -- mutated citation is caught, falls back to English + pointer.
    guarded = MultilingualPipeline(
        MockTranslator(_mutate_on_backtranslate), detector=detector,
        fallback_pointer=f"For the official details, see {_OFFICIAL_URL}",
    )
    r2 = guarded.process(_ANSWER_EN, query=_QUERY_ES)

    # 3. THE OFFICIAL LOOKUP -- the city's own translation is served outright.
    official_es = (
        "Segun la Ley Local 34 de 2020, un propietario no puede rechazar su vale de la Seccion 8. "
        "{cite:S1} Se aplica bajo el Codigo Administrativo 20-840, con multas de hasta $2,500. {cite:S2}"
    )
    glossary = Glossary(official_lookup=lambda ans, lang: official_es if lang == "es" else None)
    with_official = MultilingualPipeline(MockTranslator(), detector=detector, glossary=glossary)
    r3 = with_official.process(_ANSWER_EN, query=_QUERY_ES)

    for title, r in [
        ("1. THE FREEZE (faithful translation ships)", r1),
        ("2. THE CATCH  (mutated citation -> fall back to English)", r2),
        ("3. THE OFFICIAL LOOKUP (serve the city's human translation)", r3),
    ]:
        print(_RULE)
        print(title)
        print(f"  route    : {r.route}")
        print(f"  faithful : {r.faithful}")
        print(f"  detail   : {r.detail}")
        print(f"  shipped  : {r.answer}")


def main() -> None:
    print("=" * 92)
    print("Translate-at-edge pipeline - freeze the citation, catch the mutation (MockTranslator, NO model)")
    print("=" * 92)
    _show_masking()
    print()
    _show_pipeline()
    print(_RULE)


if __name__ == "__main__":
    main()
