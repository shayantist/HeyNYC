"""Unit tests for the multilingual translate-at-edge pipeline (heynyc.core.multilingual).

Every test here is fully offline: MockLanguageDetector and MockTranslator drive the router and the
translation/back-translation deterministically, and PromptedTranslator is exercised against an
INJECTED fake completion. No test loads a real model, imports langdetect, or touches the network.
The real LangDetectDetector / PromptedTranslator backends are exercised only by the demo script.

Mirrors tests/test_nli.py: a tiny Protocol + a Mock backend for tests + a lazy/injected real backend.
"""
from __future__ import annotations

from types import SimpleNamespace

from heynyc.core.multilingual import (
    Glossary,
    LangDetectDetector,
    MockLanguageDetector,
    MockTranslator,
    MultilingualPipeline,
    PromptedTranslator,
    Term,
    detect_language,
    entity_roundtrip_gate,
    mask_entities,
    needs_translation,
    unmask,
)

# A verified English answer with the exact classes of entity the copy-don't-translate rule freezes:
# a Local Law citation, an Administrative Code section, a dollar amount, a year, and a {cite:Sn} marker.
# Real facts: Local Law 34 of 2020 / Admin Code section 20-840 is NYC's source-of-income protection.
_ANSWER_EN = (
    "Under Local Law 34 of 2020, landlords cannot refuse your Section 8 voucher. {cite:S1} "
    "This is enforced under Administrative Code 20-840, with penalties up to $2,500. {cite:S2}"
)


# --- 1. Language router: detect + switch -----------------------------------------------------------


def test_detect_language_uses_injected_detector():
    det = MockLanguageDetector({"hola, necesito ayuda con mi vivienda": "es"})
    assert detect_language("Hola, necesito ayuda con mi vivienda", det) == "es"


def test_detect_language_callable_rule():
    det = MockLanguageDetector(lambda text: "zh" if "你" in text else "en")
    assert detect_language("你好", det) == "zh"
    assert detect_language("hello", det) == "en"


def test_needs_translation_is_a_switch_not_a_committee():
    assert needs_translation("es") is True
    assert needs_translation("zh") is True
    assert needs_translation("en") is False   # English -> passthrough
    assert needs_translation("und") is False  # undetermined -> safe passthrough (answer in English)
    assert needs_translation("") is False


def test_langdetect_backend_is_lazy_and_imports_nothing_on_construction():
    # Constructing the real backend must NOT import langdetect (mirrors MiniCheckNLI). If langdetect
    # is not installed, .detect() would raise ImportError -- but construction alone stays clean.
    det = LangDetectDetector()
    assert det.backend == "langdetect"


# --- 2. Copy-don't-translate masking ---------------------------------------------------------------


def test_mask_freezes_law_section_money_year_and_cite_markers():
    masked, entities = mask_entities(_ANSWER_EN)
    kinds = {e.kind for e in entities}
    # every load-bearing entity class is frozen
    assert "law" in kinds
    assert "cite" in kinds
    assert "money" in kinds
    frozen = {e.text for e in entities}
    assert "Local Law 34 of 2020" in frozen
    assert "{cite:S1}" in frozen
    assert "$2,500" in frozen
    # the masked prose no longer contains the raw citations/numbers -- the translator never sees them
    assert "Local Law 34" not in masked
    assert "{cite:S1}" not in masked
    assert "$2,500" not in masked
    assert "20-840" not in masked


def test_year_inside_a_law_citation_is_not_double_masked():
    # "Local Law 34 of 2020" must be ONE frozen span, not a law span plus a separate "2020" date span.
    _masked, entities = mask_entities("See Local Law 34 of 2020 for details.")
    law = [e for e in entities if e.kind == "law"]
    assert len(law) == 1
    assert law[0].text == "Local Law 34 of 2020"
    # no stray date entity carved out of the middle of the law citation
    assert not any(e.kind == "date" and e.text == "2020" for e in entities)


def test_mask_unmask_roundtrip_preserves_a_law_number_verbatim_through_translation():
    """The structural fix: mask -> translate (placeholders preserved) -> unmask restores the citation
    EXACTLY. 'Local Law 34' can never become 'Ley Local 56' because the translator never authors it."""
    masked, entities = mask_entities(_ANSWER_EN)
    translated = MockTranslator().translate(masked, "es")  # default mock preserves [[N]] placeholders
    restored = unmask(translated, entities)
    assert "Local Law 34 of 2020" in restored
    assert "Administrative Code" in restored  # the glossary term still reads correctly around the freeze
    assert "20-840" in restored
    assert "{cite:S1}" in restored and "{cite:S2}" in restored
    assert "$2,500" in restored


def test_unmask_leaves_an_unknown_placeholder_untouched():
    # Fail-safe: a placeholder with no matching entity is left as-is (the gate will then catch the loss).
    assert unmask("prefix [[9]] suffix", []) == "prefix [[9]] suffix"


# --- 3. Entity round-trip faithfulness gate --------------------------------------------------------


def test_gate_passes_a_faithful_translation():
    _masked, entities = mask_entities(_ANSWER_EN)
    # A faithful target answer: the default mock back-translates it keeping every entity verbatim.
    target = "[es] " + _ANSWER_EN
    gate = entity_roundtrip_gate(target, entities, MockTranslator())
    assert gate.ok is True
    assert gate.missing == []


def test_gate_fails_a_mutated_citation():
    _masked, entities = mask_entities(_ANSWER_EN)

    def mutate_on_backtranslate(text, target_lang):
        # Simulate a translator that mutates the statute on the way back to English: Law 34 -> Law 56,
        # exactly the "Ley Local 56" failure the pipeline exists to catch.
        if target_lang == "en":
            return text.replace("Local Law 34 of 2020", "Local Law 56 of 2021")
        return f"[{target_lang}] {text}"

    target = "[es] " + _ANSWER_EN
    gate = entity_roundtrip_gate(target, entities, MockTranslator(mutate_on_backtranslate))
    assert gate.ok is False
    assert any(e.text == "Local Law 34 of 2020" for e in gate.missing)


# --- 4. Full pipeline: passthrough / official / translate / fallback -------------------------------


def _es_detector():
    return MockLanguageDetector(lambda text: "es" if "ayuda" in text.lower() else "en")


def test_pipeline_english_query_passes_through_untranslated():
    pipe = MultilingualPipeline(MockTranslator(), detector=_es_detector())
    res = pipe.process(_ANSWER_EN, query="What law protects my voucher?")
    assert res.route == "passthrough"
    assert res.answer == _ANSWER_EN
    assert res.faithful is True


def test_pipeline_translates_and_preserves_the_citation_verbatim():
    pipe = MultilingualPipeline(MockTranslator(), detector=_es_detector())
    res = pipe.process(_ANSWER_EN, query="necesito ayuda con mi voucher")
    assert res.route == "translated"
    assert res.lang == "es"
    assert res.faithful is True
    # the frozen citation and section survive verbatim in the shipped Spanish answer
    assert "Local Law 34 of 2020" in res.answer
    assert "20-840" in res.answer
    assert "{cite:S1}" in res.answer


def test_pipeline_falls_back_to_english_when_the_gate_fails():
    def mutate_on_backtranslate(text, target_lang):
        if target_lang == "en":
            return text.replace("Local Law 34 of 2020", "Local Law 56 of 2021")
        return f"[{target_lang}] {text}"

    pipe = MultilingualPipeline(
        MockTranslator(mutate_on_backtranslate), detector=_es_detector(),
        fallback_pointer="See the official page: https://www.nyc.gov/site/hra/",
    )
    res = pipe.process(_ANSWER_EN, query="necesito ayuda con mi voucher")
    assert res.route == "fallback"
    assert res.faithful is False
    # never ship the unfaithful translation: fall back to the English answer + a pointer
    assert _ANSWER_EN in res.answer
    assert "nyc.gov" in res.answer


def test_pipeline_prefers_official_translation_when_one_exists():
    official_es = "Segun la Ley Local 34 de 2020, su casero no puede rechazar su vale. {cite:S1}"

    def official_lookup(answer_en, lang):
        return official_es if lang == "es" else None

    glossary = Glossary(official_lookup=official_lookup)
    pipe = MultilingualPipeline(MockTranslator(), detector=_es_detector(), glossary=glossary)
    res = pipe.process(_ANSWER_EN, query="necesito ayuda con mi voucher")
    assert res.route == "official"
    assert res.answer == official_es
    assert res.faithful is True


def test_pipeline_accepts_an_explicit_target_lang_without_a_detector():
    pipe = MultilingualPipeline(MockTranslator())
    res = pipe.process(_ANSWER_EN, target_lang="es")
    assert res.route == "translated"
    assert "Local Law 34 of 2020" in res.answer


# --- 5. Glossary / termbase seam -------------------------------------------------------------------


def test_glossary_separates_frozen_proper_nouns_from_translated_terms():
    g = Glossary(terms=[
        Term("Housing Connect", freeze=True),
        Term("Administrative Code", {"es": "Codigo Administrativo"}),
    ])
    assert g.freeze_terms() == ["Housing Connect"]
    assert g.renderings_for("es") == {"Administrative Code": "Codigo Administrativo"}
    assert g.renderings_for("zh") == {}  # no Chinese rendering pinned -> empty, not a crash


def test_glossary_freeze_term_is_masked_verbatim():
    g = Glossary(terms=[Term("Housing Connect", freeze=True)])
    masked, entities = mask_entities("Apply on Housing Connect today.", glossary=g)
    assert "Housing Connect" not in masked
    assert any(e.kind == "term" and e.text == "Housing Connect" for e in entities)


# --- 6. PromptedTranslator: prompt build + parse, against an injected fake completion ---------------


def _fake_completion(content: str, capture: dict):
    def fn(model, messages):
        capture["model"] = model
        capture["messages"] = messages
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    return fn


def test_prompted_translator_builds_prompt_and_returns_translation():
    capture: dict = {}
    fake = _fake_completion("[[0]] es la ley aplicable.", capture)
    tr = PromptedTranslator(model="ollama/qwen3.5:9b", completion_fn=fake)
    out = tr.translate("[[0]] is the applicable law.", "es", glossary={"Administrative Code": "Codigo Administrativo"})

    assert out == "[[0]] es la ley aplicable."
    system = "\n".join(m["content"] for m in capture["messages"] if m["role"] == "system")
    user = "\n".join(m["content"] for m in capture["messages"] if m["role"] == "user")
    # the prompt must (a) name the target language, (b) instruct placeholder preservation,
    # (c) pass the glossary rendering through.
    assert "Spanish" in system or "es" in system
    assert "[[" in system  # placeholder-preservation rule mentions the token shape
    assert "Codigo Administrativo" in system
    assert "[[0]] is the applicable law." in user


def test_prompted_translator_strips_whitespace_from_the_completion():
    tr = PromptedTranslator(completion_fn=_fake_completion("  traducido  \n", {}))
    assert tr.translate("translated", "es") == "traducido"


# --- MockTranslator construction modes -------------------------------------------------------------


def test_mock_translator_default_tags_language_and_preserves_placeholders():
    out = MockTranslator().translate("prefix [[0]] suffix", "es")
    assert "[[0]]" in out
    assert out.startswith("[es]")


def test_mock_translator_custom_transform():
    out = MockTranslator(lambda text, lang: text.upper()).translate("hola", "en")
    assert out == "HOLA"
