"""Multilingual answers: reason in English, translate at the EDGE with a copy-don't-translate rule.

The failure this fixes is narrow and repeatable: models understand a Spanish question fine but MUTATE
a specific citation when they GENERATE the grounded answer in Spanish -- qwen3.5 turned "Local Law 34"
into "Ley Local 56 de 2021", Sonnet invented "Local Law 68", both while their English answers stayed
clean (see the design spec, 2026-07-05-multilingual-translate-at-edge-design.md). The fix keeps all
grounding + verification in English and translates only the finished answer, so the translator never
gets to author a law number.

Structure mirrors core.nli (a tiny Protocol + a Mock backend for tests + a lazy/injected real backend,
off by default): nothing wires this into the live agent loop yet (that is the follow-on, gated on
config.HEYNYC_MULTILINGUAL). The pieces, each small and testable:

  1. LANGUAGE ROUTER   -- detect the user's language, then a SWITCH: English passes through, non-English
                          takes the pivot path. `LanguageDetector` Protocol + `MockLanguageDetector`
                          (tests) + `LangDetectDetector` (lazy `langdetect`, pure-Python, no model).
  2. TRANSLATOR        -- `Translator` Protocol + `MockTranslator` (tests) + `PromptedTranslator`
                          (litellm seam, lazy/injected, NEVER run this session).
  3. COPY-DON'T-TRANSLATE MASKING -- freeze law/section citations, dollar amounts, dates, {cite:Sn}
                          markers, and pinned proper nouns into [[N]] placeholders; translate the prose
                          around them; unmask verbatim. This is the structural fix.
  4. GLOSSARY / TERMBASE seam -- one canonical rendering per civic-legal term, and an official-
                          translation lookup preferred over machine translation when the city has
                          already published the human translation.
  5. ENTITY ROUND-TRIP GATE -- back-translate the target answer to English and assert every frozen
                          entity round-trips EXACTLY; on any mismatch, FALL BACK to the English answer
                          plus a pointer to the official page. The translation-side analog of the
                          Phase 1 grounding guard: grounded or it abstains, across the language boundary.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Protocol, Sequence, Union

_WS_RE = re.compile(r"\s+")


# ==================================================================================================
# 1. Language router
# ==================================================================================================

# ISO 639-1 codes ("en", "es", "zh", ...). "und" == undetermined (too short / no features).
class LanguageDetector(Protocol):
    def detect(self, text: str) -> str: ...


def _norm_text(text: str) -> str:
    """The mapping key space for MockLanguageDetector: lowercase, whitespace collapsed."""
    return _WS_RE.sub(" ", str(text).strip().lower())


# A MockLanguageDetector rule is a callable (text) -> code, a mapping normalized-text -> code, or a
# fixed code string (always return that language).
MockLangRule = Union[Callable[[str], str], Mapping[str, str], str]


class MockLanguageDetector:
    """Deterministic detector for tests: no network, no model. Pass a callable ``(text) -> code``, a
    mapping keyed by the NORMALIZED text, or a fixed code string. Falls back to ``default`` for an
    unmapped input."""

    backend = "mock"

    def __init__(self, rule: MockLangRule, *, default: str = "en"):
        self._rule = rule
        self._default = default

    def detect(self, text: str) -> str:
        if isinstance(self._rule, str):
            return self._rule
        if callable(self._rule):
            return self._rule(text)
        return self._rule.get(_norm_text(text), self._default)


class LangDetectDetector:
    """Real backend: the pure-Python `langdetect` library (a port of Google's language-detection). Its
    n-gram language profiles ship INSIDE the wheel, so it downloads no model and needs no network --
    verified against the langdetect docs (Context7 /mimino666/langdetect, 2026-07-11). The import is
    lazy (inside ``detect``) so importing THIS module drags in nothing, exactly like MiniCheckNLI.

    langdetect is non-deterministic on short/ambiguous text unless a seed is pinned, so we set
    ``DetectorFactory.seed`` once. An undetectable input (empty / punctuation-only) maps to "und", which
    the router treats as a safe English passthrough rather than a risky guess."""

    backend = "langdetect"

    def __init__(self, *, seed: Optional[int] = 0):
        self._seed = seed
        self._seeded = False

    def detect(self, text: str) -> str:
        # Heavy-ish import kept out of module load (mirrors the [nli] / [whatsapp] extras pattern).
        from langdetect import DetectorFactory, LangDetectException
        from langdetect import detect as _detect

        if not self._seeded and self._seed is not None:
            DetectorFactory.seed = self._seed
            self._seeded = True
        try:
            return _detect(text)
        except LangDetectException:
            return "und"


def detect_language(text: str, detector: LanguageDetector) -> str:
    """The router's first half: ask the injected detector for the text's language code."""
    return detector.detect(text)


def needs_translation(lang: str) -> bool:
    """The router's second half: a SWITCH, not a committee. English (and undetermined) pass through on
    the normal English path; everything else takes the pivot/translate path."""
    return bool(lang) and lang not in ("en", "und")


# ==================================================================================================
# 3. Copy-don't-translate masking
# ==================================================================================================

@dataclass(frozen=True)
class Entity:
    """A span frozen verbatim through translation. ``kind`` is for telemetry / the gate's messages;
    ``text`` is the exact surface string restored on unmask and checked by the round-trip gate."""
    kind: str   # law | section | admin_code | money | date | cite | term
    text: str


# Placeholders the translator must copy through unchanged. Double brackets + a bare index: distinctive
# (never a real word), ASCII (survives any MT engine), and trivially round-tripped by regex.
_PLACEHOLDER = "[[{}]]"
_PLACEHOLDER_RE = re.compile(r"\[\[(\d+)\]\]")

# --- entity patterns, most-specific first (priority only breaks exact-start ties; see the scanner) ---
_CITE_RE = re.compile(r"\{cite:S\d+\}")
# "Local Law 34" / "Local Law 34 of 2020" -- freeze the whole citation (incl. the year) as ONE span.
_LOCAL_LAW_RE = re.compile(r"\bLocal Law\s+\d+(?:\s+of\s+\d{4})?", re.IGNORECASE)
# A section symbol + code, e.g. "section 20-840", "20-840". The bare code form is the common one.
_SECTION_RE = re.compile(r"(?:section|sec\.?)\s*\d+(?:[.\-]\d+)+", re.IGNORECASE)
# An Administrative-Code-style section number: "20-840", "8-107.1". Bounded so it can't grab a phone
# fragment or a date range.
_ADMIN_CODE_RE = re.compile(r"(?<!\d)\d{1,3}-\d{2,4}(?:\.\d+)?(?!\d)")
# A dollar amount (reuses the grounding.py shape): "$2,500", "$ 50.00".
_MONEY_RE = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?")
_MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|November|December"
    "|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
)
# A date: a month-name date, a numeric date, or a bare 4-digit year. Freezing a year verbatim is safe
# (worst case a harmless freeze); it is what stops "de 2020" drifting to "de 2021".
_DATE_RE = re.compile(
    rf"\b(?:{_MONTHS})\.?\s+\d{{1,2}}(?:,?\s+\d{{4}})?"
    r"|\b\d{1,2}/\d{1,2}/\d{2,4}\b"
    r"|\b(?:19|20)\d{2}\b",
    re.IGNORECASE,
)

# (kind, compiled pattern). Priority = index here; used only to break EXACT-start ties. Overlap across
# different starts is resolved by preferring the earlier-starting, then longer, span (see _select).
_BASE_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("cite", _CITE_RE),
    ("law", _LOCAL_LAW_RE),
    ("section", _SECTION_RE),
    ("admin_code", _ADMIN_CODE_RE),
    ("money", _MONEY_RE),
    ("date", _DATE_RE),
]


@dataclass
class _Cand:
    start: int
    end: int
    kind: str
    priority: int
    text: str


def _select(cands: list[_Cand]) -> list[_Cand]:
    """Pick a non-overlapping set of frozen spans. Sort by (start, longest, priority) and greedily take
    a span whose start is at/after the last taken span's end. Longest-first at a shared start is what
    keeps "Local Law 34 of 2020" whole instead of letting a "2020" date span carve out its middle."""
    cands.sort(key=lambda c: (c.start, -(c.end - c.start), c.priority))
    chosen: list[_Cand] = []
    cursor = -1
    for c in cands:
        if c.start >= cursor:
            chosen.append(c)
            cursor = c.end
    return chosen


def mask_entities(text: str, *, glossary: Optional["Glossary"] = None) -> tuple[str, list[Entity]]:
    """Replace every citation / section / dollar amount / date / {cite:Sn} marker, plus the glossary's
    pinned freeze-terms (program & agency proper nouns), with an ordered [[N]] placeholder. Returns the
    masked prose (what the translator sees) and the ordered list of frozen Entities (restored on
    unmask). Pure string work: no model, no network."""
    glossary = glossary or Glossary()
    patterns = list(_BASE_PATTERNS)
    # Freeze-terms are proper nouns that stay identical across languages (e.g. "Housing Connect"): mask
    # them so the translator can't paraphrase a program name. Lowest priority: a real citation wins a tie.
    for term in glossary.freeze_terms():
        patterns.append(("term", re.compile(re.escape(term))))

    cands: list[_Cand] = []
    for priority, (kind, rx) in enumerate(patterns):
        for m in rx.finditer(text):
            if m.end() > m.start():  # never emit an empty span
                cands.append(_Cand(m.start(), m.end(), kind, priority, m.group()))

    chosen = _select(cands)
    chosen.sort(key=lambda c: c.start)  # emit left-to-right so placeholder index == appearance order
    out: list[str] = []
    entities: list[Entity] = []
    last = 0
    for i, c in enumerate(chosen):
        out.append(text[last:c.start])
        out.append(_PLACEHOLDER.format(i))
        entities.append(Entity(kind=c.kind, text=c.text))
        last = c.end
    out.append(text[last:])
    return "".join(out), entities


def unmask(text: str, entities: Sequence[Entity]) -> str:
    """Restore every [[N]] placeholder to its frozen entity text, verbatim. A placeholder with no
    matching entity (a lost / duplicated token) is left AS-IS on purpose -- the round-trip gate then
    sees the entity is missing and falls back, rather than this step silently papering over the loss."""
    def repl(m: re.Match) -> str:
        idx = int(m.group(1))
        return entities[idx].text if 0 <= idx < len(entities) else m.group(0)

    return _PLACEHOLDER_RE.sub(repl, text)


# ==================================================================================================
# 4. Glossary / termbase seam
# ==================================================================================================

@dataclass(frozen=True)
class Term:
    """One civic-legal term. ``freeze=True`` marks a proper noun that is IDENTICAL across languages
    (a program / agency name) -> masked verbatim. Otherwise ``renderings`` gives the one canonical
    per-language translation (lang code -> rendering) handed to the translator so terminology can't
    drift between two answers to the same question."""
    en: str
    renderings: Mapping[str, str] = field(default_factory=dict)
    freeze: bool = False


# A small seed termbase (extend as modules need). Real NYC civic terms; the renderings prefer the
# city's own official Spanish where it publishes one.
DEFAULT_TERMBASE: list[Term] = [
    Term("Housing Connect", freeze=True),
    Term("Notify NYC", freeze=True),
    Term("ACCESS HRA", freeze=True),
    Term("Administrative Code", {"es": "Codigo Administrativo"}),
    Term("source of income", {"es": "fuente de ingresos"}),
    Term("Human Resources Administration", {"es": "Administracion de Recursos Humanos"}),
]

# An official-translation lookup: (english_answer, lang) -> the city's published human translation, or
# None. Injected; default None. This is the seam the compliance work needs -- serve the official Spanish
# outright when it exists, machine-translate only as the fallback.
OfficialLookup = Callable[[str, str], Optional[str]]


class Glossary:
    """The termbase + the official-translation seam. Off-the-shelf defaults; inject ``terms`` and/or
    ``official_lookup`` to override."""

    def __init__(
        self,
        terms: Optional[Sequence[Term]] = None,
        *,
        official_lookup: Optional[OfficialLookup] = None,
    ):
        self._terms = list(terms) if terms is not None else list(DEFAULT_TERMBASE)
        self._official_lookup = official_lookup

    def freeze_terms(self) -> list[str]:
        """Proper-noun terms to mask verbatim (longest first, so a multi-word name masks before any
        substring of it)."""
        return sorted((t.en for t in self._terms if t.freeze), key=len, reverse=True)

    def renderings_for(self, lang: str) -> dict[str, str]:
        """{english_term: canonical rendering} for the translatable (non-frozen) terms in ``lang``.
        Empty when nothing is pinned for that language -- never a crash."""
        return {t.en: t.renderings[lang] for t in self._terms if not t.freeze and lang in t.renderings}

    def official_translation(self, answer_en: str, lang: str) -> Optional[str]:
        """The city's own published translation of this answer if one exists, else None."""
        if self._official_lookup is None:
            return None
        return self._official_lookup(answer_en, lang)


# ==================================================================================================
# 2. Translator
# ==================================================================================================

class Translator(Protocol):
    def translate(self, text: str, target_lang: str, *, glossary: Optional[Mapping[str, str]] = None) -> str: ...


# Injected transform for MockTranslator: (text, target_lang) -> str.
MockTransform = Callable[[str, str], str]


class MockTranslator:
    """Deterministic translator for tests: no network, no model. The DEFAULT tags the language and
    leaves [[N]] placeholders (and any already-verbatim entity) untouched -- enough to prove masking
    and the round-trip gate. Pass a ``transform`` callable to simulate a specific behavior (e.g. a
    back-translation that mutates a citation, to drive the gate's failure path)."""

    backend = "mock"

    def __init__(self, transform: Optional[MockTransform] = None):
        self._transform = transform

    def translate(self, text: str, target_lang: str, *, glossary: Optional[Mapping[str, str]] = None) -> str:
        if self._transform is not None:
            return self._transform(text, target_lang)
        return f"[{target_lang}] {text}"


# Human-readable language names for the prompt (fall back to the raw code for anything unlisted).
_LANG_NAMES = {
    "es": "Spanish", "zh": "Chinese", "ru": "Russian", "bn": "Bengali", "ht": "Haitian Creole",
    "ko": "Korean", "ar": "Arabic", "fr": "French", "ur": "Urdu", "pl": "Polish", "en": "English",
}

_TRANSLATE_SYSTEM = (
    "You are a translator for a New York City civic assistant. Translate the user's text into {lang}. "
    "Rules you must follow exactly:\n"
    "1. FROZEN TOKENS: any token of the form [[N]] (double square brackets around a number) stands for "
    "a law, citation, dollar amount, date, or program name. Copy every [[N]] token into your output "
    "UNCHANGED and in place. Never translate it, renumber it, reorder its digits, or drop it.\n"
    "2. TERMINOLOGY: use these exact renderings for civic terms (English -> {lang}):\n{glossary}\n"
    "3. Preserve meaning faithfully and return ONLY the translated text, with no notes or explanation."
)


def _parse_translation(content: str) -> str:
    """The translator returns prose, not JSON: the completion body IS the translation. Strip surrounding
    whitespace only."""
    return (content or "").strip()


# Injected completion fn: (model, messages) -> an OpenAI-shaped response with .choices[0].message.content.
CompletionFn = Callable[[str, list], object]


class PromptedTranslator:
    """The real backend: translation via the existing litellm seam with a tight, placeholder-preserving
    prompt. Point ``model`` at ``ollama/<model>`` for a self-hosted, zero-egress translation (the data-
    sovereignty track). A general LLM is the machine-translation FALLBACK, used only after the official-
    translation lookup misses. Inject ``completion_fn`` in tests to run with no network; the litellm
    import stays lazy so importing this module never pulls it. NOT run this session."""

    backend = "prompted"

    def __init__(
        self,
        model: str = "ollama/qwen3.5:9b",
        *,
        completion_fn: Optional[CompletionFn] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.0,
    ):
        self._model = model
        self._completion_fn = completion_fn
        self._api_base = api_base
        self._temperature = temperature

    def _build_messages(self, text: str, target_lang: str, glossary: Optional[Mapping[str, str]]) -> list[dict]:
        lang = _LANG_NAMES.get(target_lang, target_lang)
        gloss_lines = "\n".join(f"  - {en} -> {tgt}" for en, tgt in (glossary or {}).items()) or "  (none)"
        system = _TRANSLATE_SYSTEM.format(lang=lang, glossary=gloss_lines)
        return [{"role": "system", "content": system}, {"role": "user", "content": text}]

    def _complete(self, messages: list[dict]):
        if self._completion_fn is not None:
            return self._completion_fn(self._model, messages)
        from litellm import completion  # lazy: keep litellm import out of module load

        kwargs: dict = {"model": self._model, "messages": messages, "temperature": self._temperature}
        if self._api_base:
            kwargs["api_base"] = self._api_base
        return completion(**kwargs)

    def translate(self, text: str, target_lang: str, *, glossary: Optional[Mapping[str, str]] = None) -> str:
        resp = self._complete(self._build_messages(text, target_lang, glossary))
        return _parse_translation(resp.choices[0].message.content)


# ==================================================================================================
# 5. Entity round-trip faithfulness gate
# ==================================================================================================

@dataclass
class GateResult:
    ok: bool                       # did every frozen entity round-trip through back-translation?
    missing: list                 # list[Entity] absent from the back-translation (the caught mutations)
    back_translation: str          # the English back-translation the gate inspected (for telemetry)


def entity_roundtrip_gate(target_answer: str, entities: Sequence[Entity], translator: Translator) -> GateResult:
    """Back-translate the target-language answer to English and assert every frozen entity survives
    EXACTLY. This is the translation-side analog of the grounding guard: a fluency score would wave a
    pretty sentence with the wrong law number through, so we check the ENTITIES, not the prose. Any
    missing entity means the translation is unfaithful and the caller must NOT ship it."""
    back = translator.translate(target_answer, "en")
    missing = [e for e in entities if e.text not in back]
    return GateResult(ok=not missing, missing=missing, back_translation=back)


# ==================================================================================================
# The pipeline: router -> (official | translate) -> gate -> ship or fall back
# ==================================================================================================

@dataclass
class MultilingualResult:
    answer: str          # the text to ship to the user
    lang: str            # the resolved target language code
    route: str           # passthrough | official | translated | fallback
    faithful: bool       # entity gate verdict (True on passthrough/official/faithful translation)
    entities: list       # list[Entity] frozen through the translation ([] on passthrough/official)
    detail: str = ""     # human-readable note (what happened / why a fallback)


class MultilingualPipeline:
    """Wires the pieces into one call. Off by default -- gated on config.HEYNYC_MULTILINGUAL, which this
    module does not read; the agent-loop wiring is the follow-on. The translator is required; the
    detector is optional (pass ``target_lang`` to ``process`` instead)."""

    def __init__(
        self,
        translator: Translator,
        *,
        detector: Optional[LanguageDetector] = None,
        glossary: Optional[Glossary] = None,
        fallback_pointer: str = "",
    ):
        self._translator = translator
        self._detector = detector
        self._glossary = glossary or Glossary()
        self._fallback_pointer = fallback_pointer

    def process(
        self,
        answer_en: str,
        *,
        query: Optional[str] = None,
        target_lang: Optional[str] = None,
    ) -> MultilingualResult:
        """Take a VERIFIED English answer and return what to ship in the user's language. ``target_lang``
        wins if given; otherwise the language is detected from ``query`` via the injected detector."""
        lang = target_lang or (detect_language(query, self._detector) if (query and self._detector) else "en")

        # ROUTE: English (or undetermined) ships the English answer as-is.
        if not needs_translation(lang):
            return MultilingualResult(
                answer=answer_en, lang="en", route="passthrough", faithful=True, entities=[],
                detail="English path: no translation needed",
            )

        # PREFER the city's own official human translation when it exists.
        official = self._glossary.official_translation(answer_en, lang)
        if official is not None:
            return MultilingualResult(
                answer=official, lang=lang, route="official", faithful=True, entities=[],
                detail="served the city's official translation",
            )

        # MASK -> translate the prose only -> unmask the frozen entities verbatim.
        masked, entities = mask_entities(answer_en, glossary=self._glossary)
        translated_masked = self._translator.translate(
            masked, lang, glossary=self._glossary.renderings_for(lang)
        )
        final = unmask(translated_masked, entities)

        # GATE: back-translate and require every frozen entity to round-trip exactly.
        gate = entity_roundtrip_gate(final, entities, self._translator)
        if not gate.ok:
            missing = ", ".join(f"{e.kind} '{e.text}'" for e in gate.missing)
            answer = answer_en + (f"\n\n{self._fallback_pointer}" if self._fallback_pointer else "")
            return MultilingualResult(
                answer=answer, lang=lang, route="fallback", faithful=False, entities=entities,
                detail=f"entity round-trip failed ({missing}); fell back to English + official pointer",
            )

        return MultilingualResult(
            answer=final, lang=lang, route="translated", faithful=True, entities=entities,
            detail=f"translated with {len(entities)} frozen entit{'y' if len(entities) == 1 else 'ies'}",
        )
