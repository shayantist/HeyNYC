"""Verified crisis-line translations for the Local Law 30 languages (crisis screen, phase 1).

This is DATA, not translation. The deterministic crisis-response floor is authored in English (see
`heynyc/core/agent.py`, `_emergency_backstop` and the self-harm / 988 constants). This module makes
that floor available in the ten Local Law 30 citywide languages by carrying, per language, the city's
OWN or SAMHSA's OWN official crisis copy, COPIED VERBATIM from the official page with its source URL
and verification date. Nothing here is machine-translated: for a script this session cannot read
(Arabic, Chinese, Bengali, Hangul, Cyrillic) the only safe copy is bytes lifted from an official
government page, and that is exactly what these records are.

WHAT WAS VERIFIED LIVE (2026-07-20), and what was NOT:

  - The clean "988" pointer comes from the national 988 Suicide & Crisis Lifeline language pages
    (988lifeline.org/es, /interpretation-services/{russian,french,chinese}). Those pages publish
    human-translated copy for Spanish, Russian, French, and Chinese only. The Russian / French /
    Chinese lines are the Lifeline's official "call 988 for interpretation in 240+ languages"
    sentence: verbatim, official, and correctly route the caller to 988 in their own language.
  - The clean "911" emergency line comes from NYC's own translated 988 pages
    (nyc988.cityofnewyork.us/es, /zh, /ar). IMPORTANT HONEST FINDING: those NYC pages still carry
    LEGACY "NYC Well" contact copy (1-888-692-9355 / text WELL to 65173) in the body, not a clean
    "988" line, so only their 911 emergency sentence is usable here.
  - NO official human-translated crisis copy was located for Bengali, Haitian Creole, Korean, Urdu,
    or Polish. For those languages NYC 988 offers phone interpretation only (call 988, 200+
    languages). They therefore carry NO verified in-language line and fall back to the English floor,
    which itself carries 988 and 911. That gap is stated in each record's `note`, not papered over.

Net coverage: Spanish is fully covered by the English-authored Spanish floor already in agent.py
(`_SELF_HARM_RESPONSE_ES`, human-reviewed) plus the verified pointer here. Chinese, Russian, French,
and Arabic get the English floor plus a verified in-language pointer. The remaining five get the
English floor, honestly, because no verified translation exists yet.

Detection (which language a message is in) is deterministic and lives beside the floor in agent.py
(`_crisis_language`, dominant-non-Latin-script based). Semantic crisis DETECTION across languages is
the scope-classifier preflight (phase 2), separate from this response floor.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Verification date for every string in this module (live doc check via HTTP).
VERIFIED_ON = "2026-07-20"

# The ten Local Law 30 designated citywide languages (NYC Charter, LL30 language access), as
# ISO 639-1 code -> English name. This is the exact set the crisis screen must cover.
LL30_LANGUAGES: dict[str, str] = {
    "es": "Spanish",
    "zh": "Chinese",
    "ru": "Russian",
    "bn": "Bengali",
    "ht": "Haitian Creole",
    "ko": "Korean",
    "ar": "Arabic",
    "ur": "Urdu",
    "fr": "French",
    "pl": "Polish",
}

# Official sources the copy below was lifted from, verbatim.
_LIFELINE_ES = "https://988lifeline.org/es/inicio/"
_LIFELINE_RU = "https://988lifeline.org/interpretation-services/russian/"
_LIFELINE_FR = "https://988lifeline.org/interpretation-services/french/"
_LIFELINE_ZH = "https://988lifeline.org/interpretation-services/chinese/"
_NYC988_ES = "https://nyc988.cityofnewyork.us/es/"
_NYC988_ZH = "https://nyc988.cityofnewyork.us/zh/"
_NYC988_AR = "https://nyc988.cityofnewyork.us/ar/"

_INTERPRETATION_ONLY = (
    "No official human-translated crisis copy located; NYC 988 offers phone interpretation only "
    "(call 988, 200+ languages). Falls back to the English floor, which carries 988 and 911."
)


@dataclass(frozen=True)
class CrisisLine:
    """One LL30 language's verified crisis copy. `lifeline_988` / `emergency_911` are VERBATIM
    official copy (or "" when none was located); the `source_*` URL and `verified_on` date are the
    provenance for each. `note` states honestly what could not be verified."""

    lang: str
    name: str
    lifeline_988: str = ""
    emergency_911: str = ""
    source_988: str = ""
    source_911: str = ""
    verified_on: str = ""
    note: str = ""

    @property
    def has_verified_copy(self) -> bool:
        return bool(self.lifeline_988 or self.emergency_911)


CRISIS_LINES: dict[str, CrisisLine] = {
    "es": CrisisLine(
        lang="es",
        name="Spanish",
        lifeline_988=(
            "Si necesitas apoyo emocional, comunícate con la línea nacional de salud mental: 988."
        ),
        emergency_911=(
            "¿Está en peligro o necesita atención médica de inmediato? Llame al 911 ahora mismo."
        ),
        source_988=_LIFELINE_ES,
        source_911=_NYC988_ES,
        verified_on=VERIFIED_ON,
        note=(
            "Also fully covered by the human-reviewed Spanish self-harm floor in agent.py "
            "(_SELF_HARM_RESPONSE_ES), which the Spanish-detection path serves directly."
        ),
    ),
    "zh": CrisisLine(
        lang="zh",
        name="Chinese",
        lifeline_988=(
            "对于说其他语言的客户，请拨打988，Language Line Solutions将为您提供超过240种语言的翻译服务。"
        ),
        emergency_911="发生危险或者需要立即就医？请立即拨打 911。",
        source_988=_LIFELINE_ZH,
        source_911=_NYC988_ZH,
        verified_on=VERIFIED_ON,
    ),
    "ru": CrisisLine(
        lang="ru",
        name="Russian",
        lifeline_988=(
            "Если вы говорите на другом языке, позвоните по номеру 988, и Language Line Solutions "
            "предоставит вам услуги переводчика на более чем 240 языков."
        ),
        source_988=_LIFELINE_RU,
        verified_on=VERIFIED_ON,
        note="No clean in-language 911 line located on an official page; English floor carries 911.",
    ),
    "fr": CrisisLine(
        lang="fr",
        name="French",
        lifeline_988=(
            "Pour les personnes parlant d’autres langues, appelez le 988 afin de bénéficier d’une "
            "traduction dans plus de 240 autres langues par l’intermédiaire de Language Line "
            "Solutions."
        ),
        source_988=_LIFELINE_FR,
        verified_on=VERIFIED_ON,
        note=(
            "Verified copy exists, but French is Latin-script with no deterministic single-language "
            "signal here, so _crisis_language does not route to it in phase 1. No in-language 911 "
            "line located; English floor carries 911."
        ),
    ),
    "ar": CrisisLine(
        lang="ar",
        name="Arabic",
        emergency_911="هل أنت في خطر أو تحتاج إلى رعاية طبية في الحال؟ اتصل الآن برقم 911",
        source_911=_NYC988_AR,
        verified_on=VERIFIED_ON,
        note=(
            "NYC's Arabic 988 page still shows legacy NYC Well contact copy, not a clean 988 line, "
            "and no Arabic Lifeline page exists; English floor carries 988."
        ),
    ),
    "bn": CrisisLine(lang="bn", name="Bengali", note=_INTERPRETATION_ONLY),
    "ht": CrisisLine(lang="ht", name="Haitian Creole", note=_INTERPRETATION_ONLY),
    "ko": CrisisLine(lang="ko", name="Korean", note=_INTERPRETATION_ONLY),
    "ur": CrisisLine(lang="ur", name="Urdu", note=_INTERPRETATION_ONLY),
    "pl": CrisisLine(lang="pl", name="Polish", note=_INTERPRETATION_ONLY),
}


def compose_crisis_floor(english_floor: str, lang: Optional[str]) -> str:
    """Return the crisis floor to serve for a message detected in `lang`.

    When a verified in-language pointer exists (zh/ru/fr/ar; es too, though Spanish is served by its
    own full floor), append the verbatim official 988 and/or 911 line(s) to the English floor. When
    no verified copy exists (bn/ht/ko/ur/pl, unknown, or None), return the English floor UNCHANGED,
    byte-identical, so the fallback is honest and still carries 988 and 911."""
    line = CRISIS_LINES.get(lang or "")
    if line is None or not line.has_verified_copy:
        return english_floor
    parts = [english_floor]
    if line.lifeline_988:
        parts.append(line.lifeline_988)
    if line.emergency_911:
        parts.append(line.emergency_911)
    return "\n\n".join(parts)
