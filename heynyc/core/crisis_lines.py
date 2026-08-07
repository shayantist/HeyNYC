"""Verified crisis-line translations for the Local Law 30 languages (crisis screen, phase 1).

This is DATA, not translation. The deterministic crisis-response floor is authored in English (see
`heynyc/core/agent.py`, `_emergency_backstop` and the self-harm / 988 constants). This module makes
that floor available in the ten Local Law 30 citywide languages by carrying, per language, the city's
OWN or SAMHSA's OWN official crisis copy, COPIED VERBATIM from the official page with its source URL
and verification date. Nothing in `CRISIS_LINES` is machine-translated: for a script this session
cannot read (Arabic, Chinese, Bengali, Hangul, Cyrillic), the records contain only bytes lifted
from an official government page.

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
  - NYS OMH and ACCESS NYC publish current Bengali 988 and 911 instructions.
  - CORRECTED 2026-07-30 (F149): an earlier version of this docstring said no official
    human-translated crisis copy existed for Haitian Creole, Korean, Urdu, or Polish. That was
    WRONG, and an Urdu resident disclosing imminent self-harm got an all-English reply because of
    it. NYS OMH publishes this crisis page in ALL of them at `<lang>.omh.ny.gov`, the same host
    the Bengali record already used; the original survey only tried that pattern for Bengali. All
    four now carry verbatim official copy. Their 911 text is an explanatory "988 versus 911"
    comparison rather than an instruction, so they carry no `emergency_911` and the English floor
    supplies 911, matching the Russian and French records.

`compose_crisis_floor` coverage: Spanish has human-reviewed assistant copy plus the verified
pointer here. Every other LL30 language now gets the English floor plus a verified in-language
988 pointer, except Arabic, whose official page carries a clean 911 line but no clean 988 line;
Arabic and any unrecognized language get the cited interpretation fact instead.

The legacy floor's language detection lives in `agent.py`. The Pydantic runtime's semantic safety
classifier returns a typed risk and language label, then `crisis_response` selects deterministic
copy without accepting model-authored response text.

`_FULL_CRISIS_RESPONSES` is deterministic assistant copy for the current Bengali and Chinese
release cases, not verbatim government translation. It uses only the cited response actions and
must receive human language review before HeyNYC makes a broad public language-quality claim.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

SELF_HARM_RESPONSE_EN = (
    "I'm really sorry you're hurting. I can't help with ways to end your life. "
    "Call 988 right now. If you might act now or cannot stay safe, call 911. "
    "Move away from anything you could use to hurt yourself and contact someone you trust."
)
IMMINENT_SELF_HARM_RESPONSE_EN = (
    "I'm really sorry you're in this much pain. Call 911 right now. Call 988 now too. "
    "Move away from anything you could use to hurt yourself and contact someone you trust who "
    "can stay with you. I'm an AI and can't call or monitor emergency help for you."
)
SELF_HARM_RESPONSE_ES = (
    "Siento mucho que estés pasando por esto. No puedo ayudar con formas de quitarte la vida. "
    "Llama o envía un mensaje al 988 ahora. Si podrías actuar ahora o no puedes mantenerte "
    "a salvo, llama al 911. Aléjate de cualquier cosa con la que puedas lastimarte y contacta "
    "a alguien de confianza."
)

_FULL_CRISIS_RESPONSES = {
    "bn": (
        "আপনি এত কষ্টে আছেন শুনে আমি খুবই দুঃখিত। এখনই 988-এ কল করুন। আপনি যদি এখনই "
        "নিজেকে আঘাত করতে পারেন বা নিরাপদে থাকতে না পারেন, 911-এ কল করুন। নিজেকে আঘাত "
        "করতে পারেন এমন জিনিস থেকে দূরে সরে যান এবং একজন বিশ্বস্ত মানুষকে এখনই আপনার "
        "সঙ্গে থাকতে বলুন। আমি আপনার হয়ে কল করতে বা সাহায্য পর্যবেক্ষণ করতে পারি না।"
    ),
    "zh": (
        "我很抱歉你现在这么痛苦。请立刻打 988 求助。如果你觉得自己可能马上会行动，或者无法"
        "保证安全，请立刻打 911。请马上远离任何你可能用来伤害自己的东西，并联系一位你信任的"
        "人，让他或她现在来陪着你。我不能替你打电话，也不能持续监看你。"
    ),
}

# Verification date for every string in this module (live doc check via HTTP).
VERIFIED_ON = "2026-07-20"
# The four LL30 languages harvested from NYS OMH on this date (F149)
VERIFIED_ON_OMH_LL30 = "2026-07-30"

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
_NYS988_BN = "https://bn.omh.ny.gov/omhweb/crisis/what-is-988.html"
_NYS988_UR = "https://ur.omh.ny.gov/omhweb/crisis/what-is-988.html"
_NYS988_KO = "https://ko.omh.ny.gov/omhweb/crisis/what-is-988.html"
_NYS988_PL = "https://pl.omh.ny.gov/omhweb/crisis/what-is-988.html"
_NYS988_HT = "https://ht.omh.ny.gov/omhweb/crisis/what-is-988.html"
_ACCESS_NYC_988_BN = "https://access.nyc.gov/bn/programs/nyc-988/"

_INTERPRETATION_ONLY = (
    "No official human-translated crisis copy located; NYC 988 offers phone interpretation only "
    "(call 988, 200+ languages). Falls back to the English floor, which carries 988 and 911."
)

# SAMHSA's own 988 FAQ, verified live 2026-07-30. Serves the languages that still lack a clean
# in-language 988 pointer (Arabic, and any language the router does not recognize): the module
# recorded 988's interpreter availability in a `note` a resident never saw, so the floor now
# states and cites it
SAMHSA_988_FAQ_URL = "https://www.samhsa.gov/mental-health/988/faqs"
SAMHSA_988_INTERPRETATION_SNIPPET = (
    "988 call, chat, and text services are available in English and Spanish. Call services with "
    "interpreters are available in more than 240 languages. If you speak a language other than "
    "English or Spanish, the 988 Lifeline uses Language Line Solutions to provide interpretation "
    "to callers in more than 240 additional languages. There is no cost to you for language "
    "interpretation."
)
# English, because it is the floor's language, so it does NOT serve a monolingual reader; it
# serves a partial-English reader or whoever is helping them. Preferred order is always verified
# official copy in the resident's own language (see the NYS OMH records below); this is the
# fallback for a language that has none
_INTERPRETER_LINE_EN = (
    "988 has interpreters in more than 240 languages, at no cost to you."
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
    "bn": CrisisLine(
        lang="bn",
        name="Bengali",
        lifeline_988=(
            "যখন আপনার কারো সাথে কথা বলার প্রয়োজন হয় - 988 আপনার জন্য এখানে আছে৷ "
            "এখন সাহায্য প্রয়োজন? ডায়াল করুন 988।"
        ),
        emergency_911=(
            "যদি আপনি তাৎক্ষণিক বিপদের মধ্যে থাকেন অথবা আপনার জরুরি চিকিৎসা সহায়তার "
            "প্রয়োজন হয়, তবে 911 নম্বরে কল করুন।"
        ),
        source_988=_NYS988_BN,
        source_911=_ACCESS_NYC_988_BN,
        verified_on=VERIFIED_ON,
        note=(
            "ACCESS NYC advertises call, text, and chat, but identifies English, Spanish, and "
            "Chinese counselors plus 200-language interpretation. Calling is the verified "
            "Bengali-language route."
        ),
    ),
    "ht": CrisisLine(
        lang="ht",
        name="Haitian Creole",
        lifeline_988=(
            "Bezwen Èd Kounye a? Rele oswa voye tèks 988 oswa chat sou entènèt"
        ),
        source_988=_NYS988_HT,
        verified_on=VERIFIED_ON_OMH_LL30,
        note=(
            "NYS OMH publishes this page in the language. Calling is the verified in-language route: the page advertises call, text, and chat, but 988 identifies English and Spanish responders plus phone interpretation, so text and chat may not be answered in this language. No clean in-language 911 directive on the page (its 911 text is an explanatory 988-versus-911 comparison, not an instruction); the English floor carries 911."
        ),
    ),
    "ko": CrisisLine(
        lang="ko",
        name="Korean",
        lifeline_988=(
            "지금 도움이 필요하세요? 전화 또는 문자 988 또는 온라인 채팅"
        ),
        source_988=_NYS988_KO,
        verified_on=VERIFIED_ON_OMH_LL30,
        note=(
            "NYS OMH publishes this page in the language. Calling is the verified in-language route: the page advertises call, text, and chat, but 988 identifies English and Spanish responders plus phone interpretation, so text and chat may not be answered in this language. No clean in-language 911 directive on the page (its 911 text is an explanatory 988-versus-911 comparison, not an instruction); the English floor carries 911."
        ),
    ),
    "ur": CrisisLine(
        lang="ur",
        name="Urdu",
        lifeline_988=(
            "ابھی مدد کی ضرورت ہے؟ ڈائل کریں یا ٹیکسٹ کریں 988 یا آن لائن چیٹ کریں۔"
        ),
        source_988=_NYS988_UR,
        verified_on=VERIFIED_ON_OMH_LL30,
        note=(
            "NYS OMH publishes this page in the language. Calling is the verified in-language route: the page advertises call, text, and chat, but 988 identifies English and Spanish responders plus phone interpretation, so text and chat may not be answered in this language. No clean in-language 911 directive on the page (its 911 text is an explanatory 988-versus-911 comparison, not an instruction); the English floor carries 911."
        ),
    ),
    "pl": CrisisLine(
        lang="pl",
        name="Polish",
        lifeline_988=(
            "Potrzebujesz pomocy teraz? Wybierz numer lub wyślij SMS 988 lub porozmawiaj online"
        ),
        source_988=_NYS988_PL,
        verified_on=VERIFIED_ON_OMH_LL30,
        note=(
            "NYS OMH publishes this page in the language. Calling is the verified in-language route: the page advertises call, text, and chat, but 988 identifies English and Spanish responders plus phone interpretation, so text and chat may not be answered in this language. No clean in-language 911 directive on the page (its 911 text is an explanatory 988-versus-911 comparison, not an instruction); the English floor carries 911."
        ),
    ),
}


def compose_crisis_floor(english_floor: str, lang: Optional[str]) -> str:
    """Return the crisis floor to serve for a message detected in `lang`.

    When a verified in-language pointer exists (zh/ru/fr/ar; es too, though Spanish is served by its
    own full floor), append the verbatim official 988 and/or 911 line(s) to the English floor.

    When the resident's language is a KNOWN LL30 language with no verified 988 pointer
    (ht/ko/ur/pl, and ar which has only a 911 line), append the cited interpretation fact. F149:
    an Urdu resident disclosing imminent self-harm previously received the English floor with no
    indication that 988 can serve them in Urdu, a fact this module recorded in a `note` and never
    told the resident.

    When the language is unknown or None, return the English floor UNCHANGED, byte-identical:
    there is no established language to promise interpretation for."""
    line = CRISIS_LINES.get(lang or "")
    if line is None:
        return english_floor
    parts = [english_floor]
    if line.lifeline_988:
        parts.append(line.lifeline_988)
    elif "988" in english_floor:
        parts.append(_INTERPRETER_LINE_EN)
    if line.emergency_911:
        parts.append(line.emergency_911)
    return "\n\n".join(parts)


def crisis_response(risk: str, lang: Optional[str]) -> str:
    """Return deterministic response copy; model output never supplies resident-facing text."""
    if lang == "es":
        return SELF_HARM_RESPONSE_ES
    if lang in _FULL_CRISIS_RESPONSES:
        return _FULL_CRISIS_RESPONSES[lang]
    english = (
        IMMINENT_SELF_HARM_RESPONSE_EN
        if risk == "imminent_self_harm"
        else SELF_HARM_RESPONSE_EN
    )
    return compose_crisis_floor(english, lang)
