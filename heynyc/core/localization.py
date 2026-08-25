from __future__ import annotations

import gettext
from collections.abc import Iterable
from pathlib import Path

from babel import Locale
from babel.core import UnknownLocaleError
from babel.lists import format_list

_DOMAIN = "heynyc"
_LOCALE_DIR = Path(__file__).with_name("locale")


def _(message: str) -> str:
    return message


_WELCOME_FOOTER = _(
    "First time here? I'm HeyNYC. I help with {categories} across NYC using current city "
    "information, and I show the source links.\n"
    "Anytime, text HELP to see what I can do and all chat controls."
)
_SMS_CONTROLS = _(
    "On SMS, text MENU for the full menu, STOP to stop messages, or START to resume."
)
_HELP_TEXT = _(
    "Hi! I'm HeyNYC. I help you find and use NYC services with current city information, and I "
    "show the source links.\n\nTell me what you need in your language.\n\nChat controls:\n"
    "• HELP or MENU: show this menu\n• NEW: start fresh without using the earlier chat\n"
    "• PRIVACY: see how your messages are handled\n"
    "• REPORT or 👎: flag the last exchange for review, after you confirm\n"
    "• DELETE MY DATA: erase the conversation data I keep, after you confirm\n"
    "• STOP / START: stop or resume SMS messages (SMS only)\n\n"
    "Heads up: I'm an AI assistant, not a City employee or caseworker. Double-check anything "
    "important against the official source."
)
_REGULAR_HOURS_SOURCE_LIMIT = _(
    "These are regular hours. Confirm holiday or temporary schedule exceptions before traveling."
)
_SOURCE_LIMITS = frozenset({_REGULAR_HOURS_SOURCE_LIMIT})


def _locale(language: str | None) -> str | None:
    if not isinstance(language, str):
        return "en" if language is None else None
    value = language.strip()
    if not value:
        return "en"
    try:
        return str(Locale.parse(value, sep="-"))
    except (UnknownLocaleError, ValueError):
        return None


def localize(message: str, language: str | None) -> str:
    locale = _locale(language)
    if locale is None:
        return message
    translation = gettext.translation(
        _DOMAIN,
        localedir=str(_LOCALE_DIR),
        languages=[locale, locale.split("_", 1)[0]],
        fallback=True,
    )
    return translation.gettext(message)


def sms_controls(language: str | None = None) -> str:
    return localize(_SMS_CONTROLS, language)


def help_text(language: str | None = None) -> str | None:
    locale = _locale(language)
    if locale is None or locale.split("_", 1)[0] == "en":
        return None
    translated = localize(_HELP_TEXT, locale)
    return translated if translated != _HELP_TEXT else None


def localized_source_limit(message: str, language: str | None) -> str | None:
    if message not in _SOURCE_LIMITS:
        return None
    locale = _locale(language)
    if locale is None:
        return None
    if locale.split("_", 1)[0] == "en":
        return message
    translated = localize(message, locale)
    return translated if translated != message else None


def welcome_footer(categories: Iterable[str], language: str | None = None) -> str | None:
    locale = _locale(language)
    if locale is None:
        return None
    base_language = locale.split("_", 1)[0]
    if base_language != "en" and gettext.find(
        _DOMAIN, localedir=str(_LOCALE_DIR), languages=[locale]
    ) is None:
        return None
    translation = gettext.translation(
        _DOMAIN,
        localedir=str(_LOCALE_DIR),
        languages=[locale],
        fallback=True,
    )
    listed = format_list(tuple(categories) or ("NYC services",), locale=locale)
    translated = translation.gettext(_WELCOME_FOOTER)
    if base_language != "en" and translated == _WELCOME_FOOTER:
        return None
    return translated.format(categories=listed)
