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
    "First time here? I'm HeyNYC. I help with {categories} across NYC, grounded in real city "
    "data, and I cite my sources.\n"
    "Anytime, text HELP for what I can do, PRIVACY for how your info is handled, REPORT to "
    "flag a bad answer, or DELETE MY DATA to erase everything I keep."
)


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
