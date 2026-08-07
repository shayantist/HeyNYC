from __future__ import annotations

import gettext
from collections.abc import Iterable
from pathlib import Path

from babel.lists import format_list

_DOMAIN = "heynyc"
_LOCALE_DIR = Path(__file__).with_name("locale")
_SUPPORTED_LOCALES = frozenset({"en"})


def _(message: str) -> str:
    return message


_WELCOME_FOOTER = _(
    "First time here? I'm HeyNYC. I help with {categories} across NYC, grounded in real city "
    "data, and I cite my sources.\n"
    "Anytime, text HELP for what I can do, PRIVACY for how your info is handled, REPORT to "
    "flag a bad answer, or DELETE MY DATA to erase everything I keep."
)


def _locale(language: str | None) -> str:
    value = language.strip().lower() if isinstance(language, str) else ""
    return value if value in _SUPPORTED_LOCALES else "en"


def welcome_footer(categories: Iterable[str], language: str | None = None) -> str:
    locale = _locale(language)
    translation = gettext.translation(
        _DOMAIN,
        localedir=str(_LOCALE_DIR),
        languages=[locale],
        fallback=True,
    )
    listed = format_list(tuple(categories) or ("NYC services",), locale=locale)
    return translation.gettext(_WELCOME_FOOTER).format(categories=listed)
