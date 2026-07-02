"""Regression tests for the P0 URL-grounding fix.

The benefits tool must surface REAL per-program deep links (extracted from the dataset's HTML
prose) instead of letting the model invent one, and the Sources list must show only the citations
the answer actually used — never a tangential source (e.g. a World Cup link under a SNAP answer).
"""
from heynyc.core.citations import used_citations
from heynyc.modules.benefits.tools import (
    SOURCE_URL,
    _best_url,
    _first_href,
    _help_url,
)


def test_first_href_extracts_real_link():
    html = '<p>Visit <a href="https://access.nyc.gov/programs/eitc" target="_blank">here</a></p>'
    assert _first_href(html) == "https://access.nyc.gov/programs/eitc"


def test_first_href_adds_scheme_to_bare_domain():
    assert _first_href('<a href="nyc.gov/taxprep">x</a>') == "https://nyc.gov/taxprep"


def test_first_href_skips_mailto_tel_and_missing():
    assert _first_href('<a href="mailto:a@b.com">x</a>') == ""
    assert _first_href('<a href="tel:311">x</a>') == ""
    assert _first_href("no link here") == ""
    assert _first_href("") == ""


def test_help_url_pulls_href_from_help_fields():
    row = {"get_help_online": '<p><strong>Visit </strong><a href="https://www.foodhelp.nyc/">map</a></p>'}
    assert _help_url(row) == "https://www.foodhelp.nyc/"


def test_best_url_prefers_apply_then_help_then_landing():
    assert _best_url({"url_of_online_application": "https://apply.example"}) == "https://apply.example"
    assert _best_url({"get_help_summary": '<a href="https://help.example">h</a>'}) == "https://help.example"
    # no real url anywhere -> the dataset landing page, NEVER a model invention
    assert _best_url({}) == SOURCE_URL


def test_used_citations_drops_uncited_sources():
    cits = {"S1": {"url": "events"}, "S4": {"url": "snap"}, "S9": {"url": "wic"}}
    used = used_citations("apply for SNAP {cite:S4} or WIC {cite:S9}", cits)
    assert set(used) == {"S4", "S9"}  # the tangential S1 is dropped


def test_used_citations_empty_when_nothing_cited():
    assert used_citations("no markers at all", {"S1": {"url": "x"}}) == {}
