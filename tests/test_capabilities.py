"""Capability discovery — data-driven from each module's `examples:` (multichannel spec §14)."""
from __future__ import annotations

import re

from heynyc.__main__ import CAPABILITIES_END, CAPABILITIES_START
from heynyc.core import config
from heynyc.core.registry import Registry


def _reg():
    return Registry.discover(config.MODULES_DIR)


def test_modules_carry_examples():
    benefits = next(m for m in _reg().modules if m.name == "benefits")
    assert len(benefits.examples) >= 2
    assert all(isinstance(e, str) and e for e in benefits.examples)


def test_welcome_examples_are_spread_and_capped():
    ex = _reg().welcome_examples(4)
    assert 1 <= len(ex) <= 4
    assert all(e for e in ex)
    assert len(set(ex)) == len(ex)        # no duplicates


def test_capability_menu_rows_are_category_blurb_examples():
    menu = _reg().capability_menu()
    assert menu, "expected at least the benefits module"
    assert "benefits" in {cat for cat, _blurb, _ex in menu}
    assert all(isinstance(ex, list) for _cat, _blurb, ex in menu)


def test_welcome_text_leads_with_examples_never_how_can_i_help():
    text = _reg().welcome_text()
    assert "How can I help" not in text          # the anti-MyCity / "show, don't ask" rule
    assert "•" in text                            # concrete example bullets
    assert "SNAP" in text or "groceries" in text  # real, groundable capabilities


def test_welcome_text_discloses_it_is_an_ai_not_a_city_employee():
    # EU AI Act Art 50 + NYC GenAI transparency: first contact must disclose it's an AI, not staff.
    low = _reg().welcome_text().lower()
    assert "ai assistant" in low
    assert "not a city employee" in low


def test_capability_table_is_one_row_per_top_level_module():
    reg = _reg()
    rows = reg.capability_table()
    top_level = [m for m in reg.modules if not m.parent]
    assert len(rows) == len(top_level)                 # submodules fold into their parent
    services = [r.service for r in rows]
    assert "Events" in services
    assert "Cooling centers" in services
    # world_cup is a submodule of events, so it never gets its own row.
    assert not any("World Cup" in s or "world_cup" in s for s in services)


def test_capability_table_derives_source_and_link_from_manifest():
    rows = {r.service: r for r in _reg().capability_table()}
    # ArcGIS dataset binding names itself by its title.
    assert rows["Cooling centers"].grounded_in == "NYC Emergency Management - Cooling Centers"
    # Socrata binding is labelled by its open-data id.
    assert rows["SNAP centers"].grounded_in == "NYC Open Data (tc6u-8rnp)"
    # Official link is the first seed when present, else the first allowlist domain.
    assert rows["Food pantries"].official_link == "https://finder.nyc.gov/foodhelp/"
    assert rows["Benefits & programs"].official_link == "https://access.nyc.gov"
    assert all(r.asks for r in rows.values())          # every row shows at least one example


def test_capability_markdown_has_the_four_columns():
    md = _reg().capability_markdown()
    assert md.splitlines()[0] == "| Service | What you can ask | Grounded in | Official link |"


def test_readme_capabilities_block_matches_generated_table():
    """Drift guard: the table committed to README must equal the freshly generated one.
    Add or remove a module without regenerating (`heynyc capabilities --write-readme`)
    and this fails, so the README single-sources from the manifests, same as the index."""
    readme = (config.PROJECT_ROOT / "README.md").read_text()
    match = re.search(
        re.escape(CAPABILITIES_START) + r"(.*?)" + re.escape(CAPABILITIES_END),
        readme,
        re.DOTALL,
    )
    assert match, "README is missing the CAPABILITIES markers"
    assert match.group(1).strip() == _reg().capability_markdown().strip()


def test_is_help_detects_greetings_but_not_real_questions():
    from heynyc.channels.orchestrator import is_help
    assert is_help("hi") and is_help("Help!") and is_help("what can you do?")
    assert is_help("/menu") and is_help("  hello  ")
    assert not is_help("where's the nearest cooling center?")   # a real question is NOT help
    assert not is_help("am i eligible for snap")
