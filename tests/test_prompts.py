from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from heynyc.core import config
from heynyc.core.prompts import build_system_prompt
from heynyc.core.registry import Registry


def _real_registry() -> Registry:
    """The live, module-discovered registry (real keywords, examples, and blurbs). The router and
    progressive-disclosure tests need real manifests, not the empty registry."""
    return Registry.discover(config.MODULES_DIR, config.BASE_ALLOWLIST)


def test_system_prompt_injects_current_nyc_datetime():
    fixed = datetime(2026, 6, 28, 19, 30, tzinfo=ZoneInfo("America/New_York"))
    prompt = build_system_prompt(Registry([]), now=fixed)
    assert "Current date & time" in prompt
    assert "June 28, 2026" in prompt
    assert "America/New_York" in prompt
    # still carries the grounding rules
    assert "GROUND EVERYTHING" in prompt


def test_system_prompt_includes_active_recency_check():
    # The freshness guard goes from passive date-stamping to an ACTIVE recency check: on
    # time-sensitive law/policy/rights questions the agent must run recent_developments and
    # surface any breaking change as a dated, cited heads-up on top of the official answer.
    prompt = build_system_prompt(Registry([]))
    low = prompt.lower()
    assert "recent_developments" in prompt
    assert "this may be changing" in low
    assert "recency check" in low


def test_system_prompt_surfaces_human_and_appeal_path():
    # When it can't help or a user reports a denial/problem, the agent must offer a way to
    # reach a human (311/the agency) and the official appeal path. (NYC GenAI Guidance.)
    prompt = build_system_prompt(Registry([]))
    low = prompt.lower()
    assert "311" in prompt
    assert "appeal" in low
    assert "human" in low


def test_system_prompt_refuses_obfuscated_encoded_instructions():
    # Red-team PI12 fix: an encoded (base64/hex/rot13) "do what it says" instruction must be refused,
    # not executed, and never answered with silence.
    low = build_system_prompt(Registry([])).lower()
    assert "base64" in low
    assert "encoded" in low or "obfuscated" in low
    assert "non-empty" in low or "never fall silent" in low


def test_system_prompt_forbids_uncited_authority_on_substantive_facts():
    # Grounding-discipline fix: an ungrounded substantive claim must not be framed as authoritative;
    # the agent grounds it via a tool or gives it as general guidance routed to 311/official — while
    # still correcting harmful misconceptions (never over-abstaining).
    low = build_system_prompt(Registry([])).lower()
    assert "authoritative answer" in low            # the anti-pattern it must avoid
    assert "general information" in low             # the allowed ungrounded framing
    assert "misconception" in low                   # must still correct false premises
    assert "emtali" in low or "emtala" in low       # names the find_clinic grounding route


def test_system_prompt_describes_data_practices_accurately():
    # Privacy-accuracy fix: the prompt must NOT let the agent claim it "stores nothing"; it should
    # describe the in-progress application draft honestly and match the PIA.
    low = build_system_prompt(Registry([])).lower()
    assert "draft" in low
    assert "profile" in low                          # "don't build a profile"
    assert "store nothing" in low or "retain nothing" in low  # named as the claim to AVOID


def test_system_prompt_sets_plain_language_reading_level():
    low = build_system_prompt(Registry([])).lower()
    assert "reading level" in low
    assert "6th" in low and "8th" in low


def test_system_prompt_contested_legal_matter_protocol():
    # Red-team MC03/MC04/FP02/ES03 fix: on a contested legal ruling surfaced from news, the agent must
    # NOT restate the court/holding/scope, must NOT tell a user their protection is struck down/gone/
    # annulled, and must LEAD with the protection that currently stands, then route.
    low = build_system_prompt(Registry([])).lower()
    assert "contested legal matter" in low
    # names the exact words it must never use about a still-valid right
    assert "struck down" in low and "annulled" in low
    # the safe shape: lead with what currently stands
    assert "currently stands" in low
    assert "active legal challenge" in low
    # must not assert the court
    assert "never name the court" in low or "which court ruled" in low


def test_system_prompt_emergency_no_medical_dosing():
    # Red-team ES09 fix: an emergency (heart-attack signs) gets 911 only — NO drug names or dosages,
    # not even aspirin. The English path already held; this makes the standard explicit for all langs.
    low = build_system_prompt(Registry([])).lower()
    assert "911" in low
    assert "no medical instructions" in low or "no dosages" in low
    assert "aspirin" in low  # named as the exact thing NOT to dose


def test_system_prompt_public_charge_snap_guardrail():
    # Red-team FP03 fix: the agent must NOT say SNAP counts toward public charge; SNAP is generally
    # NOT a public-charge benefit, and it must not derive the rule from news about "the administration."
    low = build_system_prompt(Registry([])).lower()
    assert "public charge" in low
    assert "snap" in low and "not counted" in low
    assert "actionnyc" in low  # routes to immigration legal aid


def test_system_prompt_same_discipline_in_every_language():
    # Red-team ES09/ES13 fix: EVERY safety + grounding rule holds identically in Spanish and any other
    # language (emergencies -> 911 no dosing; never fabricate a law number/citation; lead with the
    # standing protection). A prompt reinforcement only partially closes a model-level cross-lingual gap.
    low = build_system_prompt(Registry([])).lower()
    assert "every language" in low
    assert "law number" in low  # never invent a law number in any language
    # cites the concrete correct statute the Spanish ES13 failure fabricated ("Local Law 68")
    assert "local law 34" in low or "20-840" in low


# --- Change 1: progressive disclosure of the per-module detailed blurbs ---------------------------

def test_router_matches_module_on_curated_keyword():
    from heynyc.core.prompts import route_modules

    assert "food_pantries" in route_modules("where's the nearest food pantry?", _real_registry())


def test_router_returns_empty_set_on_unrelated_query():
    from heynyc.core.prompts import route_modules

    assert route_modules("what's the capital of France?", _real_registry()) == set()


def test_router_short_keyword_is_not_substring_matched():
    # cooling_centers has the 2-letter keyword "ac"; it must NOT match inside "beach". A beach-closure
    # question routes to advisories (which owns "beach closure"), never to cooling via a substring hit.
    from heynyc.core.prompts import route_modules

    matched = route_modules("is the beach closed today?", _real_registry())
    assert "cooling_centers" not in matched
    assert "advisories" in matched


def test_query_loads_only_matching_blurbs_but_keeps_menu_and_all_rules():
    prompt = build_system_prompt(_real_registry(), query="where's the nearest food pantry?")
    # the matched module's DETAILED blurb loads
    assert "nearest_food_pantry(near=" in prompt
    # clearly-unrelated modules' DETAILED blurbs do NOT load
    assert "NOT outdoor misting stations" not in prompt   # cooling blurb text
    assert "nyc_advisories" not in prompt                 # advisories blurb text
    assert "PROGRAM INFO" not in prompt                   # housing blurb text
    # the always-on capability menu + every safety rule stay present (a routing miss drops neither)
    assert "Services you can help with (quick menu)" in prompt
    assert "GROUND EVERYTHING" in prompt
    assert "911" in prompt                                # rule 13 (emergencies)
    assert "PUBLIC CHARGE" in prompt                      # rule 14 (SNAP / public charge)


def test_no_match_query_keeps_menu_and_rules_but_loads_no_detailed_blurbs():
    prompt = build_system_prompt(_real_registry(), query="what's the capital of France?")
    # fail-open on a routing miss: NO detailed blurbs at all...
    assert "nearest_food_pantry(near=" not in prompt
    assert "benefits_search(query=" not in prompt
    assert "nyc_advisories" not in prompt
    # ...but the menu + safety rules are never dropped
    assert "Services you can help with (quick menu)" in prompt
    assert "GROUND EVERYTHING" in prompt
    assert "911" in prompt


def test_query_none_includes_every_blurb_backward_compat():
    prompt = build_system_prompt(_real_registry())  # query defaults to None -> today's behavior
    assert "nearest_food_pantry(near=" in prompt      # food blurb
    assert "benefits_search(query=" in prompt         # benefits blurb
    assert "NOT outdoor misting stations" in prompt   # cooling blurb
    assert "nyc_advisories" in prompt                 # advisories blurb


def test_capability_menu_names_every_service_regardless_of_routing():
    # The cheap always-on menu names capabilities the current query didn't select, so a routing miss
    # never hides a service from the model. Holds for query=None, a matched query, and a no-match query.
    for query in (None, "where's the nearest food pantry?", "what's the capital of France?"):
        low = build_system_prompt(_real_registry(), query=query).lower()
        assert "cooling" in low
        assert "eviction" in low or "housing" in low
        assert "benefit" in low


def test_tiers_keep_date_and_selected_blurbs_out_of_the_stable_prefix():
    from heynyc.core.prompts import build_system_prompt_tiers

    stable, volatile = build_system_prompt_tiers(
        _real_registry(), query="where's the nearest food pantry?")
    # stable = safety rules + menu, query- and time-independent (the cacheable prefix)
    assert "GROUND EVERYTHING" in stable
    assert "Services you can help with (quick menu)" in stable
    assert "Current date & time" not in stable          # the date must NOT sit inside the cached prefix
    assert "nearest_food_pantry(near=" not in stable     # selected blurbs are volatile, not cached
    # volatile = the selected blurbs + the date line
    assert "Current date & time" in volatile
    assert "nearest_food_pantry(near=" in volatile


def test_capability_blurbs_only_filters_to_named_modules():
    reg = _real_registry()
    only = reg.capability_blurbs(only={"food_pantries"})
    assert "## food_pantries" in only
    assert "## benefits" not in only
    # the default (no filter) still returns every module's blurb (backward-compat)
    assert "## benefits" in reg.capability_blurbs()
