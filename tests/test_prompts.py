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


def test_system_prompt_teaches_orient_then_verify():
    """RULED (2026-07-18): when a reference is ambiguous or abbreviated, the first tool call
    is one broad allowlisted web_search with a short noun-phrase query, orient, then gather
    citable evidence. Audited live: searching the resident's whole sentence returns garbage."""
    prompt = build_system_prompt(Registry([]))
    low = prompt.lower()
    assert "orient" in low
    assert "noun-phrase" in low or "noun phrase" in low
    assert "whole sentence" in low or "full sentence" in low
    assert "then" in low and "evidence" in low


def test_system_prompt_bans_internal_jargon_in_replies():
    """Observed live: the assistant told a resident about its 'grounded NYC match-related
    item'. Plumbing words stay out of resident-facing copy."""
    prompt = build_system_prompt(Registry([]))
    low = prompt.lower()
    assert "grounded" in low  # the internal rules still use the concept
    assert "never say" in low or "plumbing" in low or "internal words" in low
    assert '"grounded"' in prompt  # the ban names the exact word residents saw
    # Luna-medium runs long for SMS (2026-07-18 review): the voice caps answer size.
    assert "text-message sized" in low
    assert "about 5 items" in low   # consolidated count home (prompt diet block 3)


def test_system_prompt_includes_active_recency_check():
    # The freshness guard goes from passive date-stamping to an ACTIVE recency check: on
    # time-sensitive law/policy/rights questions the agent must run recent_developments and
    # surface any breaking change as a dated, cited heads-up on top of the official answer.
    prompt = build_system_prompt(Registry([]))
    low = prompt.lower()
    assert "recent_developments" in prompt
    assert "check for recent changes" in low
    # The heads-up shape ("this may be changing") rehomed to the tool description with the
    # rest of the protocol (prompt diet, 2026-07-22); pinned there by the contested test.


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


def test_system_prompt_requires_citations_on_the_supported_sentence_or_bullet():
    low = build_system_prompt(Registry([])).lower()
    assert "same sentence or bullet" in low
    assert "elsewhere in a paragraph or list does not count" in low
    assert "never put braces around a url" in low


def test_system_prompt_describes_data_practices_accurately():
    # Privacy-accuracy fix: the prompt must NOT let the agent claim it "stores nothing"; it should
    # describe the in-progress application draft honestly and match the PIA.
    low = build_system_prompt(Registry([])).lower()
    assert "draft" in low
    assert "profile" in low                          # "don't build a profile"
    assert "store nothing" in low or "retain nothing" in low  # named as the claim to AVOID
    assert "encrypted conversation" in low
    assert "configured retention period" in low
    assert "deletion through the channel controls" not in low
    assert "don't carry a chat history between separate conversations" not in low


def test_system_prompt_sets_plain_language_reading_level():
    low = build_system_prompt(Registry([])).lower()
    assert "reading level" in low
    assert "6th" in low and "8th" in low


def test_contested_legal_matter_protocol_lives_in_its_one_home():
    # Red-team MC03/MC04/FP02/ES03 fix, rehomed by the prompt diet (2026-07-22): the FULL
    # contested-legal protocol (never restate court/holding/scope from news, never struck
    # down/annulled, lead with what currently stands, active-legal-challenge framing) now
    # lives ONCE, in the recent_developments tool description, the point of decision. The
    # stable prompt keeps only the short never-cross line. Both are pinned so the policy
    # can never silently vanish from either surface.
    from heynyc.core.tools.web_search import web_search_tools

    tools = {t.name: t for t in web_search_tools(["nyc.gov"], {}, set())}
    desc = tools["recent_developments"].description.lower()
    assert "contested legal matter" in desc
    assert "struck" in desc and "annulled" in desc
    assert "currently stands" in desc
    assert "active legal challenge" in desc
    assert "never name the court" in desc or "characterize the outcome" in desc

    low = build_system_prompt(Registry([])).lower()
    assert "currently stands" in low                      # the never-cross line survives
    assert "never a repeal" in low
    assert "recent_developments" in low                   # rule 9 points at the one home


def test_system_prompt_emergency_no_medical_dosing():
    # Red-team ES09 fix: an emergency (heart-attack signs) gets 911 only — NO drug names or dosages,
    # not even aspirin. The English path already held; this makes the standard explicit for all langs.
    low = build_system_prompt(Registry([])).lower()
    assert "911" in low
    assert "no medical instructions" in low or "no dosages" in low
    assert "aspirin" in low  # named as the exact thing NOT to dose


def test_system_prompt_public_charge_snap_guardrail():
    # Keep public charge available as a retrieved topic-specific answer, not as a detailed answer
    # injected into every conversation. The latter hijacked a Bengali SNAP-loss question.
    low = build_system_prompt(Registry([])).lower()
    assert "public charge" in low
    assert "only when the resident asks" in low
    assert "never introduce immigration or public charge" in low
    assert "retrieve current official guidance" in low
    assert "snap (food stamps)" not in low
    assert "cash assistance" not in low
    assert "institutional care" not in low


def test_system_prompt_same_discipline_in_every_language():
    # Red-team ES09/ES13 fix: EVERY safety + grounding rule holds identically in Spanish and any other
    # language (emergencies -> 911 no dosing; never fabricate a law number/citation; lead with the
    # standing protection). A prompt reinforcement only partially closes a model-level cross-lingual gap.
    low = build_system_prompt(Registry([])).lower()
    assert "every language" in low
    assert "law number" in low  # never invent a law number in any language
    # cites the concrete correct statute the Spanish ES13 failure fabricated ("Local Law 68")
    assert "local law 34" in low or "20-840" in low


def test_system_prompt_carries_ambient_equal_dignity_values_in_stable_tier():
    # RULED (2026-07-21): the equal-dignity values baseline is ambient in the standing prompt,
    # carried into every generated reply, not trapped in a canned denial template. Owner constraint:
    # NO named groups (legal exposure); open-ended in equal dignity, justice, and civil rights, and
    # never taking sides in or adjudicating a contested political or armed conflict.
    from heynyc.core.prompts import build_system_prompt_tiers

    stable, volatile = build_system_prompt_tiers(Registry([]))
    low = stable.lower()
    assert "equal dignity" in low
    assert "civil rights" in low
    assert "never take sides" in low
    assert "contested" in low
    # No named groups anywhere in the values baseline.
    assert "palestinian" not in low
    assert "jewish" not in low
    assert "israel" not in low
    # Ambient in the cacheable stable prefix, not the volatile per-turn suffix.
    assert "equal dignity" not in volatile.lower()


def test_system_prompt_teaches_per_turn_composition_in_stable_tier():
    # HYBRID (RULED 2026-07-21): composition guidance with the worked commute example lands in the
    # cacheable stable tier; it unlocks advisories+closures composition without a forced lane, and
    # keeps the retrieve-first floor for high-stakes situations in any language.
    from heynyc.core.prompts import build_system_prompt_tiers

    stable, volatile = build_system_prompt_tiers(Registry([]))
    low = stable.lower()
    assert "tool menu" in low
    assert "commute" in low
    assert "current advisories and disruptions" in low
    assert "cooling centers near their route" in low
    assert "official guidance first, in any language" in low
    # Stable, not volatile.
    assert "cooling centers near their route" not in volatile.lower()


# --- Change 1: progressive disclosure of the per-module detailed blurbs ---------------------------

def test_router_matches_module_on_curated_keyword():
    from heynyc.core.prompts import route_modules

    assert "food_pantries" in route_modules("where's the nearest food pantry?", _real_registry())


def test_router_loads_cooling_and_water_details_for_combined_request():
    from heynyc.core.prompts import route_modules

    matched = route_modules(
        "Where are the nearest cooling centers and water stations to Rockefeller Center?",
        _real_registry(),
    )

    assert {"cooling_centers", "drinking_fountains"} <= matched


def test_phone_style_limits_combined_default_without_capping_user_requests():
    from heynyc.core.prompts import BASE_SYSTEM_PROMPT

    assert "about 6 across categories" in BASE_SYSTEM_PROMPT
    assert "honor the user's requested count" in BASE_SYSTEM_PROMPT


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
    assert "cool_options_lookup" not in prompt             # cooling blurb text
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
    assert "cool_options_lookup" in prompt             # cooling blurb
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


def test_static_conversation_and_language_rules_live_in_the_stable_prefix():
    # Cache-layout fix (2026-07-21): the conversation-interpretation and reply-language rules are
    # byte-static (query- and time-independent), so they belong in the cacheable stable prefix, not
    # in the volatile suffix that changes every turn. Only the true mutables (the date line and the
    # query-selected blurbs) stay in the volatile suffix.
    from heynyc.core.prompts import build_system_prompt_tiers

    stable, volatile = build_system_prompt_tiers(
        _real_registry(), query="where's the nearest food pantry?")
    assert "Interpret the latest message using the conversation" in stable
    assert "Reply in the same language as the resident" in stable
    # the static rules are NOT duplicated into the volatile suffix
    assert "Interpret the latest message using the conversation" not in volatile
    assert "Reply in the same language as the resident" not in volatile
    # the volatile suffix is only the true mutables: the date line and the selected blurbs
    assert "Current date & time" in volatile
    assert "nearest_food_pantry(near=" in volatile
    assert "Current date & time" not in stable
    assert "nearest_food_pantry(near=" not in stable


def test_conversation_rules_preserve_transform_only_followups_without_retrieval():
    from heynyc.core.prompts import build_system_prompt_tiers

    stable, _ = build_system_prompt_tiers(_real_registry())
    low = stable.lower()
    assert "translate, repeat, shorten, or reformat" in low
    assert "do not call a discovery or retrieval tool" in low
    assert "preserve the same items" in low
    assert "retrieve again only when the resident asks for updated or new facts" in low
    assert "earlier answer lacks the evidence" in low
    assert "when a new or current factual answer is needed" in low


def test_capability_blurbs_only_filters_to_named_modules():
    reg = _real_registry()
    only = reg.capability_blurbs(only={"food_pantries"})
    assert "## food_pantries" in only
    assert "## benefits" not in only
    # the default (no filter) still returns every module's blurb (backward-compat)
    assert "## benefits" in reg.capability_blurbs()
