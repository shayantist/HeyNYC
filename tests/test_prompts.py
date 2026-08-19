from __future__ import annotations

import inspect
from datetime import datetime
from zoneinfo import ZoneInfo

from heynyc.core import config
from heynyc.core.prompts import build_system_prompt
from heynyc.core.registry import Registry


def _real_registry() -> Registry:
    """The live, module-discovered registry with real capability blurbs."""
    return Registry.discover(config.MODULES_DIR, config.BASE_ALLOWLIST)


def test_system_prompt_injects_current_nyc_datetime():
    fixed = datetime(2026, 6, 28, 19, 30, tzinfo=ZoneInfo("America/New_York"))
    prompt = build_system_prompt(Registry([]), now=fixed)
    assert "Current date & time" in prompt
    assert "June 28, 2026" in prompt
    assert "America/New_York" in prompt
    # still carries the grounding rules
    assert "GROUND EVERYTHING" in prompt


def test_system_prompt_teaches_broad_search_for_an_ambiguous_reference():
    """An ambiguous reference starts with a short broad search, not the whole message."""
    prompt = build_system_prompt(Registry([]))
    low = prompt.lower()
    assert "ambiguous or unfamiliar reference" in low
    assert "start with one broad `web_search`" in low
    assert "reference itself plus at most a date or \"nyc\"" in low
    assert "not the resident's whole sentence" in low


def test_system_prompt_routes_retrieval_by_evidence_shape_instead_of_a_fixed_chain():
    low = build_system_prompt(Registry([])).lower()

    assert "first `index_search`" not in low
    assert "index_search" not in low
    assert "choose the available tool whose operation matches the evidence gap" in low


def test_system_prompt_does_not_date_filter_standing_official_guidance():
    low = build_system_prompt(Registry([])).lower()

    assert "standing official guidance without publication bounds" in low
    assert "recent change itself" in low


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


def test_system_prompt_includes_active_publication_freshness_check():
    # The freshness guard goes from passive date-stamping to an active check: on
    # time-sensitive law/policy/rights questions the agent uses the one web search tool.
    prompt = build_system_prompt(Registry([]))
    low = prompt.lower()
    assert "web_search" in prompt
    assert "published_after" in low
    assert "published_before" in low
    assert "publication date" in low
    assert "recent_developments" not in prompt
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
    # Grounding discipline has one rule: retrieve and cite, otherwise state the gap and route.
    low = build_system_prompt(Registry([])).lower()
    assert "general information, not an official ruling" not in low
    assert "if retrieval does not support the claim, state the gap" in low
    assert "an uncited substantive claim is not" in low
    assert "misconception" in low


def test_system_prompt_requires_citations_on_the_supported_sentence_or_bullet():
    low = build_system_prompt(Registry([])).lower()
    assert "same sentence or bullet" in low
    assert "elsewhere in a paragraph or list does not count" in low
    assert "separate facts from different sources into separate sentences or bullets" in low
    assert "one factual or procedural claim" in low
    assert "one factual claim per `groundedanswer` block" in low
    assert "never put braces around a url" in low


def test_system_prompt_describes_data_practices_accurately():
    # Product-policy details live behind the eagerly available documentation tool.
    low = build_system_prompt(Registry([])).lower()
    assert "about_heynyc" in low
    assert "product-policy questions from memory" in low
    assert "ssn" in low and "sensitive id" in low
    assert "encrypted conversation" not in low
    assert "configured retention period" not in low
    assert "delete my data" not in low


def test_system_prompt_sets_plain_language_reading_level():
    low = build_system_prompt(Registry([])).lower()
    assert "reading level" in low
    assert "6th" in low and "8th" in low


def test_contested_legal_matter_protocol_lives_in_global_policy():
    from heynyc.core.tools.web_search import web_search_tools

    tools = {t.name: t for t in web_search_tools(["nyc.gov"], {}, set())}
    desc = tools["web_search"].description.lower()
    assert "contested legal matter" not in desc
    assert "active legal challenge" not in desc

    low = build_system_prompt(Registry([])).lower()
    assert "currently stands" in low                      # the never-cross line survives
    assert "never a repeal" in low
    assert "recent_developments" not in low
    assert "web_search" in low


def test_web_search_query_contract_separates_independent_facts():
    from heynyc.core.tools.web_search import web_search_tools

    tool = web_search_tools(["nyc.gov"], {}, set())[0]
    description = tool.parameters["properties"]["query"]["description"].lower()

    assert "one fact-finding objective" in description
    assert "parallel" in description
    assert "independent" in description


def test_system_prompt_emergency_no_medical_dosing():
    # Red-team ES09 fix: an emergency (heart-attack signs) gets 911 only — NO drug names or dosages,
    # not even aspirin. The English path already held; this makes the standard explicit for all langs.
    low = build_system_prompt(Registry([])).lower()
    assert "911" in low
    assert "no medical instructions" in low or "no dosages" in low
    assert "aspirin" in low  # named as the exact thing NOT to dose


def test_system_prompt_does_not_generalize_unknown_medication_instructions():
    low = build_system_prompt(Registry([])).lower()

    assert "unknown medication" in low
    assert "do not infer its instructions from other drugs" in low
    assert "dispensing pharmacist or prescriber" in low
    assert "poison control" in low and "extra dose" in low


def test_system_prompt_does_not_make_personalized_medical_mobility_decisions():
    low = build_system_prompt(Registry([])).lower()

    assert "health condition or recovery" in low
    assert "do not recommend walking, driving, or another transport mode based on medical facts" in low
    assert "clinician's instructions" in low
    assert "verified logistical facts" in low
    assert "do not follow that limitation with a vehicle, escort, or walking recommendation" in low


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


def test_system_prompt_translates_resident_facing_source_language():
    low = build_system_prompt(Registry([])).lower()
    assert "translate resident-facing labels and suggested phrases" in low
    assert "required official keyword" in low
    assert "keep official names, addresses, and links exact" in low


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


def test_system_prompt_answers_broadly_without_topic_suppression():
    low = build_system_prompt(Registry([])).lower()

    assert "answer broadly" in low
    assert "do not suppress" in low
    assert "stay in scope" not in low
    assert "outside what you help with" not in low


def test_system_prompt_teaches_per_turn_composition_in_stable_tier():
    # Composition guidance belongs in the cacheable stable tier without a topic-specific example.
    from heynyc.core.prompts import build_system_prompt_tiers

    stable, volatile = build_system_prompt_tiers(Registry([]))
    low = stable.lower()
    assert "tool menu" in low
    assert "what each result means on its own and alongside the last one" in low
    assert "parallelize only independent tool calls" in low
    assert "return the final resident answer immediately" in low
    assert "official guidance first, in any language" in low
    assert "return the final resident answer immediately" not in volatile.lower()


def test_system_prompt_sequences_dependent_tool_calls():
    low = build_system_prompt(Registry([])).lower()
    assert "parallelize only independent tool calls" in low
    assert "wait for that result" in low


def test_system_prompt_stops_retrieval_once_requested_constraints_are_supported():
    low = build_system_prompt(Registry([])).lower()
    assert low.count("return the final resident answer immediately") == 1
    assert "call `final_answer`" not in low
    assert "call the real tool most likely to resolve that specific gap" in low
    assert "no available tool can resolve it" in low
    assert "supported information" in low
    assert "state the limit plainly" in low
    assert "do not repeat a search or fetch that already answered the same question" in low


def test_system_prompt_does_not_prescribe_removed_regeneration_or_deferred_module_tools():
    low = build_system_prompt(Registry([])).lower()

    assert "regenerated once" not in low
    assert "find_clinics" not in low
    assert "search_benefits" not in low
    assert "cooling centers near their route" not in low


def test_system_prompt_separates_partial_matches_from_exact_results():
    low = build_system_prompt(Registry([])).lower()

    assert "treat partial matches as alternatives, not exact matches" in low
    assert "do not infer a missing property" in low
    assert "related fallback" not in low
    assert "systemwide statement supports a specific location only" in low
    assert "names that location" in low


def test_system_prompt_preserves_source_time_and_population_scope():
    low = build_system_prompt(Registry([])).lower()

    assert "source's as-of date" in low
    assert "sample or shortlist" in low
    assert "complete population" in low
    assert "current, permanent, temporary, or citywide" in low
    assert "does not substantiate a premise" in low
    assert "do not assert the opposite" in low


def test_system_prompt_keeps_complaint_privacy_rules_procedure_specific():
    low = build_system_prompt(Registry([])).lower()

    assert "anonymous, confidential, or disclosed" in low
    assert "exact complaint or application procedure" in low
    assert "do not transfer a privacy rule" in low
    assert "if that procedure's source is silent" in low


def test_housing_privacy_situation_fetches_each_exact_complaint_page():
    _module, hint = _real_registry().situation_hints()["housing_complaint_identity"]

    assert hint.high_stakes is True
    assert "KA-02025" in " ".join(hint.urls)
    assert "KA-01036" in " ".join(hint.urls)
    assert any("heat-and-hot-water-information" in url for url in hint.urls)
    assert any("report-a-maintenance-issue" in url for url in hint.urls)
    assert "exact complaint type" in hint.reminder
    assert "source is silent" in hint.reminder
    assert "render=true" in hint.reminder
    assert "managing-agent contact" in hint.reminder
    assert "lead with the conflict" in hint.reminder.lower()
    assert "do not reconcile" in hint.reminder.lower()
    assert "for a heat complaint, fetch exactly" in hint.reminder.lower()
    assert "do not fetch the conversion page" in hint.reminder.lower()
    assert "each page's anonymity statement in its own sentence" in hint.reminder.lower()
    assert "conflict itself as a separate limitation" in hint.reminder.lower()


def test_chronic_repair_situation_uses_current_city_organizing_routes():
    _module, hint = _real_registry().situation_hints()["chronic_tenant_repairs"]

    assert hint.high_stakes is True
    assert any("talk-to-tenants" in url for url in hint.urls)
    assert any("partners-in-preservation" in url for url in hint.urls)
    assert "does not file" in hint.reminder
    assert "ordinary single-request lookup" in hint.reminder
    assert "load the parent housing capability" in hint.reminder.lower()
    assert "get_hpd_building_records" in hint.focus_tools
    assert "request number is supplied" in hint.reminder
    assert "one factual or procedural claim" in hint.reminder.lower()
    assert "call 311 ask for tenant helpline other questions" in hint.reminder.lower()
    assert "do not add a recordkeeping checklist" in hint.reminder.lower()
    assert "state the read-only limitation separately" in hint.reminder.lower()
    assert "do not name program partners" in hint.reminder.lower()
    assert "report only the tenant helpline sentence" in hint.reminder.lower()
    assert "do not add a purpose qualifier" in hint.reminder.lower()
    assert "do not state common-area or retaliation details" in hint.reminder.lower()
    assert "load_capability(id=\"nyc311_status\")" in hint.reminder
    assert "do not use search_tools" not in hint.reminder
    assert 'complaint_terms=["rodent"]' in hint.reminder
    assert 'evidence_scope="tenant helpline route"' in hint.reminder.lower()


def test_311_status_hands_unresolved_building_repairs_to_composed_records():
    module = next(
        module for module in _real_registry().modules
        if module.name == "nyc311_status"
    )
    low = module.prompt.lower()

    assert "housing-chronic-tenant-repairs" in low
    assert "do not also search nearby" in low
    assert "ordinary status lookup" in low
    assert "30-day" in low
    assert "800-meter" in low
    assert "within_days" in low
    assert "radius_meters" in low


def test_system_prompt_preserves_official_handoff_without_reasking_named_landmarks():
    low = build_system_prompt(Registry([])).lower()

    assert "best retrieved official next step" in low
    assert "named venue or landmark" in low
    assert "already-supplied endpoint" in low


def test_system_prompt_minimizes_missing_attachment_recovery():
    low = build_system_prompt(Registry([])).lower()
    assert "attachment was not received" in low
    assert "paste only the redacted text" in low
    assert "redacted image or text summary" not in low
    assert "case or client numbers" in low
    assert "never ask for a full case number" in low


def test_phone_style_limits_combined_default_without_capping_user_requests():
    from heynyc.core.prompts import BASE_SYSTEM_PROMPT

    assert "about 6 across categories" in BASE_SYSTEM_PROMPT
    assert "honor the user's requested count" in BASE_SYSTEM_PROMPT


def test_legacy_module_guidance_is_complete_and_query_independent():
    assert "query" not in inspect.signature(build_system_prompt).parameters
    prompt = build_system_prompt(_real_registry())
    assert "find_foodhelp_locations(near=" in prompt      # food blurb
    assert "search_benefits(query=" in prompt         # benefits blurb
    assert "find_cool_options" in prompt             # cooling blurb
    assert "check_notify_nyc" in prompt                 # advisories blurb


def test_capability_menu_names_every_service_regardless_of_query():
    low = build_system_prompt(_real_registry()).lower()
    assert "cooling" in low
    assert "eviction" in low or "housing" in low
    assert "benefit" in low


def test_tiers_keep_date_and_module_blurbs_out_of_the_stable_prefix():
    from heynyc.core.prompts import build_system_prompt_tiers

    stable, volatile = build_system_prompt_tiers(
        _real_registry())
    # stable = safety rules + menu, query- and time-independent (the cacheable prefix)
    assert "GROUND EVERYTHING" in stable
    assert "Services you can help with (quick menu)" in stable
    assert "Current date & time" not in stable          # the date must NOT sit inside the cached prefix
    assert "find_foodhelp_locations(near=" not in stable     # module blurbs are volatile, not cached
    # volatile = every module blurb + the date line
    assert "Current date & time" in volatile
    assert "find_foodhelp_locations(near=" in volatile
    assert "find_cool_options" in volatile
    assert "check_notify_nyc" in volatile


def test_static_conversation_and_language_rules_live_in_the_stable_prefix():
    # Cache-layout fix (2026-07-21): the conversation-interpretation and reply-language rules are
    # byte-static (query- and time-independent), so they belong in the cacheable stable prefix, not
    # in the volatile suffix that changes every turn. Only the true mutables (the date line and the
    # module blurbs) stay in the volatile suffix.
    from heynyc.core.prompts import build_system_prompt_tiers

    stable, volatile = build_system_prompt_tiers(
        _real_registry())
    assert "Interpret the latest message using the conversation" in stable
    assert "Reply in the same language as the resident" in stable
    # the static rules are NOT duplicated into the volatile suffix
    assert "Interpret the latest message using the conversation" not in volatile
    assert "Reply in the same language as the resident" not in volatile
    # the volatile suffix contains the date line and module blurbs
    assert "Current date & time" in volatile
    assert "find_foodhelp_locations(near=" in volatile
    assert "Current date & time" not in stable
    assert "find_foodhelp_locations(near=" not in stable


def test_conversation_rules_preserve_transform_only_followups_without_retrieval():
    from heynyc.core.prompts import build_system_prompt_tiers

    stable, _ = build_system_prompt_tiers(_real_registry())
    low = stable.lower()
    assert "translate, repeat, shorten, or reformat" in low
    assert "do not call a discovery or retrieval tool" in low
    assert "preserve the same items" in low
    assert "inclusive or exclusive boundary" in low
    assert "negation, quantity, date, and eligibility condition" in low
    assert "retrieve again only when the resident asks for updated or new facts" in low
    assert "earlier answer lacks the evidence" in low
    assert "when a new or current factual answer is needed" in low
