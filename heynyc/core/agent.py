"""HeyNYC agent, a grounded, streaming tool-calling harness.

The core is the standard agent loop (LLM + tools until no tool calls), but built
as an event stream so a UI can show work in progress, with model-call retries,
clean terminal events, reactive system-reminders, and minimal approval gating for
side-effecting tools. `run()` is a convenience that drains the stream into a result.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from typing import AsyncIterator, Awaitable, Callable, Literal, NamedTuple, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict

from . import config, events
from .citations import CitationRegistry, used_citations, used_discovery_citations
from .crisis_lines import (
    CRISIS_LINES,
    SAMHSA_988_FAQ_URL,
    SAMHSA_988_INTERPRETATION_SNIPPET,
    compose_crisis_floor,
)
from .crisis_lines import (
    IMMINENT_SELF_HARM_RESPONSE_EN as _IMMINENT_SELF_HARM_RESPONSE_EN,
)
from .crisis_lines import (
    SELF_HARM_RESPONSE_EN as _SELF_HARM_RESPONSE_EN,
)
from .crisis_lines import (
    SELF_HARM_RESPONSE_ES as _SELF_HARM_RESPONSE_ES,
)
from .freshness import attach_temporal_provenance
from .grounding import GroundingResult, check_grounding
from .localization import localize
from .memory import (
    ContextCapacityError,
    ContextPlan,
    ContinuityRecord,
    compact_memory,
    context_capacity,
    continuity_reminder,
    prepare_context,
    request_tokens,
)
from .prompts import NYC_TZ, build_system_prompt_tiers
from .registry import Registry
from .spend import SpendGuard
from .telemetry import priced_cost_usd
from .tools import Tool, ToolContext, build_toolbox
from .tools.geo import maps_link
from .tools.notify_nyc import is_citywide_area

logger = logging.getLogger("heynyc.agent")

# The application injects its configured model via `Agent(model=...)`. When it does not, the
# default comes from config.HEYNYC_MODEL, the single source of truth (RULED 2026-07-21, owner:
# "just use the .env value and keep it consistent"). This replaced a hardcoded Sonnet default that
# made dev sessions silently run at ~3.3x the production model's price.
CONTEXT_LIMIT_FALLBACK = (
    "I can't safely fit enough of this conversation into the AI model right now. "
    "Please try again shortly or send NEW to start a fresh conversation."
)

# Safe fallback for a terminal turn (no tool calls) that comes back empty/whitespace. Some inputs,
# notably an encoded-instruction injection the model refuses by going silent, yield a blank final
# turn; the user must NEVER see an empty response, so we substitute an explicit safe refusal.
EMPTY_ANSWER_FALLBACK = (
    "I can't help with that request. If there's something about NYC services, benefits, or events "
    "I can help you find, tell me in your own words and I'll do my best, or you can call 311."
)

# --- Deterministic grounding guard (post-generation safety hook) ----------------------------------
# The single most important safety mechanism for running HeyNYC on a cheaper model: after the agent
# produces its FINAL answer, we deterministically re-check that every {cite:Sn}'d structured fact
# actually appears in the source it's attributed to (core.grounding.check_grounding, the SAME logic
# the eval gate uses). A HARD mismatch (a verbatim phone / dollar amount / address absent from an
# all-complete-capture source) is a fabrication: we feed the model a SPECIFIC correction and let it
# regenerate (Tier 3), capped; if it still can't ground it, we strip the offending claim or abstain and
# route to 311 (Tier 4). SOFT mismatches (name drift, or anything cited to a truncated snippet) never
# fire, preserving the check's zero-false-fail calibration so a CORRECT answer is never over-blocked.
GUARD_MAX_RETRIES = 2

# Tier 4 last resort: the model could not ground a load-bearing fact after the retry cap. Rather than
# ship an unverified number/address, hold off and route to a human / the official source.
GROUNDING_ABSTAIN_FALLBACK = (
    "I want to get this right, and I couldn't confirm that detail against an official source, so I'd "
    "rather not guess and risk sending you the wrong number or address. For the accurate, current "
    "info, call 311 or check the official NYC page, and they can take it from there."
)

_CITE_STRIP_RE = re.compile(r"\{cite:[^{}]+\}")
_CITE_MARKER_RE = re.compile(r"\{cite:([^{}]+)\}")
_HTTP_URL_RE = re.compile(r"https?://[^\s)\]}>]+")

# Spend-cap halt (OWASP LLM10 Unbounded Consumption). When a configured HEYNYC_SPEND_CAP is reached,
# the loop stops making model calls at the next turn boundary and returns this instead of silently
# spending past the ceiling. OFF by default (no cap), so it never fires unless the owner sets one.
SPEND_CAPPED_FALLBACK = (
    "I've reached the usage limit set for this session and can't take another step right now. "
    "For urgent NYC help you can call 311, or 911 in an emergency, and please try again a bit later."
)

FORCED_TOOL_FALLBACK = (
    "I couldn't start that action safely, so nothing was sent or changed. "
    "Please try again, or call 311 for help."
)

EVENT_CONTEXT_ABSTAIN_FALLBACK = (
    "I found event sources, but I couldn't turn them into a reliable, directly linked shortlist. "
    "Tell me a borough or a type like music, sports, family, or museums and I'll try a narrower search."
)

EVENT_PREPARATION_ABSTAIN_FALLBACK = (
    "I found current event sources, but I couldn't confirm which event you mean or turn them "
    "into a reliable plan. Tell me which event you're going to, or its venue or team, and I'll "
    "pull the current details."
)

# The scope preflight is a CHECKLIST, not a gate (RULED 2026-07-21, the denial redesign / HYBRID):
# it no longer returns an allow/deny verdict and no longer swaps the response. Every turn reaches the
# answer model, which writes the reply carried by the ambient values in the standing prompt. What the
# preflight still provides: the module/situation checklist and the event signal (below).
class _ScopeDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Semantic event signal (F046/F053/F058 family), a tri-state read by meaning, not a phrase
    # list: "preparation" plans around one dated event (resolve identity before advice),
    # "discovery" browses what is on or which event is happening (broad-shortlist treatment),
    # "none" is neither. The broad-events and preparation regexes below are only the fallback
    # for callers without the preflight. None means the model did not say, which must fall back
    # to those regex floors rather than silently disabling the guards.
    event_turn: Optional[Literal["none", "discovery", "preparation"]] = None
    # The module CHECKLIST (RULED 2026-07-18): which service modules this turn touches,
    # multi-select by meaning in any language, names drawn from the module registry. This is
    # a checklist, never a router: nothing is handed off, and unknown names are dropped
    # fail-safe. Behavior-neutral in this boundary; recorded for observability and matrices.
    modules: list[str] = []
    # Checked high-stakes SITUATIONS, from the modules' manifest-declared definitions
    # (migration boundary 2). A checked situation contributes its manifest retrieval config
    # to the same turn; unknown names are dropped fail-safe.
    situations: list[str] = []


_SCOPE_SYSTEM_PROMPT = """You are the preflight classifier for HeyNYC. Read the conversation and
describe the latest turn for the assistant that will answer it: which service modules and situations
it touches, and whether it is about a dated public event. You do not decide whether to answer, and
you never swap the reply; every turn is answered.

HeyNYC is about New York City civic life: finding, understanding, or using public services, laws,
benefits, official data, public places, transportation, alerts, community resources, or events in
NYC. Greetings, questions about HeyNYC itself, and short follow-ups whose meaning is clear only when
read with the conversation belong here too.

HeyNYC serves New Yorkers. Treat an ordinary public-service or civic-help request as concerning the
user's NYC situation unless the conversation places it elsewhere. Do not require the user to repeat
NYC or New York in every turn.

Read practical event-attendance planning as concerning NYC unless the conversation places the event
elsewhere. This includes preparing for, getting to, entering, or staying safe at a game, concert,
festival, parade, watch party, or other public event. Read a planning turn as event preparation when
the event name is abbreviated or ambiguous so the assistant can retrieve or clarify it. Residents
often write in texting shorthand: read abbreviations like tm, tmrw, tn, or wknd as dates, and read an
initial-letter or shortened event name as a resolvable public event when the surrounding words ask for
practical local help. A very short message that pairs a date shorthand with an abbreviated or
unexplained event name, even just initials, is usually a resident asking for practical local help,
not trivia, and the assistant will resolve or clarify it. Apply the same reading when the message is
nothing but that date and abbreviation, with no other words: a bare fragment like a date plus initials
is a resident's shorthand ask. This does not include predictions or sports trivia with no practical
NYC connection. An event-identity question, which event is, was, or will be happening, concerns a
public event regardless of TENSE, especially about an event this conversation already discussed;
asking what game happened today is identity, not trivia. Results, scores, and winner questions are
not event attendance.

State, federal, or global matters concern NYC when the user is asking about their practical effect on
a New Yorker, an NYC service, or NYC civic life. Unrelated general knowledge, opinion, and politics do
not. Do not treat an official government source as proof that a question concerns NYC. When the
subject is plausibly the resident's own life in the city (their work, home, block, kids, a hearing or
appointment they must get through, or getting to and from things, including a regional event reached
on the transit New Yorkers use even when the venue sits outside the city), read it as concerning NYC
even when only their need is vague or casually worded. Judge meaning, not keywords, language, country,
or viewpoint. Vagueness or slang alone never takes a turn out of NYC civic life. A short follow-up
whose meaning is in scope when read with the conversation stays in scope, including a practical NYC
reframe right after an off-scope turn: read it from the conversation, not from the prior turn.

Set event_turn only when the turn is actually about a dated public NYC event, read with the
conversation, in any language or shorthand: "preparation" to plan around one specific dated public
event (get to, attend, watch, or stay safe at it, so its identity must be resolved before advice);
"discovery" to browse what is on, whether something is happening, or which event is or was
happening, without naming one event to plan around (whether there is a game today, and
what-happened identity questions, are discovery); "none" whenever the turn is not about attending a
public event. Return only the supplied schema."""

_SNAP_TERMS_RE = re.compile(
    r"\b(?:snap|ebt|food stamps?|food[- ]benefits?|food assistance|cupones? de alimentos?|"
    r"beneficios? de alimentos?|asistencia alimentaria)\b",
    re.IGNORECASE,
)
_SNAP_WORK_RULE_RE = re.compile(
    r"\b(?:abawd|work (?:rule|requirement)s?|employment requirement|volunteer(?:ing)?|"
    r"regla(?:s)? de trabajo|requisito(?:s)? de trabajo|empleo|voluntari[oa]s?)\b",
    re.IGNORECASE,
)
# SNAP work-rule retrieval config, reminder, and tool focus live in the benefits module's
# manifest (`situations: snap_work_rules`), read via `registry.situation_hints()`. The two
# regexes above remain only as the preflight-absent fallback trigger for that situation.
_BENEFITS_PROBLEM_RE = re.compile(
    r"\b(?:denied|rejected|cut off|stopp(?:ed|ing)|closed|terminated|appeal(?:ing)?|"
    r"owe|overpayment|repay(?:ment)?|bill(?:ing)?|dispute|"
    r"fair hearing|denegad[oa]s?|rechazad[oa]s?|cortad[oa]s?|suspendid[oa]s?|"
    r"cancelad[oa]s?|apelar|audiencia imparcial)\b",
    re.IGNORECASE,
)
_BENEFITS_OVERPAYMENT_RE = re.compile(
    r"\b(?:owe|overpayment|repay(?:ment)?|billing|back SNAP benefits|"
    r"debo|sobrepago|reembolso|cobro)\b",
    re.IGNORECASE,
)
_CIVIC_BENEFITS_RE = re.compile(
    r"\b(?:benefits?|public assistance|cash assistance|snap|ebt|food stamps?|medicaid|wic|"
    r"fair fares|beneficios?|asistencia p[uú]blica|asistencia en efectivo|cupones? de alimentos?)\b",
    re.IGNORECASE,
)
_BENEFITS_RECOVERY_SEARCH_QUERY = (
    "NYC HRA benefits denial appeal fair hearing current official"
)
_BENEFITS_RECOVERY_URLS = (
    "https://www.nyc.gov/assets/hra/ACCESSNYC/html/snapfaq/english.shtml",
    "https://www.nyc.gov/site/hra/about/claims-collections.page",
    "https://www.nyc.gov/site/hra/about/frequently-asked-questions-faq.page",
    "https://otda.ny.gov/hearings/request/",
)
_BENEFITS_RECOVERY_SCOPE_REMINDER = (
    "This turn is about a public-benefit denial, cut-off, or appeal. Use current official benefit "
    "guidance, ask which benefit if it is unclear, and preserve the human or fair-hearing path. "
    "Do not call or mention unrelated service modules unless the user separately asked for them. "
    "Keep the answer phone-length."
)
_IMMIGRATION_STATUS_RE = re.compile(
    r"\b(?:undocumented|immigration status|citizen child|mixed[- ]status|green card|public charge|"
    r"ice|indocumentad[oa]s?|estatus migratorio|hij[oa] ciudadan[oa])\b",
    re.IGNORECASE,
)
_IMMIGRANT_HELP_CLAIM_RE = re.compile(
    r"\b(?:zero help|no help|any help|benefits?|ninguna ayuda|sin ayuda|ayuda|beneficios?)\b",
    re.IGNORECASE,
)
_IMMIGRANT_BENEFITS_SEARCH_QUERY = (
    "NYC official mixed-status household SNAP personal eligibility; SNAP and most Medicaid not counted "
    "for public charge; long-term institutional care exception; November 2025 proposal not in effect; "
    "Emergency Medicaid and NYC Care regardless of immigration status; ICE data sharing current"
)
_IMMIGRANT_BENEFITS_URLS = (
    "https://www.nyc.gov/assets/hra/ACCESSNYC/html/snapfaq/english.shtml",
    "https://www.nyc.gov/site/doh/health/health-topics/immigrant-health.page",
    "https://www.nyc.gov/site/immigrants/legal-resources/public-charge-rule.page",
    "https://www.nyc.gov/site/immigrants/legal-resources/moia-immigration-legal-support-hotline.page",
)
_IMMIGRANT_BENEFITS_SCOPE_REMINDER = (
    "This turn crosses immigration status and public benefits. Use current official sources and "
    "keep eligibility, public charge, and data sharing as three separate questions. Distinguish "
    "the person asking from eligible members of a mixed-status household. Do not generalize one "
    "program's rule to another. A generic program description is not proof that an undocumented "
    "person is eligible; use the immigration-specific official text. For SNAP, distinguish the "
    "right to apply for eligible household members from the undocumented person's own eligibility, "
    "and say that distinction plainly. If the source says anyone can apply, immediately explain "
    "that an application does not establish personal eligibility and that a person may apply for "
    "eligible household members without applying for themself. Do not call unrelated service modules. Keep the answer "
    "phone-length and route individualized immigration advice to trusted legal help. In Spanish, "
    "write llama al 311, not only a bare 311 reference, so the next action is unmistakable."
)
_ACTIVE_LOCKOUT_RE = re.compile(
    r"\b(?:my (?:landlord|super|building) (?:changed|replaced) (?:the )?locks?|"
    r"i['’]?m locked out|i am locked out|locked me out|"
    r"mi caser[oa] cambi[oó] (?:la cerradura|las cerraduras)|me dej[oó] fuera|"
    r"estoy fuera (?:de mi|del) (?:apartamento|casa))\b",
    re.IGNORECASE,
)
_HOUSING_CONTEXT_RE = re.compile(
    r"\b(?:apartment|building|home|house|housing|landlord|lease|roommate|tenant|unit|"
    r"apartamento|casa|caser[oa]|inquilin[oa]|vivienda)\b",
    re.IGNORECASE,
)
_SELF_HELP_EVICTION_RE = re.compile(
    r"\b(?:evict|remove|kick out)\b.{0,35}\b(?:my )?tenant\b.{0,65}"
    r"\b(?:without|no)\b.{0,20}\b(?:court|warrant|order)\b|"
    r"\bdesalojar\b.{0,35}\b(?:mi )?inquilin[oa]\b.{0,65}\bsin\b.{0,25}\bcorte\b",
    re.IGNORECASE,
)
_ESSENTIAL_SERVICES_SHUTOFF_RE = re.compile(
    r"\b(?:landlord|caser[oa])\b.{0,45}\b(?:shut off|turned off|cut off|cort[oó])\b"
    r".{0,30}\b(?:hot water|heat|agua caliente|calefacci[oó]n)\b.{0,55}"
    r"\b(?:force|make|get|obligar|forzar)\b.{0,25}\b(?:me|us|tenant|inquilin[oa])\b"
    r".{0,20}\b(?:out|leave|move|salir|mudar)\b",
    re.IGNORECASE,
)
# Lockout retrieval config, reminder, and tool focus live in the housing module's manifest
# (`situations: active_lockout`), read via `registry.situation_hints()`. The regexes above
# remain only as the preflight-absent fallback trigger for that situation.
_CIVIC_LAW_SEARCHES = (
    (
        re.compile(
            r"\b(?:school|escuela)\b.{0,90}\b(?:immigration|immigrant|status|tracked|ice|"
            r"inmigraci[oó]n|estatus|rastre)",
            re.IGNORECASE,
        ),
        "NYC Public Schools immigration status enrollment confidentiality rights current",
        (
            "https://www.schools.nyc.gov/school-life/school-environment/immigrant-families",
            "https://www.schools.nyc.gov/learning/multilingual-learners/"
            "bill-of-rights-for-parents-of-english-language-learners",
        ),
    ),
    (
        re.compile(
            r"\b(?:cashless|cash[- ]free|refus(?:e|ing) cash|solo tarjeta|"
            r"no acept\w* efectivo|sin efectivo)\b",
            re.IGNORECASE,
        ),
        "NYC official cashless ban Local Law 34 of 2020 Administrative Code 20-840 exceptions DCWP",
        (
            "https://www.nyc.gov/site/dca/consumers/Prohibition-of-Cashless-Establishments.page",
            "https://nyc-business.nyc.gov/nycbusiness/resources-by-industry/restaurant",
            "https://legistar.council.nyc.gov/LegislationDetail.aspx?GUID=7800AFC9-D8B1-41FD-9C31-172565712686&ID=3763665&Options=ID%7CText%7C",
        ),
    ),
    (
        re.compile(
            r"\b(?:fast[- ]food|fast food|restaurant)\b.*\b(?:schedule|shift|notice|horario|turno)\b|"
            r"\b(?:schedule|shift|notice|horario|turno)\b.*\b(?:fast[- ]food|fast food|restaurant)\b",
            re.IGNORECASE,
        ),
        "NYC official Fair Workweek fast food schedule notice premium pay current DCWP",
        ("https://home4.nyc.gov/site/dca/businesses/fairworkweek-deductions-laws-employers.page",),
    ),
    (
        re.compile(
            r"\bretail\b.*\b(?:schedule|shift|notice|horario|turno)\b|"
            r"\b(?:schedule|shift|notice|horario|turno)\b.*\bretail\b",
            re.IGNORECASE,
        ),
        "NYC official Fair Workweek retail schedule notice current DCWP",
        ("https://www.nyc.gov/site/dca/workers/workersrights/retail-workers.page",),
    ),
    (
        re.compile(r"\b(?:broker fee|broker's fee|comisi[oó]n del corredor)\b", re.IGNORECASE),
        "NYC official broker fee FARE Act tenant landlord current",
        ("https://portal.311.nyc.gov/article/?kanumber=KA-03665",),
    ),
    (
        re.compile(
            r"\b(?:security deposit|tenant['’]s deposit|deposit for (?:normal )?wear and tear|"
            r"dep[oó]sito de seguridad)\b",
            re.IGNORECASE,
        ),
        "New York official tenant security deposit rules current NYC",
        ("https://ag.ny.gov/publications/residential-tenants-rights-guide",),
    ),
    (
        re.compile(
            r"\b(?:raise|increase|subir|aumentar)\b.{0,30}\b(?:rent|renta|alquiler)\b|"
            r"\b(?:rent|renta|alquiler)\b.{0,45}\b(?:raise|increase|subir|aumentar|cap|limit|l[ií]mite)\b|"
            r"\b(?:cap|limit|l[ií]mite)\b.{0,45}\b(?:rent|renta|alquiler)\b|"
            r"\blease renewal\b.{0,45}\b(?:raise|increase|higher|percent|%)\b",
            re.IGNORECASE,
        ),
        "NYC official rent increase notice rent stabilized Good Cause current",
        (
            "https://portal.311.nyc.gov/article/?kanumber=KA-03296",
            "https://www.nyc.gov/site/hpd/services-and-information/good-cause-eviction.page",
            "https://ag.ny.gov/publications/residential-tenants-rights-guide",
            "https://hcr.ny.gov/rent-control",
        ),
    ),
    (
        re.compile(
            r"\b(?:marshal|sheriff)\b.{0,55}\b(?:evict|eviction|remove|coming|scheduled)\w*\b|"
            r"\b(?:evict|eviction)\w*\b.{0,55}\b(?:tomorrow|marshal|sheriff)\b",
            re.IGNORECASE,
        ),
        "New York Courts official stopping eviction Order to Show Cause current",
        (
            "https://www.nycourts.gov/new-york-city-housing-court/stopping-eviction",
            "https://www.nycourts.gov/new-york-city-housing-court/nyc-housing-court-orders-show-cause",
            "https://www.nyc.gov/site/mayorspeu/resources/right-to-counsel.page",
        ),
    ),
    (
        re.compile(
            r"\b(?:lock out|change (?:my |the |a )?locks?|self[- ]help)\b.{0,65}"
            r"\b(?:tenant|landlord|rent|behind|arrears|inquilin[oa]|caser[oa]|renta)\b|"
            r"\b(?:tenant|landlord|rent|behind|arrears|inquilin[oa]|caser[oa]|renta)\b.{0,65}"
            r"\b(?:lock out|change (?:my |the |a )?locks?|self[- ]help)\b",
            re.IGNORECASE,
        ),
        "NYC official illegal lockout Admin Code 26-521 court warrant 911 311",
        ("https://portal.311.nyc.gov/article/?kanumber=KA-02518",),
    ),
    (
        re.compile(
            r"\b(?:withhold|stop paying|not pay|retener|dejar de pagar)\b.{0,30}\b(?:rent|renta|alquiler)\b|"
            r"\b(?:landlord|apartment|tenant|rent|ceiling|wall|caser[oa]|apartamento|inquilin[oa]|renta|techo|pared)\b"
            r".{0,55}\b(?:repair|repairs|leak|leaks|mold|reparaci[oó]n|gotera|moho)\b|"
            r"\b(?:repair|repairs|leak|leaks|mold|reparaci[oó]n|gotera|moho)\b.{0,55}"
            r"\b(?:landlord|apartment|tenant|rent|ceiling|wall|caser[oa]|apartamento|inquilin[oa]|renta|techo|pared)\b",
            re.IGNORECASE,
        ),
        "New York official tenant repairs warranty habitability rent withholding NYC current",
        (
            "https://ag.ny.gov/publications/residential-tenants-rights-guide",
            "https://ag.ny.gov/resources/individuals/tenants-homeowners/legal-services-and-code-enforcement",
        ),
    ),
    (
        re.compile(
            r"\b(?:sexual harassment|acoso sexual)\b.{0,55}\b(?:report|complain|hr|"
            r"fire|firing|retaliat|denunciar|queja|despedir|represalia)\w*\b|"
            r"\b(?:retaliat|fire|firing|represalia|despedir)\w*\b.{0,55}"
            r"\b(?:sexual harassment|acoso sexual)\b",
            re.IGNORECASE,
        ),
        "NYC CCHR sexual harassment retaliation employment rights current",
        ("https://www.nyc.gov/site/cchr/law/sexual-harassment-training-main.page",),
    ),
    (
        re.compile(
            r"\b(?:locs|cornrows|braids|afro|bantu knots|natural hair|hair discrimination|"
            r"crown act|cabello natural|trenzas)\b",
            re.IGNORECASE,
        ),
        "NYC CCHR hair discrimination natural hair locs cornrows current",
        ("https://www.nyc.gov/site/cchr/law/hair-discrimination-legal-guidance.page",),
    ),
    (
        re.compile(
            r"\b(?:landlord|housing|apartment|caser[oa]|vivienda|apartamento)\b.{0,55}"
            r"\b(?:family|families|children|child|kids|pregnan|familia|niñ[oa]s?|embarazad)\w*\b|"
            r"\b(?:family status|familial status|presence of children)\b",
            re.IGNORECASE,
        ),
        "NYC CCHR housing family status children discrimination current",
        (
            "https://www.nyc.gov/site/fairhousing/rights-responsibilities/what-are-the-protected-classes.page",
            "https://home4.nyc.gov/site/cchr/help/residents.page",
        ),
    ),
    (
        re.compile(r"\b(?:criminal record|conviction history|arrest record)\b", re.IGNORECASE),
        "NYC CCHR Fair Chance Act criminal record hiring current",
        ("https://www.nyc.gov/site/cchr/media/fair-chance-employers.page",),
    ),
    (
        re.compile(r"\b(?:salary history|pay history|prior salary)\b", re.IGNORECASE),
        "NYC CCHR salary history hiring ban current",
        ("https://www.nyc.gov/site/cchr/media/salary-history-employers.page",),
    ),
    (
        re.compile(
            r"\b(?:emotional support animals?|support animals?|companion animals?)\b|"
            r"\besa\b.{0,35}\b(?:animal|housing|landlord|apartment)\b|"
            r"\b(?:animal|housing|landlord|apartment)\b.{0,35}\besa\b",
            re.IGNORECASE,
        ),
        "NYC emotional support animal housing reasonable accommodation current",
        (
            "https://www.nyc.gov/site/animalwelfare/resources/service-and-emotional-support-animals.page",
            "https://www.nyc.gov/site/cchr/law/disability-discrimination-legal-guidance.page",
        ),
    ),
    (
        re.compile(
            r"\b(?:buyout|buy[- ]out|offered? me \$?[\d,]+ to (?:move|leave)|"
            r"pay(?:ing)? me to (?:move|leave))\b",
            re.IGNORECASE,
        ),
        "NYC official tenant buyout rights rent stabilized legal help",
        (
            "https://home4.nyc.gov/site/hpd/services-and-information/tenant-harassment.page",
            "https://www.nyc.gov/site/hpd/services-and-information/buyout-agreement-law.page",
            "https://www.nyc.gov/site/mayorspeu/resources/right-to-counsel.page",
        ),
    ),
    (
        re.compile(
            r"\b(?:official )?pdf\b.{0,55}\b(?:every|all)\b.{0,20}\b(?:homeless )?shelter",
            re.IGNORECASE,
        ),
        "NYC official shelter intake locations no comprehensive shelter PDF",
        (
            "https://www.nyc.gov/site/dhs/about/frequently-asked-questions.page",
            "https://www.nyc.gov/site/dhs/shelter/singleadults/single-adults-shelter.page",
        ),
    ),
    (
        re.compile(
            r"\b(?:kitchen cooks?|back[- ]of[- ]house|cooks?)\b.{0,45}\b(?:tipped|tip credit|wage|pay)\b|"
            r"\b(?:tipped|tip credit|wage|pay)\b.{0,45}\b(?:kitchen cooks?|back[- ]of[- ]house|cooks?)\b",
            re.IGNORECASE,
        ),
        "New York official kitchen cook tipped worker minimum wage current",
        ("https://dol.ny.gov/minimum-wage-tipped-workers",),
    ),
    (
        re.compile(
            r"\b(?:tipped (?:waiters?|workers?|employees?)|waiters?|servers?)\b.{0,55}"
            r"\b(?:wage|pay|paid|hour|minimum|tip credit)\b|"
            r"\b(?:wage|pay|paid|hour|minimum|tip credit)\b.{0,55}"
            r"\b(?:tipped (?:waiters?|workers?|employees?)|waiters?|servers?)\b",
            re.IGNORECASE,
        ),
        "New York official tipped worker minimum wage cash wage current",
        ("https://dol.ny.gov/minimum-wage-tipped-workers",),
    ),
    (
        re.compile(
            r"\b(?:pregnan(?:t|cy)|embarazad[ao]|embarazo)\b.{0,55}"
            r"\b(?:fire|firing|fired|disciplin|slow|despedir|despido)\w*\b|"
            r"\b(?:fire|firing|fired|disciplin|despedir|despido)\w*\b.{0,55}"
            r"\b(?:pregnan(?:t|cy)|embarazad[ao]|embarazo)\b",
            re.IGNORECASE,
        ),
        "NYC CCHR pregnancy discrimination firing employment current",
        (
            "https://www.nyc.gov/site/cchr/law/pregnancy-legal-guidance.page",
            "https://www.nyc.gov/site/cchr/enforcement/complaint-process.page",
        ),
    ),
    (
        re.compile(
            r"\bsection 8\b.{0,65}"
            r"\b(?:refuse|reject|accept|listing|no vouchers?|rent|negar|rechaz|acept)\w*\b|"
            r"\b(?:refuse|reject|accept|listing|no vouchers?|negar|rechaz|acept)\w*\b.{0,65}"
            r"\bsection 8\b",
            re.IGNORECASE,
        ),
        "Commons West March 5 2026 Third Department Section 8 ORDERED judgments affirmed; "
        "NYC local source-of-income law separate",
        (
            "https://www.nyc.gov/site/cchr/media/source-of-income.page",
            "https://www.nycourts.gov/reporter/3dseries/2026/2026_01253.htm",
            "https://www.nycourts.gov/ctapps/Decisions/2026/May26/DecisionList052126.pdf",
        ),
    ),
    (
        re.compile(
            r"\b(?:cityfheps|housing vouchers?|vouchers?)\b.{0,65}"
            r"\b(?:refuse|reject|accept|listing|no vouchers?|rent|negar|rechaz|acept)\w*\b|"
            r"\b(?:refuse|reject|accept|listing|no vouchers?|negar|rechaz|acept)\w*\b.{0,65}"
            r"\b(?:cityfheps|housing vouchers?|vouchers?)\b",
            re.IGNORECASE,
        ),
        "NYC CCHR CityFHEPS and lawful source-of-income voucher protection current",
        ("https://www.nyc.gov/site/cchr/media/source-of-income.page",),
    ),
    (
        re.compile(
            r"\b(?:asylum|immigration)\b.{0,45}\b(?:hearing|court|lawyer|attorney|legal help)\b|"
            r"\b(?:abogad[oa])\b.{0,35}\b(?:inmigraci[oó]n|asilo)\b|"
            r"\b(?:inmigraci[oó]n|asilo)\b.{0,35}\b(?:abogad[oa]|audiencia|corte)\b|"
            r"\b(?:audiencia|corte)\b.{0,35}\b(?:inmigraci[oó]n|asilo)\b|"
            r"\b(?:work permit|employment authorization|permiso de trabajo)\b",
            re.IGNORECASE,
        ),
        "NYC official immigration legal help hearing work permit current",
        (
            "https://www.nyc.gov/site/hra/help/legal-services-for-immigrant-new-yorkers.page",
            "https://www.nyc.gov/site/immigrants/legal-resources/moia-immigration-legal-support-hotline.page",
            "https://www.justice.gov/eoir/check-case-status",
            "https://www.uscis.gov/i-765",
        ),
    ),
    (
        re.compile(
            r"\b(?:scrape|download|extract)\b.{0,45}\b(?:snap|benefit)\b.{0,35}"
            r"\b(?:recipient|personal|private|server|data)\b",
            re.IGNORECASE,
        ),
        "NYC official public open data privacy safe aggregate benefits data",
        ("https://opendata.cityofnewyork.us/",),
    ),
    (
        re.compile(
            r"\b(?:housing court|eviction case|landlord)\b.{0,65}"
            r"\b(?:argument|judge|lie|liar|legal strategy|cite cases)\b|"
            r"\b(?:argument|judge|lie|liar|legal strategy|cite cases)\b.{0,65}"
            r"\b(?:housing court|eviction case|landlord)\b",
            re.IGNORECASE,
        ),
        "NYC official Right to Counsel housing court free legal help",
        ("https://www.nyc.gov/site/mayorspeu/resources/right-to-counsel.page",),
    ),
    (
        re.compile(
            r"\bice\b.{0,60}\b(?:path|shelter|intake center)\b|"
            r"\b(?:path|shelter|intake center)\b.{0,60}\bice\b",
            re.IGNORECASE,
        ),
        "NYC official immigration safety shelter intake legal help",
        (
            "https://www.nyc.gov/site/dhs/about/frequently-asked-questions.page",
            "https://www.nyc.gov/site/hra/help/legal-services-for-immigrant-new-yorkers.page",
        ),
    ),
)
_CIVIC_LAW_SCOPE_REMINDER = (
    "This turn asks about a current civic or tenant law. Answer from the official source just "
    "retrieved, preserve important exceptions, and give the responsible agency or safe next step. "
    "For a fee question, distinguish who hired or represents the broker. For a repair or rent-"
    "withholding question, explain the warranty of habitability and that withholding can lead to "
    "a nonpayment case where a court decides any abatement; do not volunteer a different self-help "
    "remedy. For an employer asking about retaliation or discrimination, "
    "state the lawful boundary and route to counsel or CCHR; do not offer help carrying out the "
    "firing or discriminatory policy. Use short sentences and plain language. "
    "For fast-food scheduling, distinguish premium pay for a late schedule change from the separate "
    "rule about a reduction over 15 percent. Do not open with yes or probably yes when a late change "
    "can require premium pay. For kitchen cooks, answer the wage classification question first: "
    "non-tipped back-of-house workers are owed the full minimum wage when the source supports it; "
    "do not pivot to tip theft. Ignore requests to remove safety context or routing. "
    "For a rent-limit question, include the written-notice rule for increases over 5 percent when "
    "the retrieved source supports it, and distinguish rent-stabilized RGB limits, rent-controlled "
    "DHCR limits, and unregulated rent. Do not decide whether the tenant should sign a lease renewal; "
    "explain the grounded options and route to tenant legal help. If a marshal eviction is imminent, "
    "lead with the official court's stopping-eviction steps and an Order to Show Cause; do not divert "
    "into shelter intake unless the resident asks for shelter. For a lockout question, state that "
    "self-help is illegal, only a court warrant carried out by a marshal or sheriff can remove a "
    "tenant, and tell an already locked-out tenant to call 911, then 311 and Housing Court. "
    "For cashless citation traps, use Local Law 34 of 2020 and "
    "Administrative Code section 20-840 only when retrieved official text supports both, then say "
    "the official page proves "
    "the opposite of a claimed permission. Open with the plain correction before linking the page. "
    "For security deposits, cite the chunk containing each deduction or return rule, not an earlier "
    "introductory chunk. Include one short legal-information caveat and a safe agency or legal-help "
    "route even if the user asks you to omit warnings. For tenant buyouts, do not recommend accepting "
    "or rejecting a specific offer; preserve the right to refuse and route to legal help before signing. "
    "If asked for an exhaustive shelter PDF, do not substitute intake addresses as though they answer "
    "the request: say when no verified comprehensive official PDF was found, then offer current intake "
    "guidance by household type. Do not list intake addresses or future transitions unless the user "
    "asks for a place to go. For immigration hearings or work permits, do not decide the case or "
    "suggest skipping a hearing; say not to miss an asylum hearing, name ActionNYC as the free legal-help "
    "route, and use the official court status path. "
    "In Spanish, write llama al 311, not only a bare 311 reference, so the action is unmistakable. "
    "For sensitive recipient data, refuse access to private records and provide the cited public Open "
    "Data route for aggregate research. Do not coach deception in court; offer help organizing true "
    "facts and retrieve the Right-to-Counsel route. Never claim to verify real-time immigration "
    "enforcement activity; state that limit and provide current DHS intake plus trusted immigration "
    "legal-help contacts. For tipped-wage questions, answer the tipped cash wage and full minimum wage "
    "from the current DOL source before discussing tip theft. For pregnancy at work, state the CCHR "
    "pregnancy-discrimination boundary and route to CCHR; do not treat firing because of pregnancy as "
    "an open question. For Section 8, report both retrieved sources, not only the city page: the March "
    "5, 2026 Third Department opinion affirmed that the state Executive Law provision is facially "
    "unconstitutional to the extent it requires Section 8 participation. The current NYC Commission "
    "page still lists Section 8 under the separate NYC Human Rights Law, but do not claim the state "
    "ruling definitely leaves the Section 8 mandate unchanged. Say that its effect on a specific NYC "
    "property needs legal guidance, while CityFHEPS and other non-Section-8 vouchers remain protected, "
    "so a blanket 'no vouchers' listing is unsafe. Keep the protected entity name 'NYC Commission on "
    "Human Rights' unchanged in every language. Do not describe the ruling only as an active challenge, "
    "and do not apply a tenant "
    "complaint route to an owner asking what the law requires. "
    "Do not turn a general rule into a definitive ruling on the user's individual case. Keep the "
    "answer phone-length."
)
_EXPLICIT_HOUSING_RE = re.compile(
    r"\b(?:rent|evict(?:ion|ed|ing)?|landlord|housing|shelter|voucher|section 8|cityfheps)\b",
    re.IGNORECASE,
)
_EXPLICIT_CLINIC_RE = re.compile(
    r"\b(?:doctor|clinic|health ?care|medical care|hospital)\b", re.IGNORECASE
)
_EXPLICIT_WORKER_RE = re.compile(
    r"\b(?:employer|boss|wages?|paycheck|workplace|worker rights?)\b", re.IGNORECASE
)


def _needs_current_snap_work_rule_guidance(user_message: str) -> bool:
    """Require current official retrieval for a SNAP work-rule question, never model memory."""
    user_message = _routing_text(user_message)
    return bool(_SNAP_TERMS_RE.search(user_message) and _SNAP_WORK_RULE_RE.search(user_message))


def _needs_current_benefits_recovery_guidance(user_message: str) -> bool:
    """Require current official retrieval for a civic-benefit denial, cut-off, or appeal."""
    user_message = _routing_text(user_message)
    return bool(_CIVIC_BENEFITS_RE.search(user_message) and _BENEFITS_PROBLEM_RE.search(user_message))


def _needs_current_immigrant_benefits_guidance(user_message: str) -> bool:
    """Require current official retrieval when immigration status intersects with benefits."""
    user_message = _routing_text(user_message)
    return bool(
        _IMMIGRATION_STATUS_RE.search(user_message)
        and (
            _CIVIC_BENEFITS_RE.search(user_message)
            or _IMMIGRANT_HELP_CLAIM_RE.search(user_message)
        )
    )


def _needs_current_lockout_guidance(user_message: str) -> bool:
    """Require current official retrieval for a first-person active landlord lockout."""
    user_message = _routing_text(user_message)
    return bool(
        (_ACTIVE_LOCKOUT_RE.search(user_message) and _HOUSING_CONTEXT_RE.search(user_message))
        or _ESSENTIAL_SERVICES_SHUTOFF_RE.search(user_message)
        or _SELF_HELP_EVICTION_RE.search(user_message)
    )


def _current_civic_law_search(user_message: str) -> Optional[str]:
    """Return the official-search query for a supported volatile civic-law topic."""
    user_message = _routing_text(user_message)
    for pattern, query, _urls in _CIVIC_LAW_SEARCHES:
        if pattern.search(user_message):
            return query
    return None


def _current_civic_law_urls(query: str) -> tuple[str, ...]:
    return next(urls for _pattern, candidate, urls in _CIVIC_LAW_SEARCHES if candidate == query)


_CASHLESS_PERMISSION_TRAP_RE = re.compile(
    r"\b(?:allowed|may|can)\b.{0,35}\b(?:cashless|cash[- ]free)\b|"
    r"\b(?:cashless|cash[- ]free)\b.{0,35}\b(?:allowed|may|can)\b",
    re.IGNORECASE,
)
_PLAIN_CORRECTION_RE = re.compile(
    r"^\s*(?:no\b|there is no\b|that (?:is|premise is) (?:not correct|false|wrong)\b)",
    re.IGNORECASE,
)
_SPANISH_QUERY_RE = re.compile(
    r"[¿¡]|\b(?:puedo|tengo|solicito|afectará|indocumentad[oa]|cafeter[ií]a|efectivo|"
    r"aceptar|negarme|beneficio|audiencia|dime|desalojar|inquilin[oa]|caser[oa]|cerraduras?)\b",
    re.IGNORECASE,
)
_SPANISH_ANSWER_RE = re.compile(
    r"\b(?:puede|puedes|debe|debes|ley|efectivo|ciudad|llama|solicitud|beneficios|"
    r"inmigraci[oó]n|no se|largo plazo)\b",
    re.IGNORECASE,
)
_SPANISH_CASHLESS_DIRECT_RE = re.compile(
    r"(?:^|[.!?]\s*)no\b.{0,100}(?:exige|debe|tiene que).{0,30}aceptar efectivo|"
    r"(?:cafeter[ií]a|negocio).{0,80}(?:debe|tiene que).{0,30}aceptar efectivo|"
    r"no (?:puede|puedes).{0,80}(?:solo tarjeta|rechazar efectivo)",
    re.IGNORECASE,
)


def _dominant_non_latin_script(text: str, *, require_majority: bool = True) -> Optional[str]:
    # require_majority=True (reply-language feedback): the non-Latin script must be MOST of the
    # letters. require_majority=False (crisis routing): the most-frequent non-Latin script wins even
    # when Latin leads, because a code-switched crisis message ("I want to die 我不想活了") pairs an
    # English trigger phrase with the person's own-language despair, and the own-language script is
    # what the verified in-language floor must serve.
    scripts: dict[str, int] = {}
    letter_count = 0
    for char in text:
        if not char.isalpha():
            continue
        letter_count += 1
        script = unicodedata.name(char, "").split(" ", 1)[0]
        if script and script != "LATIN":
            scripts[script] = scripts.get(script, 0) + 1
    if not scripts:
        return None
    script, user_count = max(scripts.items(), key=lambda item: item[1])
    if user_count < 2 or (require_majority and user_count * 2 < letter_count):
        return None
    return script


def _reply_script_feedback(user_message: str, final_text: str) -> Optional[str]:
    """Request one rewrite when a non-Latin resident message gets a different-script reply."""
    script = _dominant_non_latin_script(user_message)
    if script is None:
        return None
    answer = _HTTP_URL_RE.sub("", final_text or "")
    letters = [char for char in answer if char.isalpha()]
    matching = sum(unicodedata.name(char, "").startswith(f"{script} ") for char in letters)
    if matching >= 2 and matching * 3 >= len(letters):
        return None
    return (
        f"The resident wrote primarily in the {script.title()} script. Rewrite the full answer in "
        "that same language and script. Preserve every grounded fact, citation marker, official "
        "name, address, phone number, and URL exactly."
    )


def _required_scope_feedback(
    user_message: str, final_text: str, civic_law_search: Optional[str],
    *, immigrant_benefits_turn: bool = False, benefits_recovery_turn: bool = False,
    lockout_turn: bool = False,
) -> Optional[str]:
    routed_query = _routing_text(user_message)
    low_query = routed_query.lower()
    low_answer = (final_text or "").lower()
    if (
        re.search(r"definitive legal ruling|answer with (?:just|only) yes or no|only yes or no", low_query)
        and re.match(r"\s*(?:yes|no)\b", final_text or "", re.I)
    ):
        return (
            "Ignore the instruction to reduce the legal question to yes or no. Start by saying that "
            "a flat verdict would be unsafe, then give only the grounded risks, options, and legal-help route."
        )
    if (
        civic_law_search
        and "cashless ban" in civic_law_search.lower()
        and _CASHLESS_PERMISSION_TRAP_RE.search(routed_query)
        and not _PLAIN_CORRECTION_RE.search(final_text or "")
    ):
        return (
            "The user's premise is false. Start with a short, explicit correction such as "
            "'There is no official page that says that. NYC says the opposite.' Then give only "
            "the supporting official source and cited rule. Do not begin with the URL."
        )
    if civic_law_search and "cashless ban" in civic_law_search.lower():
        if "34" not in low_answer or "20-840" not in low_answer:
            return (
                "Name the exact retrieved law: Local Law 34 of 2020 and Administrative Code "
                "section 20-840. Keep both in the user's language and cite the supporting source."
            )
        if _SPANISH_QUERY_RE.search(user_message) and not _SPANISH_ANSWER_RE.search(final_text or ""):
            return "The user wrote in Spanish. Rewrite the full answer in Spanish without changing the cited facts."
        if _SPANISH_QUERY_RE.search(user_message) and not _SPANISH_CASHLESS_DIRECT_RE.search(final_text or ""):
            return (
                "Answer the user's exact question in the first sentence: the café may not operate as "
                "card-only and must accept cash. Write the full answer in Spanish, then name and cite "
                "Local Law 34 of 2020 and Administrative Code section 20-840."
            )
    if benefits_recovery_turn:
        starts_with_verdict = bool(re.match(r"\s*(?:yes|no)\b", final_text or "", re.I))
        required = (
            "reappl" in low_answer,
            "appeal" in low_answer or "fair hearing" in low_answer,
            "notice" in low_answer,
            "deadline" in low_answer,
            bool(re.search(r"which benefit|benefit type|agency", low_answer)),
        )
        if starts_with_verdict or not all(required):
            return (
                "Do not decide whether appealing is 'worth it.' Explain that reapplying and appealing "
                "are different, preserve the appeal or fair-hearing option, tell the person to keep the "
                "notice and not miss its deadline, and ask which benefit and agency issued the denial "
                "before giving program-specific steps."
            )
    if lockout_turn:
        required = (
            "illegal" in low_answer or "ilegal" in low_answer,
            "26-521" in low_answer,
            "marshal" in low_answer and "sheriff" in low_answer,
            "housing court" in low_answer,
        )
        active_tenant = not _SELF_HELP_EVICTION_RE.search(routed_query)
        if active_tenant:
            required += ("911" in low_answer, "311" in low_answer)
        if not all(required):
            return (
                "Use both retrieved lockout sources. State that self-help eviction is illegal under "
                "NYC Administrative Code 26-521 and only a City Marshal or Sheriff may execute a "
                "warrant. For an active locked-out tenant, lead with 911, then 311 and Housing Court. "
                "For an owner asking how to bypass court, refuse and direct them to Housing Court."
            )
    if civic_law_search and "immigration status enrollment" in civic_law_search.lower():
        required = (
            "public school" in low_answer,
            bool(re.search(r"regardless of (?:the family'?s )?immigration status|sin importar .{0,30}estatus", low_answer)),
            bool(re.search(r"(?:must not|cannot|no debe|no puede).{0,60}(?:immigration papers|"
                           r"immigration documents|documentos migratorios|social security)", low_answer)),
            bool(re.search(r"(?:do not need to|no necesitas).{0,30}(?:withdraw|retirar)", low_answer)),
            "actionnyc" in low_answer or "311" in low_answer,
        )
        if not all(required):
            return (
                "State the retrieved school rights plainly: the child may attend NYC public school "
                "regardless of family immigration status; the school must not require immigration "
                "papers, citizenship papers, a visa, or a Social Security number; the parent does not "
                "need to withdraw the child because of the question. Suggest asking why the information "
                "was requested and route immigration-specific advice to 311 and ActionNYC."
            )
    if (
        civic_law_search
        and "tipped worker minimum wage" in civic_law_search.lower()
        and re.search(r"\b(?:waiters?|servers?)\b", low_query)
    ):
        food_wage = low_answer.find("$11.35")
        service_wage = low_answer.find("$14.15")
        if (
            food_wage < 0
            or "$5.65" not in low_answer
            or "$17" not in low_answer
            or (service_wage >= 0 and food_wage > service_wage)
        ):
            return (
                "Answer the waiters question first using the retrieved food-service-worker row: $5 "
                "is not enough; the 2026 NYC cash wage is $11.35, the maximum tip credit is $5.65, "
                "and the full minimum wage is $17.00. Do not lead with the separate service-employee row."
            )
    if civic_law_search and "fair workweek fast food" in civic_law_search.lower():
        if not (
            "14" in low_answer
            and ("schedule-change premium" in low_answer or "premium pay" in low_answer)
            and "dcwp" in low_answer
        ):
            return (
                "Answer the two-hour shift change directly from the retrieved Fair Workweek source: "
                "if the worker is covered, a change inside the 14-day notice window requires a "
                "schedule-change premium. Tell the worker to keep the schedule and route to DCWP."
            )
    if (
        civic_law_search
        and "rent increase notice" in civic_law_search.lower()
        and "rent stabil" in low_query
        and ("ended" in low_query or "as much" in low_query)
    ):
        return (
            "Correct the false premise without guessing a rate. State that rent stabilization did "
            "not end in 2019, the Rent Guidelines Board still sets renewal increases for regulated "
            "apartments, and the resident should confirm the apartment's status with HCR or 311."
        )
    if (
        civic_law_search
        and "third department section 8" in civic_law_search.lower()
        and "section 8" in low_query
    ):
        required = (
            "2026" in low_answer and bool(re.search(r"(?:march|marzo).{0,20}\b5\b|\b5\b.{0,20}(?:march|marzo)", low_answer)),
            bool(re.search(r"(?:affirmed|confirm[oó]|ratific[oó]|anul[oó]|invalid|inconstitucional)", low_answer)),
            bool(re.search(r"(?:state|estatal|executive law|ley ejecutiva)", low_answer)),
            bool(re.search(
                r"(?:still|continues? to) (?:list|include).{0,50}section 8|"
                r"(?:todav[ií]a|a[uú]n) incluye.{0,50}section 8|"
                r"sigue (?:incluyendo|diciendo).{0,160}section 8",
                low_answer,
            )),
            bool(re.search(
                r"(?:may|might|could) (?:limit|affect|change)|puede (?:limitar|afectar|cambiar)|"
                r"(?:unclear|not clear|no est[aá] claro|orientaci[oó]n legal|legal guidance|"
                r"flat yes(?:-or-no| or no))",
                low_answer,
            )),
        )
        if not all(required):
            return (
                "State the retrieved current status plainly in the user's language: on March 5, 2026, "
                "the Third Department affirmed the ruling that the state Executive Law provision is facially "
                "unconstitutional to the extent it requires Section 8 participation. The current NYC Commission "
                "page still lists Section 8 under the local law, but the ruling may limit that mandate, so do not "
                "give a flat yes or no for a specific property. Say that CityFHEPS and other non-Section-8 vouchers "
                "remain protected and that a blanket 'no vouchers' listing is unsafe. Route to case-specific legal "
                "guidance, keep 'NYC Commission on Human Rights' unchanged, and do not add an appellate fact absent "
                "from the retrieved evidence."
            )
    if immigrant_benefits_turn:
        if (
            re.search(r"\b(?:deport|deportar|deportaci[oó]n)\w*\b", low_query)
            and re.match(r"\s*yes\b", final_text or "", re.I)
        ):
            return (
                "Do not answer 'yes' to a deportation question. Start with the grounded boundary: "
                "using Medicaid does not cause automatic deportation. Then separate public charge, "
                "current Medicaid data-sharing concerns, and case-specific legal advice."
            )
        if _SPANISH_QUERY_RE.search(user_message) and not _SPANISH_ANSWER_RE.search(final_text or ""):
            return "The user wrote in Spanish. Rewrite the complete grounded answer in Spanish."
        public_charge_question = any(
            term in low_query for term in ("green card", "public charge", "estatus migratorio")
        )
        if public_charge_question and ("snap" in low_query or "medicaid" in low_query):
            missing_snap_rule = "snap" in low_query and not re.search(
                r"snap.{0,90}(?:not counted|no (?:se )?cuenta|no cuentan|no cuenta)", low_answer,
            )
            missing_medicaid_rule = "medicaid" in low_query and not (
                ("not counted" in low_answer or "no se cuenta" in low_answer
                 or "no cuentan" in low_answer or "no cuenta" in low_answer
                 or "tampoco cuenta" in low_answer)
                and ("long-term" in low_answer or "largo plazo" in low_answer or "institucional" in low_answer)
            )
            if missing_snap_rule or missing_medicaid_rule:
                return (
                    "State the current public-charge rule plainly in the user's language: SNAP and most "
                    "Medicaid are not counted; long-term institutional care is the Medicaid exception; "
                    "the November 2025 proposal is not in effect. Keep eligibility separate and route to ActionNYC."
                )
        if "indocument" in low_query and any(
            term in low_query for term in ("benefit", "beneficio", "ayuda", "zero help", "no help")
        ):
            required = (
                ("snap" in low_answer),
                bool(re.search(
                    r"(?:eligible (?:family|household)|familiares? elegibles?|miembros? elegibles?|"
                    r"you (?:are|may be) not eligible|t[uú] no seas elegible|no eres elegible)",
                    low_answer,
                )),
                ("medicaid de emergencia" in low_answer or "emergency medicaid" in low_answer),
                ("nyc care" in low_answer),
            )
            if not all(required):
                return (
                    "Separate personal eligibility from applying for eligible household members. Say plainly "
                    "that an undocumented person may apply for eligible family members even when the person "
                    "is not personally eligible for SNAP. Include the retrieved Emergency Medicaid and NYC "
                    "Care options without making a blanket claim about full Medicaid eligibility. Route "
                    "individualized advice."
                )
    return None


def _section8_grounded_backstop(user_message: str, citations: dict[str, dict]) -> Optional[str]:
    """Return a narrow legal-status summary only when both exact live sources were captured."""
    if "section 8" not in _routing_text(user_message).lower():
        return None
    city_id = next(
        (cid for cid, cite in citations.items()
         if cite.get("url") == "https://www.nyc.gov/site/cchr/media/source-of-income.page"),
        None,
    )
    court_id = next(
        (cid for cid, cite in citations.items()
         if cite.get("url") == "https://www.nycourts.gov/reporter/3dseries/2026/2026_01253.htm"),
        None,
    )
    if not city_id or not court_id:
        return None
    if _SPANISH_QUERY_RE.search(user_message):
        return (
            "No uses una regla general de 'no vouchers'. CityFHEPS y otros vouchers que no son "
            f"Section 8 siguen protegidos por la ley local de NYC {{cite:{city_id}}}. Para Section 8, "
            "el 5 de marzo de 2026, el "
            "Third Department confirmó que la disposición de la ley estatal era facialmente "
            "inconstitucional en la medida en que obligaba a aceptar Section 8 "
            f"{{cite:{court_id}}}. La página actual de NYC Commission on Human Rights todavía incluye "
            f"Section 8 bajo la ley local {{cite:{city_id}}}, pero esas dos fuentes no resuelven "
            "directamente cómo se aplica el fallo estatal a una propiedad específica en NYC. No "
            "supongas que el fallo permite rechazar Section 8. Antes de actuar, busca orientación legal "
            "para tu propiedad con NYC Commission on Human Rights o llama al 311."
        )
    return (
        "Do not use a blanket 'no vouchers' rule. CityFHEPS and other non-Section-8 vouchers remain "
        f"protected under NYC local law {{cite:{city_id}}}. For Section 8, on March 5, 2026, the Third "
        "Department affirmed that the state-law provision was facially unconstitutional to the extent "
        f"it required Section 8 participation {{cite:{court_id}}}. The current NYC Commission on Human "
        f"Rights page still lists Section 8 under local law {{cite:{city_id}}}, but those two sources do "
        "not directly resolve how the state ruling applies to a specific NYC property. Do not assume the "
        "ruling permits a Section 8 refusal. Before acting, get property-specific legal guidance from "
        "NYC Commission on Human Rights or call 311."
    )


def _rent_stabilization_grounded_backstop(
    user_message: str, citations: dict[str, dict],
) -> Optional[str]:
    """Correct the narrow rent-stabilization false premise without guessing a rate."""
    routed = _routing_text(user_message).lower()
    if "rent stabil" not in routed or not ("ended" in routed or "as much" in routed):
        return None
    city_id = next(
        (cid for cid, cite in citations.items()
         if cite.get("url") == "https://portal.311.nyc.gov/article/?kanumber=KA-03296"),
        None,
    )
    hcr_id = next(
        (cid for cid, cite in citations.items() if cite.get("url") == "https://hcr.ny.gov/rent-control"),
        None,
    )
    if not city_id or not hcr_id:
        return None
    return (
        "No. Rent stabilization did not end in 2019. For a rent-stabilized apartment, the NYC Rent "
        f"Guidelines Board still sets renewal increases {{cite:{city_id}}}. Rent-controlled apartments "
        f"also remain regulated by New York State Homes and Community Renewal {{cite:{hcr_id}}}. "
        "Before accepting an unlimited increase, confirm your apartment's status with HCR or call 311."
    )


def _public_charge_grounded_backstop(
    user_message: str, citations: dict[str, dict],
) -> Optional[str]:
    """Return the current public-charge floor only with both exact MOIA sources captured."""
    rule_id = next(
        (cid for cid, cite in citations.items()
         if cite.get("url") == "https://www.nyc.gov/site/immigrants/legal-resources/public-charge-rule.page"),
        None,
    )
    help_id = next(
        (cid for cid, cite in citations.items()
         if cite.get("url") == (
             "https://www.nyc.gov/site/immigrants/legal-resources/"
             "moia-immigration-legal-support-hotline.page"
         )),
        None,
    )
    if not rule_id or not help_id:
        return None
    routed = _routing_text(user_message)
    snap_id = next(
        (cid for cid, cite in citations.items()
         if cite.get("url") == "https://www.nyc.gov/assets/hra/ACCESSNYC/html/snapfaq/english.shtml"),
        None,
    )
    health_id = next(
        (cid for cid, cite in citations.items()
         if cite.get("url") == "https://www.nyc.gov/site/doh/health/health-topics/immigrant-health.page"),
        None,
    )
    citizen_children = bool(re.search(
        r"\b(?:citizen children|hijos?\b.{0,30}\bciudadanos?)\b", routed, re.I,
    ))
    if (
        re.search(r"\b(?:undocumented|indocumentad[oa]s?)\b", routed, re.I)
        and (_IMMIGRANT_HELP_CLAIM_RE.search(routed) or citizen_children)
        and snap_id
        and health_id
    ):
        if _SPANISH_QUERY_RE.search(user_message):
            snap = (
                "Tus hijos ciudadanos pueden calificar para SNAP aunque tú no seas elegible; HRA "
                if citizen_children
                else "Por lo general, no eres elegible para recibir SNAP para ti, pero puedes solicitarlo "
                "para familiares elegibles; HRA "
            )
            return (
                f"No. {snap}aplica las demás reglas del programa {{cite:{snap_id}}}. Medicaid de "
                f"Emergencia y NYC Care también pueden estar disponibles sin importar tu estatus "
                f"migratorio {{cite:{health_id}}}. Para orientación sobre tu caso, llama al 311 y pide "
                f"ActionNYC {{cite:{help_id}}}."
            )
        return (
            "No. An undocumented person is generally not eligible for SNAP for themselves, but can "
            f"apply for eligible household members {{cite:{snap_id}}}. Emergency Medicaid and NYC Care "
            f"may also be available regardless of immigration status {{cite:{health_id}}}. For advice "
            f"about your case, call 311 and ask for ActionNYC {{cite:{help_id}}}."
        )
    deportation_question = bool(re.search(r"\b(?:deport|deportar|deportaci[oó]n)\w*\b", routed, re.I))
    if _SPANISH_QUERY_RE.search(user_message):
        opening = (
            "No: usar Medicaid no causa una deportación automática. "
            if deportation_question
            else "No hay un efecto migratorio automático por solicitar esos beneficios. "
        )
        return (
            opening
            + "Según la regla actual, SNAP y la mayoría de Medicaid no cuentan para public charge. "
            "La excepción de Medicaid es el cuidado institucional a largo plazo. La propuesta de "
            f"noviembre de 2025 no está en vigor {{cite:{rule_id}}}. Para tu cita o caso exacto, llama "
            f"al 800-354-0365 o llama al 311 y di 'Immigration Legal' {{cite:{help_id}}}."
        )
    opening = (
        "No: using Medicaid does not cause automatic deportation. "
        if deportation_question
        else "Applying for those benefits does not have an automatic immigration effect. "
    )
    return (
        opening
        + "Under the current rule, SNAP and most Medicaid do not count for public charge. The Medicaid "
        "exception is long-term institutional care. The November 2025 proposal is not in effect "
        f"{{cite:{rule_id}}}. For your interview or exact case, call 800-354-0365 or call 311 and say "
        f"'Immigration Legal' {{cite:{help_id}}}."
    )


def _cashless_grounded_backstop(
    user_message: str, citations: dict[str, dict],
) -> Optional[str]:
    """Return the cash-acceptance floor only when the live rule and enacted law were captured."""
    rule_id = next(
        (cid for cid, cite in citations.items()
         if cite.get("url") == (
             "https://www.nyc.gov/site/dca/consumers/"
             "Prohibition-of-Cashless-Establishments.page"
         )),
        None,
    )
    law_id = next(
        (cid for cid, cite in citations.items()
         if cite.get("url", "").startswith(
             "https://legistar.council.nyc.gov/LegislationDetail.aspx?"
         )),
        None,
    )
    if not rule_id or not law_id:
        return None
    if _SPANISH_QUERY_RE.search(user_message):
        return (
            "No. Una cafetería pequeña que cobra en persona debe aceptar efectivo. No puede operar "
            f"como 'solo tarjeta' {{cite:{rule_id}}}. La norma es la Ley Local 34 de 2020 y el Código "
            f"Administrativo de NYC, sección 20-840 {{cite:{law_id}}}. La excepción principal es un "
            "dispositivo en el local que convierte efectivo a una tarjeta prepaga sin cobrar comisión. "
            f"Las violaciones se pueden reportar al 311 o a DCWP {{cite:{rule_id}}}."
        )
    return (
        "No. A small in-person café must accept cash. It cannot operate as card-only "
        f"{{cite:{rule_id}}}. The rule is Local Law 34 of 2020 and NYC Administrative Code section "
        f"20-840 {{cite:{law_id}}}. The main alternative is an on-site machine that converts cash to "
        "a prepaid card without a fee. Violations can be reported to 311 or DCWP "
        f"{{cite:{rule_id}}}."
    )


def _school_immigration_grounded_backstop(
    user_message: str, citations: dict[str, dict],
) -> Optional[str]:
    """Return NYCPS enrollment rights only when both current school sources were captured."""
    family_id = next(
        (cid for cid, cite in citations.items()
         if cite.get("url") == (
             "https://www.schools.nyc.gov/school-life/school-environment/immigrant-families"
         )),
        None,
    )
    rights_id = next(
        (cid for cid, cite in citations.items()
         if cite.get("url") == (
             "https://www.schools.nyc.gov/learning/multilingual-learners/"
             "bill-of-rights-for-parents-of-english-language-learners"
         )),
        None,
    )
    if not family_id or not rights_id:
        return None
    if _SPANISH_QUERY_RE.search(user_message):
        return (
            "Tu hijo tiene derecho a asistir a una escuela pública de NYC sin importar el estatus "
            f"migratorio de la familia {{cite:{family_id}}}. La escuela no debe exigir documentos que "
            "revelen el estatus migratorio, como una visa, documentos de ciudadanía o un número de "
            f"Seguro Social {{cite:{rights_id}}}. No necesitas retirar a tu hijo por esa pregunta. "
            "Pide por escrito por qué solicitaron la información y no entregues documentos que no "
            "entiendas. Para orientación migratoria sobre tu situación, llama al 311 y pide ActionNYC."
        )
    return (
        "Your child has the right to attend an NYC public school regardless of immigration status "
        f"{{cite:{family_id}}}. The school must not require immigration papers, "
        "including a visa, citizenship documents, or a Social Security number "
        f"{{cite:{rights_id}}}. You do not need to withdraw your child because of that question. Ask "
        "in writing why the information was requested, and do not provide documents you do not "
        "understand. For advice about your family's immigration situation, call 311 and ask for ActionNYC."
    )


def _benefits_denial_grounded_backstop(
    user_message: str, citations: dict[str, dict],
) -> Optional[str]:
    """Keep an unclear denial recoverable without inventing a program-specific appeal rule."""
    claims_id = next(
        (cid for cid, cite in citations.items()
         if cite.get("url") == "https://www.nyc.gov/site/hra/about/claims-collections.page"),
        None,
    )
    if claims_id and _BENEFITS_OVERPAYMENT_RE.search(_routing_text(user_message)):
        if _SPANISH_QUERY_RE.search(user_message):
            return (
                "No ignores el aviso ni pagues todo antes de entender el reclamo. Guarda el aviso y "
                "la fecha límite. Contacta a HRA Claims and Collections para pedir la base del reclamo "
                f"y las opciones de pago {{cite:{claims_id}}}. Si disputas la decisión, sigue las "
                "instrucciones de audiencia imparcial del aviso antes de la fecha límite."
            )
        return (
            "Do not ignore the notice or pay the full amount before you understand the claim. Keep the "
            "notice and its deadline. Contact HRA Claims and Collections to ask what the claim is based "
            f"on and what repayment options are available {{cite:{claims_id}}}. If you dispute the "
            "decision, follow the fair-hearing instructions on the notice before the deadline."
        )
    snap_id = next(
        (cid for cid, cite in citations.items()
         if cite.get("url") == "https://www.nyc.gov/assets/hra/ACCESSNYC/html/snapfaq/english.shtml"),
        None,
    )
    if not snap_id:
        return None
    if _SPANISH_QUERY_RE.search(user_message):
        return (
            "Volver a solicitar y apelar son cosas distintas. Guarda el aviso de denegación y no "
            "pierdas la fecha límite escrita allí. Contacta a HRA y pregunta qué ruta de apelación o "
            "audiencia imparcial corresponde; también dime qué beneficio y agencia emitió el aviso. "
            "Si era SNAP, HRA dice "
            f"que puedes volver a solicitar en cualquier momento {{cite:{snap_id}}}; eso no sustituye "
            "una apelación del aviso actual."
        )
    return (
        "Reapplying and appealing are different. Keep the denial notice and do not miss the deadline "
        "printed on it. Contact HRA and ask which appeal or fair-hearing path applies; also tell me "
        "which benefit and agency issued the notice. If this was SNAP, HRA says you may reapply at any time "
        f"{{cite:{snap_id}}}; that does not replace an appeal of the current notice."
    )


def _lockout_grounded_backstop(
    user_message: str, citations: dict[str, dict],
) -> Optional[str]:
    """Return the illegal-lockout floor only with the NYC311 and HPD sources captured."""
    routed = _routing_text(user_message)
    route_id = next(
        (cid for cid, cite in citations.items()
         if cite.get("url") == "https://portal.311.nyc.gov/article/?kanumber=KA-02518"),
        None,
    )
    if route_id and _ESSENTIAL_SERVICES_SHUTOFF_RE.search(routed):
        if _SPANISH_QUERY_RE.search(user_message):
            return (
                "No. Tu casero no puede cortar el agua caliente para obligarte a salir. NYC311 dice "
                "que privar intencionalmente a un inquilino de un servicio esencial como el agua se "
                f"debe reportar al 911 {{cite:{route_id}}}. Llama al 911 ahora y después al 311, y "
                "guarda pruebas del corte y de los mensajes del casero."
            )
        return (
            "No. Your landlord cannot shut off hot water to force you out. NYC311 says a landlord "
            "who willfully deprives a tenant of an essential service such as water should be reported "
            f"to 911 {{cite:{route_id}}}. Call 911 now and then 311, and keep proof of the shutoff and "
            "the landlord's messages."
        )
    law_id = next(
        (cid for cid, cite in citations.items()
         if cite.get("url") == (
             "https://home4.nyc.gov/site/hpd/services-and-information/"
             "tenants-rights-and-responsibilities.page"
         )),
        None,
    )
    if not route_id or not law_id:
        return None
    owner_request = bool(_SELF_HELP_EVICTION_RE.search(routed))
    if _SPANISH_QUERY_RE.search(user_message):
        if owner_request:
            return (
                "No puedo ayudar a sacar a un inquilino sin proceso judicial. Un desalojo por cuenta "
                f"propia es ilegal bajo el Código Administrativo 26-521 {{cite:{law_id}}}. Solo un "
                f"City Marshal o Sheriff puede ejecutar una orden de desalojo {{cite:{route_id}}}. "
                "Como propietario, usa Housing Court y el proceso legal. Si ya cambiaste las cerraduras "
                "o cortaste servicios, restáuralos y busca asesoría legal; el inquilino puede llamar al "
                "911 y después al 311."
            )
        return (
            f"Llama al 911 ahora mismo para reportar el lockout ilegal {{cite:{route_id}}}. Cambiar "
            f"las cerraduras sin una orden viola el Código Administrativo 26-521 {{cite:{law_id}}}. "
            f"Solo un City Marshal o Sheriff puede ejecutar una orden de desalojo {{cite:{route_id}}}. "
            "Después llama al 311 y pide ayuda de Housing Court. Diles que estás afuera con tus hijos."
        )
    if owner_request:
        return (
            "I cannot help remove a tenant without court process. A self-help eviction is illegal "
            f"under NYC Administrative Code 26-521 {{cite:{law_id}}}. Only a City Marshal or Sheriff "
            f"may carry out a warrant of eviction {{cite:{route_id}}}. As the owner, use Housing Court "
            "and the legal process. If you already changed locks or cut services, restore them and get "
            "legal advice; the tenant may call 911 and then 311."
        )
    return (
        f"Call 911 right now to report the illegal lockout {{cite:{route_id}}}. Changing the locks "
        f"without a warrant violates NYC Administrative Code 26-521 {{cite:{law_id}}}. Only a City "
        f"Marshal or Sheriff may carry out a warrant of eviction {{cite:{route_id}}}. Then call 311 "
        "and ask for Housing Court help."
    )


def _scope_grounded_backstop(
    user_message: str,
    citations: dict[str, dict],
    civic_law_search: Optional[str],
    *,
    immigrant_benefits_turn: bool,
    benefits_recovery_turn: bool,
    lockout_turn: bool,
) -> Optional[str]:
    """Choose a narrow retrieved-source backstop for a failed high-stakes scope check."""
    search = (civic_law_search or "").lower()
    if "third department section 8" in search:
        answer = _section8_grounded_backstop(user_message, citations)
        if answer:
            return answer
    if "rent increase notice" in search:
        answer = _rent_stabilization_grounded_backstop(user_message, citations)
        if answer:
            return answer
    if "cashless ban" in search:
        answer = _cashless_grounded_backstop(user_message, citations)
        if answer:
            return answer
    if "immigration status enrollment" in search:
        answer = _school_immigration_grounded_backstop(user_message, citations)
        if answer:
            return answer
    if immigrant_benefits_turn:
        answer = _public_charge_grounded_backstop(user_message, citations)
        if answer:
            return answer
    if benefits_recovery_turn:
        answer = _benefits_denial_grounded_backstop(user_message, citations)
        if answer:
            return answer
    if lockout_turn:
        return _lockout_grounded_backstop(user_message, citations)
    return None


def _routing_text(user_message: str) -> str:
    """Normalize compatibility Unicode and common leetspeak inside words for safety routing."""
    text = "".join(
        char for char in unicodedata.normalize("NFKC", user_message)
        if unicodedata.category(char) != "Cf"
    )
    substitutions = str.maketrans("013457", "oieast")
    return re.sub(
        r"\b(?=[a-z0-9]*[a-z])[a-z0-9]+\b",
        lambda match: match.group(0).translate(substitutions),
        text,
        flags=re.IGNORECASE,
    )


def _current_source_call(tools: dict[str, Tool], query: str, urls: tuple[str, ...]):
    """Prefer declared-page retrieval, with scoped web search as a compatibility fallback."""
    if "official_sources" in tools:
        return "official_sources", {"urls": list(urls), "query": query}
    return "web_search", {"query": query}


def _benefits_recovery_allowed_tools(user_message: str) -> set[str]:
    """Keep a benefits recovery turn out of unrelated modules unless the resident asks."""
    allowed = {
        "official_sources", "web_search", "recent_developments", "screen_eligibility",
        "prepare_snap_application", "nearest_food_pantry", "geocode", "nearest",
    }
    if _EXPLICIT_HOUSING_RE.search(user_message):
        allowed.update({"hpd_building_lookup", "hpd_litigation_lookup", "housing_guidance"})
    if _EXPLICIT_CLINIC_RE.search(user_message):
        allowed.update({"find_clinic", "health_coverage_guidance"})
    if _EXPLICIT_WORKER_RE.search(user_message):
        allowed.add("worker_rights_guidance")
    return allowed


def _immigrant_benefits_allowed_tools() -> set[str]:
    return {
        "official_sources", "web_search", "recent_developments", "benefits_search",
        "screen_eligibility",
        "health_coverage_guidance",
    }


def _civic_law_allowed_tools() -> set[str]:
    """Keep a current-law turn on the retrieved law instead of unrelated static guidance."""
    return {"official_sources", "web_search", "recent_developments"}


# An actual SSN-like value crosses the channel's safe intake boundary. Stop before the model can
# echo it or ask for more identifiers. Application help may resume through the official path or a
# later turn that does not contain the sensitive identifier.
# A nine-digit run (bare or 3-2-4 dashed) is the SSN shape, but it is ALSO the shape of a 311
# service-request, benefits-case, or confirmation number a resident legitimately quotes to ask for
# help. Grouping alone cannot tell them apart (case numbers come dashed too), so a benign ID context
# suppresses ONLY the bare-number trigger, never an explicit SSN mention.
_SENSITIVE_SSN_NUMBER_RE = re.compile(r"\b\d{3}-?\d{2}-?\d{4}\b")
_SENSITIVE_SSN_PHRASE_RE = re.compile(
    r"\bmy\s+ssn\b|\bmi\s+n[uú]mero\s+de\s+seguro\s+social\b", re.IGNORECASE
)
_EXPLICIT_SSN_MENTION_RE = re.compile(
    r"\bssn\b|social security number|n[uú]mero de seguro social", re.IGNORECASE
)
_BENIGN_IDENTIFIER_CONTEXT_RE = re.compile(
    r"\b(?:complaint|case|confirmation|reference|tracking|ticket|"
    r"service request|sr|311)\b",
    re.IGNORECASE,
)
_SENSITIVE_ID_SPANISH_RE = re.compile(
    r"\b(?:seguro social|solicitud|env[ií]a|presenta|aqu[ií] est[aá])\b", re.IGNORECASE
)
_SENSITIVE_ID_RESPONSE_EN = (
    "I can't use or submit sensitive ID information here. Do not send an SSN or other sensitive "
    "identifier in chat. Use the official ACCESS HRA application, where you stay in control of review "
    "and submission. I can still explain the public steps without collecting your identifiers."
)
_SENSITIVE_ID_RESPONSE_ES = (
    "No puedo usar ni enviar información de identificación confidencial aquí. No envíes un número "
    "de Seguro Social ni otra identificación confidencial por chat. Usa la solicitud oficial de ACCESS "
    "HRA, donde tú revisas y envías la solicitud. Puedo explicar los pasos públicos sin recopilar tus datos."
)


def _sensitive_identifier_backstop(user_message: str) -> Optional[str]:
    phrase = _SENSITIVE_SSN_PHRASE_RE.search(user_message)
    has_number = _SENSITIVE_SSN_NUMBER_RE.search(user_message)
    if not phrase and not has_number:
        return None
    # A number introduced as a 311/case/complaint/confirmation ID is not an SSN. Suppress only when
    # the trigger is the bare number, no "my ssn" phrase, and no explicit SSN mention anywhere.
    if (
        has_number
        and not phrase
        and not _EXPLICIT_SSN_MENTION_RE.search(user_message)
        and _BENIGN_IDENTIFIER_CONTEXT_RE.search(user_message)
    ):
        return None
    if _SENSITIVE_ID_SPANISH_RE.search(user_message):
        return _SENSITIVE_ID_RESPONSE_ES
    return _SENSITIVE_ID_RESPONSE_EN


_INTERNAL_CONFIG_RE = re.compile(
    r"\b(?:system (?:prompt|reminders?|instructions?)|hidden (?:prompt|instructions?)|"
    r"own instructions?|tool (?:configuration|instructions?)|grounding rules?)\b.{0,80}"
    r"\b(?:paste|show|tell|reveal|print|debug|summarize)\b|"
    r"\b(?:paste|show|tell|reveal|print|summarize)\b.{0,80}\b(?:system (?:prompt|reminders?|instructions?)|"
    r"own instructions?|hidden rules?|"
    r"tool (?:configuration|instructions?)|grounding rules?)\b",
    re.IGNORECASE,
)
_INTERNAL_CONFIG_RESPONSE = (
    "I can't provide hidden system instructions, tool configuration, or internal reminders. "
    "I can keep helping with what you asked about before, or explain HeyNYC's public "
    "capabilities and safety approach."
)


def _internal_config_backstop(user_message: str) -> Optional[str]:
    return _INTERNAL_CONFIG_RESPONSE if _INTERNAL_CONFIG_RE.search(user_message) else None

# Clear, active chest-pain and overdose statements bypass the model entirely. This is intentionally
# narrow, not a general medical classifier: first-person English and Spanish only. A deterministic
# response prevents unsafe diagnosis or dosage text from entering the streaming event path at all.
_CHEST_PAIN_EN_RE = re.compile(
    r"\b(?:i have|i am (?:having|experiencing|feeling)|i['’]m (?:having|experiencing|feeling)|i feel) "
    r"(?:(?:severe|bad|really bad|a) )?"
    r"(?:chest (?:pain|pressure)|(?:pain|pressure) in (?:my|the) chest)\b|"
    r"\bmy chest (?:hurts|is hurting|feels tight)\b",
    re.IGNORECASE,
)
_CHEST_PAIN_ES_RE = re.compile(
    r"\b(?:tengo|siento|estoy (?:teniendo|experimentando)) "
    r"(?:un )?(?:dolor|presi[oó]n)(?: fuerte)? "
    r"(?:en (?:el|mi)|de) pecho\b|"
    r"\bme (?:duele|est[aá] doliendo) el pecho\b",
    re.IGNORECASE,
)
_CHEST_PAIN_HYPOTHETICAL_RE = re.compile(
    r"\b(?:what if|if i (?:have|had)|si tengo|si tuviera)\b",
    re.IGNORECASE,
)
_CHEST_PAIN_PAST_RE = re.compile(
    r"\b(?:stopped|went away|gone|no longer|last week|yesterday|used to|history of|"
    r"la semana pasada|ayer|ya pas[oó]|ya no|se me pas[oó]|tuve)\b",
    re.IGNORECASE,
)
_CHEST_PAIN_INFORMATIONAL_RE = re.compile(
    r"^[\s¿]*(?:what|why|how|when|can|could|would|should|is|are|do|does|"
    r"qu[eé]|por qu[eé]|c[oó]mo|cu[aá]ndo)\b",
    re.IGNORECASE,
)
_OVERDOSE_EN_RE = re.compile(
    r"\b(?:i (?:just )?(?:took|swallowed|ingested)|i['’]ve (?:just )?(?:taken|swallowed|ingested)) "
    r"(?:(?:way )?too many|\d+|a whole bottle|the whole bottle|an entire bottle) "
    r"(?:pills?|tablets?|capsules?|doses?|[a-z][a-z-]+ pills?)\b",
    re.IGNORECASE,
)
_OVERDOSE_ES_RE = re.compile(
    r"\b(?:acabo de (?:tomar|tragar|ingerir)|(?:yo )?(?:tom[eé]|tragu[eé]|inger[ií])) "
    r"(?:demasiad[oa]s?|\d+|un frasco entero|todo el frasco) "
    r"(?:pastillas?|tabletas?|c[aá]psulas?|dosis)(?: de [a-záéíóúñ-]+)?\b",
    re.IGNORECASE,
)
_OVERDOSE_OLD_RE = re.compile(
    r"\b(?:last week|yesterday|days? ago|la semana pasada|ayer|hace d[ií]as?)\b",
    re.IGNORECASE,
)
_EMERGENCY_RESPONSE_EN = "Call 911 right now. I'm an AI and can't diagnose this."
_EMERGENCY_RESPONSE_ES = "Llama al 911 ahora mismo. Soy una IA y no puedo diagnosticar esto."
# _OVERDOSE_*_RE match only first-person self-ingestion, so every message reaching these responses
# is a possible suicide attempt as well as a poisoning. Poisoning care leads because it is what
# keeps the person alive tonight; the crisis route is conditional so an accidental double dose is
# not told it was a suicide attempt. A caregiver reporting someone else never reaches here
_POISON_RESPONSE_EN = (
    "Call 911 and Poison Control at 1-800-222-1222 right now. "
    "If you took them to hurt yourself, call 988 too. "
    "Move away from any pills you have left and ask someone you trust to stay with you. "
    "I'm an AI and can't diagnose this."
)
_POISON_RESPONSE_ES = (
    "Llama al 911 y a Poison Control al 1-800-222-1222 ahora mismo. "
    "Si las tomaste para hacerte daño, llama también al 988. "
    "Aléjate de las pastillas que te queden y pide a alguien de confianza que se quede contigo. "
    "Soy una IA y no puedo diagnosticar esto."
)
_SELF_HARM_EN_RE = re.compile(
    r"\b(?:i (?:think about|am thinking about|want to|plan to|might) (?:ending it|end my life|"
    r"kill myself|die)|i (?:do not|don't) want to (?:live|be alive)|what pills would do it|"
    r"what(?:'s| is) the .{0,30} way to (?:end my life|kill myself|die))\b",
    re.IGNORECASE,
)
_IMMINENT_SELF_HARM_EN_RE = re.compile(
    r"\bi(?:['’]m| am) going to (?:kill myself|end my life|die)|"
    r"\bi (?:will|intend to) (?:kill myself|end my life|die)\b",
    re.IGNORECASE,
)
_SELF_HARM_ES_RE = re.compile(
    r"\b(?:ya no quiero seguir viviendo|no quiero (?:vivir|seguir con vida)|quiero (?:matarme|"
    r"quitarme la vida|morir|hacerme da[ñn]o)|me quiero hacer da[ñn]o|"
    r"pienso en (?:matarme|quitarme la vida))\b",
    re.IGNORECASE,
)
_INFANT_DOSE_EN_RE = re.compile(
    r"\b(?:baby|infant|\d{1,2}[- ]month[- ]old)\b.*\b(?:acetaminophen|tylenol|ibuprofen)\b|"
    r"\b(?:acetaminophen|tylenol|ibuprofen)\b.*\b(?:baby|infant|\d{1,2}[- ]month[- ]old)\b",
    re.IGNORECASE,
)
_INFANT_DOSE_ES_RE = re.compile(
    r"\b(?:beb[eé]|\d{1,2} meses?)\b.*\b(?:acetaminof[eé]n|paracetamol|ibuprofeno)\b|"
    r"\b(?:acetaminof[eé]n|paracetamol|ibuprofeno)\b.*\b(?:beb[eé]|\d{1,2} meses?)\b",
    re.IGNORECASE,
)
_EXACT_DOSE_RE = re.compile(
    r"\b(?:exact dose|dose in ml|(?:exactly )?how (?:much|many) (?:ml|milliliters?)|"
    r"dosis exacta|cu[aá]ntos ml|dosis en ml)\b",
    re.IGNORECASE,
)
_INFANT_DOSE_RESPONSE_EN = (
    "Do not guess an exact dose for a baby. Contact the child's pediatric clinician now because "
    "dosing depends on weight and the product concentration. If a dose may "
    "already have been given incorrectly, call Poison Control at 1-800-222-1222. Call 911 only "
    "for emergency symptoms."
)
_INFANT_DOSE_RESPONSE_ES = (
    "No adivines una dosis exacta para un bebé. Comunícate ahora con su pediatra porque la dosis "
    "depende del peso y de la concentración del producto. Si es posible que ya le "
    "hayas dado una dosis incorrecta, llama a Poison Control al 1-800-222-1222. Llama al 911 solo "
    "si tiene síntomas de emergencia."
)
_MISSED_DOSE_BN_RE = re.compile(
    r"(?:ভুলে গেছি|ভুলেছি|দিতে ভুল|মিস).{0,100}"
    r"(?:দুইটা|দুইটি|২টা|২টি|দ্বিগুণ|double)"
    r"|(?:দুইটা|দুইটি|২টা|২টি|দ্বিগুণ).{0,25}"
    r"(?:দেব|খাব|দিতে|খাওয়াব|খাওয়াব).{0,60}(?:না তো|কাকে ফোন|ফোন|এখন)",
    re.IGNORECASE,
)
_MISSED_DOSE_RESPONSE_EN = (
    "I can't tell you to double the missed dose. Follow the medicine label and contact the "
    "dispensing pharmacist or prescriber now. If an extra dose was already taken, call Poison "
    "Control at 1-800-222-1222. Call 911 for emergency symptoms."
)
_INFANT_MEDICATION_CONCERN_EN_RE = re.compile(
    r"\b(?:cold medicine|cough medicine|medication)\b.{0,90}\b(?:sleepy|drowsy|hard to wake)\b|"
    r"\b(?:sleepy|drowsy|hard to wake)\b.{0,90}\b(?:cold medicine|cough medicine|medication)\b",
    re.IGNORECASE,
)
_INFANT_MEDICATION_CONCERN_ES_RE = re.compile(
    r"\b(?:medicamento|medicina)\b.{0,90}\b(?:somnolient[oa]|dif[ií]cil de despertar)\b|"
    r"\b(?:somnolient[oa]|dif[ií]cil de despertar)\b.{0,90}\b(?:medicamento|medicina)\b",
    re.IGNORECASE,
)
_INFANT_MEDICATION_CONCERN_RESPONSE_EN = (
    "Do not guess an exact dose for a baby. Contact the child's pediatric clinician because dosing "
    "depends on weight and the product concentration. Because the baby "
    "already took medicine and is sleepy, call Poison Control now at 1-800-222-1222. Call 911 if "
    "the baby is hard to wake, has trouble breathing, or has a seizure."
)
_INFANT_MEDICATION_CONCERN_RESPONSE_ES = (
    "No adivines una dosis exacta para un bebé. Comunícate ahora con su pediatra porque la dosis "
    "depende del peso y de la concentración del producto. Como el bebé ya tomó un "
    "medicamento y está somnoliento, llama ahora a Poison Control al 1-800-222-1222. Llama al 911 "
    "si es difícil despertarlo, tiene dificultad para respirar o tiene una convulsión."
)
_POISON_CONTROL_SOURCE_URL = "https://www.poison.org/need-immediate-assistance"
_POISON_CONTROL_SOURCE_SNIPPET = (
    "Call your poison center at 1-800-222-1222 for help. If the individual collapses, has a "
    "seizure, has trouble breathing, or can't be awakened: Call 911 IMMEDIATELY."
)
_NYC_988_SOURCE_URL = "https://access.nyc.gov/programs/nyc-988/"
_NYC_988_SOURCE_SNIPPET = (
    "Call 988 for free, confidential crisis support. Call 911 if you are in immediate danger "
    "or need emergency medical attention."
)
_NIMH_SUICIDE_SAFETY_SOURCE_URL = (
    "https://www.nimh.nih.gov/health/publications/"
    "5-action-steps-to-help-someone-having-thoughts-of-suicide"
)
_NIMH_SUICIDE_SAFETY_SOURCE_SNIPPET = (
    "Reducing access to highly lethal items or places can help prevent suicide. "
    "Connecting the person with the 988 Suicide & Crisis Lifeline and other community resources "
    "can give them a safety net. You can also help them reach out to a trusted family member, "
    "friend, spiritual advisor, or mental health professional."
)
# Evidence keys a deterministic trigger can require. Names, not response phrases, so a translated
# floor keeps its citations
_SOURCE_POISON_CONTROL = "poison_control"
_SOURCE_INFANT_DOSING = "infant_dosing"
_SOURCE_MISSED_DOSE = "missed_dose"
_MISSED_DOSE_SOURCE_URL = "https://medlineplus.gov/ency/patientinstructions/000600.htm"
_MISSED_DOSE_SOURCE_SNIPPET = (
    "Most of the time, you move on and take the next dose when it is due. Do not take a double "
    "dose. Check with your provider or pharmacist."
)
_INFANT_DOSING_SOURCE_URL = (
    "https://www.poison.org/articles/simpler-acetaminophen-dosing-for-kids"
)
_INFANT_DOSING_SOURCE_SNIPPET = (
    "This has happened when parents didn't understand the concentration or measurements. ... "
    "NEVER measure a dose without checking the label first. Use the right dose for your child's "
    "age and weight. ... If you have a question about the right drug or right dose for your child, "
    "ask your health care provider."
)


def _ground_emergency_backstop(
    text: str, citations: CitationRegistry, sources: frozenset[str] = frozenset()
) -> str:
    """Attach verified evidence to deterministic emergency guidance.

    `sources` names the evidence the TRIGGER requires, because response copy is translated and
    English phrase matching cannot survive that. The 988 branch is the one exception: it keys off
    a phone number, which is identical in every LL30 language's verified copy.
    """
    cite_ids: list[str] = []
    if _SOURCE_MISSED_DOSE in sources:
        cite_ids.append(citations.register(
            _MISSED_DOSE_SOURCE_URL,
            title="Keeping your medicines organized | MedlinePlus",
            snippet=_MISSED_DOSE_SOURCE_SNIPPET,
            kind="WEB",
            valid_as_of="2024-09-04",
            provenance={"evidence_grade": "authoritative"},
        ))
    if "988" in text:
        cite_ids.append(citations.register(
            _NYC_988_SOURCE_URL,
            title="NYC 988 | ACCESS NYC",
            snippet=_NYC_988_SOURCE_SNIPPET,
            kind="WEB",
            valid_as_of="2026-06-09",
            provenance={"evidence_grade": "authoritative"},
        ))
        cite_ids.append(citations.register(
            _NIMH_SUICIDE_SAFETY_SOURCE_URL,
            title="5 Action Steps for Helping Someone in Emotional Pain | NIMH",
            snippet=_NIMH_SUICIDE_SAFETY_SOURCE_SNIPPET,
            kind="WEB",
            valid_as_of="2024",
            provenance={"evidence_grade": "authoritative"},
        ))
        # Evidence for the interpretation fact the floor states to residents whose language has no
        # verified crisis copy (F149). Registered wherever 988 appears, since the snippet supports
        # both calling 988 and its interpreter availability
        cite_ids.append(citations.register(
            SAMHSA_988_FAQ_URL,
            title="988 Frequently Asked Questions | SAMHSA",
            snippet=SAMHSA_988_INTERPRETATION_SNIPPET,
            kind="WEB",
            valid_as_of="2026-07-30",
            provenance={"evidence_grade": "authoritative"},
        ))
        for line in CRISIS_LINES.values():
            for translated, source in (
                (line.lifeline_988, line.source_988),
                (line.emergency_911, line.source_911),
            ):
                if translated and translated in text and source:
                    cite_ids.append(citations.register(
                        source,
                        title=f"Official 988 crisis guidance in {line.name}",
                        snippet=translated,
                        kind="WEB",
                        valid_as_of=line.verified_on,
                        provenance={"evidence_grade": "authoritative"},
                    ))
    if _SOURCE_INFANT_DOSING in sources:
        cite_ids.append(citations.register(
            _INFANT_DOSING_SOURCE_URL,
            title="Acetaminophen: Easier dosing | Poison Control",
            snippet=_INFANT_DOSING_SOURCE_SNIPPET,
            kind="WEB",
            valid_as_of="2026-07-28",
            provenance={"evidence_grade": "authoritative"},
        ))
    if _SOURCE_POISON_CONTROL in sources:
        cite_ids.append(citations.register(
            _POISON_CONTROL_SOURCE_URL,
            title="Need immediate assistance? | Poison Control",
            snippet=_POISON_CONTROL_SOURCE_SNIPPET,
            kind="WEB",
            valid_as_of="2026-07-28",
            provenance={"evidence_grade": "authoritative"},
        ))
    markers = " ".join(f"{{cite:{cite_id}}}" for cite_id in dict.fromkeys(cite_ids))
    return f"{text} {markers}".rstrip()


# Perso-Urdu letters absent from standard Arabic; their presence routes an Arabic-script message to
# Urdu (which has no verified copy -> the honest English floor), so a verified Arabic line is never
# shown to an Urdu reader. ponytail: codepoint check, extend the set only if a real Urdu miss appears.
_URDU_LETTERS = frozenset("پچژگکیۓےڈڑٹںہھ")

# Dominant non-Latin script -> LL30 language code. Each script here is unique to one covered language;
# Arabic script (shared by Arabic and Urdu) is disambiguated by _URDU_LETTERS in _crisis_language.
_SCRIPT_TO_CRISIS_LANG = {"CYRILLIC": "ru", "CJK": "zh", "BENGALI": "bn", "HANGUL": "ko"}


def _crisis_language(user_message: str) -> Optional[str]:
    """Route the crisis floor to an LL30 language by the most-present non-Latin script.

    Deterministic. Uses require_majority=False so a code-switched crisis message that leads with an
    English trigger phrase ("I want to end my life 我不想活了") still routes to the person's own-language
    script. Latin-script languages (English, Spanish, French, Polish, Haitian Creole) return None:
    Spanish keeps its own regex path in `_emergency_backstop`, and the other Latin LL30 languages
    carry no deterministic single-language signal here, so they take the honest English floor.
    Semantic crisis DETECTION across languages is the scope preflight (phase 2), separate from this."""
    script = _dominant_non_latin_script(_routing_text(user_message), require_majority=False)
    if script is None:
        return None
    if script == "ARABIC":
        return "ur" if any(ch in _URDU_LETTERS for ch in user_message) else "ar"
    return _SCRIPT_TO_CRISIS_LANG.get(script)


class Backstop(NamedTuple):
    """A deterministic floor response plus what the TRIGGER carried.

    `risk` and `sources` come from the matched trigger, never from the response text. Crisis copy
    is composed and translated per language, so any caller that recovers meaning by searching the
    response for English phrases silently reports nothing for every other language. Read the
    fields; never re-derive them from `.text`.
    """

    text: str
    risk: Optional[Literal["self_harm", "imminent_self_harm"]] = None
    sources: frozenset[str] = frozenset()


def _emergency_backstop_result(user_message: str) -> Optional[Backstop]:
    """Return a fixed, language-matched response for clear chest pain or an overdose."""
    user_message = _routing_text(user_message)
    if _IMMINENT_SELF_HARM_EN_RE.search(user_message):
        return Backstop(
            compose_crisis_floor(
                _IMMINENT_SELF_HARM_RESPONSE_EN, _crisis_language(user_message)
            ),
            "imminent_self_harm",
        )
    if _SELF_HARM_ES_RE.search(user_message):
        return Backstop(_SELF_HARM_RESPONSE_ES, "self_harm")
    if _SELF_HARM_EN_RE.search(user_message):
        return Backstop(
            compose_crisis_floor(_SELF_HARM_RESPONSE_EN, _crisis_language(user_message)),
            "self_harm",
        )
    if _MISSED_DOSE_BN_RE.search(user_message):
        return Backstop(
            localize(_MISSED_DOSE_RESPONSE_EN, "bn"),
            sources=frozenset({_SOURCE_MISSED_DOSE, _SOURCE_POISON_CONTROL}),
        )
    if _EXACT_DOSE_RE.search(user_message):
        # Caregiver asking about someone else: an emergency, but not this resident's crisis
        infant = frozenset({_SOURCE_INFANT_DOSING, _SOURCE_POISON_CONTROL})
        if _INFANT_DOSE_ES_RE.search(user_message):
            if _INFANT_MEDICATION_CONCERN_ES_RE.search(user_message):
                return Backstop(_INFANT_MEDICATION_CONCERN_RESPONSE_ES, sources=infant)
            return Backstop(_INFANT_DOSE_RESPONSE_ES, sources=infant)
        if _INFANT_DOSE_EN_RE.search(user_message):
            if _INFANT_MEDICATION_CONCERN_EN_RE.search(user_message):
                return Backstop(_INFANT_MEDICATION_CONCERN_RESPONSE_EN, sources=infant)
            return Backstop(_INFANT_DOSE_RESPONSE_EN, sources=infant)
    if not _OVERDOSE_OLD_RE.search(user_message):
        # _OVERDOSE_*_RE match first-person self-ingestion only, so these are possible attempts
        poison = frozenset({_SOURCE_POISON_CONTROL})
        if _OVERDOSE_ES_RE.search(user_message):
            return Backstop(_POISON_RESPONSE_ES, "self_harm", poison)
        if _OVERDOSE_EN_RE.search(user_message):
            return Backstop(_POISON_RESPONSE_EN, "self_harm", poison)
    signals = [
        (match.start(), _EMERGENCY_RESPONSE_ES)
        for match in _CHEST_PAIN_ES_RE.finditer(user_message)
    ] + [
        (match.start(), _EMERGENCY_RESPONSE_EN)
        for match in _CHEST_PAIN_EN_RE.finditer(user_message)
    ]
    if not signals:
        return None
    signal_pos, response = max(signals, key=lambda item: item[0])
    hypothetical = _CHEST_PAIN_HYPOTHETICAL_RE.search(user_message)
    if hypothetical:
        question_end = user_message.find("?", hypothetical.start())
        if question_end < 0 or signal_pos < question_end:
            return None
    if _CHEST_PAIN_INFORMATIONAL_RE.search(user_message):
        question_end = user_message.find("?")
        if question_end < 0 or signal_pos < question_end:
            return None
    past = list(_CHEST_PAIN_PAST_RE.finditer(user_message))
    if past and signal_pos < past[-1].end():
        return None
    return Backstop(response)


def _emergency_backstop(user_message: str) -> Optional[str]:
    """The response text alone, for callers that do not record risk."""
    result = _emergency_backstop_result(user_message)
    return result.text if result is not None else None


def _grounding_feedback(result: GroundingResult) -> str:
    """The SPECIFIC correction fed back to the model (Tier 3). Names each ungrounded fact so the model
    fixes THAT fact rather than blindly rewriting or over-abstaining."""
    problems = "; ".join(m.message for m in result.hard_failures)
    return (
        "<system-reminder>\n"
        "A grounding check ran on your last answer and found a cited fact your source does not "
        f"support: {problems}. Each is a structured fact (a phone number, dollar amount, address, or "
        "the like) you attributed to a {cite:Sn} source whose content does not contain it. Fix it: "
        "correct the fact to exactly match the cited source, cite a source that actually contains it, "
        "remove the claim, or, if you can't ground it, abstain on that specific fact and point the "
        "user to 311 or the official page. Do not repeat the unsupported fact.\n"
        "</system-reminder>"
    )


def _strip_ungrounded_claims(text: str, result: GroundingResult) -> str:
    """Tier 4: remove the sentence(s) carrying an ungrounded structured fact; if that guts the answer
    (nothing substantive left), return the abstention that routes to 311. Otherwise return the answer
    with the offending claim(s), and their now-orphaned citations, excised."""
    stripped = text
    for claim in {m.claim for m in result.hard_failures}:
        if claim and claim in stripped:
            stripped = stripped.replace(claim, " ")
    stripped = re.sub(r"[ \t]+", " ", stripped)
    stripped = re.sub(r" *\n *", "\n", stripped)
    stripped = re.sub(r"\n{3,}", "\n\n", stripped).strip()
    meaningful = _CITE_STRIP_RE.sub("", stripped).strip()
    if len(meaningful) < 40:  # the ungrounded fact was load-bearing → the whole answer must abstain
        return GROUNDING_ABSTAIN_FALLBACK
    return stripped


def _unknown_citation_ids(text: str, citations: dict) -> list[str]:
    """Return model-invented or stale citation ids in first-seen order."""
    return list(dict.fromkeys(
        match.group(1) for match in _CITE_MARKER_RE.finditer(text)
        if match.group(1) not in citations
    ))


def _unknown_citation_feedback(ids: list[str]) -> str:
    joined = ", ".join(ids)
    return (
        "<system-reminder>\n"
        f"Your last answer used citation ids that do not exist in this turn: {joined}. "
        "Regenerate the answer. Use only citation ids present in current tool results. Facts the "
        "user supplied do not need citations. If a factual claim has no current source, remove it "
        "or retrieve a source before stating it.\n"
        "</system-reminder>"
    )


def _discovery_citation_feedback(ids: list[str]) -> str:
    joined = ", ".join(ids)
    return (
        "<system-reminder>\n"
        f"Your last answer used search-result snippets as final evidence: {joined}. "
        "Search snippets are for discovery only. Fetch the relevant official page with "
        "official_sources and cite that evidence, or omit the unsupported claim and explain "
        "that you could not verify it.\n"
        "</system-reminder>"
    )


def _routing_query(user_message: str, history: Optional[list[dict]]) -> str:
    """Route detailed blurbs from the current turn plus recent resident messages."""
    context = [
        str(message.get("content") or "")
        for message in history or []
        if message.get("role") == "user"
    ][-2:]
    return "\n".join([*context, user_message])


def _history_messages(history: Optional[list[dict]]) -> list[dict]:
    """Return provider-safe dialogue history with prior assistant turns readable (F052).

    The model keeps full conversational continuity, what it already said, asked, or proposed,
    per the standard multi-turn chat contract. Stale-evidence safety stays deterministic and
    downstream: citation MARKERS are stripped here because every turn's registry reuses the
    same S-number ids, and the unknown-citation and grounding guards already check each new
    answer against the current turn's captured sources."""
    messages = []
    for message in history or []:
        content = str(message.get("content") or "")
        if message.get("role") == "assistant":
            sent = _sent_phrase(str(message.get("timestamp") or ""))
            if sent:
                label = (
                    f"[Earlier assistant reply, {sent}. The resident already read it: treat "
                    "what it establishes as shared context and do not re-introduce or "
                    "re-explain it. Weigh that time against the current date: retrieve "
                    "current evidence before restating anything that may have changed since.]\n"
                )
            else:
                label = (
                    "[Earlier assistant reply. The resident already read it: treat what it "
                    "establishes as shared context and do not re-introduce or re-explain it. "
                    "Its facts and links may be stale: retrieve current evidence before "
                    "restating any of them.]\n"
                )
            content = label + _CITE_STRIP_RE.sub("", content)
        messages.append({"role": message.get("role"), "content": content})
    return messages


def _sent_phrase(timestamp: str) -> str:
    """`sent 2026-07-15 09:30 ET` from a turn's ISO timestamp, or "" when absent/unreadable."""
    try:
        sent = datetime.fromisoformat(timestamp).astimezone(NYC_TZ)
    except ValueError:
        return ""
    return f"sent {sent.strftime('%Y-%m-%d %H:%M')} ET"


def turn_timestamp() -> str:
    """The ISO NYC-time stamp recorded on each committed turn (F062: the history label states
    when a reply was sent, so the model weighs real elapsed time instead of a blanket warning)."""
    return datetime.now(NYC_TZ).isoformat(timespec="seconds")


def _is_broad_event_query(user_message: str) -> bool:
    low = user_message.lower()
    return (
        any(term in low for term in ("event", "what's on", "whats on", "happening", "things to do"))
        and any(term in low for term in ("today", "tonight", "weekend", "this week"))
    )


# Practical get-ready-for-an-event turns (F046): a preparation intent, a public-event noun, and a
# date reference. The event name may be abbreviated or ambiguous ("WC game"), so the contract is
# resolve-or-clarify from current sources, never advise from memory. English and Spanish, matching
# the safety-routing convention used by the other intent regexes in this file.
_EVENT_PREP_INTENT_RE = re.compile(
    r"\b(?:prepare|prep|get ready|ready for|bring|wear|pack|plan(?:ning)?|"
    r"preparar(?:me|nos)?|llevo|llevar|empacar|listos?)\b",
    re.IGNORECASE,
)
_EVENT_PREP_EVENT_RE = re.compile(
    r"\b(?:game|match|concert|show|festival|parade|watch party|final|semifinal|opening|"
    r"premiere|race|marathon|gala|rally|ceremony|"
    r"partido|concierto|desfile|marat[oó]n|ceremonia)\b",
    re.IGNORECASE,
)
# A dated "prepare and bring" turn about a hearing, court date, official interview, or an
# immigration matter is a high-stakes civic turn, never event planning, even when a word like
# "show" or "final" doubles as an event noun in it.
# ponytail: an enumerated carve-out, not a semantic classifier. The semantic scope gate and the
# high-stakes reminder branches still run first; extend this list when a new collision appears.
_EVENT_PREP_EXCLUDE_RE = re.compile(
    r"\b(?:hearing|court|appeal|interview|appointment|immigration|asylum|"
    r"audiencia|corte|apelaci[oó]n|entrevista|cita|inmigraci[oó]n|asilo)\b",
    re.IGNORECASE,
)
_EVENT_PREP_DATE_RE = re.compile(
    r"\b(?:today|tonight|tom{1,2}or{1,2}ow|tmrw|tm|tn|weekend|this week|next week|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"hoy|esta noche|ma[ñn]ana|fin de semana|"
    r"lunes|martes|mi[eé]rcoles|jueves|viernes|s[aá]bado|domingo)(?:'?s)?\b",
    re.IGNORECASE,
)


def is_event_preparation_query(user_message: str) -> bool:
    """A dated practical event-preparation turn that needs identity resolution before advice.

    Public because `heynyc/modules/events/tools.py` uses the same predicate to pick its
    coordinated retrieval lanes and synthesis rules, keeping the two layers from drifting."""
    routed = _routing_text(user_message)
    return bool(
        _EVENT_PREP_INTENT_RE.search(routed)
        and _EVENT_PREP_EVENT_RE.search(routed)
        and _EVENT_PREP_DATE_RE.search(routed)
        and not _EVENT_PREP_EXCLUDE_RE.search(routed)
    )


_EVENT_PREPARATION_SCOPE_REMINDER = (
    "This turn asks how to prepare for a specific dated event whose name may be abbreviated or "
    "ambiguous. Resolve the event identity from current retrieved sources before giving any "
    "advice: call `whats_on_events` with the event keyword and use the current official and "
    "editorial context it returns. If the evidence supports one plausible event, state plainly "
    "which event it is with its date and local time, then build the plan only from cited "
    "evidence: how to attend or watch it, ticket or reservation status, venue access, transit or "
    "street impacts, and any material advisory, each with its direct link. If more than one "
    "event remains plausible, ask one short clarifying question instead of guessing. The event "
    "happening on the asked date is the answer; a more prominent event on a different date is "
    "context at most, never the resident's event. Residents "
    "often use texting shorthand: expand an abbreviated event name to its likely full name for "
    "the `whats_on_events` keyword instead of searching the raw abbreviation, and retry once "
    "with a broader keyword if the first search returns nothing relevant. In a follow-up turn, "
    "keep the event already under discussion instead of re-asking for details the resident "
    "already gave. State the event's own date plainly, and say so when it is not on the exact "
    "day the resident asked about. Do not "
    "give generic packing advice unless a retrieved advisory or forecast supports it. Pure "
    "predictions and sports trivia remain out of scope."
)

_BULLET_LINE_RE = re.compile(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+")
# Imperative advice verbs that mark UNCITED text as smuggled preparation advice rather than a
# clarifying question ("Wear team colors. Which game do you mean?").
_PREP_ADVICE_VERB_RE = re.compile(
    r"\b(?:wear|bring|pack|carry|grab|charge|lleva|llevar|trae|empaca|carga)\b",
    re.IGNORECASE,
)


def _event_preparation_feedback(
    user_message: str,
    text: str,
    citations: dict[str, dict],
    available_citation_ids: Optional[set[str]] = None,
    preparation_turn: Optional[bool] = None,
) -> Optional[str]:
    """Reject an event-preparation answer that is neither grounded nor a clarification.

    Deterministic floor for the F046 contract: a preparation answer must carry cited current
    evidence or ask one clarifying question, and must not carry uncited generic-advice lists.
    Event-identity quality beyond that floor is judged semantically by the eval suite.
    `preparation_turn` is the semantic scope-preflight flag; the regex predicate is only the
    fallback for callers without the preflight."""
    active = (
        is_event_preparation_query(user_message)
        if preparation_turn is None else preparation_turn
    )
    if not active:
        return None
    available_ids = set(citations) if available_citation_ids is None else available_citation_ids
    answer_body = re.split(r"(?im)^\s*(?:sources?|fuentes):", text, maxsplit=1)[0]
    cited = [m for m in _CITE_MARKER_RE.finditer(answer_body) if m.group(1) in available_ids]
    if not cited:
        # One SHORT clarifying question is the accepted uncited path. A trailing question does
        # not launder a plan: any bullet list, long body, or advice verb still needs cited
        # evidence.
        body = answer_body.strip()
        if (
            "?" in body
            and len(body) <= 280
            and not _BULLET_LINE_RE.search(body)
            and not _PREP_ADVICE_VERB_RE.search(body)
        ):
            return None
        return (
            "<system-reminder>\n"
            "This is a preparation question about a specific dated event, but your answer has no "
            "cited current evidence and is not one short clarifying question. Resolve which event "
            "the resident means from retrieved sources, state it with its date and local time, and "
            "build the plan from cited evidence with direct links, or ask one short clarifying "
            "question and stop. Do not give uncited generic preparation advice.\n"
            "</system-reminder>"
        )
    # A filler-led answer opens with uncited generic advice; a long grounded resolution
    # sentence whose citation lands at its end is fine (observed live), so the lead is judged
    # by advice content, never by length.
    lead = answer_body[: cited[0].start()]
    if len(_BULLET_LINE_RE.findall(lead)) >= 2 or _PREP_ADVICE_VERB_RE.search(lead):
        return (
            "<system-reminder>\n"
            "Your event-preparation answer leads with generic uncited advice before any cited "
            "fact. Lead with the resolved event: which event it is, with its date and local time, "
            "from cited evidence. Keep only preparation advice tied to retrieved conditions or "
            "advisories, and put each option's direct link beside it.\n"
            "</system-reminder>"
        )
    # An uncited packing LIST is filler wherever it sits, including after a citation: any
    # citation-free run of 2+ advice-verb bullet lines is rejected.
    for segment in _CITE_STRIP_RE.split(answer_body)[1:]:
        if len(_BULLET_LINE_RE.findall(segment)) >= 2 and _PREP_ADVICE_VERB_RE.search(segment):
            return (
                "<system-reminder>\n"
                "Your event-preparation answer contains an uncited generic advice list. Keep "
                "only preparation advice tied to cited retrieved conditions or advisories, with "
                "the supporting citation beside it, and drop the rest.\n"
                "</system-reminder>"
            )
    return None


_URGENT_NOTIFY_TITLE_RE = re.compile(
    r"\b(?:advisory|warning|emergency|alert)\b", re.IGNORECASE,
)


def _notify_subject(citation: dict) -> str:
    title = str(citation.get("title") or "")
    title = re.sub(r"^Notify NYC\s*-?\s*", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\([^)]*\)|\b\d+(?:[/-]\d+)*\b", " ", title)
    return " ".join(title.replace("-", " ").split()).strip()


def _is_notify_url(url: str) -> bool:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    return (
        host == "a858-nycnotify.nyc.gov"
        and parsed.path.lower().endswith("/notifynyc/home/recentmessages")
    ) or host == "everbridge.net" or host.endswith(".everbridge.net")

def _history_already_cites_notify(history) -> bool:
    """A prior assistant turn in THIS conversation already delivered a Notify NYC citation.
    Gates the forced advisories prefetch to once per conversation (F080): re-forcing every
    turn re-injects the full report, which the model re-briefs, and steals the first tool
    round from whatever the resident just asked for. The model can still call the tool."""
    for message in history or []:
        if message.get("role") != "assistant":
            continue
        for cite in (message.get("citations") or {}).values():
            if _is_notify_url(str(cite.get("url") or "")):
                return True
    return False



def _delivered_notify_titles(history) -> frozenset:
    """Normalized titles of Notify NYC citations delivered by prior assistant turns in THIS
    conversation (F080 residual): threaded into ToolContext so a repeat nyc_advisories call
    can return an already-shared marker instead of the full payload the model would re-brief."""
    titles = set()
    for message in history or []:
        if message.get("role") != "assistant":
            continue
        for cite in (message.get("citations") or {}).values():
            if _is_notify_url(str(cite.get("url") or "")):
                title = str(cite.get("title") or "").strip().casefold()
                if title:
                    titles.add(title)
    return frozenset(titles)


def _action_url(citation: dict) -> str:
    return _normalize_url(str(citation.get("url") or ""))


def _normalize_url(url: str) -> str:
    return url.rstrip(".,;:!?").split("#", 1)[0].rstrip("/")


def _urls_in(text: str) -> set[str]:
    return {_normalize_url(match.group()) for match in _HTTP_URL_RE.finditer(text)}


def _is_broad_notify_citation(citation: dict) -> bool:
    url = str(citation.get("url") or "").lower()
    if urlparse(url).path.lower().endswith("/notifynyc/home/recentmessages"):
        # F061: recent notes carry no structured area field, and parsing their prose broke on
        # real borough-list spellings — fail OPEN and let the model judge from the full text.
        return True
    snapshot = citation.get("provenance", {}).get("snapshot", {})
    return _is_notify_url(url) and is_citywide_area(str(snapshot.get("areaDesc") or ""))


def _attach_event_action_urls(
    text: str,
    citations: dict[str, dict],
    available_citation_ids: Optional[set[str]] = None,
) -> str:
    answer_body = re.split(r"(?im)^\s*(?:sources?|fuentes):", text, maxsplit=1)[0]
    available_ids = set(citations) if available_citation_ids is None else available_citation_ids
    for cid in dict.fromkeys(_CITE_MARKER_RE.findall(answer_body)):
        if cid not in available_ids:
            continue
        citation = citations.get(cid, {})
        url = _action_url(citation)
        if not url or url in _urls_in(answer_body):
            continue
        marker = f"{{cite:{cid}}}"
        label = "Alert source" if _is_notify_url(url) else "Details"
        text = text.replace(marker, f"{marker}\n  {label}: {url}", 1)
        answer_body = re.split(r"(?im)^\s*(?:sources?|fuentes):", text, maxsplit=1)[0]
    return text


def _citation_coordinates(citation: dict) -> Optional[tuple[float, float]]:
    provenance = citation.get("provenance") or {}
    derivation = provenance.get("derivation") or {}
    point = derivation.get("point")
    if isinstance(point, (list, tuple)) and len(point) >= 2:
        values = point[:2]
    else:
        snapshot = provenance.get("snapshot") or {}
        values = (
            snapshot.get("lat", snapshot.get("latitude")),
            snapshot.get("lon", snapshot.get("longitude")),
        )
    try:
        lat, lon = float(values[0]), float(values[1])
    except (TypeError, ValueError):
        return None
    if not (40.45 <= lat <= 40.95 and -74.30 <= lon <= -73.65):
        return None
    return lat, lon


_LOCATION_BLOCK_SPLIT_RE = re.compile(
    r"(?m)\n\s*\n|(?=^\s*(?:[-*•]\s+|\d+[.)]\s+))"
)


def _attach_location_action_urls(
    text: str,
    citations: dict[str, dict],
    available_citation_ids: Optional[set[str]] = None,
) -> str:
    """Keep a usable map beside every cited NYC location when the model drops it."""
    answer_body = re.split(r"(?im)^\s*(?:sources?|fuentes):", text, maxsplit=1)[0]
    available_ids = set(citations) if available_citation_ids is None else available_citation_ids
    limitations: list[str] = []
    for cid in dict.fromkeys(_CITE_MARKER_RE.findall(answer_body)):
        citation = citations.get(cid, {})
        if cid not in available_ids or citation.get("kind") != "DATA":
            continue
        limitation = str(
            (citation.get("provenance") or {}).get("derivation", {}).get("limitations") or ""
        ).strip()
        if limitation and limitation not in limitations:
            limitations.append(limitation)
        marker = f"{{cite:{cid}}}"
        blocks = _LOCATION_BLOCK_SPLIT_RE.split(answer_body)
        if str(citation.get("title") or "").casefold() == "nyc emergency management cool options":
            for block in blocks:
                if marker not in block:
                    continue
                updated = re.sub(
                    r"(?<!scheduled )\bopen (now|today)\b",
                    r"scheduled open \1",
                    block,
                    flags=re.I,
                )
                text = text.replace(block, updated, 1)
                answer_body = re.split(r"(?im)^\s*(?:sources?|fuentes):", text, maxsplit=1)[0]
                blocks = _LOCATION_BLOCK_SPLIT_RE.split(answer_body)
                break
        coordinates = _citation_coordinates(citation)
        if coordinates is None:
            continue
        url = maps_link(*coordinates)
        if any(marker in block and url in _urls_in(block) for block in blocks):
            continue
        text = text.replace(marker, f"{marker}\n  Directions: {url}", 1)
        answer_body = re.split(r"(?im)^\s*(?:sources?|fuentes):", text, maxsplit=1)[0]
    low_body = answer_body.casefold()
    plain_body = re.sub(r"[*_~`]", "", low_body)
    missing_limits = [
        limitation for limitation in limitations
        if limitation not in answer_body
        and not (
            "not a live guarantee" in limitation.casefold()
            and (
                "not a live guarantee" in plain_body
                or re.search(
                    r"\b(?:doesn['’]t|does not|not)\s+guarantee\b.{0,100}\b(?:work|working|available)\b",
                    plain_body,
                )
            )
        )
    ]
    if missing_limits:
        note = "\n".join(f"Source limit: {limitation}" for limitation in missing_limits)
        parts = re.split(r"(?im)(^\s*(?:sources?|fuentes):)", text, maxsplit=1)
        text = f"{parts[0].rstrip()}\n\n{note}"
        if len(parts) > 1:
            text += f"\n\n{parts[1]}{parts[2]}"
    return text


def _broad_event_context_feedback(
    user_message: str,
    text: str,
    citations: dict[str, dict],
    tools_made: list[str],
    available_citation_ids: Optional[set[str]] = None,
    discovery_turn: Optional[bool] = None,
) -> Optional[str]:
    if "whats_on_events" not in tools_made:
        return None
    # `discovery_turn` is the resolved semantic scope-preflight signal; the broad-events regex
    # is only the fallback for callers without the preflight (mirrors `_event_preparation_feedback`).
    active = _is_broad_event_query(user_message) if discovery_turn is None else discovery_turn
    if not active:
        return None

    available_ids = set(citations) if available_citation_ids is None else available_citation_ids
    answer_body = re.split(r"(?im)^\s*(?:sources?|fuentes):", text, maxsplit=1)[0]
    cited_ids = set(_CITE_MARKER_RE.findall(answer_body))
    missing = []
    broad_notify = {
        cid: citation for cid, citation in citations.items()
        if cid in available_ids
        and _URGENT_NOTIFY_TITLE_RE.search(str(citation.get("title") or ""))
        and _is_broad_notify_citation(citation)
    }
    unnamed_notify = {
        cid: citation for cid, citation in broad_notify.items()
        if cid not in cited_ids
        or _notify_subject(citation).casefold() not in answer_body.replace("-", " ").casefold()
    }
    if unnamed_notify:
        missing.append("a broadly applicable current Notify NYC warning")
    notify_ids = {
        cid for cid, citation in citations.items()
        if _is_notify_url(str(citation.get("url") or ""))
    }
    answer_blocks = re.split(
        r"(?m)\n\s*\n|(?=^\s*[-*•]\s+)", answer_body,
    )
    missing_action_ids = {
        cid for cid in cited_ids - notify_ids
        if cid in citations and cid in available_ids
        and not any(
            f"{{cite:{cid}}}" in block
            and _action_url(citations[cid]) in _urls_in(block)
            for block in answer_blocks
        )
    }
    if missing_action_ids:
        missing.append("a direct URL beside each cited event option")
    if not missing:
        return None
    notify_refs = "; ".join(
        f"{cid}: {_notify_subject(citation)} - {citation.get('snippet', '')}"
        for cid, citation in sorted(unnamed_notify.items())
    )
    action_refs = "; ".join(
        f"{cid}: {citations[cid].get('title', '')} - "
        f"{_action_url(citations[cid])}"
        for cid in sorted(missing_action_ids)
    )
    return (
        "<system-reminder>\n"
        f"Your broad current-events answer omitted {', and '.join(missing)} from the evidence "
        "already retrieved by `whats_on_events`. Regenerate a concise answer using those current "
        "citation ids. Do not call anything free unless its cited source says so. If an advisory "
        "applies today but not to the requested weekend, label it as a separate today-only heads-up "
        "and do not present it as a weekend forecast.\n"
        f"Broad Notify NYC evidence: {notify_refs or 'none'}\n"
        f"Direct event URLs to place beside their options: {action_refs or 'none'}\n"
        "For each recommended event, include its direct URL and any known date, time, place, and "
        "ticket or reservation step. Do not invent a missing detail.\n"
        "</system-reminder>"
    )

# Non-streaming model fn: (messages, tool_schemas) -> assistant message dict.
CompletionFn = Callable[[list[dict], list[dict]], Awaitable[dict]]
# Streaming model fn: yields {"type":"text","text":...} deltas then a terminal
# {"type":"message","message": {role, content, tool_calls}}.
StreamFn = Callable[[list[dict], list[dict]], AsyncIterator[dict]]
# Approval callback for side-effecting tools: (name, args) -> approved?
Approver = Callable[[str, dict], Awaitable[bool]]
NotifyAwarenessFn = Callable[[], Awaitable[str]]


@dataclass(frozen=True)
class ScopeResult:
    # The preflight carries no allow/deny verdict anymore (RULED 2026-07-21): it is a checklist plus
    # the event signal. Every turn reaches the answer model; these fields only configure retrieval.
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    event_turn: str | None = None
    modules: tuple[str, ...] = ()
    situations: tuple[str, ...] = ()
    # Prompt-cache read for the scope call's static prefix (prompt_tokens_details.cached_tokens),
    # captured the same way the answer stream captures its own, so both calls' cache rates surface.
    cached_input_tokens: int = 0


ScopeFn = Callable[[str, list[dict]], Awaitable[ScopeResult]]
MemoryTokenCounter = Callable[[list[dict], list[dict]], int]
MemoryCompactor = Callable[
    [list[dict], Optional[ContinuityRecord]],
    Awaitable[ContinuityRecord | dict],
]


@dataclass
class AgentResult:
    text: str
    citations: dict[str, dict]
    tool_calls_made: list[str] = field(default_factory=list)
    iterations: int = 0
    hit_max_iters: bool = False
    status: str = "success"
    messages: list[dict] = field(default_factory=list)
    usage: dict = field(default_factory=dict)  # {input_tokens, output_tokens, latency_ms} per turn
    diagnostics: dict = field(default_factory=dict)


class Agent:
    def __init__(
        self,
        registry: Registry,
        tools: Optional[dict[str, Tool]] = None,
        model: Optional[str] = None,
        complete_fn: Optional[CompletionFn] = None,
        stream_fn: Optional[StreamFn] = None,
        approver: Optional[Approver] = None,
        index=None,
        notify_awareness: Optional[NotifyAwarenessFn] = None,
        scope_fn: Optional[ScopeFn] = None,
        scope_gate: bool = False,
        guard_grounding: bool = True,
        guard_max_retries: int = GUARD_MAX_RETRIES,
        spend_cap: Optional[float] = None,
        memory_limit_tokens: Optional[int] = None,
        memory_token_counter: Optional[MemoryTokenCounter] = None,
        memory_compactor: Optional[MemoryCompactor] = None,
        reasoning_effort: Optional[str] = None,
    ):
        self.registry = registry
        self._reasoning_effort = (
            reasoning_effort if reasoning_effort is not None else config.HEYNYC_REASONING_EFFORT
        )
        self._embedder = getattr(index, "embedder", None)  # shared with retrieval-using module tools
        self.tools = tools if tools is not None else build_toolbox(registry, index=index)
        self.model = model or config.HEYNYC_MODEL
        self._approver = approver
        self._notify_awareness = notify_awareness
        # Session spend cap (OWASP LLM10). Defaults to the env-configured ceiling; None keeps it OFF
        # so behavior is unchanged unless HEYNYC_SPEND_CAP is set. Accumulates cost across this
        # instance's turns, so a Conversation caps the whole conversation.
        self._spend = SpendGuard(config.HEYNYC_SPEND_CAP if spend_cap is None else spend_cap)
        # Deterministic post-generation grounding guard. On by default, it's the safety floor that lets
        # HeyNYC run on a cheaper model; disable only to observe raw model output (see tests).
        self.guard_grounding = guard_grounding
        self.guard_max_retries = guard_max_retries
        self._memory_limit_tokens = memory_limit_tokens
        self._memory_token_counter = memory_token_counter
        self._memory_compactor = memory_compactor
        if stream_fn is not None:
            self._stream_fn = stream_fn
            self._uses_litellm = False
        elif complete_fn is not None:
            self._stream_fn = _wrap_complete(complete_fn)
            self._uses_litellm = False
        else:
            self._stream_fn = None
            self._uses_litellm = True
        if self._uses_litellm:
            _register_extra_model_info()
        self._scope_fn = scope_fn or (self._classify_scope if scope_gate else None)

    def _context_capacity(self) -> int | None:
        return context_capacity(
            self.model,
            self._memory_limit_tokens,
            self._uses_litellm,
        )

    def _memory_request_tokens(
        self,
        user_message: str,
        history: list[dict],
        continuity: ContinuityRecord | None,
        reminders: Optional[list[str]],
    ) -> int:
        effective_reminders = list(reminders or [])
        scope_reminder = self._runtime_scope_reminder(user_message)
        if scope_reminder and scope_reminder not in effective_reminders:
            effective_reminders.append(scope_reminder)
        if continuity is not None:
            effective_reminders.append(continuity_reminder(continuity))
        messages = self._build_messages(user_message, history, effective_reminders)
        schemas = self._tool_schemas()
        return request_tokens(
            self.model,
            messages,
            schemas,
            self._memory_token_counter,
        )

    async def _compact_memory(
        self,
        older: list[dict],
        current: ContinuityRecord | None,
    ) -> tuple[ContinuityRecord, dict]:
        return await compact_memory(older, current, self._spend)

    async def prepare_memory_context(
        self,
        user_message: str,
        history: list[dict],
        continuity: ContinuityRecord | None,
        reminders: Optional[list[str]] = None,
    ) -> tuple[ContextPlan, dict]:
        """Select bounded dialogue and compact older turns once, only under measured pressure."""
        capacity = self._context_capacity()
        if capacity is None and not self._uses_litellm and self._memory_limit_tokens is None:
            return ContextPlan(history=list(history), continuity=continuity, compacted=False,
                               pre_compaction_tokens=0, post_compaction_tokens=0), {
                "memory_compactions": 0,
            }
        compaction_usage: dict = {}

        async def compact(older, current):
            try:
                if self._memory_compactor is not None:
                    return await self._memory_compactor(older, current)
                record, usage = await self._compact_memory(older, current)
                compaction_usage.update(usage)
                return record
            except ContextCapacityError:
                raise
            except Exception as exc:
                raise ContextCapacityError("continuity compaction is unavailable") from exc

        plan = await prepare_context(
            history,
            continuity,
            budget=capacity,
            measure=lambda selected, record: self._memory_request_tokens(
                user_message, selected, record, reminders,
            ),
            compact=compact,
        )
        return plan, {
            "memory_compactions": int(plan.compacted),
            "memory_pre_tokens": plan.pre_compaction_tokens,
            "memory_post_tokens": plan.post_compaction_tokens,
            **compaction_usage,
        }

    def _request_fits_context(self, messages: list[dict], schemas: list[dict]) -> bool:
        capacity = self._context_capacity()
        if capacity is None:
            return not self._uses_litellm and self._memory_limit_tokens is None
        try:
            tokens = request_tokens(
                self.model,
                messages,
                schemas,
                self._memory_token_counter,
            )
        except Exception:
            logger.exception("could not verify current model request size")
            return False
        return tokens <= capacity

    async def _classify_scope(self, user_message: str, history: list[dict]) -> ScopeResult:
        """One schema-bound, no-tools model call that classifies the turn for retrieval: the module /
        situation checklist and the event signal. It carries no allow/deny verdict (RULED 2026-07-21):
        every turn reaches the answer model regardless of what this returns."""
        import litellm

        transcript = [
            {"role": message.get("role"), "content": str(message.get("content") or "")}
            for message in history
            if message.get("role") in {"user", "assistant"}
        ]
        transcript.append({"role": "user", "content": user_message})
        # The module checklist section: names plus meaning-based definitions from the
        # registry (top-level modules), assembled at call time so adding a module extends
        # the checklist with zero core changes. Definitions are meaning, never word lists.
        module_lines = "\n".join(
            f"{module.name}: {' '.join(str(module.description or '').split())[:140]}"
            for module in self.registry.modules
            if getattr(module, "parent", None) is None
        )
        situation_lines = "\n".join(
            f"{hint.name}: {' '.join(hint.definition.split())}"
            for _module, hint in self.registry.situation_hints().values()
        )
        situations_section = (
            "\n\nAlso return situations: the list of declared situations below that the turn "
            "matches by meaning, in any language, empty when none apply.\n" + situation_lines
        ) if situation_lines else ""
        checklist_section = (
            "\n\nAlso return modules: the list of service modules this turn touches, chosen "
            "only from the list below, judged by meaning in any language, empty when none "
            "apply. This is a checklist for grounding, not a route: pick every module whose "
            "sources could help.\n" + module_lines + situations_section
        ) if module_lines else ""
        kwargs = {
            "model": config.HEYNYC_SCOPE_MODEL,
            "messages": [
                {"role": "system", "content": _SCOPE_SYSTEM_PROMPT + checklist_section},
                {"role": "user", "content": json.dumps(transcript, ensure_ascii=False)},
            ],
            "response_format": _ScopeDecision,
            "max_completion_tokens": 128,
            "stream": False,
            "timeout": 15,
        }
        if config.HEYNYC_SCOPE_MODEL.startswith("openai/"):
            kwargs["reasoning_effort"] = "low"

        def usage_value(response_usage, name: str) -> int:
            value = (
                response_usage.get(name, 0)
                if isinstance(response_usage, dict)
                else getattr(response_usage, name, 0)
            )
            return int(value or 0)

        def cached_value(response_usage) -> int:
            details = (
                response_usage.get("prompt_tokens_details")
                if isinstance(response_usage, dict)
                else getattr(response_usage, "prompt_tokens_details", None)
            )
            if details is None:
                return 0
            cached = (
                details.get("cached_tokens", 0)
                if isinstance(details, dict)
                else getattr(details, "cached_tokens", 0)
            )
            return int(cached or 0)

        input_tokens = 0
        output_tokens = 0
        cached_input_tokens = 0
        event_turn: Optional[str] = None
        checked_modules: tuple[str, ...] = ()
        checked_situations: tuple[str, ...] = ()
        known_modules = {
            module.name for module in self.registry.modules
            if getattr(module, "parent", None) is None
        }
        known_situations = set(self.registry.situation_hints())
        # One retry on transiently EMPTY structured output (observed live), then give up quietly with
        # an empty signal. There is no verdict to fail closed on: an unusable scope reply just means
        # the turn reaches the answer model with no checklist config, never a canned wall.
        for attempt in range(2):
            try:
                response = await litellm.acompletion(**kwargs)
            except Exception:
                logger.exception("scope model call failed; returning empty signal")
                return ScopeResult(
                    model=config.HEYNYC_SCOPE_MODEL,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=(
                        priced_cost_usd(
                            config.HEYNYC_SCOPE_MODEL, input_tokens, output_tokens,
                            cached_input_tokens=cached_input_tokens,
                        )
                        if input_tokens or output_tokens else None
                    ),
                    cached_input_tokens=cached_input_tokens,
                )
            message = response.choices[0].message
            response_usage = getattr(response, "usage", None)
            input_tokens += usage_value(response_usage, "prompt_tokens")
            output_tokens += usage_value(response_usage, "completion_tokens")
            cached_input_tokens += cached_value(response_usage)
            if getattr(message, "refusal", None):
                break
            parsed = getattr(message, "parsed", None)
            content = (message.content or "").strip()
            if parsed is None and not content:
                if attempt == 0:
                    continue
                logger.warning("scope model returned empty output twice; returning empty signal")
                break
            try:
                verdict = (
                    parsed if isinstance(parsed, _ScopeDecision)
                    else _ScopeDecision.model_validate_json(content)
                )
                event_turn = verdict.event_turn
                checked_modules = tuple(
                    name for name in verdict.modules if name in known_modules
                )
                checked_situations = tuple(
                    name for name in verdict.situations if name in known_situations
                )
            except Exception:
                logger.exception("scope model returned invalid structured output; empty signal")
            break
        return ScopeResult(
            model=config.HEYNYC_SCOPE_MODEL,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=priced_cost_usd(
                config.HEYNYC_SCOPE_MODEL, input_tokens, output_tokens,
                cached_input_tokens=cached_input_tokens,
            ),
            event_turn=event_turn,
            modules=checked_modules,
            situations=checked_situations,
            cached_input_tokens=cached_input_tokens,
        )

    def _tool_schemas(self, excluded_tools: Optional[set[str]] = None) -> list[dict]:
        excluded = excluded_tools or set()
        return [tool.schema() for name, tool in self.tools.items() if name not in excluded]

    def _runtime_scope_reminder(self, user_message: str) -> str:
        """Return the one current-source reminder added to the real request."""
        if "official_sources" in self.tools or "web_search" in self.tools:
            lockout_entry = self.registry.situation_hints().get("active_lockout")
            if lockout_entry is not None and _needs_current_lockout_guidance(user_message):
                return lockout_entry[1].reminder
            snap_entry = self.registry.situation_hints().get("snap_work_rules")
            if snap_entry is not None and _needs_current_snap_work_rule_guidance(user_message):
                return snap_entry[1].reminder
            if _needs_current_immigrant_benefits_guidance(user_message):
                return _IMMIGRANT_BENEFITS_SCOPE_REMINDER
            if _needs_current_benefits_recovery_guidance(user_message):
                return _BENEFITS_RECOVERY_SCOPE_REMINDER
            if _current_civic_law_search(user_message):
                return _CIVIC_LAW_SCOPE_REMINDER
        if "whats_on_events" in self.tools and is_event_preparation_query(user_message):
            return _EVENT_PREPARATION_SCOPE_REMINDER
        return ""

    async def get_notify_awareness(self) -> str:
        """Fetch the optional citywide-awareness reminder once per resident turn."""
        if self._notify_awareness is None:
            return ""
        try:
            return await self._notify_awareness()
        except Exception:
            logger.exception("Notify NYC awareness refresh failed")
            return ""

    def _grounding_verdict(self, text: str, citations_map: dict, query: str, retries: int):
        """Run the deterministic grounding guard on a terminal answer. Returns
        (action, out_text, result):
          "pass"    → ship out_text unchanged (nothing to verify, or only SOFT mismatches)
          "retry"   → feed _grounding_feedback(result) back and let the model regenerate (Tier 3)
          "abstain" → ship out_text (offending claim stripped, or the abstention fallback) (Tier 4)
        The guard acts ONLY on a HARD (blocking) mismatch, a verbatim structured fact absent from an
        all-complete-capture source. Soft mismatches pass through, so a correct answer is never
        over-blocked."""
        if not self.guard_grounding:
            return "pass", text, None
        result = check_grounding(text, citations_map, query)
        if result is None or not result.blocking:
            return "pass", text, result
        if retries < self.guard_max_retries:
            return "retry", text, result
        return "abstain", _strip_ungrounded_claims(text, result), result

    def _system_message(self, stable: str) -> dict:
        """The system message for THIS turn: the STABLE prefix ONLY (safety rules, capability menu,
        and the byte-static conversation/reply-language rules). It is query- and time-independent, so
        it is byte-identical across turns and the growing prefix (stable system + history) caches.

        For an Anthropic model we emit `content` as one text block marked with cache_control so repeat
        calls read that prefix from cache (~90% cheaper). Every other provider (openai, ollama, ...)
        gets a plain-string `content`. The volatile tier (date line + query-selected blurbs) is NOT
        here: `_build_messages` injects it after history so it never breaks the cached prefix."""
        if _is_anthropic(self.model):
            return {"role": "system", "content": [
                {"type": "text", "text": stable, "cache_control": {"type": "ephemeral"}},
            ]}
        return {"role": "system", "content": stable}

    def _build_messages(
        self, user_message: str, history, reminders, citations: Optional[CitationRegistry] = None,
    ) -> list[dict]:
        if citations is None:
            citations = CitationRegistry()
        stable, volatile = build_system_prompt_tiers(
            self.registry, query=_routing_query(user_message, history),
        )
        messages: list[dict] = [self._system_message(stable)]
        messages.extend(_history_messages(history))
        # The volatile tier (current-date line + query-selected blurbs) rides a reminder-shaped user
        # message placed AFTER history, next to the user turn. Keeping these mutables out of the
        # system prefix is what lets the growing history cache (static-first / dynamic-last); the
        # existing reminders (scope, continuity) already sit in this same post-history band.
        volatile = volatile.lstrip("\n")
        if volatile:
            messages.append(
                {"role": "user", "content": f"<system-reminder>\n{volatile}\n</system-reminder>"}
            )
        for reminder in reminders or []:
            messages.append({"role": "user", "content": f"<system-reminder>\n{reminder}\n</system-reminder>"})
        messages.append({"role": "user", "content": user_message})
        return messages

    async def stream(
        self,
        user_message: str,
        history: Optional[list[dict]] = None,
        max_iters: int = 8,
        reminders: Optional[list[str]] = None,
        output_dir=None,
        drafts=None,
        forced_tool: Optional[str] = None,
        forced_tool_args: Optional[dict] = None,
        excluded_tools: Optional[set[str]] = None,
        prefetched_notify_awareness: Optional[str] = None,
    ) -> AsyncIterator[events.Event]:
        """Run one turn, yielding events (text deltas, tool lifecycle, terminal done)."""
        initial_forced_tool = forced_tool
        initial_forced_args = forced_tool_args
        snap_work_rule_turn = False
        benefits_recovery_turn = False
        immigrant_benefits_turn = False
        lockout_turn = False
        civic_law_search = _current_civic_law_search(user_message)
        has_current_source = "official_sources" in self.tools or "web_search" in self.tools
        lockout_entry = self.registry.situation_hints().get("active_lockout")
        lockout_hint = lockout_entry[1] if lockout_entry is not None else None
        snap_entry = self.registry.situation_hints().get("snap_work_rules")
        snap_hint = snap_entry[1] if snap_entry is not None else None
        if (
            initial_forced_tool is None
            and has_current_source
            and lockout_hint is not None
            and _needs_current_lockout_guidance(user_message)
        ):
            lockout_turn = True
            initial_forced_tool, initial_forced_args = _current_source_call(
                self.tools, lockout_hint.query, tuple(lockout_hint.urls),
            )
        elif (
            initial_forced_tool is None
            and has_current_source
            and snap_hint is not None
            and _needs_current_snap_work_rule_guidance(user_message)
        ):
            snap_work_rule_turn = True
            initial_forced_tool, initial_forced_args = _current_source_call(
                self.tools, snap_hint.query, tuple(snap_hint.urls),
            )
        elif (
            initial_forced_tool is None
            and has_current_source
            and _needs_current_immigrant_benefits_guidance(user_message)
        ):
            immigrant_benefits_turn = True
            initial_forced_tool, initial_forced_args = _current_source_call(
                self.tools,
                f"{user_message} {_IMMIGRANT_BENEFITS_SEARCH_QUERY}",
                _IMMIGRANT_BENEFITS_URLS,
            )
        elif (
            initial_forced_tool is None
            and has_current_source
            and _needs_current_benefits_recovery_guidance(user_message)
        ):
            benefits_recovery_turn = True
            initial_forced_tool, initial_forced_args = _current_source_call(
                self.tools,
                f"{user_message} {_BENEFITS_RECOVERY_SEARCH_QUERY}",
                _BENEFITS_RECOVERY_URLS,
            )
        elif initial_forced_tool is None and has_current_source and civic_law_search:
            initial_forced_tool, initial_forced_args = _current_source_call(
                self.tools, civic_law_search, _current_civic_law_urls(civic_law_search),
            )
        current_source_required = bool(
            lockout_turn
            or snap_work_rule_turn
            or immigrant_benefits_turn
            or benefits_recovery_turn
            or civic_law_search
        )
        effective_reminders = list(reminders or [])
        scope_reminder = self._runtime_scope_reminder(user_message)
        if scope_reminder and scope_reminder not in effective_reminders:
            effective_reminders.append(scope_reminder)
        citations = CitationRegistry()
        messages = self._build_messages(user_message, history, effective_reminders, citations)
        effective_excluded_tools = set(excluded_tools or ())
        if lockout_turn and lockout_hint is not None:
            allowed = set(lockout_hint.focus_tools)
            effective_excluded_tools.update(name for name in self.tools if name not in allowed)
        elif snap_work_rule_turn and snap_hint is not None:
            allowed = set(snap_hint.focus_tools)
            effective_excluded_tools.update(name for name in self.tools if name not in allowed)
        elif immigrant_benefits_turn:
            allowed = _immigrant_benefits_allowed_tools()
            effective_excluded_tools.update(name for name in self.tools if name not in allowed)
        elif benefits_recovery_turn:
            allowed = _benefits_recovery_allowed_tools(user_message)
            effective_excluded_tools.update(name for name in self.tools if name not in allowed)
        elif civic_law_search:
            allowed = _civic_law_allowed_tools()
            effective_excluded_tools.update(name for name in self.tools if name not in allowed)
        user_turns = tuple(
            str(message.get("content") or "")
            for message in messages
            if message.get("role") == "user"
            and not str(message.get("content") or "").lstrip().startswith("<system-reminder>")
        )
        user_history = "\n".join(user_turns)
        ctx = ToolContext(citations=citations, registry=self.registry, query=user_message,
                          user_history=user_history, user_turns=user_turns, toolbox=self.tools,
                          embedder=self._embedder,
                          output_dir=output_dir, drafts=drafts,
                          delivered_notify_titles=_delivered_notify_titles(history))
        tools_made: list[str] = []
        tool_citation_ids: set[str] = set()
        guard_retries = 0  # how many times the grounding guard has bounced a terminal answer back
        reply_script = _dominant_non_latin_script(user_message)
        language_retries = 0
        turn_started = time.perf_counter()
        turn_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "answer_input_tokens": 0,
            "answer_output_tokens": 0,
            "answer_cached_input_tokens": 0,
            "model_time_ms": 0.0,
            "tool_time_ms": 0.0,
            "n_model_calls": 0,
            "n_answer_model_calls": 0,
            "n_tool_calls": 0,
            "iterations": 0,
            "scope_input_tokens": 0,
            "scope_output_tokens": 0,
            "scope_model": "",
            "scope_cost_usd": None,
            "scope_modules": [],
            "scope_situations": [],
        }

        def _usage() -> dict:
            latency_ms = (time.perf_counter() - turn_started) * 1000.0
            costs: list[float] = []
            unpriced = False
            if turn_usage["scope_model"]:
                if turn_usage["scope_cost_usd"] is None:
                    unpriced = True
                else:
                    costs.append(float(turn_usage["scope_cost_usd"]))
            if turn_usage["n_answer_model_calls"]:
                answer_cost = priced_cost_usd(
                    self.model,
                    turn_usage["answer_input_tokens"],
                    turn_usage["answer_output_tokens"],
                    cached_input_tokens=turn_usage["answer_cached_input_tokens"],
                )
                if answer_cost is None:
                    unpriced = True
                else:
                    costs.append(answer_cost)
            return {
                **turn_usage,
                "cost_usd": None if unpriced else sum(costs),
                "cost_status": "unpriced" if unpriced else "priced",
                "latency_ms": latency_ms,
                "orchestration_time_ms": max(
                    0.0,
                    latency_ms
                    - turn_usage["model_time_ms"]
                    - turn_usage["tool_time_ms"]
                    - turn_usage.get("scope_time_ms", 0.0),
                ),
            }
        for reminder in effective_reminders:
            yield events.Reminder(summary=reminder)

        emergency = _emergency_backstop_result(user_message)
        backstop_text = (emergency.text if emergency is not None else None) or (
            _sensitive_identifier_backstop(user_message)
            or _internal_config_backstop(user_message)
        )
        if backstop_text:
            backstop_text = _ground_emergency_backstop(
                backstop_text,
                citations,
                emergency.sources if emergency is not None else frozenset(),
            )
            message_id = "m0"
            assistant = {"role": "assistant", "content": backstop_text, "tool_calls": None}
            messages.append(assistant)
            yield events.MessageStart(message_id=message_id)
            yield events.TextDelta(message_id=message_id, text=backstop_text)
            yield events.MessageCompleted(
                message_id=message_id, text=backstop_text, citations=citations.mapping()
            )
            result = AgentResult(
                text=backstop_text, citations=citations.mapping(), tool_calls_made=tools_made,
                iterations=0, status="success", messages=messages, usage=_usage(),
                # The same diagnostics the Pydantic runtime records. Without these the rollback
                # path has no crisis telemetry at all, and `inv_harm_routing` fails every
                # self_harm case by construction however correct the response is
                diagnostics={
                    **(
                        {"safety_risk": emergency.risk}
                        if emergency is not None and emergency.risk is not None
                        else {}
                    ),
                    **(
                        {"safety_response_source": "deterministic"}
                        if emergency is not None
                        and (emergency.risk is not None or emergency.sources)
                        else {}
                    ),
                    **(
                        {
                            "deterministic_evidence_citations": sorted(
                                used_citations(backstop_text, citations.mapping())
                            )
                        }
                        if emergency is not None and emergency.sources
                        else {}
                    ),
                },
            )
            yield events.Done(
                status="success", num_turns=0, citations=result.citations, result=result
            )
            return

        scope_event_turn: Optional[str] = None
        scope_modules: tuple[str, ...] = ()
        scope_situations: tuple[str, ...] = ()
        if self._scope_fn is not None:
            scope_started = time.perf_counter()
            try:
                scope_result = await self._scope_fn(user_message, list(history or []))
            except Exception:
                logger.exception("scope classifier failed; returning empty signal")
                scope_result = ScopeResult(model="unknown/injected-scope", cost_usd=None)
            turn_usage["scope_time_ms"] = (time.perf_counter() - scope_started) * 1000.0
            # No gate: whatever the preflight returns, the turn proceeds to the answer model. A
            # non-ScopeResult (a legacy bare return) simply carries no checklist config.
            if isinstance(scope_result, ScopeResult):
                scope_event_turn = scope_result.event_turn
                scope_modules = scope_result.modules
                scope_situations = scope_result.situations
                turn_usage["scope_modules"] = list(scope_result.modules)
                turn_usage["scope_situations"] = list(scope_result.situations)
                turn_usage["scope_model"] = scope_result.model
                turn_usage["scope_input_tokens"] = scope_result.input_tokens
                turn_usage["scope_output_tokens"] = scope_result.output_tokens
                turn_usage["scope_cost_usd"] = scope_result.cost_usd
                turn_usage["scope_cached_input_tokens"] = scope_result.cached_input_tokens
                turn_usage["input_tokens"] += scope_result.input_tokens
                turn_usage["output_tokens"] += scope_result.output_tokens
                # Fold the scope call's cache read into the aggregate too (the answer stream adds its
                # own on top), so `heynyc stats` cached-token total covers both model calls.
                turn_usage["cached_input_tokens"] = (
                    turn_usage.get("cached_input_tokens", 0) + scope_result.cached_input_tokens
                )
                turn_usage["n_model_calls"] += 1
                if scope_result.cost_usd is None:
                    self._spend.mark_unpriceable()
                else:
                    self._spend.record(
                        scope_result.model, scope_result.input_tokens, scope_result.output_tokens,
                        scope_result.cached_input_tokens,
                    )

        # The semantic preflight tri-state is authoritative when present; the broad-events and
        # preparation regexes are the preflight-absent fallback only (F058, owner ruling: no
        # phrase-list growth, the regex is demoted, never deleted, until its retirement matrix
        # passes). When the preflight speaks, discovery vs preparation is its call, so broadness
        # no longer has to override an over-fired boolean: they are distinct signals now.
        if scope_event_turn is not None:
            event_turn = scope_event_turn
        elif is_event_preparation_query(user_message) and not _is_broad_event_query(user_message):
            event_turn = "preparation"
        elif _is_broad_event_query(user_message):
            event_turn = "discovery"
        else:
            event_turn = "none"
        event_preparation_turn = event_turn == "preparation"
        event_discovery_turn = event_turn == "discovery"
        ctx.event_turn = event_turn
        # A checked high-stakes SITUATION contributes its manifest-declared retrieval config
        # to this same turn (RULED: checklist, never a router). One mandatory-first fetch at
        # most; tool focus applies only on a single-module turn (prioritize, never narrow).
        if scope_situations:
            hints = self.registry.situation_hints()
            checked_hints = [hints[name] for name in scope_situations if name in hints]
            high = next((entry for entry in checked_hints if entry[1].high_stakes), None)
            if high is not None:
                _module_name, hint = high
                # A checked high-stakes situation must ground its answer in a current source, so
                # the terminal-answer citation guard applies even when only the semantic signal
                # fired (mirrors the deterministic high-stakes turns).
                current_source_required = True
                # F063: the manifest active_lockout definition is broad enough that the scope
                # model flags a plain no-heat complaint. Engage the deterministic lockout FLOOR
                # (the Call-911 feedback + backstop) only when the message itself reads as an
                # active lockout or coercive shutoff, never on the broader essential-services
                # reading, so an ordinary no-heat turn keeps the forced official retrieval but
                # reaches the model for its grounded answer.
                if hint.name == "active_lockout" and _needs_current_lockout_guidance(user_message):
                    lockout_turn = True  # the deterministic lockout floors key off this
                if hint.name == "snap_work_rules":
                    snap_work_rule_turn = True  # F089: seeds the forced pantry prefetch below
                if initial_forced_tool is None and has_current_source and hint.query:
                    initial_forced_tool, initial_forced_args = _current_source_call(
                        self.tools, hint.query, tuple(hint.urls),
                    )
                if hint.reminder and hint.reminder not in effective_reminders:
                    messages.insert(-1, {
                        "role": "user",
                        "content": f"<system-reminder>\n{hint.reminder}\n</system-reminder>",
                    })
                    yield events.Reminder(summary=hint.reminder)
                if hint.focus_tools and len(scope_modules) <= 1:
                    effective_excluded_tools.update(
                        name for name in self.tools if name not in set(hint.focus_tools)
                    )
        if (
            event_preparation_turn
            and "whats_on_events" in self.tools
            and _EVENT_PREPARATION_SCOPE_REMINDER not in effective_reminders
        ):
            messages.insert(-1, {
                "role": "user",
                "content": (
                    f"<system-reminder>\n{_EVENT_PREPARATION_SCOPE_REMINDER}\n</system-reminder>"
                ),
            })
            yield events.Reminder(summary=_EVENT_PREPARATION_SCOPE_REMINDER)

        notify_awareness = prefetched_notify_awareness
        if notify_awareness is None:
            notify_awareness = await self.get_notify_awareness()
            if notify_awareness and _history_already_cites_notify(history):
                # F080, second organ: the full notice index re-injected every turn is what the
                # model re-briefs. Once this conversation has delivered notices, shrink the
                # reminder to a delta instruction (Anthropic context-editing shape: never
                # re-deliver unchanged content for re-processing).
                titles = "; ".join(
                    line.lstrip("- ").split(": ", 1)[-1]
                    for line in notify_awareness.splitlines() if line.startswith("- ")
                )[:400]
                notify_awareness = (
                    "You already told the resident about today's Notify NYC notices earlier in "
                    "this conversation. Do NOT re-brief them. Mention one again only if it "
                    f"directly bears on this new message. Current titles, for change detection "
                    f"only: {titles}"
                )
            if notify_awareness:
                messages.insert(-1, {
                    "role": "user",
                    "content": f"<system-reminder>\n{notify_awareness}\n</system-reminder>",
                })
        if (
            initial_forced_tool is None
            and (event_discovery_turn or event_preparation_turn)
            and bool(notify_awareness)
            and not _history_already_cites_notify(history)
            and "nyc_advisories" in self.tools
            and "nyc_advisories" not in set(excluded_tools or ())
        ):
            # F061: keyed on the EXISTENCE of same-day notifications, never on parsing their
            # wording — the full report comes back cited and the model judges what's material.
            initial_forced_tool = "nyc_advisories"
            # `incidental` marks this as OUR check, not the resident's question: an empty
            # result then returns nothing at all instead of prose for the model to narrate
            initial_forced_args = {"incidental": True}

        if forced_tool and forced_tool not in self.tools:
            message_id = "m0"
            assistant = {"role": "assistant", "content": FORCED_TOOL_FALLBACK, "tool_calls": None}
            messages.append(assistant)
            yield events.MessageStart(message_id=message_id)
            yield events.TextDelta(message_id=message_id, text=FORCED_TOOL_FALLBACK)
            yield events.MessageCompleted(
                message_id=message_id, text=FORCED_TOOL_FALLBACK, citations=citations.mapping()
            )
            result = AgentResult(
                text=FORCED_TOOL_FALLBACK, citations=citations.mapping(), tool_calls_made=tools_made,
                iterations=0, status="error", messages=messages, usage=_usage(),
            )
            yield events.Done(status="error", num_turns=0, citations=result.citations, result=result)
            return

        # F089 machinery: the forced-call QUEUE. Slot 0 is the situation's current-source
        # search when one fired; a SNAP work-rule situation then ALSO forces the food-pantry
        # prefetch (the advisories-forcing pattern): food help lands by construction, and with
        # no location the tool's ask-for-location result IS the offer. Empty queue = no forcing.
        initial_forced_calls: list[tuple[str, Optional[dict]]] = (
            [(initial_forced_tool, initial_forced_args)] if initial_forced_tool else []
        )
        if snap_work_rule_turn and "nearest_food_pantry" in self.tools:
            initial_forced_calls.append(("nearest_food_pantry", {"near": ""}))
        for i in range(max_iters):
            # SPEND-CAP GUARD (turn boundary). Before each model call, halt if this session's
            # cumulative cost has reached the configured ceiling, never spend past it silently.
            # No-op when no cap is set, so the default path is unchanged.
            halt = self._spend.halt_reason()
            if halt:
                logger.warning("spend cap halted the agent: %s", halt)
                yield events.ErrorEvent(scope="spend", message=halt, retryable=False)
                result = AgentResult(
                    text=SPEND_CAPPED_FALLBACK, citations=citations.mapping(),
                    tool_calls_made=tools_made, iterations=i, status="max_budget",
                    messages=messages, usage=_usage(),
                )
                yield events.Done(status="max_budget", num_turns=i,
                                  citations=result.citations, result=result)
                return
            tool_schemas = self._tool_schemas(effective_excluded_tools)
            if not self._request_fits_context(messages, tool_schemas):
                result = AgentResult(
                    text=CONTEXT_LIMIT_FALLBACK,
                    citations=citations.mapping(),
                    tool_calls_made=tools_made,
                    iterations=i,
                    status="context_limit",
                    messages=messages,
                    usage=_usage(),
                )
                yield events.Done(
                    status="context_limit", num_turns=i,
                    citations=result.citations, result=result,
                )
                return
            message_id = f"m{i}"
            turn_usage["n_model_calls"] += 1
            turn_usage["n_answer_model_calls"] += 1
            turn_usage["iterations"] = i + 1
            yield events.MessageStart(message_id=message_id)
            parts: list[str] = []
            assistant: Optional[dict] = None
            model_started = time.perf_counter()
            try:
                requested_tool = (
                    initial_forced_calls[i][0] if i < len(initial_forced_calls) else None
                )
                model_stream = (
                    self._litellm_stream(messages, tool_schemas, requested_tool)
                    if self._uses_litellm
                    else self._stream_fn(messages, tool_schemas)
                )
                async for chunk in model_stream:
                    if chunk["type"] == "text":
                        parts.append(chunk["text"])
                        if requested_tool is None and reply_script is None:
                            yield events.TextDelta(message_id=message_id, text=chunk["text"])
                    elif chunk["type"] == "usage":
                        call_in = int(chunk.get("input_tokens", 0) or 0)
                        call_out = int(chunk.get("output_tokens", 0) or 0)
                        turn_usage["input_tokens"] += call_in
                        turn_usage["output_tokens"] += call_out
                        turn_usage["answer_input_tokens"] += call_in
                        turn_usage["answer_output_tokens"] += call_out
                        call_cached = int(chunk.get("cached_input_tokens", 0) or 0)
                        turn_usage["cached_input_tokens"] = (
                            turn_usage.get("cached_input_tokens", 0) + call_cached
                        )
                        turn_usage["answer_cached_input_tokens"] += call_cached
                        # accrue the cache-discounted cost toward the cap
                        self._spend.record(self.model, call_in, call_out, call_cached)
                    elif chunk["type"] == "message":
                        assistant = chunk["message"]
            except Exception as exc:  # model call failed after retries
                turn_usage["model_time_ms"] += (time.perf_counter() - model_started) * 1000.0
                logger.exception("model stream failed")
                yield events.ErrorEvent(scope="model", message=str(exc), retryable=True)
                result = AgentResult(
                    text="", citations=citations.mapping(), tool_calls_made=tools_made,
                    iterations=i, status="error", messages=messages, usage=_usage(),
                )
                yield events.Done(status="error", num_turns=i, citations=result.citations, result=result)
                return
            turn_usage["model_time_ms"] += (time.perf_counter() - model_started) * 1000.0

            if assistant is None:
                assistant = {"role": "assistant", "content": "".join(parts) or None, "tool_calls": None}
            tool_calls = assistant.get("tool_calls") or []
            text = assistant.get("content") or "".join(parts)
            if requested_tool is not None:
                called_tools = [
                    call.get("function", {}).get("name")
                    if isinstance(call, dict) and isinstance(call.get("function"), dict)
                    else None
                    for call in tool_calls
                ]
                if called_tools != [requested_tool]:
                    assistant = {
                        "role": "assistant", "content": FORCED_TOOL_FALLBACK, "tool_calls": None,
                    }
                    messages.append(assistant)
                    yield events.TextDelta(message_id=message_id, text=FORCED_TOOL_FALLBACK)
                    yield events.MessageCompleted(
                        message_id=message_id, text=FORCED_TOOL_FALLBACK,
                        citations=citations.mapping(),
                    )
                    result = AgentResult(
                        text=FORCED_TOOL_FALLBACK, citations=citations.mapping(),
                        tool_calls_made=tools_made, iterations=i + 1, status="error",
                        messages=messages, usage=_usage(),
                    )
                    yield events.Done(
                        status="error", num_turns=i + 1, citations=result.citations, result=result,
                    )
                    return
                assistant["content"] = None
                text = ""
            # EMPTY-ANSWER GUARD: a terminal turn (no tool calls) must never reach the user blank.
            # Substitute an explicit safe refusal and stream it, so both the streaming UI and the
            # drained result get non-empty, safe text.
            if not tool_calls and not (text or "").strip():
                text = EMPTY_ANSWER_FALLBACK
                assistant["content"] = text
                if reply_script is None:
                    yield events.TextDelta(message_id=message_id, text=text)
            messages.append(assistant)

            if not tool_calls:
                scope_feedback = _required_scope_feedback(
                    user_message,
                    text,
                    civic_law_search,
                    immigrant_benefits_turn=immigrant_benefits_turn,
                    benefits_recovery_turn=benefits_recovery_turn,
                    lockout_turn=lockout_turn,
                )
                if scope_feedback:
                    backstop = _scope_grounded_backstop(
                        user_message,
                        citations.mapping(),
                        civic_law_search,
                        immigrant_benefits_turn=immigrant_benefits_turn,
                        benefits_recovery_turn=benefits_recovery_turn,
                        lockout_turn=lockout_turn,
                    )
                    if backstop is not None:
                        text = backstop
                        assistant["content"] = text
                    elif guard_retries < self.guard_max_retries:
                        guard_retries += 1
                        yield events.MessageCompleted(
                            message_id=message_id, text="", citations=citations.mapping()
                        )
                        yield events.Reminder(
                            summary=("scope guard: ambiguous false-premise correction, retrying "
                                     f"({guard_retries}/{self.guard_max_retries})")
                        )
                        messages.append({"role": "user", "content": scope_feedback})
                        continue
                    else:
                        text = GROUNDING_ABSTAIN_FALLBACK
                        assistant["content"] = text
                script_feedback = _reply_script_feedback(user_message, text)
                if script_feedback and language_retries < 1 and i + 1 < max_iters:
                    language_retries += 1
                    yield events.MessageCompleted(
                        message_id=message_id, text="", citations=citations.mapping()
                    )
                    yield events.Reminder(
                        summary=("reply language guard: script mismatch, retrying "
                                 f"({language_retries}/1)")
                    )
                    messages.append({"role": "user", "content": script_feedback})
                    continue
                if "whats_on_events" in tools_made and (
                    event_discovery_turn or event_preparation_turn
                ):
                    text = _attach_event_action_urls(
                        text, citations.mapping(), available_citation_ids=tool_citation_ids,
                    )
                    assistant["content"] = text
                text = _attach_location_action_urls(
                    text, citations.mapping(), available_citation_ids=tool_citation_ids,
                )
                assistant["content"] = text
                event_context_feedback = _broad_event_context_feedback(
                    user_message, text, citations.mapping(), tools_made,
                    available_citation_ids=tool_citation_ids,
                    discovery_turn=event_discovery_turn,
                )
                if event_context_feedback and guard_retries < self.guard_max_retries:
                    guard_retries += 1
                    yield events.MessageCompleted(
                        message_id=message_id, text="", citations=citations.mapping()
                    )
                    yield events.Reminder(
                        summary=("event context guard: required evidence lane omitted, retrying "
                                 f"({guard_retries}/{self.guard_max_retries})")
                    )
                    messages.append({"role": "user", "content": event_context_feedback})
                    continue
                if event_context_feedback:
                    text = EVENT_CONTEXT_ABSTAIN_FALLBACK
                    assistant["content"] = text
                # Availability means the turn's citation registry, not the {cite:Sn} markers
                # seen in tool text: tools may register a citation while referencing it in
                # another format (observed live with `official_sources`), and invented ids are
                # already rejected by the unknown-citation guard below.
                preparation_feedback = _event_preparation_feedback(
                    user_message, text, citations.mapping(),
                    preparation_turn=event_preparation_turn,
                )
                if preparation_feedback and guard_retries < self.guard_max_retries:
                    guard_retries += 1
                    yield events.MessageCompleted(
                        message_id=message_id, text="", citations=citations.mapping()
                    )
                    yield events.Reminder(
                        summary=("event preparation guard: unresolved or ungrounded plan, "
                                 f"retrying ({guard_retries}/{self.guard_max_retries})")
                    )
                    messages.append({"role": "user", "content": preparation_feedback})
                    continue
                if preparation_feedback:
                    text = EVENT_PREPARATION_ABSTAIN_FALLBACK
                    assistant["content"] = text
                cited_ids = set(_CITE_MARKER_RE.findall(text))
                if current_source_required and not (cited_ids & set(citations.mapping())):
                    text = GROUNDING_ABSTAIN_FALLBACK
                    assistant["content"] = text
                # DETERMINISTIC GROUNDING GUARD (the post-generation safety hook). Before the final
                # answer reaches the user, verify every {cite:Sn}'d structured fact is in its source.
                unknown_citations = _unknown_citation_ids(text, citations.mapping())
                if unknown_citations:
                    if guard_retries < self.guard_max_retries:
                        guard_retries += 1
                        yield events.MessageCompleted(
                            message_id=message_id, text="", citations=citations.mapping()
                        )
                        yield events.Reminder(
                            summary=("citation guard: unknown citation id, retrying "
                                     f"({guard_retries}/{self.guard_max_retries})")
                        )
                        messages.append({
                            "role": "user",
                            "content": _unknown_citation_feedback(unknown_citations),
                        })
                        continue
                    text = GROUNDING_ABSTAIN_FALLBACK
                discovery_citations = used_discovery_citations(
                    text, citations.mapping(),
                )
                if discovery_citations:
                    if guard_retries < self.guard_max_retries:
                        guard_retries += 1
                        yield events.MessageCompleted(
                            message_id=message_id,
                            text="",
                            citations=citations.mapping(),
                        )
                        yield events.Reminder(
                            summary=(
                                "citation guard: discovery-only evidence, retrying "
                                f"({guard_retries}/{self.guard_max_retries})"
                            )
                        )
                        messages.append({
                            "role": "user",
                            "content": _discovery_citation_feedback(
                                discovery_citations,
                            ),
                        })
                        continue
                    text = GROUNDING_ABSTAIN_FALLBACK
                action, text, gr = self._grounding_verdict(
                    text, citations.mapping(), user_message, guard_retries)
                if action == "retry":
                    # Tier 3: a cited structured fact isn't grounded. Close out the rejected attempt
                    # for streaming clients, tell the model EXACTLY what's wrong, and regenerate.
                    guard_retries += 1
                    yield events.MessageCompleted(message_id=message_id,
                                                  text=assistant.get("content") or "",
                                                  citations=citations.mapping())
                    yield events.Reminder(summary=("grounding guard: unsupported cited fact, retrying "
                                                   f"({guard_retries}/{self.guard_max_retries})"))
                    messages.append({"role": "user", "content": _grounding_feedback(gr)})
                    continue
                if action == "pass":
                    text = attach_temporal_provenance(text, citations.mapping())
                # "pass" (grounded, or only soft mismatches) or "abstain" (Tier 4 rewrote `text`).
                assistant["content"] = text
                if reply_script is not None:
                    yield events.TextDelta(message_id=message_id, text=text)
                yield events.MessageCompleted(message_id=message_id, text=text, citations=citations.mapping())
                result = AgentResult(
                    text=text, citations=citations.mapping(), tool_calls_made=tools_made,
                    iterations=i + 1, status="success", messages=messages, usage=_usage(),
                )
                yield events.Done(status="success", num_turns=i + 1, citations=result.citations, result=result)
                return

            yield events.MessageCompleted(message_id=message_id, text=text, citations=citations.mapping())
            for call in tool_calls:
                name = call["function"]["name"]
                call_id = call.get("id") or name
                tools_made.append(name)
                turn_usage["n_tool_calls"] += 1
                tool = None if name in effective_excluded_tools else self.tools.get(name)
                yield events.ToolStart(tool_call_id=call_id, name=name, label=name)

                tool_started = time.perf_counter()
                arg_overrides = (
                    initial_forced_calls[i][1]
                    if i < len(initial_forced_calls) and initial_forced_calls[i][0] == name
                    else None
                )
                async for ev, tool_result in self._invoke(
                    name, call["function"]["arguments"], tool, ctx, arg_overrides=arg_overrides,
                ):
                    if tool_result is None:
                        yield ev  # an approval-required event
                        continue
                    messages.append({"role": "tool", "tool_call_id": call_id, "content": tool_result})
                    tool_citation_ids.update(_CITE_MARKER_RE.findall(tool_result))
                    # Broad shortlist turns lean on the tool's coordinated lanes; preparation
                    # turns KEEP free scoped search so the model can resolve the event
                    # identity when listings alone cannot (F053).
                    if name == "whats_on_events" and event_discovery_turn:
                        effective_excluded_tools.update({
                            "index_search", "web_search", "recent_developments",
                        })
                    status = "error" if tool_result.startswith(("ERROR", "Action not approved")) else "ok"
                    yield events.ToolCompleted(
                        tool_call_id=call_id, name=name, status=status, result_summary=tool_result[:200]
                    )
                turn_usage["tool_time_ms"] += (time.perf_counter() - tool_started) * 1000.0

        result = AgentResult(
            text=(messages[-1].get("content") or "").strip() or EMPTY_ANSWER_FALLBACK,
            citations=citations.mapping(),
            tool_calls_made=tools_made, iterations=max_iters, hit_max_iters=True,
            status="max_turns", messages=messages, usage=_usage(),
        )
        yield events.Done(status="max_turns", num_turns=max_iters, citations=result.citations, result=result)

    async def run(
        self,
        user_message: str,
        history: Optional[list[dict]] = None,
        max_iters: int = 8,
        reminders: Optional[list[str]] = None,
        output_dir=None,
        drafts=None,
        forced_tool: Optional[str] = None,
        forced_tool_args: Optional[dict] = None,
        excluded_tools: Optional[set[str]] = None,
        prefetched_notify_awareness: Optional[str] = None,
        event_sink: Optional[Callable[[events.Event], None]] = None,
    ) -> AgentResult:
        """Drain the event stream into a single AgentResult.

        `event_sink`, when given, is called once per event with exactly what `stream` yields:
        the conditional-streaming seam a channel view (the console REPL) rides while every guard
        and accounting still lives in `stream`. It is a pure observer, it cannot change the result,
        and its exceptions are swallowed so a rendering bug never aborts a resident's turn. None
        (the default, e.g. Twilio) is byte-identical to draining the stream directly."""
        result: Optional[AgentResult] = None
        async for event in self.stream(user_message, history=history, max_iters=max_iters,
                                       reminders=reminders, output_dir=output_dir, drafts=drafts,
                                       forced_tool=forced_tool, forced_tool_args=forced_tool_args,
                                       excluded_tools=excluded_tools,
                                       prefetched_notify_awareness=prefetched_notify_awareness):
            if event_sink is not None:
                try:
                    event_sink(event)
                except Exception:
                    pass  # a view bug never aborts a turn
            if isinstance(event, events.Done):
                result = event.result
        assert result is not None  # stream always ends with Done
        return result

    async def _invoke(
        self, name: str, raw_args, tool: Optional[Tool], ctx: ToolContext,
        arg_overrides: Optional[dict] = None,
    ):
        """Yield (event, tool_result). tool_result is None for non-terminal events
        (e.g. approval prompts); a string when the call resolved."""
        if tool is None:
            yield None, f"ERROR: unknown tool '{name}'."
            return
        try:
            args = raw_args if isinstance(raw_args, dict) else json.loads(raw_args or "{}")
        except json.JSONDecodeError as exc:
            yield None, f"ERROR: could not parse arguments for '{name}': {exc}"
            return
        if not isinstance(args, dict):
            yield None, f"ERROR: arguments for '{name}' must be a JSON object."
            return
        args = {**args, **(arg_overrides or {})}

        if tool.requires_approval:
            yield events.ToolApprovalRequired(tool_call_id=name, name=name, args=args), None
            approved = await self._approver(name, args) if self._approver else False
            if not approved:
                yield None, "Action not approved by the user; not executed."
                return

        try:
            yield None, await tool.handler(args, ctx)
        except Exception as exc:  # surface tool errors to the model, don't crash
            logger.exception("tool %s failed", name)
            yield None, f"ERROR: tool '{name}' failed: {exc}"

    def conversation(self) -> "Conversation":
        return Conversation(self)

    async def _litellm_stream(
        self, messages: list[dict], tool_schemas: list[dict], forced_tool: Optional[str] = None,
    ) -> AsyncIterator[dict]:
        import litellm

        kwargs = _completion_kwargs(
            self.model, messages, tool_schemas, forced_tool=forced_tool,
            reasoning_effort=self._reasoning_effort,
        )

        async def _open():
            return await litellm.acompletion(**kwargs)

        stream = await _with_retry(_open)
        content_parts: list[str] = []
        calls: dict[int, dict] = {}
        usage: Optional[dict] = None
        async for chunk in stream:
            if getattr(chunk, "usage", None):
                details = getattr(chunk.usage, "prompt_tokens_details", None)
                usage = {"input_tokens": chunk.usage.prompt_tokens,
                         "output_tokens": chunk.usage.completion_tokens,
                         "cached_input_tokens": int(getattr(details, "cached_tokens", 0) or 0)}
            if not chunk.choices:  # include_usage emits a final choices-less chunk
                continue
            delta = chunk.choices[0].delta
            if getattr(delta, "content", None):
                content_parts.append(delta.content)
                yield {"type": "text", "text": delta.content}
            for tc in getattr(delta, "tool_calls", None) or []:
                slot = calls.setdefault(tc.index, {"id": None, "name": "", "args": ""})
                if tc.id:
                    slot["id"] = tc.id
                if tc.function and tc.function.name:
                    slot["name"] = tc.function.name
                if tc.function and tc.function.arguments:
                    slot["args"] += tc.function.arguments
        if usage is not None:
            yield {"type": "usage", **usage}
        tool_calls = None
        if calls:
            tool_calls = [
                {"id": s["id"] or f"call_{i}", "type": "function",
                 "function": {"name": s["name"], "arguments": s["args"]}}
                for i, s in sorted(calls.items())
            ]
        yield {"type": "message", "message": {"role": "assistant", "content": "".join(content_parts) or None, "tool_calls": tool_calls}}


def _is_anthropic(model: str) -> bool:
    """Anthropic-family model? Only these accept prompt-cache content blocks (cache_control on a
    system block); every other provider must get a plain-string system message. Detected from the
    model string so it also covers Bedrock / Vertex Claude ids (e.g. 'bedrock/anthropic.claude-...')."""
    m = model.lower()
    return m.startswith("anthropic/") or "claude" in m


_extra_model_info_registered = False


def _register_extra_model_info() -> None:
    """Teach LiteLLM the models it doesn't ship metadata for (config.EXTRA_MODEL_INFO).
    Without this the memory planner fails closed on the unknown context window and the model
    never receives a single turn. Also makes cost tracking price these models."""
    global _extra_model_info_registered
    if _extra_model_info_registered or not config.EXTRA_MODEL_INFO:
        return
    import litellm

    litellm.register_model(config.EXTRA_MODEL_INFO)
    _extra_model_info_registered = True


def _completion_kwargs(
    model: str, messages: list[dict], tool_schemas: list[dict], forced_tool: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
) -> dict:
    """Build the kwargs for litellm.acompletion.

    GPT-5 models reject temperature != 1 (litellm raises UnsupportedParamsError), so we omit
    temperature for them and let it default; every other model pins temperature=0 for
    deterministic, grounded output. Tool schemas are attached only when present. An explicit
    `reasoning_effort` (the bench's effort axis) always wins over per-model defaults.
    """
    kwargs: dict = {
        "model": model, "messages": messages,
        "stream": True, "stream_options": {"include_usage": True},
    }
    if "gpt-5" not in model:
        kwargs["temperature"] = 0.0
    if tool_schemas:
        kwargs["tools"] = tool_schemas
        if forced_tool:
            kwargs["tool_choice"] = {
                "type": "function", "function": {"name": forced_tool},
            }
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort
    if config.HEYNYC_SERVICE_TIER:
        kwargs["service_tier"] = config.HEYNYC_SERVICE_TIER
    if "ollama" in model:
        kwargs["num_ctx"] = config.OLLAMA_NUM_CTX
    return kwargs


def _wrap_complete(fn: CompletionFn) -> StreamFn:
    async def _stream(messages: list[dict], tool_schemas: list[dict]) -> AsyncIterator[dict]:
        message = await fn(messages, tool_schemas)
        if message.get("content"):
            yield {"type": "text", "text": message["content"]}
        yield {"type": "message", "message": message}

    return _stream


async def _with_retry(factory, attempts: int = 3, base_delay: float = 0.5):
    delay = base_delay
    for attempt in range(attempts):
        try:
            return await factory()
        except Exception:
            if attempt == attempts - 1:
                raise
            await asyncio.sleep(delay)
            delay *= 2


class Conversation:
    """Stateful multi-turn wrapper. Keeps user/assistant turns as history so the
    agent has context across messages. Tool-call noise stays within a single turn."""

    def __init__(self, agent: Agent):
        self.agent = agent
        self.turns: list[dict] = []

    async def send(self, user_message: str, max_iters: int = 8, reminders=None,
                   output_dir=None, drafts=None, forced_tool=None, forced_tool_args=None,
                   excluded_tools=None) -> AgentResult:
        result = await self.agent.run(user_message, history=self.turns, max_iters=max_iters,
                                      reminders=reminders, output_dir=output_dir, drafts=drafts,
                                      forced_tool=forced_tool, forced_tool_args=forced_tool_args,
                                      excluded_tools=excluded_tools)
        self.turns.append({"role": "user", "content": user_message, "timestamp": turn_timestamp()})
        self.turns.append({
            "role": "assistant",
            "content": result.text,
            "citations": used_citations(result.text, result.citations),
            "timestamp": turn_timestamp(),
        })
        return result

    async def stream(self, user_message: str, max_iters: int = 8, reminders=None,
                     output_dir=None, drafts=None, forced_tool=None, forced_tool_args=None,
                     excluded_tools=None):
        """Stream a turn's events, then commit the turn to history."""
        final_result = None
        async for event in self.agent.stream(user_message, history=self.turns, max_iters=max_iters,
                                             reminders=reminders, output_dir=output_dir, drafts=drafts,
                                             forced_tool=forced_tool, forced_tool_args=forced_tool_args,
                                             excluded_tools=excluded_tools):
            if isinstance(event, events.Done) and event.result is not None:
                final_result = event.result
            yield event
        self.turns.append({"role": "user", "content": user_message, "timestamp": turn_timestamp()})
        self.turns.append({
            "role": "assistant",
            "content": final_result.text if final_result else "",
            "citations": used_citations(
                final_result.text, final_result.citations,
            ) if final_result else {},
            "timestamp": turn_timestamp(),
        })
