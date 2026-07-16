"""workers module tool: `worker_rights_guidance`, NYC/NYS worker-rights facts, each cited to its
official source page.

Closes the red-team tip-theft gap: no existing module covered worker rights, so a question like
"my boss is keeping our tips" got a safe abstention instead of a grounded, cited answer. Like
housing_guidance, the facts are STATIC but OFFICIAL (the statute + the enforcement path), so they
live here as grounded _Fact records and are returned WITH a DOC citation to the official page each
one comes from, never stated from the model's memory.

Every URL + fact below was verified against the source page on VERIFIED_ON. `snippet` is a subset of
`body`'s wording so the eval's faithfulness check (snippet ⊆ tool output) holds. Re-verify the live
pages before editing any fact.
"""
from __future__ import annotations

from dataclasses import dataclass

from heynyc.core.tools.base import Tool, ToolContext

VERIFIED_ON = "2026-07-12"


@dataclass(frozen=True)
class _Fact:
    url: str      # official source page (verified)
    title: str    # citation title
    snippet: str  # short cite label, a subset of `body` wording
    body: str     # the grounded fact text to report, cited


_GUIDANCE: dict[str, tuple[str, tuple[_Fact, ...]]] = {
    "tips": (
        "Your tips belong to you, and an employer can't take them:",
        (
            _Fact(
                url="https://www.nysenate.gov/legislation/laws/LAB/196-D",
                title="New York Labor Law section 196-d (Gratuities), The Laws of New York",
                snippet=("Under New York Labor Law section 196-d, no employer or their agent may "
                         "demand, accept, or keep any part of a tip or gratuity that belongs to an "
                         "employee; the law does allow tip-sharing among the employees who provide "
                         "the service; file a wage claim with the New York State Department of Labor"),
                body=("Under New York Labor Law section 196-d, no employer or their agent may demand, "
                      "accept, or keep any part of a tip or gratuity that belongs to an employee, or "
                      "keep any charge that a customer was led to believe is a gratuity. Your tips "
                      "are yours. The law does allow tip-sharing (a tip pool) among the employees who "
                      "provide the service, and a narrow exception for the checking of hats and "
                      "coats. If an employer is taking your tips, you can file a wage claim with the "
                      "New York State Department of Labor."),
            ),
            _Fact(
                url="https://dol.ny.gov/tips-and-gratuities-faq",
                title=("Tips and Gratuities Frequently Asked Questions, New York State Department of "
                       "Labor"),
                snippet=("The New York State Department of Labor enforces the tip rules and "
                         "investigates stolen tips; you can file a claim to recover them at "
                         "dol.ny.gov; in New York City the Department of Consumer and Worker "
                         "Protection can also help, through 311"),
                body=("The New York State Department of Labor enforces the tip rules and investigates "
                      "stolen tips. If your employer has taken tips that belong to you, you can file "
                      "a claim to recover them; call the Department of Labor or file a claim at "
                      "dol.ny.gov. If it happened in New York City, the NYC Department of Consumer "
                      "and Worker Protection can also help, through 311."),
            ),
            _Fact(
                url="https://ag.ny.gov/immigrants-rights/immigrant-workers-rights",
                title="Immigrant Workers' Rights, New York State Office of the Attorney General",
                snippet=("These protections apply no matter your immigration status; New York protects "
                         "all workers, including undocumented workers; an employer must pay you for "
                         "every hour you worked and cannot keep your tips, and cannot use your "
                         "immigration status as an excuse to underpay you; the state Department of "
                         "Labor does not report workers or witnesses to immigration authorities"),
                body=("These protections apply no matter your immigration status. New York protects "
                      "all workers, including undocumented workers and workers paid off the books: an "
                      "employer must pay you for every hour you worked and cannot keep your tips, and "
                      "cannot use your immigration status as an excuse to underpay you. You can file a "
                      "wage claim, and the state Department of Labor does not report workers or "
                      "witnesses to immigration authorities."),
            ),
        ),
    ),
}

# free-text → canonical topic. The `topic` arg SHOULD be a key, but the model may hand us the user's
# words ("my boss is keeping our tips"); map those to a topic rather than fail.
_TOPIC_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("tips", ("tip", "tips", "stolen tips", "tip theft", "gratuity", "gratuities", "took my tips",
              "keeping our tips", "keep our tips", "pockets our tips", "tip pool", "tipped",
              "tip jar", "cut of our tips", "propina", "propinas", "se queda con mis propinas")),
)


def _resolve_topic(raw: str) -> str | None:
    """Map the `topic` arg (a canonical key or free text) to one of the guidance topics."""
    key = (raw or "").strip().lower().replace("-", "_").replace(" ", "_")
    if key in _GUIDANCE:
        return key
    text = (raw or "").lower()
    for topic, needles in _TOPIC_KEYWORDS:
        if any(n in text for n in needles):
            return topic
    return None


async def _guidance_handler(args: dict, ctx: ToolContext) -> str:
    topic = _resolve_topic(args.get("topic", ""))
    if topic is None:
        return ("I don't have grounded guidance for that worker-rights topic. Use "
                "worker_rights_guidance with topic = 'tips' (an employer taking or keeping your "
                "tips). For anything else, point the user to the NYS Department of Labor "
                "(dol.ny.gov) or 311.")
    intro, facts = _GUIDANCE[topic]
    lines = [intro]
    for fact in facts:
        cite = ctx.citations.register(
            fact.url, snippet=fact.snippet, title=fact.title, kind="DOC", valid_as_of=VERIFIED_ON,
        )
        lines.append(f"- {fact.body} {{cite:{cite}}}")
    lines.append("Report ONLY these grounded facts, each with its {cite:Sn}. Do not add or change a "
                 "law section, an agency, a phone number, or a dollar figure; if the user needs "
                 "more, send them to the NYS Department of Labor (dol.ny.gov) or 311.")
    return "\n".join(lines)


def get_tools() -> list[Tool]:
    return [
        Tool(
            name="worker_rights_guidance",
            description=(
                "Return NYC/NYS official, grounded guidance for worker-rights questions, each WITH a "
                "citation to the official source page. Topic `tips`: an employer who takes, keeps, or "
                "skims an employee's tips or gratuities (New York Labor Law section 196-d) and how "
                "to file a wage claim with the NYS Department of Labor (in NYC, also the Department "
                "of Consumer and Worker Protection via 311). Pass `topic` = 'tips' (free text like "
                "'my boss is keeping our tips' or 'they took my gratuity' is mapped to it). ALWAYS "
                "use this instead of stating the law, the agency, or a phone number about stolen "
                "tips from your own knowledge; report only what it returns, cited."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": ("tips: the worker-rights situation (free text about stolen or "
                                        "withheld tips is mapped to it)."),
                    },
                },
                "required": ["topic"],
            },
            handler=_guidance_handler,
            open_world=False,  # static official facts baked in + cited; no network call
        ),
    ]
