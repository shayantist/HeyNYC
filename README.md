# HeyNYC

Talk to the city in plain English: *"Hey NYC, where's the nearest cooling center?"* or *"what can I do this weekend?"* and HeyNYC answers from real city data and shows you exactly where every fact came from. When nothing in the data backs an answer, it tells you it doesn't know instead of guessing.

**A grounded, no-hallucination assistant for NYC.gov services & events.**

That last part is the whole point. Every specific it gives you (an address, a dollar amount, an eligibility rule, a deadline) has to trace to a real grounded source (NYC Open Data, the city's own Benefits Screening API, geocoding, or a scoped search over trusted NYC domains) and it ships with an inline citation you can click and check yourself. When nothing grounds the answer, it abstains and points you to 311 or the right agency. This is deliberate: HeyNYC is built for the questions where a confident wrong answer costs someone money, their home, or their immigration status, so "grounded or it says it doesn't know" isn't a nice-to-have, it's the contract.

Right now (summer 2026, with heat advisories and the World Cup both live) you can ask it things like *"where's the nearest cooling center?"* or *"where can I watch the World Cup this weekend?"* and it answers from live city data, walks you through it, and cites every source. It covers a growing subset of the city's services today, around eight so far, not the whole catalogue yet (see [What you can ask](#what-you-can-ask)).

Goal is to build this up into a conversational front door for NYC.gov's rich catalogue of services.

> Open-source passion project. Not affiliated with the City of New York.

## Why

The city's data is rich but scattered across dozens of sites and portals. [MyCity](https://mycity.nyc.gov/) was meant to be the official push to centralize it, but ended up a [skeleton of links](https://nysfocus.com/2025/03/19/mycity-eric-adams-child-care) that's not the easiest to navigate, plus a chatbot that [confidently told business owners they could break the law](https://themarkup.org/artificial-intelligence/2024/03/29/nycs-ai-chatbot-tells-businesses-to-break-the-law) (take workers' tips, refuse Section 8 tenants, go cash-free).

Thus, I wanted to make HeyNYC as a way to make government services accessible through natural language by leveraging LLMs while making sure to **minimize hallucinations** and cite its sources. A key underpinning of the agent is deterministically enforcing that every fact comes from a specific grounded tool (NYC Open Data, geocoding, scoped web search) and ships with a citation. When nothing grounds an answer, it HAS to abstain.

If you find any failure modes, please feel free to **make an issue** as we're also making a database of failures to test against and make sure we [fill any and all holes](https://www.instagram.com/p/DWMD6wGD6-O/) as we go.

## What you can ask

Here's what you can ask me today, one row per service module. This table is generated straight from the module manifests, so it can't drift as modules are added or removed; if it ever looks stale, `uv run python -m heynyc capabilities --write-readme` regenerates it in place. It's a growing subset of the city's services, not the whole catalogue yet:

<!-- CAPABILITIES:START -->

| Service | What you can ask | Grounded in | Official link |
| --- | --- | --- | --- |
| **Emergency advisories** | "Are there any advisories right now?"<br>"Is it safe to be outside today?" | Notify NYC / NYC Emergency Management feed | [nyc.gov](https://www.nyc.gov/notifynyc) |
| **Benefits & programs** | "I'm struggling to afford groceries, what can I get?"<br>"Am I eligible for SNAP?" | NYC Benefits & Programs dataset + Screening API | [access.nyc.gov](https://access.nyc.gov) |
| **Clinics** | "Where can I see a doctor without insurance?"<br>"Free clinic near me" | HRSA Primary Health Care service-delivery sites | [access.nyc.gov](https://access.nyc.gov/programs/nyc-care/) |
| **Cooling centers** | "Where's the nearest cooling center?"<br>"It's too hot, where can I cool off indoors today?" | NYC Emergency Management - Cooling Centers | [finder.nyc.gov](https://finder.nyc.gov/coolingcenters/) |
| **Events** | "What's happening in NYC this weekend?"<br>"Any free events tonight?" | Ticketmaster + NYC Parks | [nyctourism.com](https://www.nyctourism.com/events/) |
| **Food pantries** | "Where's the nearest food pantry?"<br>"I need free food today near me" | NYC FoodHelp finder | [finder.nyc.gov](https://finder.nyc.gov/foodhelp/) |
| **Housing & eviction help** | "I got an eviction notice, where can I get help near me?"<br>"My landlord won't turn on the heat, what do I do?" | NYC DSS/DHS - Homebase (eviction prevention) offices | [access.nyc.gov](https://access.nyc.gov) |
| **SNAP centers** | "Where's the nearest SNAP center?"<br>"Where do I apply for food stamps in person?" | NYC Open Data (tc6u-8rnp) | [access.nyc.gov](https://access.nyc.gov) |

<!-- CAPABILITIES:END -->

## How it works

```
you ──▶ agent (streaming tool-calling loop)
            ├─ nearest()       NYC Open Data + geocoding + distance   (never guessed)
            ├─ index_search()  curated official pages (hybrid RAG)
            ├─ web_search()    trusted NYC domains, ranked by source trust
            └─ module tools    e.g. benefits_search, whats_on_events (live city data)
            ▼
        grounded answer with {cite:S1} sources + links out
```

Services are **pluggable modules**: each is a self-contained folder (manifest + optional tool + its own eval), so adding a service is adding a folder and deleting one is deleting the folder. The built-in modules are the ones listed in **[What you can ask](#what-you-can-ask)** above. See the **[module authoring guide](heynyc/modules/README.md)**.

## Safety

HeyNYC answers questions where a confident wrong answer can cost someone money, their home, or their immigration status, so safety is the point, not a footnote. Here's how we keep it honest: every fact is grounded in an official source and cited, or it abstains. On top of that, a **runtime grounding guard** re-checks each answer before it reaches you. Phase 1 (shipped) verifies that every cited fact (a dollar amount, a law or section number, an address, a date) actually appears in the source it's cited to, and drops, hedges, or abstains on anything that doesn't; a Phase 2 NLI faithfulness checker for looser prose claims the deterministic check can't parse is designed but not yet built. It never decides eligibility (the city's own screener does) and never acts as your lawyer or doctor; it routes those to a human. And we adversarially red-team it and publish the failures, including the seven real grounding slips we found and fixed. Because that guard is architectural, the safety guarantee doesn't rest on any one model. The full write-up (the grounded design, the no-hallucination eval, the MyCity safety subset, and a 137-query red-team with independent grading, honest results and all) is in **[SAFETY.md](SAFETY.md)**, with the underlying eval docs (including the [model comparison](docs/eval/model-comparison.md)) in [`docs/eval/`](docs/eval/).

## Quickstart

```bash
cd heynyc
uv sync --extra dev
cp .env.example .env        # add an LLM key; others are optional
uv run python -m heynyc index-build      # build the RAG index from module seeds
uv run python -m heynyc repl             # interactive, streaming chat
```

Other commands: `modules` (list), `new-module <name>` (scaffold), `chat "..."` (one-shot), `index-search "..."` (query the index), `capabilities --write-readme` (regenerate the table above), `eval` (run the no-hallucination gate).

- **Messaging on-ramp (WhatsApp/SMS):** `uv run python -m heynyc serve` runs the server so you can text the agent from WhatsApp or SMS. Setup and design in the **[channels guide](heynyc/channels/README.md)**.

## Contributing

On top of contributing to the code, one great way to contribute is adding service modules. See **[CONTRIBUTING.md](CONTRIBUTING.md)** and the **[module authoring guide](heynyc/modules/README.md)**. You can also [request a module](.github/ISSUE_TEMPLATE/new_service_module.yml)!

## Status

Standalone Python package, fully offline-tested and live-verified against the real NYC APIs. Built and working: the agent core, the geo / RAG / web-search tools, the service modules above, the [no-hallucination eval gate](heynyc/eval/README.md) (currently green), the runtime grounding guard (Phase 1), and a WhatsApp/SMS on-ramp (the [channels guide](heynyc/channels/README.md) has setup). It replies in the user's language on a best-effort basis, surfacing the city's own official translation where one exists (Notify NYC advisories, benefit programs) and translating the rest itself, and it flags data that's gone stale. Full history is in **[CHANGELOG.md](CHANGELOG.md)**.

**Next:** a web chat UI with a map, the Phase 2 faithfulness checker, and a fair self-hosted / open-weight model behind the guard for data sovereignty (the [model comparison](docs/eval/model-comparison.md) lays out where that stands).

## Known limitations

- **Intersections geocode poorly.** NYC GeoSearch sends "116 St and Broadway" to the wrong neighborhood, and its confidence scores don't flag it. HeyNYC echoes back the address it resolved and asks you to confirm before trusting it.
- **Some datasets are thin.** A few finders don't publish everything: the SNAP-center list, for instance, has no per-center hours or phone numbers, so HeyNYC won't guess them and points you to ACCESS HRA and 311 instead.