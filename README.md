# HeyNYC

**HeyNYC** helps New Yorkers find, understand, and apply for New York City government services, grounded in real city data, in your language, over the channels you already use (SMS, WhatsApp, web).

It answers questions about benefits, housing, food, health, immigration, and events. Every fact has an official citation, or HeyNYC abstains and points you to 311.

> Open-source civic project. Not affiliated with the City of New York.

## What you can ask

- "Where's the nearest cooling center to the Bronx 10453?"
- "My landlord won't take my CityFHEPS voucher, is that legal?"
- "I'm undocumented and my boss keeps our tips, what can I do?"
- "Where can I watch the World Cup final, and how do I get a ticket?"
- "Am I eligible for SNAP?" HeyNYC uses the city's real, PII-free benefits screener.

The table below is the full, generated list of current service modules.

<!-- CAPABILITIES:START -->

| Service | What you can ask | Grounded in | Official link |
| --- | --- | --- | --- |
| **Emergency advisories** | "Are there any advisories right now?"<br>"Is it safe to be outside today?" | Notify NYC / NYC Emergency Management feed | [nyc.gov](https://www.nyc.gov/notifynyc) |
| **Benefits & programs** | "I'm struggling to afford groceries, what can I get?"<br>"Am I eligible for SNAP?" | NYC Benefits & Programs dataset + Screening API | [access.nyc.gov](https://access.nyc.gov) |
| **Child care** | "Where's the nearest day care?"<br>"I need child care for my 2-year-old near Fordham, the Bronx" | NYC Open Data (gy3q-4tzp) | [nyc.gov](https://www.nyc.gov/site/doh/services/child-care.page) |
| **Clinics** | "Where can I see a doctor without insurance?"<br>"Free clinic near me" | HRSA Primary Health Care service-delivery sites | [access.nyc.gov](https://access.nyc.gov/programs/nyc-care/) |
| **Cooling centers** | "Where's the nearest cooling center?"<br>"It's too hot, where can I cool off indoors today?" | NYC Emergency Management - Cooling Centers | [finder.nyc.gov](https://finder.nyc.gov/coolingcenters/) |
| **Events** | "What's happening in NYC this weekend?"<br>"Any free events tonight?" | Ticketmaster + NYC Parks | [nyctourism.com](https://www.nyctourism.com/events/) |
| **Food pantries** | "Where's the nearest food pantry?"<br>"I need free food today near me" | NYC FoodHelp finder | [finder.nyc.gov](https://finder.nyc.gov/foodhelp/) |
| **Housing & eviction help** | "I got an eviction notice, where can I get help near me?"<br>"My landlord won't turn on the heat, what do I do?" | NYC DSS/DHS - Homebase (eviction prevention) offices | [access.nyc.gov](https://access.nyc.gov) |
| **Housing Connect** | "What affordable housing lotteries are open right now?"<br>"How do I apply for an apartment through the housing lottery?" | NYC Open Data (vy5i-a666) | [housingconnect.nyc.gov](https://housingconnect.nyc.gov) |
| **SNAP centers** | "Where's the nearest SNAP center?"<br>"Where do I apply for food stamps in person?" | NYC Open Data (tc6u-8rnp) | [access.nyc.gov](https://access.nyc.gov) |
| **WIC** | "Where's the nearest WIC office?"<br>"I'm pregnant and need WIC near Jackson Heights, Queens" | NY State WIC directory (Health Data NY) | [health.ny.gov](https://www.health.ny.gov/prevention/nutrition/wic/) |
| **Worker rights** | "My boss is keeping our tips, is that legal?"<br>"The restaurant I work at takes a cut of our tips, can they do that?" | NY Labor Law + NYC DCWP | [dol.ny.gov](https://dol.ny.gov) |

<!-- CAPABILITIES:END -->

## What makes it different

- **Cite or abstain:** Every fact needs a citation or an abstention. A deterministic guard rechecks cited claims before an answer ships. See [SAFETY.md](SAFETY.md).
- **Does, not just tells:** It uses the city's benefits screener and can prepare the real application for review, without deciding eligibility itself.
- **Reachable:** SMS, WhatsApp, and web, in the user's language and without an account.
- **Open and self-hostable:** A deployment can keep resident data inside city infrastructure instead of sending it to a model vendor. See the [model comparison](docs/eval/model-comparison.md).

## Quickstart

```bash
cd heynyc
uv sync --extra dev
cp .env.example .env        # add an LLM key; others are optional
uv run python -m heynyc index-build      # build the RAG index from module seeds
uv run python -m heynyc repl             # interactive, streaming chat
```

Other commands: `modules`, `new-module <name>`, `chat "..."`, `index-search "..."`, `capabilities --write-readme`, and `eval`. For SMS or WhatsApp, run `uv run python -m heynyc serve`; see the [channels guide](heynyc/channels/README.md).

## How it works

HeyNYC routes a question to a service module, then uses grounded tools such as city datasets, the benefits screener, geocoding, and scoped official-source search. It returns inline citations for supported facts and sends the answer through a deterministic grounding guard before delivery. If the evidence does not support a claim, it hedges or abstains and routes people to 311 or the right agency. The [safety guide](SAFETY.md) and [system specs](docs/superpowers/specs/) cover the guard, evaluations, and module design.

## Repo layout

```text
.
├── heynyc/              Python package and CLI
│   ├── core/            Agent loop, grounding, citations, RAG, and shared tools
│   ├── modules/         Service modules, their manifests, data, and evals
│   ├── eval/            Evaluation runner, checks, and trace reporting
│   └── channels/        SMS and WhatsApp adapters
├── docs/                Product, safety, evaluation, and design docs
├── tests/               Offline test suite
├── scripts/             Development and demo scripts
└── .github/             Issue and pull-request templates
```

## Docs map

| Doc | What it covers |
| --- | --- |
| [SAFETY.md](SAFETY.md) | How the AI stays grounded and safe: guardrails, red-team results, abstention, and data freshness. |
| [SECURITY.md](SECURITY.md) | How to report a security vulnerability through the private disclosure policy. |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to add a module or contribute. |
| [CHANGELOG.md](CHANGELOG.md) | The running day-to-day log of what shipped. |

## Status and known limitations

The standalone Python package has offline tests and prior live verification against the real NYC APIs. It includes the agent core, geo, RAG, and web-search tools, the modules above, the Phase 1 runtime grounding guard, and the SMS/WhatsApp on-ramp. Replies are best-effort multilingual, preserve official translations where available, and flag stale data. Next: a web chat UI with a map, wiring the Phase 2 faithfulness checker into the live loop, and an open-weight model behind the guard.

- **Intersection geocoding can be wrong.** NYC GeoSearch can misplace an intersection, so HeyNYC echoes the resolved address and asks for confirmation.
- **Some datasets are thin.** For example, the SNAP-center list lacks per-center hours and phone numbers. HeyNYC does not guess and instead points people to ACCESS HRA or 311.

_Last updated: 2026-07-13_
