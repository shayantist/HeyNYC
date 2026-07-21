# HeyNYC

**HeyNYC** helps New Yorkers find and understand New York City government services, grounded in real city data and official sources. The current adapters are CLI, SMS, and WhatsApp. There is no resident web UI or durable hosted deployment yet.

It answers questions about benefits, housing, food, health, immigration, and events. Supported factual claims carry an official citation; when evidence is insufficient, HeyNYC abstains and points you to 311.

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
| **Cooling centers** | "Where's the nearest cooling center?"<br>"It's too hot, where can I cool off indoors today?" | NYC Emergency Management - Activated Cooling Centers + NYC Emergency Management - Year-round Cool Options | [finder.nyc.gov](https://finder.nyc.gov/coolingcenters/) |
| **Drinking fountains** | "Where is the nearest drinking fountain?"<br>"Where can I refill my water bottle near Rockefeller Center?" | NYC Parks Drinking Fountains | [data.cityofnewyork.us](https://data.cityofnewyork.us/Recreation/NYC-Parks-Drinking-Fountains-Map/622h-mkfu) |
| **Events** | "What's happening in NYC this weekend?"<br>"Any free events tonight?" | Ticketmaster + NYC Parks + NYC permitted street events | [nyctourism.com](https://www.nyctourism.com/events/) |
| **Food pantries** | "Where's the nearest food pantry?"<br>"I need free food today near me" | NYC FoodHelp finder | [finder.nyc.gov](https://finder.nyc.gov/foodhelp/) |
| **Housing & eviction help** | "I got an eviction notice, where can I get help near me?"<br>"My landlord won't turn on the heat, what do I do?" | NYC DSS/DHS - Homebase (eviction prevention) offices | [access.nyc.gov](https://access.nyc.gov) |
| **Housing Connect** | "What affordable housing lotteries are open right now?"<br>"How do I apply for an apartment through the housing lottery?" | NYC Open Data (vy5i-a666) | [housingconnect.nyc.gov](https://housingconnect.nyc.gov) |
| **311 service requests** | "Is my 311 complaint moving? My service request number is 69741503"<br>"What's happening with 311 noise complaints near Union Square?" | NYC Open Data (erm2-nwe9) | [data.cityofnewyork.us](https://data.cityofnewyork.us) |
| **Public restrooms** | "Where is the nearest public restroom?"<br>"Is there a public bathroom near Union Square?" | NYC Open Data (i7jb-7jku) | [data.cityofnewyork.us](https://data.cityofnewyork.us) |
| **SNAP centers** | "Where's the nearest SNAP center?"<br>"Where do I apply for food stamps in person?" | NYC Open Data (tc6u-8rnp) | [access.nyc.gov](https://access.nyc.gov) |
| **Street closures** | "Are any streets closed near Yankee Stadium this Saturday?"<br>"Is there construction closing streets near Union Square right now?" | NYC Open Data (i6b5-j7bu) | [nyc.gov](https://www.nyc.gov/html/dot/html/motorist/weektraf.shtml) |
| **WIC** | "Where's the nearest WIC office?"<br>"I'm pregnant and need WIC near Jackson Heights, Queens" | NY State WIC directory (Health Data NY) | [health.ny.gov](https://www.health.ny.gov/prevention/nutrition/wic/) |
| **Worker rights** | "My boss is keeping our tips, is that legal?"<br>"The restaurant I work at takes a cut of our tips, can they do that?" | NY Labor Law + NYC DCWP | [dol.ny.gov](https://dol.ny.gov) |

<!-- CAPABILITIES:END -->

## Current status

HeyNYC is an alpha release. The messaging pilot can run from an operator-managed development server, but it is not a durable public hosted service.

- **Built:** a Python CLI, SMS and WhatsApp adapters, grounded service modules, scoped official-source web search as a retrieval tool, a deterministic citation guard, and an offline evaluation suite.
- **Prototype, off by default:** the optional benefits application-form draft workflow, translate-at-edge pipeline, and Tier-2 faithfulness checker. Forms require explicit configuration and encryption settings.
- **Conversation continuity:** messaging sessions resume from encrypted local transcripts and expire after the configured inactivity period. Resident-answer context is measured before it reaches the answer model, older turns compact only under pressure, `NEW` starts fresh model context, and undelivered replies are not committed. Texting `DELETE MY DATA` and confirming erases the resident's transcript, any draft, and any pending report flags in chat; see the [privacy notice](docs/legal/HEYNYC-PRIVACY.md) and [safety guide](SAFETY.md).
- **Not yet shipped:** a resident web UI, durable hosted deployment, authenticated browser actions, automatic application submission, and demonstrated production multilingual safety.
- **Known limitations:** intersection geocoding can be wrong, so HeyNYC echoes the resolved address and asks for confirmation. Some city datasets are thin (the SNAP-center list has weekday hours but no phone numbers), and HeyNYC says so rather than filling the gap.

The assistant gives best-effort answers in the user's language when configured, but multilingual behavior is not a verified production capability. It does not decide eligibility or submit applications. When evidence is insufficient, it abstains and routes to 311 or the relevant agency.

## What makes it different

- **Cite or abstain:** Supported factual claims need a citation or an abstention. A deterministic guard rechecks cited claims before an answer ships. See [SAFETY.md](SAFETY.md).
- **Does, not just tells:** It uses the city's benefits screener and can prepare the real application for review, without deciding eligibility itself.
- **Reachable today:** CLI, SMS, and WhatsApp adapters, without a resident account. SMS and WhatsApp require provider configuration.
- **Open and self-hostable:** The package can be run by an operator who controls its infrastructure and model configuration.

## Using it

**As a resident (SMS or WhatsApp):** text the pilot number your operator shares. Ask in plain language, any language. Your first message gets a one-time note naming the built-in commands, which work in any chat, always free of model calls:

| Text | What happens |
| --- | --- |
| `HELP` | what HeyNYC can do |
| `PRIVACY` | how your info is handled, in short |
| `REPORT` (or 👎) | flag the last answer for human review, after you confirm |
| `DELETE MY DATA` | erase your conversation, draft, and pending flags, after you confirm |
| `NEW` | start a fresh conversation the assistant no longer sees |
| `STOP` / `START` | SMS opt-out and opt-in (carrier-level) |

**As an operator or developer (CLI):** every command reads `.env` for models and keys; `--model` overrides explicitly where offered.

| Command | What it does |
| --- | --- |
| `heynyc repl` | interactive chat on the SAME path texters use: commands, sessions, caps all live. `--user <name>` keys a separate identity, `--temp` is a throwaway session that persists nothing, `--raw` is the bare-agent debug view |
| `heynyc chat "..."` | one-shot question |
| `heynyc serve` | the SMS/WhatsApp webhook server (see the [channels guide](heynyc/channels/README.md); the pilot launcher is `scripts/serve_demo.sh`) |
| `heynyc eval` | the no-hallucination gate. `--list` shows every case with tags, `--tag` / `--module` / `--case` select slices, `--sample N --seed K` rotates, bare runs need `--all` (cost guard) |
| `heynyc feedback` | resident-flagged exchanges, decrypted locally for triage |
| `heynyc stats` | turns, outcomes, costs, cache rates from local telemetry |
| `heynyc bench --models a,b` | run the case set across candidate models |
| `heynyc modules` / `new-module` | list or scaffold service modules |
| `heynyc index-build` / `index-search` | build or probe the retrieval index |
| `heynyc capabilities --write-readme` | regenerate the capability table below |

## Quickstart

```bash
# Run these commands from the repository root
uv sync --extra dev
cp .env.example .env        # add an LLM key; others are optional
uv run python -m heynyc index-build      # build the RAG index from module seeds
uv run python -m heynyc repl             # interactive, streaming chat
```

Scoped web search is a retrieval tool, not a resident-facing web channel. For SMS or WhatsApp, configure `HEYNYC_PII_SALT` and `HEYNYC_PII_KEY`, then run `uv run python -m heynyc serve`; see the [channels guide](heynyc/channels/README.md).

## How it works

HeyNYC routes a question to a service module, then uses grounded tools such as city datasets, the benefits screener, geocoding, and scoped official-source search. It returns inline citations for supported facts and sends the answer through a deterministic grounding guard before delivery. If the evidence does not support a claim, it hedges or abstains and routes people to 311 or the right agency. The [safety guide](SAFETY.md) covers the guard and its boundaries.

## Repo layout

```text
.
├── heynyc/              Python package and CLI
│   ├── core/            Agent loop, grounding, citations, RAG, and shared tools
│   ├── modules/         Service modules, their manifests, data, and evals
│   ├── eval/            The live evaluation harness: runs the real model and tools against each module's eval.yaml contract and grades resident outcomes; inside the package because `heynyc eval` and `bench` are product commands
│   └── channels/        SMS and WhatsApp adapters
├── docs/                Public docs
│   ├── testing/         Generated public test records (failure register, red-team, benchmarks)
│   ├── legal/           Formal Privacy Notice and Terms of Use
│   └── internal/        Local-only specs, plans, and dev notes (gitignored, not shipped)
├── tests/               Offline pytest suite: deterministic, mocked, free; proves code contracts. Root by Python convention, never shipped
├── scripts/             Development and demo scripts
└── .github/             Issue and pull-request templates
```

Offline tests prove contracts; live evals prove behavior.

## Docs map

| Doc | What it covers |
| --- | --- |
| [SAFETY.md](SAFETY.md) | How the AI stays grounded and safe: guardrails, red-team results, abstention, and data freshness. |
| [SECURITY.md](SECURITY.md) | How to report a security vulnerability through the private disclosure policy. |
| [PRIVACY.md](PRIVACY.md) | Plain-language: what happens to your messages. The formal [Privacy Notice](docs/legal/HEYNYC-PRIVACY.md) controls. |
| [Terms of Use](docs/legal/HEYNYC-TERMS.md) | Pilot limitations, messaging terms, and user responsibilities. |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to add a module or contribute. |
| [CHANGELOG.md](CHANGELOG.md) | The running day-to-day log of what shipped. |
| [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) | The community standards for taking part. |
| [LICENSE](LICENSE) | The open-source license the project ships under. |
| [`docs/testing/`](docs/testing/) | HeyNYC's public test records; the eval harness that produces the gate results lives in [`heynyc/eval/`](heynyc/eval/README.md). Covers the [failure register](docs/testing/failure-db.md), the [red-team write-up](docs/testing/red-team.md), and the [benchmark methodology](docs/testing/benchmarks.md). |
| [heynyc/eval/README.md](heynyc/eval/README.md) | The evaluation harness: cases, gates, rubric, and how to run it. |
| [heynyc/channels/README.md](heynyc/channels/README.md) | SMS and WhatsApp adapter conventions. |
| [heynyc/modules/README.md](heynyc/modules/README.md) | Service-module structure and how to scaffold one. |

## How we test it

Every case, gate, and failure-driven regression is documented in [heynyc/eval/README.md](heynyc/eval/README.md), including the methodology and where it comes from; [SAFETY.md](SAFETY.md) covers the deterministic guardrails those evals run behind.

## FAQ

**Do people read my messages?** No one reads your conversations in the normal course of things. A human sees an exchange only if you send it to us with REPORT and confirm, or if a safety or abuse review requires it. Text DELETE MY DATA and confirm to erase your transcript, draft, and pending flags yourself. The plain-language version is [PRIVACY.md](PRIVACY.md); the formal notice is the [Privacy Notice](docs/legal/HEYNYC-PRIVACY.md).

**How do I know it isn't making things up?** Every factual claim carries a citation to an official source, checked by a deterministic guard before the answer reaches you. When the source doesn't back a claim, the answer is regenerated or HeyNYC says it can't confirm. When it can't ground an answer, it says so and points you to 311 or the official page instead of guessing.

**Is it free?** Yes. Standard messaging rates from your carrier apply, nothing from us.

**What languages?** Write in whatever language you're comfortable in: Spanish, Bengali, Chinese, Urdu, and more. Program names, addresses, and links stay exact because the official pages are in English.

**Something was wrong or unhelpful. What do I do?** Text REPORT (or just 👎) after the bad answer. You'll be asked to confirm before that one exchange is shared with a human reviewer.

_Last updated: 2026-07-21_
