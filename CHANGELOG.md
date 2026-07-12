# Changelog

Notable changes to HeyNYC, newest first. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); dates are milestones, not yet tagged
releases (this is pre-1.0 and not publicly released).

## 0.6 - 2026-07-11

- **Heat and hot water, in the `housing` module.** No-heat / no-hot-water is a code emergency, so the module now grounds the tenant's actual rights instead of the bare complaint form: the heat-season dates and indoor-temperature standard, the Housing Maintenance Code sections (§27-2029 heat, §27-2031 hot water), and the escalation ladder (311 complaint → HPD inspection → immediately-hazardous class C violation → Housing Court). A new `hpd_litigation_lookup(address)` pulls a building's HPD housing-court record and flags open Heat and Hot Water cases and any harassment finding, from [NYC Open Data (59kj-x8nc)](https://data.cityofnewyork.us/Housing-Development/Housing-Litigations/59kj-x8nc). Every fact comes from the tool with its citation; nothing is stated from memory.
- **`housing_connect` module: open affordable-housing lotteries.** A finder for the NYC Housing Connect lotteries accepting applications right now (borough, unit and bedroom mix, income bands, set-aside preferences, deadline), grounded in [NYC Open Data (vy5i-a666)](https://data.cityofnewyork.us/Housing-Development/Advertised-Lotteries-on-Housing-Connect-by-Lottery/vy5i-a666/about_data) and cited. Applying is login-gated and human-in-the-loop: it deep-links to the [Housing Connect portal](https://housingconnect.nyc.gov/PublicWeb/) and never fills or submits an application for you.
- **Runtime grounding guard, Phase 2 Tier-2 NLI checker (prototype, off by default).** A self-hosted [MiniCheck-class NLI](https://arxiv.org/abs/2404.10774) faithfulness checker for the prose claims the deterministic Tier-1 check can't parse (a made-up statute cited to a soft web source, which Tier-1 is silent on). Built and exercised offline, off by default, and not yet wired into the live loop; real-model validation and threshold calibration are still pending. Status and caveats in [SAFETY.md](SAFETY.md); design in the [Tier-2 spec](docs/superpowers/specs/2026-07-09-tier2-nli-checker-design.md).
- **`wic` module: WIC site finder.** A nearest-site finder for the WIC nutrition program (pregnant people, infants, young children), grounded in the NY State WIC directory on Health Data NY (dataset g4i5-r6zx) since WIC is state-administered, not city-run. Returns nearest sites with address, phone, and website, each cited; abstains on hours and eligibility (neither is in the data) and routes to the site by phone and to the [state WIC page](https://www.health.ny.gov/prevention/nutrition/wic/).
- **Translate-at-edge multilingual pipeline (prototype, off by default).** The Local Law 30 language-safety architecture: ground and verify in English, then translate only the finished answer with law numbers, dollar amounts, and citations frozen and passed through verbatim, gated by an entity round-trip check that falls back to the English answer plus an official pointer on any mismatch. Structurally prevents a citation like "Local Law 34" mutating into an invented "Ley Local 56" in Spanish. Self-contained, off by default (`HEYNYC_MULTILINGUAL`), not yet wired into the live loop; design in the [translate-at-edge spec](docs/superpowers/specs/2026-07-05-multilingual-translate-at-edge-design.md).
- **Reusable red-team harness with an independent grader.** A frozen 205-case adversarial suite across 8 categories, run against a candidate model and scored by a fresh-context grader in a different model family (a same-family grader is refused, so a model can never grade its own output), with reconciliation and a per-category report. Reusable across models, so it doubles as model-flip evidence. Build-only in the tree; running it live spends API keys. Methodology in [red-team-v2-methodology.md](docs/eval/red-team-v2-methodology.md).
- **Outcomes funnel and user feedback loop.** A PII-free funnel report (turns to screened to eligible-shown to apply-started to form-ready) so we can measure whether a resident actually reached the apply step, plus a `/wrong` feedback command that logs a redacted flag for a human to review without running the agent. New CLIs: `heynyc outcomes`, `heynyc feedback`.

## 0.5.2 - 2026-07-05

- **Runtime grounding guard, Phase 1 promoted to the live path.** The deterministic cited-claim check now runs online on every answer, hands the model the specific ungrounded claim to fix (capped retries, not open-ended self-reflection), then hedges or abstains on any residual, all tuned to never over-block a genuinely grounded answer. This is what lets a cheaper backend model stay safe.
- **Safety and compliance hardening.** A 137-query adversarial red-team across 8 categories, independently re-graded (the builder's own "0 failures" was caught by a fresh-context grader; honest result 130 safe / 7 grounding-accuracy fixes, all repaired), plus a Local Law 30 and accessibility audit that surfaces the city's own official non-English content instead of re-translating it, and a reading-level check.
- **`clinics` module.** A safety-net clinic finder grounded in the FQHC list plus NYC Care / Health + Hospitals, answering the "will this place see me regardless of status or ability to pay" question a map pin strips out.
- **Self-hosted open models and a leaner prompt.** A self-hosted Ollama model can now drive the agent end to end (`HEYNYC_OLLAMA_NUM_CTX`, since the default context silently truncated the prompt and broke tool-calling), the system prompt moved to progressive disclosure (only query-relevant module blurbs, with safety rules and a capability menu always on), Anthropic prompt caching rides the stable prefix, and a reusable `bench --models a,b,c` command runs the eval across candidate models. GPT-5 models get a model-aware temperature fix.
- **Channels: a generic `TWILIO_FROM`** so plain SMS works without a WhatsApp sender.
- **Advisories fail-safe.** An empty or unreachable Notify NYC feed never yields a false all-clear; it routes to nyc.gov/notifynyc, 311, and 911, and reads the live feed the Notify NYC portal itself uses.
- **Geocoder borough bias.** A borough-aware bounding box so "a street in Manhattan" stops resolving to the same-named street in another borough.

## 0.5.1 - 2026-07-03

- **`housing` module v1.** A Homebase office finder, an `hpd_building_lookup` (address to BBL to open HPD complaints and violations, cited), and Right-to-Counsel / heat / shelter routing.
- **`advisories` module.** A Notify NYC client and a queryable `nyc_advisories` tool that cites each advisory's as-of date and abstains when nothing is active.
- **`snap_centers` and `food_pantries`, on a shared ArcGIS adapter.** A SNAP-center finder and a food-pantry finder (open-now, dietary flags decoded from the coded domain, row-addressed citations) on a new generic `nearest()` adapter reused across every finder rather than a bespoke client per dataset.
- **Currency layer.** A `recent_developments` tool beside the default allowlist-only `web_search`, surfacing an on-point, dated, cited legal development as a caveat that never overrides the official answer.
- **Verifiable citations, deterministic core (Part C).** DATA citations carry a row snapshot plus SHA-256 plus a row permalink; the eval floor recomputes distances from the cited row and blocks an ungrounded structured fact.
- **Eval and geocoder fixes.** The judge split into a free interactive agent-judge and a paid API-judge, calibrated to stop false-failing current-dated, absence, and refusal answers; bare NYC zips resolve from a bundled Census ZCTA table; the cooling finder repointed to the real indoor dataset; the README capabilities table is auto-generated from manifests and drift-guarded.

## 0.5 — 2026-06-29

- **Eval overhaul → a tiered no-hallucination gate.** A deterministic *structural floor* (citation,
  faithfulness, forbidden-tool, link-liveness) that blocks CI, plus a portable **agent-as-judge** for
  the semantic verdict (abstained-for-the-right-reason, grounded + useful) that any coding agent can
  run. Retrieval upgraded to the documented standard — **hybrid dense + BM25 fused with Reciprocal
  Rank Fusion**. Principled checker fixes: refuse-with-redirect passes; a grounded, cited answer isn't
  mislabeled an abstention; only a definitive 404/410 counts as a dead link.
- **Agent voice.** Warm, direct, plain — a "Mamdani register": sincere, specific acknowledgment on
  high-stakes questions, texting-style formatting (no walls of markdown), and it surfaces the
  map/directions links the tools provide.
- **Multilingual replies.** Answers in the user's language; program names, addresses, and links kept
  as-is.
- **Staleness guard.** Every fact carries an "as of" date and is flagged when older than a module's
  tolerance (benefits = re-check annually) — the active half of the freshness guarantee.
- **REPL.** Events render in true chronological order (narration and tool calls stack as they happen).

## 0.4 — 2026-06-28

- **`benefits` navigator.** Live NYC Benefits & Programs data, hybrid retrieval, always-caveated
  eligibility that defers to the official screener; a per-program "as of" date on every fact.
- **`events` module + seasonal `world_cup` submodule.** Live Ticketmaster + NYC Parks, trust-tiered
  and ranked `web_search`, and the submodule architecture (`topics/`). Retired the standalone
  `things_to_do` and `world_cup` modules.
- **Observability.** Cost/usage telemetry (`stats`) and an in-process embedding cache.

## 0.3 — 2026-06-27

- **Scoped `web_search`** (allowlist-enforced) and the streaming `repl`; sessions / persistence.
- **Geocoder reliability.** Confidence gate + non-NYC reject behind a swappable provider.

## 0.2 — 2026-06-26

- **Eval gate.** Deterministic checks + OpenInference traces + outcome invariants + pass^k on the
  safety subset; optional cross-family LLM groundedness judge.

## 0.1 — 2026-06-25

- **Core + module SDK.** Manifest, registry, citations, streaming tool-calling agent loop, prompts.
- **Grounded tools + first modules.** `geocode` / `nearest` / `distance` and the `cooling_centers`
  module; hybrid-RAG `index_search`.
