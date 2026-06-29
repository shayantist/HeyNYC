# Changelog

Notable changes to HeyNYC, newest first. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); dates are milestones, not yet tagged
releases (this is pre-1.0 and not publicly released).

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
