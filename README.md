# HeyNYC

Talk to the city in plain English: *"Hey NYC, where's the nearest cooling center?"* or *"what can I do this
weekend?"* and HeyNYC answers from real city data and shows you exactly where every fact came from. 

**A grounded, no-hallucination assistant for NYC.gov services & events.** 

Given everything happening in NYC today (June 2026 as of writing!), you can ask it *"where's the nearest cooling center?"* or *"where can I watch the World Cup this weekend?"* (and more services to be added soon!) and it finds the answer in real city data, walks you through it, and cites every source. When nothing grounds an answer, it says it doesn't know. 

Goal is to build this up into a conversational front door for NYC.gov's rich catalogue of services! 

> Open-source passion project. Not affiliated with the City of New York.

## Why

The city's data is rich but scattered across dozens of sites and portals. [MyCity](https://mycity.nyc.gov/) was intended to be the official push to centralize it, but ended up becoming a [skeleton of links](https://nysfocus.com/2025/03/19/mycity-eric-adams-child-care) that's not the easiest to navigate through, and a chatbot that provided inaccurate info. 

Thus, I wanted to make HeyNYC as a way to make government services accessible through natural language by leveraging LLMs while making sure to **minimize hallucinations** and cite its sources. A key underpinning of the agent is deterministically enforcing that every fact comes from a specific grounded tool (NYC Open Data, geocoding, scoped web search) and ships with a citation. When nothing grounds an answer, it HAS to abstain. 

If you find any failure modes, please feel free to **make an issue** as we're also making a database of failures to test against and make sure we [fill any and all holes](https://www.instagram.com/p/DWMD6wGD6-O/) as we go. 

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

Services are **pluggable modules**: each is a self-contained folder (manifest + optional tool +
its own eval), so adding a service is adding a folder and deleting one is deleting the folder.
Built-in so far: `benefits`, `cooling_centers`, and `events` (with a seasonal `world_cup` topic).
See the **[module authoring guide](heynyc/modules/README.md)**.

## Quickstart

```bash
cd heynyc
uv sync --extra dev
cp .env.example .env        # add an LLM key; others are optional
uv run python -m heynyc index-build      # build the RAG index from module seeds
uv run python -m heynyc repl             # interactive, streaming chat
```

Other commands: `modules` (list), `new-module <name>` (scaffold), `chat "..."` (one-shot),
`index-search "..."` (query the index), `eval` (run the no-hallucination gate).

## Contributing

On top of contributing to the code, one great way to contribute is adding service modules. See **[CONTRIBUTING.md](CONTRIBUTING.md)**
and the **[module authoring guide](heynyc/modules/README.md)**. You can also
[request a module](.github/ISSUE_TEMPLATE/new_service_module.yml)! 

## Status

Standalone Python package, fully offline-tested and live-verified against the real NYC APIs. Built and working: the agent core, the geo / RAG / web-search tools, the service modules above, and the [no-hallucination eval gate](heynyc/eval/README.md) (currently green). It answers in the user's language and flags data that's gone stale. Full history is in **[CHANGELOG.md](CHANGELOG.md)**.

**Next:** a WhatsApp on-ramp — meet people where they already are — then a web chat UI with a map.

## Known limitations

- **Intersections geocode poorly.** NYC GeoSearch sends "116 St and Broadway" to the wrong neighborhood, and its confidence scores don't flag it. HeyNYC echoes back the address it resolved and asks you to confirm before trusting it.
- **Some datasets are thin.** The cooling-centers data has no street addresses or hours, so HeyNYC won't guess them. It points you to finder.nyc.gov and 311 instead. Future work involves indexing that info into the tools for the agent. 