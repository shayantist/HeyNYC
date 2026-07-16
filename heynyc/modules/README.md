# Writing a HeyNYC service module

HeyNYC is built so that **adding a city service means adding a folder**, without a need to edit the core. Each service is a self-contained *module*. This guide covers what a module is, how to write one, and how far you can get without writing any code. 

> TL;DR: `uv run python -m heynyc new-module parking` → edit the generated
> `manifest.yaml` → `uv run python -m heynyc modules` to confirm it loaded. Done.

---

## Convention (and the standard it follows)

A module is **a self-contained folder with a YAML manifest at its root** — the "one descriptor file per unit, in its own folder" idea behind [Backstage `catalog-info.yaml`](https://backstage.io/docs/features/software-catalog/descriptor-format/) and Helm's `Chart.yaml`: drop in a folder, the core stays untouched. (HeyNYC modules are plain MCP-style tools + a schema-validated manifest — deliberately **not** Anthropic Agent Skills, which are for reusable *techniques*, not project-specific data + config.)

Optional nesting: a `data/` subfolder for bundled curated data, and a `topics/<topic>/` subfolder for a **submodule** — a light, self-contained sub-service that reuses the parent module's tool and owns only its own sources + eval (e.g. `events/topics/world_cup/`, deletable with `rm -rf`). Most modules are still a single `manifest.yaml`. 

---

## How much do you need to know?

| You can… | …then you can build |
|---|---|
| Edit a text (YAML) file from a template | A full **info + web-search** module (eligibility, how-to, links) — **no code** |
| Also copy a dataset ID + column names from NYC Open Data | A **"find nearest X"** module (cooling centers, pools, senior centers) — **no code** |
| Write a little Python | A **custom-tool** module (a live API, special logic) — like `benefits` or `events` |

Most services need only the first two rows: **just pure YAML, zero code.** Custom tools are rare.

---

## What a module looks like

```
heynyc/modules/<name>/
  manifest.yaml     # required — declares the module (pure YAML)
  eval.yaml         # recommended — test questions that prove it doesn't hallucinate
  tools.py          # optional — only if the service needs custom logic
  data/             # optional — curated data files your tool reads
  topics/<topic>/   # optional — a submodule (reuses the parent's tool; own sources + eval)
```

The registry auto-discovers every folder under `modules/` at startup and wires it in: its keywords + capability blurb go into the agent's system prompt, its datasets become `nearest()` categories, its seeds get indexed, its allowlist extends web search, and its eval cases join the test gate.

---

## The manifest, field by field

```yaml
name: cooling_centers            # unique id (folder name)
category: health                 # health | transit | housing | benefits | events | tourism | ...
description: >-                   # one line; shown in listings
  Find cooling and heat-relief sites across NYC during hot weather.
keywords:                         # words that should make the agent reach for this module
  - cooling
  - heat
  - beat the heat

datasets:                         # OPTIONAL — enables "nearest X" via NYC Open Data
  - id: h2bn-gu9k                 # the dataset id from its data.cityofnewyork.us URL
    category: cooling_center      # the category the agent passes to nearest()
    field_map:                    # map the dataset's REAL column names → our common shape
      name: propertyname
      lat: y
      lon: x
      status: status              # optional
      borough: borough            # optional
    where: "status='Activated'"   # optional SoQL filter

seeds:                            # OPTIONAL — official pages indexed for how-to/eligibility
  - https://finder.nyc.gov/coolingcenters/

allowlist:                        # OPTIONAL — extra trusted domains for scoped web search
  - finder.nyc.gov

prompt: |                         # the behavior rules: how to help, what to cite, when to abstain
  For heat-relief questions, call nearest(category="cooling_center", near=<user address>).
  Report only sites the tool returns and cite them. No street address/hours in this data —
  point to finder.nyc.gov and 311 for staffed centers; don't invent hours.

eval: eval.yaml                   # OPTIONAL but recommended
```

Everything except `name` is optional. A module with just `name`, `description`, `keywords`, `seeds`, and `prompt` is a perfectly good info module. Two more optional fields exist for richer modules: **`source_tiers`** (group your `allowlist` domains by trust — `authoritative` / `editorial` / `community` — so `web_search` ranks and disclaims them) and a **`topics/`** folder for submodules. See `modules/events/` for both.

### Finding a dataset + its column names (for "nearest X")
1. Go to <https://data.cityofnewyork.us> and search (e.g. "senior centers").
2. The dataset id is in the URL, e.g. `.../Senior-Centers/3ptp-nu4r` → `3ptp-nu4r`.
3. Open the dataset's "Columns" view to see the real column names; put them in `field_map`.
   (Tip: not every dataset uses the same names, which is exactly why `field_map` exists.)

---

## The three grounding tools your module plugs into

You usually don't have to write the tools, but point the module at these shared ones via the manifest:

- **`nearest(category, near)`** — ranks dataset locations by distance from an address.
  Enabled by adding a `datasets:` entry. Used for "where's the nearest …".
- **`index_search(query)`** — semantic search over your `seeds:` pages. Used for
  "how do I…/am I eligible…/what do I bring".
- **`web_search(query)`** — scoped to your `allowlist:` (+ the global allowlist) for fresh
  or long-tail info. Always cited; abstains when nothing trusted is found.

Tell the agent which to use in your `prompt:`.

---

## Step-by-step: add a "senior centers" module (no code)

```bash
uv run python -m heynyc new-module senior_centers
```
Edit `heynyc/modules/senior_centers/manifest.yaml`:
```yaml
name: senior_centers
category: health
description: Find NYC senior centers near you.
keywords: [senior center, older adults, seniors, aging]
datasets:
  - id: 3ptp-nu4r
    category: senior_center
    field_map: { name: centername, lat: latitude, lon: longitude, borough: borough }
seeds:
  - https://www.nyc.gov/site/dfta/services/find-help.page
prompt: |
  For senior-center questions, call nearest(category="senior_center", near=<address>)
  and cite the results. For programs/eligibility, use index_search and link to the official page.
```
Confirm + (re)build the index:
```bash
uv run python -m heynyc modules          # should list senior_centers
uv run python -m heynyc index-build       # indexes the new seeds
uv run python -m heynyc chat "nearest senior center to Jackson Heights?"
```
That's a complete, grounded, cited service module, written entirely in YAML.

---

## When you need a custom tool (`tools.py`)

Only when the generic tools don't fit, e.g. a live API or special logic. See `modules/benefits/` (`benefits_search` over the live Benefits & Programs API) or `modules/events/` (`whats_on_events` over Ticketmaster + NYC Parks) for worked examples. The contract:

```python
# modules/<name>/tools.py
from heynyc.core.tools.base import Tool, ToolContext

async def _handler(args: dict, ctx: ToolContext) -> str:
    # ctx.citations.register(url, snippet=..., title=..., kind="DATA"|"DOC"|"WEB")
    # return a string the model reads; include {cite:Sn} markers
    ...

def get_tools() -> list[Tool]:
    return [Tool(name="...", description="...", parameters={...}, handler=_handler)]
```
Set `tools: tools.py` in the manifest. **Rule:** never return a fact (location, distance, hours, price) the tool didn't get from real data. 

---

## Writing eval cases (`eval.yaml`)

Every module should ship a few golden questions so the eval gate can prove it stays honest:

```yaml
- id: senior_nearest
  query: "Where's the nearest senior center to Flushing?"
  expect_tools: [nearest]
  expect_cite_kinds: [DATA]
  abstain: false
- id: senior_out_of_scope
  query: "What's the weather tomorrow?"
  abstain: true
```
Run: `uv run python -m heynyc eval`. See [the eval guide](../eval/README.md) for how the gate works.

---

## Checklist for a good module
- [ ] `keywords` cover how real people phrase it.
- [ ] `field_map` matches the dataset's actual columns (check the dataset page).
- [ ] `prompt` says which tool to use, to cite, and **when to abstain**.
- [ ] The answer exposes the source's real update or verification time, never the fetch date as a freshness substitute.
- [ ] Location and deadline results handle the practical “useful now” questions that apply: current or today's availability, holiday or exception status, access restrictions, and the exact next action.
- [ ] Evals include late-night, holiday, stale-source, closure-conflict, and access-restriction cases where those conditions could change the recommendation. See the [useful-now gate](../../docs/superpowers/specs/2026-06-30-service-coverage-map-design.md#7c-the-useful-now-gate-truth-is-necessary-not-sufficient).
- [ ] At least one `abstain: true` eval case (don't-make-things-up coverage).
- [ ] `uv run python -m heynyc modules` lists it; `pytest` stays green.
