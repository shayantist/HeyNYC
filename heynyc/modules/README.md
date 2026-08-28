# Writing a HeyNYC service module

A **module** packages one area of resident help with its sources, typed operations, instructions, limitations, and evals. Keeping those pieces in one folder lets the registry discover them without a core edit.

> TL;DR: `uv run python -m heynyc new-module parking` → edit the generated
> `manifest.yaml` → `uv run python -m heynyc modules` to confirm it loaded. Done.

---

## The internal package convention

A module is **a self-contained folder with a YAML manifest at its root**, following the same general one-descriptor-per-unit convention as [Backstage `catalog-info.yaml`](https://backstage.io/docs/features/software-catalog/descriptor-format/). This is HeyNYC's internal package boundary, not a new interoperability standard. The runtime turns its instructions and operations into [Pydantic AI capabilities](https://pydantic.dev/docs/ai/capabilities/overview/). [Agent Skills](https://agentskills.io/specification) package reusable instructions and workflows, while [MCP](https://modelcontextprotocol.io/specification/2026-07-28) is a protocol for exposing tools and data to external clients. Either can complement a HeyNYC module, but neither replaces its source mappings, normalization, or evals.

Optional nesting: a `data/` subfolder for bundled curated data, and a `topics/<topic>/` subfolder for a **submodule**, a light, self-contained sub-service that reuses the parent module's tool and owns only its own sources and evals (for example, `events/topics/world_cup/`). Cleanup candidates move to the dated quarantine described in `AGENTS.md`; never delete them.

---

## Terminology: what each layer is called

There is no external specification for the whole package HeyNYC needs, so **service module** is the project's name for that internal boundary. Standards and established patterns describe the narrower pieces inside it:

| Term | Meaning in HeyNYC | Why it is not the module |
|---|---|---|
| **Service module** | One discoverable package for an area of resident help, potentially combining sources, adapters, typed operations, instructions, limitations, and evals. Its colocated YAML descriptor follows the general pattern used by [Backstage catalog entities](https://backstage.io/docs/features/software-catalog/descriptor-format/). | This is the complete HeyNYC package and an internal convention, not a claimed industry or civic-data standard. |
| **Source adapter** | Code that translates an external API, dataset, page, or feed into HeyNYC's typed vocabulary. This matches the established [Gateway pattern](https://martinfowler.com/articles/gateway-pattern.html), which notes that such gateways are also commonly called adapters. | One module can use several adapters, and a YAML-only module may need none. |
| **Tool or operation** | A typed function the model can invoke. HeyNYC defines its internal tool boundary in [`core/tools/base.py`](../core/tools/base.py); [MCP tools](https://modelcontextprotocol.io/specification/2026-07-28/server/tools) use the same basic concepts of a name, description, input schema, and result. | It is one callable action, not the surrounding sources, policy, instructions, and evals. |
| **Capability** | The instructions and available tools projected from a module for on-demand use by the current agent, following [Pydantic AI capabilities](https://pydantic.dev/docs/ai/capabilities/on-demand/). | It is the runtime view of a module, not its source package or public interchange format. |
| **Agent Skill** | A portable directory centered on `SKILL.md` that packages reusable instructions, workflows, and optional scripts or references under the [Agent Skills specification](https://agentskills.io/specification). | It can teach an agent how to use HeyNYC, but it does not replace HeyNYC's source mappings, normalization, provenance, or eval gate. |
| **MCP server or public API** | A future external boundary through which other clients could call selected operations. [MCP exposes model-callable tools and data](https://modelcontextprotocol.io/specification/2026-07-28), while [OpenAPI describes HTTP APIs](https://www.openapis.org/). | These describe how clients reach operations, not how HeyNYC packages their implementations internally. |
| **Data profile** | A shared record contract where a real domain standard fits. For example, [Open Referral HSDS](https://docs.openreferral.org/en/latest/hsds/schema_reference.html) separates organizations, services, locations, schedules, and services at locations. | It standardizes returned data, not agent instructions, retrieval, deterministic logic, or evals. |

For consistent project vocabulary, HeyNYC uses **connector** or **client** for provider-facing code, **service module** for the package described above, and **interoperability layer** for the system as a whole. This is a local naming convention, not an external standard. We avoid **middleware** for an individual module because it is less specific about that package's purpose.

---

## How much do you need to know?

| You can… | …then you can build |
|---|---|
| Edit a text (YAML) file from a template | A module over declared pages and shared retrieval tools |
| Also map a dataset ID and its real column names | A simple location lookup through the shared `nearest` tool |
| Write typed Python | A source adapter or deterministic operation for an API, unusual record shape, time logic, or other domain behavior |

Stop at YAML when the shared tools cover the resident need. Add Python only when the source or required deterministic behavior cannot fit that existing path.

---

## What a module looks like

```
heynyc/modules/<name>/
  manifest.yaml     # required, declares the module (pure YAML)
  eval.yaml         # recommended, resident questions and outcome contracts
  tools.py          # optional, typed source adapters or deterministic operations
  data/             # optional, curated data files your tool reads
  topics/<topic>/   # optional, a submodule (reuses the parent's tool; own sources + eval)
```

The registry discovers direct module folders and one optional `topics/` level at startup. It turns descriptions into capability guidance, datasets into `nearest()` categories, seeds into the maintained local corpus, and known domains into web-search trust and ranking signals. The separate eval loader adds each package's `eval.yaml` cases to the test gate. The corpus is an internal retrieval asset, not a competing model-visible tool.

---

## The manifest, field by field

```yaml
name: cooling_centers            # unique id (folder name)
category: health                 # health | transit | housing | benefits | events | tourism | ...
description: >-                   # one line; shown in listings
  Find cooling and heat-relief sites across NYC during hot weather.
datasets:                         # OPTIONAL, enables "nearest X" via NYC Open Data
  - id: h2bn-gu9k                 # the dataset id from its data.cityofnewyork.us URL
    category: cooling_center      # the category the agent passes to nearest()
    field_map:                    # map the dataset's real columns to the shared lookup shape
      name: propertyname
      lat: y
      lon: x
      status: status              # optional
      borough: borough            # optional
    where: "status='Activated'"   # optional SoQL filter

seeds:                            # OPTIONAL, official pages indexed for how-to/eligibility
  - https://finder.nyc.gov/coolingcenters/

allowlist:                        # OPTIONAL, extra preferred domains for scoped web search
  - finder.nyc.gov

prompt: |                         # operation choice, source boundaries, and honest limitations
  For heat-relief questions, call nearest(category="cooling_center", near=<user address>).
  Report only sites the tool returns and cite them. This data has no street address or hours,
  so point to finder.nyc.gov and 311 for staffed centers; do not invent hours.

eval: eval.yaml                   # OPTIONAL but recommended
```

Everything except `name` is optional. A module with just `name`, `description`, `seeds`, and `prompt` is a perfectly good info module. Two more optional fields exist for richer modules: **`source_tiers`** (group known domains by trust so `web_search` can rank, label, and prefer them) and a **`topics/`** folder for submodules. Source tiers describe evidence; they do not decide claim risk. The runtime's current-turn high-stakes decision applies stricter evidence rules when needed. See `modules/events/` for both.

### Finding a dataset + its column names (for "nearest X")
1. Go to <https://data.cityofnewyork.us> and search (e.g. "senior centers").
2. The dataset id is in the URL, e.g. `.../Senior-Centers/3ptp-nu4r` → `3ptp-nu4r`.
3. Open the dataset's "Columns" view to see the real column names; put them in `field_map`.
   (Tip: not every dataset uses the same names, which is exactly why `field_map` exists.)

---

## The grounding tools your module plugs into

You usually don't have to write the tools, but point the module at these shared ones via the manifest:

- **`nearest(category, near)`**, ranks dataset locations by distance from an address.
  Enabled by adding a `datasets:` entry. Used for "where's the nearest …".
- **`web_search(query)`**, searches the live web for fresh or long-tail information. A module's
  known domains guide trust and ranking without blocking discovery from unlisted sources. For an
  low-stakes turn, an unverified excerpt may support only the claim it states and remains labeled
  unverified. High-stakes turns require authoritative evidence from a sufficient
  direct excerpt or fetched page
  ([source trust grading](../core/tools/web_search.py)).
- **`web_fetch(url, query)`**, opens one known result page when its search excerpt does not contain
  enough evidence.

The multilingual hybrid index over `seeds:` remains internal. Do not name it in a module prompt or
make the model choose between the cache and the live web.

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
datasets:
  - id: 3ptp-nu4r
    category: senior_center
    field_map: { name: centername, lat: latitude, lon: longitude, borough: borough }
seeds:
  - https://www.nyc.gov/site/dfta/services/find-help.page
prompt: |
  For senior-center questions, call nearest(category="senior_center", near=<address>)
  and cite the results. For programs or eligibility, search the live web with the DFTA domain
  preferred, fetch the selected official page when needed, and link it.
```
Confirm + (re)build the index:
```bash
uv run python -m heynyc modules          # should list senior_centers
uv run python -m heynyc index-build       # indexes the new seeds
uv run python -m heynyc chat "nearest senior center to Jackson Heights?"
```
That is a complete YAML-only service module.

---

## When you need a custom tool (`tools.py`)

Only when the generic tools do not fit, such as a live API, a source-specific record shape, or deterministic schedule logic. See `modules/benefits/` or `modules/events/` for worked examples. HeyNYC's current [`Tool` policy boundary](../core/tools/base.py) is adapted into a native Pydantic AI tool at runtime; use Pydantic models to generate schemas and validate structured results when practical:

```python
# modules/<name>/tools.py
from pydantic import BaseModel

from heynyc.core.location import LocationRequest
from heynyc.core.tools.base import Tool, ToolContext

class Query(LocationRequest):
    pass

class Result(BaseModel):
    name: str
    source_url: str

async def _handler(query: Query, ctx: ToolContext) -> Result:
    # ctx.citations.register(url, snippet=..., title=..., kind="DATA"|"DOC"|"WEB")
    # Fetch and normalize the real source using query.near.
    ...

def get_tools() -> list[Tool]:
    return [Tool(
        name="find_example",
        description="Find an example near a resident-supplied NYC location.",
        input_type=Query,
        return_type=Result,
        handler=_handler,
    )]
```
Set `tools: tools.py` in the manifest. Never present a location, distance, hour, price, deadline, eligibility rule, or organization name as verified unless the operation retrieved supporting evidence. Preserve useful partial records and source links while marking unsupported fields as missing or unverified.

Location-aware tools should inherit [`LocationRequest`](../core/location.py), which provides the common `near` and `max_results` names. `near` may be the resident's current conversation location or another NYC place they name. A tool may add only the constraints it actually uses, such as `visit_date`, `visit_time`, radius, accessibility, or service type. Resolve the origin through the shared [geographic tools](../core/tools/geo.py), apply hard source-backed filters first, and use their distance ranking rather than adding a module-specific nearest algorithm. Cited coordinates let the answer layer add a Google Maps link consistently. When a source lacks coordinates or schedules, keep that field unknown instead of manufacturing it.

---

## Writing eval cases (`eval.yaml`)

Every module should ship a few resident questions covering its normal path and its actual source limitations:

```yaml
- id: senior_nearest
  query: "Where's the nearest senior center to Flushing?"
  expect_tools: [nearest]
  expect_cite_kinds: [DATA]
  abstain: false
- id: senior_hours_missing
  query: "Is that senior center open at 7 tonight?"
  abstain: false
  notes: Preserve the supported center and source link, and label hours unverified when the source does not provide them.
```
Run: `uv run python -m heynyc eval`. See [the eval guide](../eval/README.md) for how the gate works.

---

## Checklist for a good module
- [ ] `field_map` matches the dataset's actual columns (check the dataset page).
- [ ] `prompt` says which operation and source to use, what can be verified, and which limitations must remain visible.
- [ ] The answer exposes the source's real update or verification time, never the fetch date as a freshness substitute.
- [ ] Location and deadline results handle the practical "useful now" questions that apply: current or today's availability, holiday or exception status, access restrictions, and the exact next action.
- [ ] Evals include late-night, holiday, stale-source, closure-conflict, and access-restriction cases where those conditions could change the recommendation (the "useful-now" gate: a grounded answer is necessary, not sufficient).
- [ ] Evals cover the normal path, relevant missing or conflicting source data, provider failure, and the inverse of any deterministic response.
- [ ] `uv run python -m heynyc modules` lists it; `pytest` stays green.
