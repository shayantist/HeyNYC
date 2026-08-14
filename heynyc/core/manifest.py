"""ServiceModule manifest, the extension contract.

Each pluggable service ("skill") lives in `heynyc/modules/<name>/manifest.yaml`
and declares its data sources, index seeds, web allowlist, capability blurb,
optional tools, and eval cases. Adding a service never touches the core.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class DatasetBinding(BaseModel):
    """Binds a structured NYC data source to a findable category.

    Two sources share one declarative shape (so `geo.nearest(category=...)` stays
    uniform regardless of backend): a NYC Open Data (Socrata) dataset addressed by
    `id`, or an ArcGIS Feature Service layer addressed by `url`. `field_map` maps the
    common shape (name/address/lat/lon/status/phone) to the source's actual column
    names.
    """

    source: str = "socrata"  # "socrata" | "arcgis"
    id: Optional[str] = None  # Socrata dataset id, e.g. "h2bn-gu9k" (socrata only)
    url: Optional[str] = None  # ArcGIS FeatureServer/<n> layer URL (arcgis only)
    category: str  # category name used by geo.nearest, e.g. "cooling_center"
    field_map: dict[str, str] = Field(default_factory=dict)
    where: Optional[str] = None  # optional default filter (SoQL $where / ArcGIS where)
    record_id_field: Optional[str] = None  # stable id column for arcgis row-addressing, e.g. "NYCEM_ID"
    title: Optional[str] = None  # citation title override
    limitations: str = ""  # source-level claim boundary preserved in tool and final output

    @model_validator(mode="after")
    def _check_source(self) -> "DatasetBinding":
        if self.source == "socrata":
            if not self.id:
                raise ValueError("socrata dataset binding requires 'id'")
        elif self.source == "arcgis":
            if not self.url:
                raise ValueError("arcgis dataset binding requires 'url'")
            if not self.record_id_field:
                raise ValueError("arcgis dataset binding requires 'record_id_field'")
        else:
            raise ValueError(f"unknown dataset source '{self.source}' (expected 'socrata' or 'arcgis')")
        return self


class SituationHint(BaseModel):
    """A module-owned high-stakes situation (RULED 2026-07-18: checklist, not router).

    The scope preflight checks situations by their meaning-based `definition` in any
    language; a checked situation contributes its forced-search config, reminder, and tool
    focus to the SAME single-agent turn. All data lives here in the module's manifest, never
    as core constants; legacy core regexes remain only as preflight-absent fallbacks."""

    model_config = ConfigDict(extra="forbid")

    name: str
    definition: str  # meaning-based, Bedrock denied-topics style: never a word list
    query: str = ""  # forced-first retrieval query
    urls: list[str] = Field(default_factory=list)  # declared official pages
    reminder: str = ""  # the runtime scope reminder for this situation
    high_stakes: bool = False  # forces the first retrieval call before answering
    # Tool focus applied ONLY when this module is the turn's sole checked module; a
    # cross-module turn never loses capability (prioritize, never narrow).
    focus_tools: list[str] = Field(default_factory=list)


class ServiceModule(BaseModel):
    """A single city service, loaded from a manifest.yaml."""

    model_config = ConfigDict(extra="forbid")

    name: str
    category: str = "general"
    description: str = ""
    keywords: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)  # user-facing example queries, the single source for capability discovery
    datasets: list[DatasetBinding] = Field(default_factory=list)
    official_link: str = ""  # public service link, independent from RAG index seeds
    seeds: list[str] = Field(default_factory=list)
    allowlist: list[str] = Field(default_factory=list)
    prompt: str = ""  # capability blurb injected into the system prompt
    # Shared tools kept visible after this module capability loads. Empty means no narrowing.
    focus_tools: list[str] = Field(default_factory=list)
    tools: Optional[str] = None  # optional module-specific tools file (e.g. "tools.py")
    eval: Optional[str] = None  # eval cases file (e.g. "eval.yaml")
    # Trust tiers for web_search ranking: {tier: [domain, ...]} where tier is one of
    # authoritative | editorial | community. Aggregated by registry.source_tiers().
    source_tiers: dict[str, list[str]] = Field(default_factory=dict)
    # RULED 2026-07-18, stakes-tiered retrieval: official modules (the DEFAULT) answer from
    # official sources only, because being wrong costs someone a benefit, a home, or their
    # safety. A module opts out (`official_only: false`) only when its answers are lifestyle
    # discovery — events atmosphere, where-to-watch, fun — where a governed editorial pool
    # matches the stakes. Enforced at load: no opt-out, no editorial/community tiers.
    official_only: bool = True
    # Submodule hint (events topics): the Ticketmaster `keyword` the agent should pass to
    # find_nyc_events for this topic. Advisory metadata; the prompt blurb drives the call.
    ticketmaster_keyword: Optional[str] = None
    # Module-owned high-stakes situations for the scope preflight checklist.
    situations: list[SituationHint] = Field(default_factory=list)

    # Populated by the loader, not from YAML.
    path: Optional[Path] = Field(default=None, exclude=True)
    parent: Optional[str] = Field(default=None, exclude=True)  # set on submodules to the parent module name

    @model_validator(mode="after")
    def _official_only_blocks_non_official_tiers(self) -> "ServiceModule":
        if self.official_only:
            declared = set(self.source_tiers) & {"editorial", "community", "news"}
            if declared:
                raise ValueError(
                    f"module {self.name!r} declares {sorted(declared)} source tiers but is "
                    "official_only (the default); set `official_only: false` in its manifest "
                    "only if this module answers lifestyle discovery, never benefits, housing, "
                    "health, or safety"
                )
        return self

    @classmethod
    def from_manifest(cls, manifest_path: Path) -> "ServiceModule":
        data = yaml.safe_load(manifest_path.read_text()) or {}
        module = cls(**data)
        module.path = manifest_path.parent
        return module


# Name alias for the shared engine-extraction contract (see the boundary spec): the generic
# concept is "Module"; HeyNYC's concrete type is ServiceModule.
Module = ServiceModule
