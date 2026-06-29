"""ServiceModule manifest — the extension contract.

Each pluggable service ("skill") lives in `heynyc/modules/<name>/manifest.yaml`
and declares its data sources, index seeds, web allowlist, capability blurb,
optional tools, and eval cases. Adding a service never touches the core.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field


class DatasetBinding(BaseModel):
    """Binds a NYC Open Data (Socrata) dataset to a findable category.

    `field_map` maps the common shape (name/address/lat/lon/status/hours) to the
    dataset's actual column names so `geo.nearest(category=...)` is uniform.
    """

    id: str  # Socrata dataset id, e.g. "h2bn-gu9k"
    category: str  # category name used by geo.nearest, e.g. "cooling_center"
    field_map: dict[str, str] = Field(default_factory=dict)
    where: Optional[str] = None  # optional default SoQL $where filter


class ServiceModule(BaseModel):
    """A single city service, loaded from a manifest.yaml."""

    model_config = ConfigDict(extra="forbid")

    name: str
    category: str = "general"
    description: str = ""
    keywords: list[str] = Field(default_factory=list)
    datasets: list[DatasetBinding] = Field(default_factory=list)
    seeds: list[str] = Field(default_factory=list)
    allowlist: list[str] = Field(default_factory=list)
    prompt: str = ""  # capability blurb injected into the system prompt
    tools: Optional[str] = None  # optional module-specific tools file (e.g. "tools.py")
    eval: Optional[str] = None  # eval cases file (e.g. "eval.yaml")
    # Trust tiers for web_search ranking: {tier: [domain, ...]} where tier is one of
    # authoritative | editorial | community. Aggregated by registry.source_tiers().
    source_tiers: dict[str, list[str]] = Field(default_factory=dict)
    # Submodule hint (events topics): the Ticketmaster `keyword` the agent should pass to
    # whats_on_events for this topic. Advisory metadata; the prompt blurb drives the call.
    ticketmaster_keyword: Optional[str] = None

    # Populated by the loader, not from YAML.
    path: Optional[Path] = Field(default=None, exclude=True)
    parent: Optional[str] = Field(default=None, exclude=True)  # set on submodules to the parent module name

    @classmethod
    def from_manifest(cls, manifest_path: Path) -> "ServiceModule":
        data = yaml.safe_load(manifest_path.read_text()) or {}
        module = cls(**data)
        module.path = manifest_path.parent
        return module
