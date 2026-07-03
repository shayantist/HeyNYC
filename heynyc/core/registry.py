"""Module discovery + assembly.

Discovers every `modules/<name>/manifest.yaml` and assembles the live system
from them: dataset bindings for geo.nearest, the web-search allowlist, the
index seed set, and the capability blurbs for the system prompt.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from .manifest import DatasetBinding, ServiceModule

# Trust-tier ordering for web_search ranking; higher = more trusted. (§10.4)
TIER_RANK = {"authoritative": 3, "editorial": 2, "community": 1}

# Friendly, user-facing service names for the capability table. Any module that is
# not listed falls back to a title-cased version of its folder name, so a newly
# added module still gets a sensible label without editing this map.
_SERVICE_NAMES = {
    "benefits": "Benefits & programs",
    "food_pantries": "Food pantries",
    "cooling_centers": "Cooling centers",
    "snap_centers": "SNAP centers",
    "housing": "Housing & eviction help",
    "advisories": "Emergency advisories",
    "events": "Events",
}

# Short, honest source labels for modules grounded in a live tool or several sources
# rather than a single declared dataset binding. Dataset-bound modules derive their
# label straight from the binding (see _grounded_in), so they are absent here.
_SOURCE_LABELS = {
    "benefits": "NYC Benefits & Programs dataset + Screening API",
    "food_pantries": "NYC FoodHelp finder",
    "advisories": "Notify NYC / NYC Emergency Management feed",
    "events": "Ticketmaster + NYC Parks",
}


@dataclass
class CapabilityRow:
    """One user-facing row of the capability table (one per top-level module)."""

    service: str          # friendly service name, e.g. "Food pantries"
    asks: list[str]       # a couple of example questions a user can ask
    grounded_in: str      # short, honest source label
    official_link: str    # a primary official URL (may be "")


def _md_cell(text: str) -> str:
    """Escape a value for a markdown table cell (a raw pipe would break the row)."""
    return text.replace("|", "\\|")


def _domain(url: str) -> str:
    """The bare host of a URL, without a leading 'www.' (a compact link label)."""
    host = urlparse(url).netloc or url
    return host[4:] if host.startswith("www.") else host


class Registry:
    def __init__(self, modules: list[ServiceModule], base_allowlist: Optional[list[str]] = None):
        self.modules = modules
        self.base_allowlist = list(base_allowlist or [])

    @classmethod
    def discover(cls, modules_dir: Path, base_allowlist: Optional[list[str]] = None) -> "Registry":
        """Scan `modules_dir` for manifests. `modules_dir` + `base_allowlist` are injected by the
        application — the engine reads no domain config module."""
        modules: list[ServiceModule] = []
        if modules_dir.exists():
            for child in sorted(modules_dir.iterdir()):
                manifest = child / "manifest.yaml"
                if not manifest.exists():
                    continue
                module = ServiceModule.from_manifest(manifest)
                modules.append(module)
                # A self-contained submodule lives at <module>/topics/<topic>/manifest.yaml
                # and reuses the parent's tool; discovering it flat lets every aggregator
                # (seeds/allowlist/source_tiers/load_cases) pick it up unchanged. (§10.1)
                topics_dir = child / "topics"
                if topics_dir.exists():
                    for topic in sorted(topics_dir.iterdir()):
                        sub_manifest = topic / "manifest.yaml"
                        if sub_manifest.exists():
                            submodule = ServiceModule.from_manifest(sub_manifest)
                            submodule.parent = module.name
                            modules.append(submodule)
        return cls(modules, base_allowlist=base_allowlist)

    def dataset_bindings(self) -> dict[str, DatasetBinding]:
        """category -> DatasetBinding. Later modules win on category collision."""
        bindings: dict[str, DatasetBinding] = {}
        for module in self.modules:
            for binding in module.datasets:
                bindings[binding.category] = binding
        return bindings

    def allowlist(self) -> list[str]:
        """Base allowlist plus every module's additions, deduped and sorted."""
        domains = set(self.base_allowlist)
        for module in self.modules:
            domains.update(module.allowlist)
        return sorted(domains)

    def seeds(self) -> list[str]:
        """All module index seeds, deduped, order preserved."""
        urls: list[str] = []
        for module in self.modules:
            urls.extend(module.seeds)
        return list(dict.fromkeys(urls))

    def source_tiers(self) -> dict[str, tuple[str, str]]:
        """domain -> (tier, module_name), aggregated across modules. Highest-trust tier
        wins on collision so a domain can never be *downgraded* by another module."""
        tiers: dict[str, tuple[str, str]] = {}
        for module in self.modules:
            for tier, domains in module.source_tiers.items():
                for domain in domains:
                    key = domain.lower()
                    current = tiers.get(key)
                    if current is None or TIER_RANK.get(tier, 0) > TIER_RANK.get(current[0], 0):
                        tiers[key] = (tier, module.name)
        return tiers

    def capability_blurbs(self) -> str:
        """Capability blurbs for the system prompt, one section per module."""
        sections = []
        for module in self.modules:
            if module.prompt.strip():
                sections.append(f"## {module.name} ({module.category})\n{module.prompt.strip()}")
        return "\n\n".join(sections)

    def welcome_examples(self, n: int = 4) -> list[str]:
        """A small, category-spread sample of example queries (first-contact / ice-breakers).
        One per module first (so the sample spans services), then fill from the extras."""
        out: list[str] = []
        for module in self.modules:
            if module.examples:
                out.append(module.examples[0])
        for module in self.modules:
            for example in module.examples[1:]:
                if len(out) >= n:
                    break
                out.append(example)
        return out[:n]

    def capability_menu(self) -> list[tuple[str, str, list[str]]]:
        """The grounded 'what can you do' for the welcome/help reply — one row per top-level
        module: (category, blurb, examples). Submodules fold into their parent. Pure function of
        installed modules → it can't drift from what's actually loaded."""
        menu: list[tuple[str, str, list[str]]] = []
        for module in self.modules:
            if module.parent:                       # submodules don't get their own menu row
                continue
            if module.examples or module.description:
                menu.append((module.category, module.description or module.name, list(module.examples)))
        return menu

    def _service_name(self, module: ServiceModule) -> str:
        return _SERVICE_NAMES.get(module.name) or module.name.replace("_", " ").capitalize()

    def _grounded_in(self, module: ServiceModule) -> str:
        """A short, honest 'grounded in' label. A dataset binding names itself (ArcGIS
        by its title, Socrata by its open-data id); everything else uses a curated
        tool/source label, falling back to a generic one for an unlisted module."""
        if module.datasets:
            labels: list[str] = []
            for binding in module.datasets:
                if binding.source == "arcgis":
                    labels.append(binding.title or "NYC ArcGIS layer")
                else:
                    labels.append(f"NYC Open Data ({binding.id})")
            return " + ".join(dict.fromkeys(labels))
        return _SOURCE_LABELS.get(module.name, "Official NYC sources")

    def _official_link(self, module: ServiceModule) -> str:
        """A primary official URL: the first index seed if any, else the first
        allowlisted domain as an https URL, else blank."""
        if module.seeds:
            return module.seeds[0]
        if module.allowlist:
            return f"https://{module.allowlist[0]}"
        return ""

    def capability_table(self, max_asks: int = 2) -> list[CapabilityRow]:
        """The user-facing capability table: one row per top-level module (submodules
        fold into their parent). A pure function of the installed modules, so it can
        never drift from what's actually loaded (the README table is generated from it)."""
        rows: list[CapabilityRow] = []
        for module in self.modules:
            if module.parent:                       # submodules fold into their parent
                continue
            rows.append(CapabilityRow(
                service=self._service_name(module),
                asks=list(module.examples[:max_asks]),
                grounded_in=self._grounded_in(module),
                official_link=self._official_link(module),
            ))
        return rows

    def capability_markdown(self) -> str:
        """The capability table as a GitHub-flavored markdown table (the block embedded
        in the README between the CAPABILITIES markers). Deterministic, so a drift test
        can assert the committed README still matches this generated output."""
        lines = [
            "| Service | What you can ask | Grounded in | Official link |",
            "| --- | --- | --- | --- |",
        ]
        for row in self.capability_table():
            asks = "<br>".join(f'"{_md_cell(a)}"' for a in row.asks)
            link = f"[{_domain(row.official_link)}]({row.official_link})" if row.official_link else ""
            lines.append(
                f"| **{_md_cell(row.service)}** | {asks} | {_md_cell(row.grounded_in)} | {link} |"
            )
        return "\n".join(lines)

    def welcome_text(self) -> str:
        """A warm, grounded 'here's what I can do' for first contact / the help intent — generated
        from the modules' examples, never a bare 'How can I help?'. Single source = the manifests,
        so it can never drift from what's actually installed."""
        lines = ["Hi! I'm HeyNYC — I help you find and use NYC services, grounded in real city data, "
                 "and I cite my sources.", "", "Here are some things you can ask me:"]
        for example in self.welcome_examples(6):
            lines.append(f"  • {example}")
        lines += [
            "",
            "Just tell me what you need — in any language.",
            "",
            "Heads up: I'm an AI assistant, not a City employee or caseworker, so please "
            "double-check anything important against the official source.",
        ]
        return "\n".join(lines)

    def load_module_tools(self) -> list:
        """Import each module's optional tools.py and collect its get_tools().

        A module ships custom tools by setting `tools: tools.py` in its manifest and
        exposing `def get_tools() -> list[Tool]`. Failures are logged, not fatal —
        one broken module shouldn't break the whole agent.
        """
        import importlib.util
        import logging
        import sys

        logger = logging.getLogger("heynyc.registry")
        tools: list = []
        for module in self.modules:
            if not module.tools or module.path is None:
                continue
            tools_path = module.path / module.tools
            if not tools_path.exists():
                logger.warning("module %s declares tools '%s' but file is missing", module.name, module.tools)
                continue
            try:
                spec = importlib.util.spec_from_file_location(f"heynyc_module_{module.name}", tools_path)
                mod = importlib.util.module_from_spec(spec)
                # Register before exec so module-level dataclasses can resolve their own
                # __module__ in sys.modules (required under `from __future__ import annotations`).
                sys.modules[spec.name] = mod
                spec.loader.exec_module(mod)
                if hasattr(mod, "get_tools"):
                    tools.extend(mod.get_tools())
            except Exception:
                logger.exception("failed loading tools for module %s", module.name)
        return tools
