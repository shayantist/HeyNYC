"""Module discovery + assembly.

Discovers every `modules/<name>/manifest.yaml` and assembles the live system
from them: dataset bindings for geo.nearest, the web-search allowlist, the
index seed set, and the capability blurbs for the system prompt.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from . import config
from .manifest import DatasetBinding, ServiceModule

# Trust-tier ordering for web_search ranking; higher = more trusted. (§10.4)
TIER_RANK = {"authoritative": 3, "editorial": 2, "community": 1}


class Registry:
    def __init__(self, modules: list[ServiceModule]):
        self.modules = modules

    @classmethod
    def discover(cls, modules_dir: Optional[Path] = None) -> "Registry":
        modules_dir = modules_dir or config.MODULES_DIR
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
        return cls(modules)

    def dataset_bindings(self) -> dict[str, DatasetBinding]:
        """category -> DatasetBinding. Later modules win on category collision."""
        bindings: dict[str, DatasetBinding] = {}
        for module in self.modules:
            for binding in module.datasets:
                bindings[binding.category] = binding
        return bindings

    def allowlist(self) -> list[str]:
        """Base allowlist plus every module's additions, deduped and sorted."""
        domains = set(config.BASE_ALLOWLIST)
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
