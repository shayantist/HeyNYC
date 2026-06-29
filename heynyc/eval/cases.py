"""Golden eval cases, aggregated from every module's eval.yaml."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import yaml

from ..core.registry import Registry


@dataclass
class EvalCase:
    id: str
    module: str
    query: str
    expect_tools: list[str] = field(default_factory=list)
    forbid_tools: list[str] = field(default_factory=list)
    expect_cite_kinds: list[str] = field(default_factory=list)
    expect_contains: list[str] = field(default_factory=list)
    abstain: bool = False
    notes: str = ""
    # --- CheckList matrix + harm taxonomy (additive, standards-based) ---
    # capability/test_type follow CheckList (Ribeiro et al., ACL 2020);
    # harm_category maps to OWASP LLM Top 10 / MLCommons AILuminate.
    capability: str = ""
    test_type: str = "MFT"  # MFT | INV | DIR
    harm_category: str = "none"  # none | injection | misinformation | specialized_advice | false_premise
    invariants: dict = field(default_factory=dict)
    base: str = ""  # for INV cases: the base case id this perturbs
    perturbation: str = ""  # paraphrase | typo | tool_reorder
    expect_same_outcome_as_base: bool = False
    safety_critical: bool = False

    def __post_init__(self) -> None:
        if not self.safety_critical:
            self.safety_critical = (
                self.harm_category != "none"
                or bool(self.invariants.get("must_not_fabricate"))
            )


def load_cases(registry: Optional[Registry] = None) -> list[EvalCase]:
    registry = registry or Registry.discover()
    cases: list[EvalCase] = []
    for module in registry.modules:
        if not module.eval or module.path is None:
            continue
        path = module.path / module.eval
        if not path.exists():
            continue
        raw = yaml.safe_load(path.read_text()) or []
        for entry in raw:
            cases.append(
                EvalCase(
                    id=entry["id"],
                    module=module.name,
                    query=entry["query"],
                    expect_tools=entry.get("expect_tools", []),
                    forbid_tools=entry.get("forbid_tools", []),
                    expect_cite_kinds=entry.get("expect_cite_kinds", []),
                    expect_contains=entry.get("expect_contains", []),
                    abstain=entry.get("abstain", False),
                    notes=entry.get("notes", ""),
                    capability=entry.get("capability", ""),
                    test_type=entry.get("test_type", "MFT"),
                    harm_category=entry.get("harm_category", "none"),
                    invariants=entry.get("invariants", {}) or {},
                    base=entry.get("base", ""),
                    perturbation=entry.get("perturbation", ""),
                    expect_same_outcome_as_base=entry.get("expect_same_outcome_as_base", False),
                    safety_critical=entry.get("safety_critical", False),
                )
            )
    return cases
