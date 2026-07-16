"""Golden eval cases, aggregated from every module's eval.yaml."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
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
    # Fairness substance-invariance: the SET of cited programs must match the base case's
    # (see check_metamorphic_programs). Peripheral personalization is allowed; the program set is not.
    expect_same_programs_as_base: bool = False
    safety_critical: bool = False
    # --- adversarial red-team fields (additive; empty for ordinary golden cases) ---
    # A case carrying a safety_criterion is graded by the strict adversarial rubric against that
    # criterion instead of the generic groundedness rubric (see judges._run_judge). redteam_category
    # is the 8-way red-team taxonomy (MC/PI/OS/FP/HS/PII/CIT/ES) used only for the per-category report.
    safety_criterion: str = ""
    adversarial_intent: str = ""
    redteam_category: str = ""
    language: str = "en"
    grounded_fact: str = ""

    def __post_init__(self) -> None:
        if not self.safety_critical:
            self.safety_critical = (
                self.harm_category != "none"
                or bool(self.invariants.get("must_not_fabricate"))
                or bool(self.safety_criterion)
            )


def load_cases(registry: Registry) -> list[EvalCase]:
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
                    expect_same_programs_as_base=entry.get("expect_same_programs_as_base", False),
                    safety_critical=entry.get("safety_critical", False),
                    language=entry.get("language", "en"),
                    grounded_fact=entry.get("grounded_fact", ""),
                )
            )
    return cases


# The 8-way adversarial taxonomy (canonical here so both the loader's validation and the judge's
# rubric label read one source; frozen from red-team-v1, docs/eval/red-team-v1.md).
CATEGORY_NAMES = {
    "MC": "MyCity-replay + expansion", "PI": "Prompt injection / jailbreak",
    "OS": "Out-of-scope harmful", "FP": "False-premise / leading",
    "HS": "Over-reliance / high-stakes", "PII": "PII / privacy",
    "CIT": "Adversarial grounding / citation", "ES": "Non-English (Spanish) safety",
}


def default_redteam_suite() -> Path:
    """The shipped, frozen adversarial suite that travels with the package."""
    return Path(__file__).resolve().parent / "redteam_suite.yaml"


def load_redteam_cases(path: Optional[Path] = None) -> list[EvalCase]:
    """Load the frozen adversarial red-team suite as EvalCases, so the 205 attack prompts run through
    the SAME runner, gate, and judge as the golden cases (they also pick up the deterministic grounding
    floor + invariants that a standalone red-team would skip). Each carries a safety_criterion (the
    grading anchor, so the judge uses the strict adversarial rubric) and redteam_category for the
    per-category report. Prompts only, never a canned answer. Raises on a malformed suite."""
    path = Path(path) if path is not None else default_redteam_suite()
    raw = yaml.safe_load(path.read_text()) or []
    cases: list[EvalCase] = []
    seen: set[str] = set()
    for i, entry in enumerate(raw):
        for req in ("id", "category", "query", "adversarial_intent", "safety_criterion"):
            if not str(entry.get(req, "")).strip():
                raise ValueError(f"redteam suite entry {i} missing '{req}': {entry.get('id', '?')}")
        cid = entry["id"]
        if cid in seen:
            raise ValueError(f"duplicate red-team case id: {cid}")
        seen.add(cid)
        if entry["category"] not in CATEGORY_NAMES:
            raise ValueError(
                f"{cid}: unknown category '{entry['category']}' (expected one of {sorted(CATEGORY_NAMES)})"
            )
        cases.append(
            EvalCase(
                id=cid,
                module="redteam",
                query=entry["query"],
                notes=entry.get("framing", "") or entry["adversarial_intent"],
                harm_category=entry.get("harm_category", "none"),
                safety_criterion=entry["safety_criterion"],
                adversarial_intent=entry["adversarial_intent"],
                redteam_category=entry["category"],
                language=entry.get("language", "en"),
                grounded_fact=entry.get("grounded_fact", ""),
                safety_critical=True,
            )
        )
    return cases
