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
    utility_criterion: str = ""
    # Free-form selection labels: failure-db ids ("F046") and category slugs, so selective
    # runs can target exactly the cases that pin recorded failures (`--tag F046`).
    tags: list[str] = field(default_factory=list)
    # Ordered resident turns for a conversational case. The runner plays every turn through
    # one Conversation so history flows (the F052 contract); every check, judge, and field
    # above applies to the FINAL turn's result, and `query` is always that final turn.
    turns: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.turns:
            self.turns = [self.query]
        if not self.safety_critical:
            self.safety_critical = (
                self.harm_category != "none"
                or bool(self.invariants.get("must_not_fabricate"))
                or bool(self.safety_criterion)
            )


def default_global_cases() -> Path:
    """Repo-level cross-module cases (conversational and multi-module contracts that no
    single module owns). Optional: the loader tolerates absence."""
    return Path(__file__).resolve().parent / "global.yaml"


def load_cases(registry: Registry, global_path: Optional[Path] = None) -> list[EvalCase]:
    cases: list[EvalCase] = []
    sources: list[tuple[str, Path]] = []
    for module in registry.modules:
        if not module.eval or module.path is None:
            continue
        path = module.path / module.eval
        if path.exists():
            sources.append((module.name, path))
    global_file = default_global_cases() if global_path is None else Path(global_path)
    if global_file.exists():
        sources.append(("global", global_file))
    for module_name, path in sources:
        raw = yaml.safe_load(path.read_text()) or []
        for entry in raw:
            turns = list(entry.get("turns", []) or [])
            cases.append(
                EvalCase(
                    id=entry["id"],
                    module=module_name,
                    query=entry.get("query") or (turns[-1] if turns else ""),
                    turns=turns,
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
                    utility_criterion=entry.get("utility_criterion", ""),
                    tags=list(entry.get("tags", []) or []),
                )
            )
    by_id = {case.id: case for case in cases}
    for case in cases:
        base = by_id.get(case.base)
        if case.test_type == "INV" and base and not case.utility_criterion:
            case.utility_criterion = base.utility_criterion
    return cases


def select_cases(
    cases: list[EvalCase],
    *,
    module: Optional[str] = None,
    case_ids: Optional[list[str]] = None,
    tags: Optional[list[str]] = None,
    sample: Optional[int] = None,
    seed: int = 0,
) -> list[EvalCase]:
    """One selection path for every eval entry point: module, exact ids, tags, then a
    deterministic seeded sample. Unknown ids abort loudly rather than silently running less."""
    import random

    selected = cases
    if module:
        selected = [case for case in selected if case.module == module]
    if case_ids:
        missing = sorted(set(case_ids) - {case.id for case in selected})
        if missing:
            noun = "id" if len(missing) == 1 else "ids"
            raise SystemExit(f"Unknown eval case {noun}: {', '.join(missing)}")
        wanted = set(case_ids)
        selected = [case for case in selected if case.id in wanted]
    if tags:
        wanted_tags = set(tags)
        selected = [case for case in selected if wanted_tags & set(case.tags)]
    if sample is not None and sample < len(selected):
        selected = random.Random(seed).sample(selected, sample)
    return selected


# The 8-way adversarial taxonomy (canonical here so both the loader's validation and the judge's
# rubric label read one source; frozen from red-team-v1, docs/internal/eval/red-team-v1.md).
CATEGORY_NAMES = {
    "MC": "MyCity-replay + expansion", "PI": "Prompt injection / jailbreak",
    "OS": "Out-of-scope harmful", "FP": "False-premise / leading",
    "HS": "Over-reliance / high-stakes", "PII": "PII / privacy",
    "CIT": "Adversarial grounding / citation", "ES": "Non-English (Spanish) safety",
}

REDTEAM_REVIEW_REQUIRED = "Fresh qualitative review of the complete red-team trace is required."


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
                turns=list(entry.get("turns", []) or []),
                notes=entry.get("framing", "") or entry["adversarial_intent"],
                harm_category=entry.get("harm_category", "none"),
                safety_criterion=entry["safety_criterion"],
                adversarial_intent=entry["adversarial_intent"],
                redteam_category=entry["category"],
                language=entry.get("language", "en"),
                grounded_fact=entry.get("grounded_fact", ""),
                utility_criterion=(
                    entry.get("utility_criterion")
                    or REDTEAM_REVIEW_REQUIRED
                ),
                safety_critical=True,
            )
        )
    return cases


def render_case_listing(registry, global_path=None) -> str:
    """The corpus on one greppable line per case: id, source, flags, tags.

    The audit surface for "what do we actually test": cases live in each module's eval.yaml
    plus the global file, and this flattens them exactly as the loader runs them."""
    cases = load_cases(registry, global_path)
    rows = [f"{'id':<44} {'source':<18} {'flags':<28} tags"]
    for c in cases:
        flags = ",".join(filter(None, [
            "abstain" if c.abstain else "",
            c.capability,
            c.harm_category if c.harm_category != "none" else "",
            "multi-turn" if len(getattr(c, "turns", None) or []) > 1 else "",
        ]))
        rows.append(f"{c.id:<44} {c.module:<18} {flags:<28} {' '.join(c.tags)}")
    return "\n".join(rows)
