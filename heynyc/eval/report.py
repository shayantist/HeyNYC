"""Aggregate case checks into a gate report with pass/fail."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Optional

from .checks import CheckResult, LinkChecker, run_checks
from .invariants import (
    build_invariant_checks,
    check_metamorphic,
    check_metamorphic_programs,
)
from .runner import CaseResult
from .trace import Trace, build_trace

Judge = Callable[[CaseResult], Awaitable[CheckResult]]


@dataclass
class CaseReport:
    case_id: str
    module: str
    checks: list[CheckResult]
    trace: Optional[Trace] = None
    qualitative_review_required: bool = False
    qualitative_review: Optional[CheckResult] = None

    @property
    def passed(self) -> bool:
        # Declared safety and structural checks block. Advisory readability, legacy abstention,
        # and metamorphic checks remain visible without deciding the gate.
        return all(c.passed for c in self.checks if c.blocking)

    @property
    def mechanical_passed(self) -> bool:
        return all(
            c.passed
            for c in self.checks
            if c.blocking and c is not self.qualitative_review
        )

    @property
    def qualitative_reviewed(self) -> bool:
        return self.qualitative_review is not None

    @property
    def promotion_ready(self) -> bool:
        return self.mechanical_passed and (
            not self.qualitative_review_required
            or (
                self.qualitative_review is not None
                and self.qualitative_review.blocking
                and self.qualitative_review.passed
            )
        )


@dataclass
class GateReport:
    reports: list[CaseReport] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.reports)

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.reports if r.passed)

    @property
    def mechanical_passed_count(self) -> int:
        return sum(1 for r in self.reports if r.mechanical_passed)

    @property
    def qualitative_pending_count(self) -> int:
        return sum(
            1
            for r in self.reports
            if r.qualitative_review_required and not r.qualitative_reviewed
        )

    @property
    def promotion_ready_count(self) -> int:
        return sum(1 for r in self.reports if r.promotion_ready)

    @property
    def passed(self) -> bool:
        return self.total > 0 and self.passed_count == self.total

    @property
    def promotion_ready(self) -> bool:
        return self.total > 0 and self.promotion_ready_count == self.total

    def failures(self) -> list[CaseReport]:
        return [r for r in self.reports if not r.passed]

    def metric_summary(self) -> dict[str, str]:
        """Per-check-type pass rates, e.g. {'abstention': '3/3', ...}."""
        totals: dict[str, list[int]] = {}
        for report in self.reports:
            for check in report.checks:
                bucket = totals.setdefault(check.name, [0, 0])
                bucket[1] += 1
                if check.passed:
                    bucket[0] += 1
        return {name: f"{ok}/{n}" for name, (ok, n) in sorted(totals.items())}

    def render(self) -> str:
        review_failed = any(
            r.qualitative_review_required
            and r.qualitative_reviewed
            and not r.promotion_ready
            for r in self.reports
        )
        if self.mechanical_passed_count != self.total or review_failed:
            status = "FAIL"
        elif self.qualitative_pending_count:
            status = "MECHANICAL PASS, QUALITATIVE REVIEW REQUIRED"
        elif self.promotion_ready:
            status = "PASS"
        else:
            status = "FAIL"
        lines = [
            f"HeyNYC eval gate: {self.mechanical_passed_count}/{self.total} "
            f"cases mechanically passed ({status})",
            "",
        ]
        for name, rate in self.metric_summary().items():
            lines.append(f"  {name:16} {rate}")
        if self.failures():
            lines.append("\nFailures:")
            for report in self.failures():
                for check in report.checks:
                    if not check.passed:
                        lines.append(f"  ✗ [{report.module}] {report.case_id}, {check.name}: {check.detail}")
        return "\n".join(lines)


async def evaluate(
    case_results: list[CaseResult],
    link_checker: Optional[LinkChecker] = None,
    judge: Optional[Judge] = None,
) -> GateReport:
    traces: dict[str, Trace] = {}
    reports: list[CaseReport] = []
    # First pass: per-case legacy checks + outcome invariants.
    for cr in case_results:
        trace = build_trace(cr)
        traces[cr.case.id] = trace
        checks = await run_checks(cr, link_checker=link_checker)
        checks.extend(build_invariant_checks(trace, cr.case))
        qualitative_review = None
        if judge is not None:
            judgment = await judge(cr)
            checks.append(judgment)
            if cr.case.utility_criterion:
                qualitative_review = judgment
        reports.append(
            CaseReport(
                case_id=cr.case.id,
                module=cr.case.module,
                checks=checks,
                trace=trace,
                qualitative_review_required=bool(cr.case.utility_criterion),
                qualitative_review=qualitative_review,
            )
        )
    # Second pass: metamorphic INV pairing (needs the base case's trace).
    for cr, report in zip(case_results, reports):
        if cr.case.test_type == "INV" and cr.case.base in traces:
            mm = check_metamorphic(traces[cr.case.id], traces[cr.case.base], cr.case)
            if mm is not None:
                report.checks.append(mm)
            # Fairness substance-invariance: the cited program SET must match the base's.
            mp = check_metamorphic_programs(traces[cr.case.id], traces[cr.case.base], cr.case)
            if mp is not None:
                report.checks.append(mp)
    return GateReport(reports=reports)


def write_run(directory, report: "GateReport", metadata: Optional[dict] = None) -> None:
    """Persist a run: report.json (the CI gate), report.txt, and OpenInference traces."""
    directory = Path(directory)
    (directory / "traces").mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata or {},
        "passed": report.passed,
        "passed_count": report.passed_count,
        "mechanical_passed": (
            report.total > 0 and report.mechanical_passed_count == report.total
        ),
        "mechanical_passed_count": report.mechanical_passed_count,
        "qualitative_review_required": any(
            r.qualitative_review_required for r in report.reports
        ),
        "qualitative_pending_count": report.qualitative_pending_count,
        "promotion_ready": report.promotion_ready,
        "promotion_ready_count": report.promotion_ready_count,
        "total": report.total,
        "metrics": report.metric_summary(),
        "cases": [
            {"case_id": r.case_id, "module": r.module, "passed": r.passed,
             "mechanical_passed": r.mechanical_passed,
             "qualitative_review_required": r.qualitative_review_required,
             "qualitative_reviewed": r.qualitative_reviewed,
             "promotion_ready": r.promotion_ready,
             "checks": [{"name": c.name, "passed": c.passed, "blocking": c.blocking,
                         "detail": c.detail} for c in r.checks]}
            for r in report.reports
        ],
    }
    (directory / "report.json").write_text(json.dumps(payload, indent=2))
    (directory / "report.txt").write_text(report.render())
    for r in report.reports:
        if r.trace is not None:
            r.trace.write(directory / "traces")
