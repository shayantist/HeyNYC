"""Hard USD spend cap for one agent session: halt, don't silently overspend.

A small, standalone guard so the agent loop stays minimal (security-audit finding F2b,
OWASP LLM10 Unbounded Consumption). It sums the LiteLLM-priced cost of each model call,
reusing `core.telemetry.cost_usd` (the SAME pricing the `heynyc stats` dashboard uses, no
reinvented price table), and reports when further model calls must halt.

OFF by default: with no cap (None, or a non-positive value) it is a no-op and agent behavior
is unchanged. Fail-safe: while a cap IS active, a cost that cannot be computed is never
treated as $0 (that would silently disable the cap); it latches the guard into a halt state
so the next turn boundary stops rather than spending blind.
"""
from __future__ import annotations

from typing import Callable, Optional

from . import telemetry

# (model, input_tokens, output_tokens, cached_input_tokens) -> USD, or None if unpriceable.
CostFn = Callable[[str, int, int, int], Optional[float]]


class SpendGuard:
    """Accumulates model-call cost for one agent session and decides when to halt.

    `cap_usd`: the USD ceiling; None or <= 0 disables the guard (default OFF).
    `cost_fn`: injected for tests; defaults to `telemetry.cost_usd` (LiteLLM pricing).
    """

    def __init__(
        self, cap_usd: Optional[float] = None, *, cost_fn: CostFn = telemetry.priced_cost_usd,
    ):
        self.cap_usd = float(cap_usd) if cap_usd else 0.0
        self._cost_fn = cost_fn
        self.spent_usd = 0.0
        # Fail-safe latch: a cost we could not compute while a cap was active.
        self._cost_unavailable = False

    @property
    def enabled(self) -> bool:
        return self.cap_usd > 0

    def record(
        self, model: str, input_tokens: int, output_tokens: int, cached_input_tokens: int = 0,
    ) -> float:
        """Add one model call's cost to the running total; return that cost.

        No-op (returns 0.0) when disabled, so a run with no cap does exactly what it did
        before. When enabled, a cost that cannot be computed (the cost_fn raises) latches
        the fail-safe halt instead of being silently counted as $0. `cached_input_tokens` is
        threaded into pricing so the cap reflects the real cache-discounted bill, not a full-rate
        overstatement that would trip the cap earlier than actual spend."""
        if not self.enabled:
            return 0.0
        try:
            priced = self._cost_fn(
                model, int(input_tokens), int(output_tokens), int(cached_input_tokens)
            )
            if priced is None:
                self._cost_unavailable = True
                return 0.0
            cost = float(priced)
        except Exception:
            self._cost_unavailable = True
            return 0.0
        self.spent_usd += cost
        return cost

    def mark_unpriceable(self) -> None:
        """Fail closed when a model call occurred but its cost cannot be established."""
        if self.enabled:
            self._cost_unavailable = True

    def halt_reason(self) -> Optional[str]:
        """A human-readable reason further model calls must stop, or None to proceed.

        Always None when the guard is disabled, so behavior is unchanged without a cap."""
        if not self.enabled:
            return None
        if self._cost_unavailable:
            return (
                f"spend guard: could not verify this session's cost against the "
                f"${self.cap_usd:.2f} cap (HEYNYC_SPEND_CAP); halting rather than spending blind"
            )
        if self.spent_usd >= self.cap_usd:
            return (
                f"spend cap reached: ${self.spent_usd:.4f} spent this session meets or exceeds the "
                f"${self.cap_usd:.2f} ceiling (HEYNYC_SPEND_CAP); halting further model calls"
            )
        return None
