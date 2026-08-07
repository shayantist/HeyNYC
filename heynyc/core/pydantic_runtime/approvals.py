from __future__ import annotations

import json
from typing import Any

from heynyc.core.agent import AgentResult

from .runtime import PydanticRunFailure, PydanticRuntimeAdapter


class PydanticApprovalFlow:
    """Persist and resume native Pydantic approvals through the shared encrypted store."""

    def __init__(
        self,
        runtime: PydanticRuntimeAdapter,
        store: Any,
        user_key: str,
        *,
        ttl_s: float,
    ) -> None:
        self.runtime = runtime
        self.store = store
        self.user_key = user_key
        self.ttl_s = ttl_s
        state = store.get_pending_approval(user_key)
        self.conversation = (
            runtime.conversation_from_state(state)
            if state is not None
            else runtime.conversation()
        )

    async def send(self, user_message: str, **kwargs: Any) -> AgentResult:
        if self.store.has_pending_approval(self.user_key):
            raise ValueError("Cannot start a new turn while approval is pending")
        result = await self.conversation.send(user_message, **kwargs)
        self._persist_if_pending(result)
        return result

    async def resume(
        self,
        decision: bool | dict[str, bool],
        *,
        persist_pending: bool = True,
        **kwargs: Any,
    ) -> AgentResult:
        expected = set(self.conversation.pending_approvals)
        if not isinstance(decision, bool) and set(decision) != expected:
            raise ValueError(
                f"Approval IDs must match pending calls: {sorted(expected)}"
            )
        if not isinstance(decision, bool) and not all(
            isinstance(value, bool) for value in decision.values()
        ):
            raise ValueError("Approval decisions must be booleans")
        decisions = (
            {
                call_id: decision
                for call_id in self.conversation.pending_approvals
            }
            if isinstance(decision, bool)
            else decision
        )
        retry_safe = all(
            not approved
            or request["tool_name"].startswith("confirm_")
            and request["tool_name"].endswith("_facts")
            or (
                (tool := self.runtime.tools.get(request["tool_name"])) is not None
                and tool.idempotent
            )
            for call_id, request in self.conversation.pending_approvals.items()
            for approved in (decisions[call_id],)
        )
        state = (
            self.store.get_pending_approval(self.user_key)
            if retry_safe
            else self.store.pop_pending_approval(self.user_key)
        )
        if state is None:
            raise ValueError("Pending approval expired or already consumed")
        self.conversation = self.runtime.conversation_from_state(state)
        incomplete = False
        try:
            result = await self.conversation.resume_approvals(decisions, **kwargs)
        except PydanticRunFailure as exc:
            result = exc.partial_result
            incomplete = True
        incomplete = incomplete or result.status in {"error", "max_turns"}
        if retry_safe:
            if incomplete:
                self.conversation = self.runtime.conversation_from_state(state)
            else:
                self.store.pop_pending_approval(self.user_key)
        self._persist_if_pending(result, persist=persist_pending)
        return result

    def _persist_if_pending(
        self,
        result: AgentResult,
        *,
        persist: bool = True,
    ) -> None:
        if result.status == "approval_required":
            if self.conversation.pending_calls:
                raise ValueError(
                    "External deferred calls are not supported by this approval flow"
                )
            if persist:
                self.store.set_pending_approval(
                    self.user_key,
                    self.conversation.dump_state(),
                    ttl_s=self.ttl_s,
                )
            result.text = self.review_text()

    def review_text(self) -> str:
        return approval_review_text(self.conversation.pending_approvals)


def approval_review_text(pending_approvals: dict[str, dict]) -> str:
    requests = tuple(pending_approvals.values())
    copies = tuple(_approval_copy(request["tool_name"]) for request in requests)
    mixed = len(set(copies)) > 1
    heading, question = (
        (
            "Review each item below:",
            "Reply YES to confirm all facts and approve all actions, "
            "or NO to correct or deny them.",
        )
        if mixed
        else copies[0]
    )
    lines = [heading]
    for request, (item_heading, _) in zip(requests, copies, strict=True):
        arguments = json.dumps(
            request["args"],
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        if mixed:
            lines.extend(("", item_heading))
        lines.extend(("", request["tool_name"], arguments))
    lines.extend(("", question))
    return "\n".join(lines)


def _approval_copy(tool_name: str) -> tuple[str, str]:
    if tool_name.startswith("confirm_") and tool_name.endswith("_facts"):
        return (
            "Review the structured facts I understood:",
            "Reply YES if these facts are accurate and run the requested read-only "
            "check, or NO to correct them.",
        )
    return (
        "Review the proposed action and exact values:",
        "Reply YES to approve, or NO to deny.",
    )
