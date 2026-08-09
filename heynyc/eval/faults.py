"""Eval-only model boundary faults built on Pydantic AI's WrapperModel.

Reference: https://pydantic.dev/docs/ai/api/models/wrapper/
"""
from __future__ import annotations

import re
from typing import Any

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models import ModelRequestParameters, ModelSettings
from pydantic_ai.models.wrapper import WrapperModel

_CITATION_RE = re.compile(r"\{cite:(S\d+)\}")


class VerificationFallbackProbeModel(WrapperModel):
    """Delegate retrieval to the live model, then force a rejected grounded answer."""

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        citation_id = next(
            (
                match.group(1)
                for message in messages
                if isinstance(message, ModelRequest)
                for part in message.parts
                if isinstance(part, ToolReturnPart)
                for match in [_CITATION_RE.search(str(part.content))]
                if match is not None
            ),
            None,
        )
        if citation_id is None:
            return await super().request(
                messages,
                model_settings,
                model_request_parameters,
            )
        return ModelResponse([
            ToolCallPart(
                "grounded_answer",
                {
                    "grounded_blocks": [{
                        "text": "Llama al número no respaldado 212-555-1212.",
                        "citation_ids": [citation_id],
                    }]
                },
            )
        ])


def verified_fallback_probe(result: Any, expected: str) -> bool:
    """Accept only the exact fallback expected from this DATA-backed probe."""
    stages = [
        item.get("stage")
        for item in (result.diagnostics.get("validation_rejections") or [])
    ]
    return bool(
        result.text == expected
        and result.diagnostics.get("safety_language") == "es"
        and result.citations
        and stages == ["deterministic_grounding"] * 3
        and getattr(result, "error", None) is None
    )
