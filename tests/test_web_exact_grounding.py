from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from heynyc.core.pydantic_runtime import PydanticRuntimeAdapter
from heynyc.core.registry import Registry
from heynyc.core.tools.base import Tool, ToolContext


async def test_authoritative_web_exact_fact_retries_against_its_cited_excerpt() -> None:
    async def fetch(_args: dict, ctx: ToolContext) -> str:
        citation_id = ctx.citations.register(
            "https://www.nyc.gov/example",
            title="Official contact",
            kind="WEB",
            snippet="Call (212) 555-0100.",
            provenance={"source_tier": "authoritative", "evidence_grade": "evidence"},
        )
        return f"Call (212) 555-0100. {{cite:{citation_id}}}"

    source = Tool(
        name="fetch_contact",
        description="Fetch an official contact",
        handler=fetch,
    )
    calls = 0

    async def model(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        if not any(
            isinstance(part, ToolReturnPart)
            for message in messages
            for part in message.parts
        ):
            return ModelResponse([ToolCallPart("fetch_contact", {}, "fetch-1")])
        phone = "(212) 555-9999" if calls == 2 else "(212) 555-0100"
        return ModelResponse([
            ToolCallPart(
                "final_answer",
                {
                    "answer": f"Call {phone}. {{cite:S1}}",
                },
                f"answer-{calls}",
            )
        ])

    runtime = PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools={source.name: source},
        structured_grounding=True,
    )

    result = await runtime.run("What is the official phone?")

    assert result.text == "Call (212) 555-0100. {cite:S1}"
    assert calls == 3
    assert result.diagnostics["validation_rejections"][0]["stage"] == "structured_grounding"
