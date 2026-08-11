from datetime import datetime, timedelta

import pytest
from babel.dates import get_day_names
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from heynyc.core.citations import data_provenance
from heynyc.core.pydantic_runtime import PydanticRuntimeAdapter
from heynyc.core.pydantic_runtime.projection import (
    GroundedAnswer,
    GroundedBlock,
    _grounded_block_text,
    _legacy_citation_ids,
    _render_grounded_answer,
)
from heynyc.core.registry import Registry
from heynyc.core.tools.base import Tool, ToolContext


def test_matching_citation_marker_variants_are_repaired_without_a_retry():
    block = GroundedBlock(
        text="Benefits may change. { CITE : s1 }",
        citation_ids=["S1"],
    )

    assert _grounded_block_text(block) == "Benefits may change."
    assert _render_grounded_answer(GroundedAnswer(grounded_blocks=[block])) == (
        "Benefits may change. {cite:S1}"
    )


def test_matching_unclosed_citation_marker_is_repaired_without_a_retry():
    block = GroundedBlock(
        text="Benefits may change. {cite:S1",
        citation_ids=["S1"],
    )

    assert _grounded_block_text(block) == "Benefits may change."
    assert _render_grounded_answer(GroundedAnswer(grounded_blocks=[block])) == (
        "Benefits may change. {cite:S1}"
    )


def test_unknown_citation_marker_is_not_repaired():
    block = GroundedBlock(
        text="Benefits may change. {cite:S2}",
        citation_ids=["S1"],
    )

    assert _grounded_block_text(block) == "Benefits may change. {cite:S2}"
    assert _legacy_citation_ids(_grounded_block_text(block)) == ["S2"]


@pytest.mark.parametrize("marker", ["{ CITE : s1 }", "{cite:S1"])
async def test_matching_marker_variant_does_not_consume_an_output_retry(marker: str):
    async def source(_args: dict, ctx: ToolContext) -> str:
        citation_id = ctx.citations.register(
            "https://www.nyc.gov/example",
            title="Official guidance",
            kind="WEB",
            snippet="Benefits may change.",
        )
        return f"Benefits may change. {{cite:{citation_id}}}"

    model_calls = 0

    async def model(
        _messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            return ModelResponse([ToolCallPart("guidance", {}, "guidance-1")])
        return ModelResponse([
            ToolCallPart(
                info.output_tools[0].name,
                {
                    "grounded_blocks": [{
                        "text": f"Benefits may change. {marker}",
                        "citation_ids": ["S1"],
                    }]
                },
                "final-1",
            )
        ])

    result = await PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools={
            "guidance": Tool(
                name="guidance",
                description="Get current official guidance",
                parameters={"type": "object", "properties": {}},
                handler=source,
            )
        },
        structured_grounding=True,
        guard_grounding=True,
    ).run("Could my benefits change?")

    assert model_calls == 2
    assert result.text == "Benefits may change. {cite:S1}"
    assert result.diagnostics["validation_rejections"] == []


@pytest.mark.parametrize("bad_marker", ["{cite:S1】", "{cite_ids:S1}"])
async def test_f192_malformed_citation_marker_consumes_an_output_retry(
    bad_marker: str,
):
    async def source(_args: dict, ctx: ToolContext) -> str:
        citation_id = ctx.citations.register(
            "https://www.nyc.gov/example",
            title="Official guidance",
            kind="WEB",
            snippet="Benefits may change.",
        )
        return f"Benefits may change. {{cite:{citation_id}}}"

    model_calls = 0

    async def model(
        _messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            return ModelResponse([ToolCallPart("guidance", {}, "guidance-1")])
        marker = f" {bad_marker}" if model_calls == 2 else ""
        return ModelResponse([
            ToolCallPart(
                info.output_tools[0].name,
                {
                    "grounded_blocks": [{
                        "text": f"Benefits may change.{marker}",
                        "citation_ids": ["S1"],
                    }]
                },
                f"final-{model_calls}",
            )
        ])

    result = await PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools={
            "guidance": Tool(
                name="guidance",
                description="Get current official guidance",
                parameters={"type": "object", "properties": {}},
                handler=source,
            )
        },
        structured_grounding=True,
        guard_grounding=True,
    ).run("Could my benefits change?")

    assert model_calls == 3
    assert result.text == "Benefits may change. {cite:S1}"
    assert [
        rejection["stage"]
        for rejection in result.diagnostics["validation_rejections"]
    ] == ["citation_marker"]


async def test_f215_mechanical_validator_does_not_parse_a_date_clause():
    async def source(_args: dict, ctx: ToolContext) -> str:
        citation_id = ctx.citations.register(
            "https://www.schools.nyc.gov/enrollment",
            title="NYC Public Schools enrollment",
            kind="WEB",
            snippet=(
                "Call NYC Public Schools at 718-935-2009 Monday through Friday, "
                "8 AM to 6 PM. You may also call 311."
            ),
        )
        return f"Official enrollment contacts. {{cite:{citation_id}}}"

    tomorrow = datetime.now().date() + timedelta(days=1)
    weekday = get_day_names("wide", locale="es")[tomorrow.weekday()]
    model_calls = 0

    async def model(
        _messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            return ModelResponse([ToolCallPart("guidance", {}, "guidance-1")])
        grounded_blocks = (
            [
                {
                    "text": (
                        f"Mañana, {weekday}, llama a NYC Public Schools al "
                        "718-935-2009, de 8:00 a. m. a 6:00 p. m."
                    ),
                    "citation_ids": ["S1"],
                },
                {
                    "text": "También puedes llamar al 311.",
                    "citation_ids": ["S1"],
                },
            ]
            if model_calls == 2
            else [
                {
                    "text": (
                        "Llama a NYC Public Schools al 718-935-2009, de 8:00 a. m. "
                        "a 6:00 p. m."
                    ),
                    "citation_ids": ["S1"],
                },
                {
                    "text": "También puedes llamar al 311.",
                    "citation_ids": ["S1"],
                },
            ]
        )
        return ModelResponse([
            ToolCallPart(
                info.output_tools[0].name,
                {"grounded_blocks": grounded_blocks},
                "final-1",
            )
        ])

    result = await PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools={
            "guidance": Tool(
                name="guidance",
                description="Get current official guidance",
                parameters={"type": "object", "properties": {}},
                handler=source,
            )
        },
        structured_grounding=True,
        guard_grounding=True,
    ).run("¿A quién llamo mañana?")

    assert model_calls == 2
    assert "718-935-2009" in result.text
    assert "311" in result.text
    assert weekday in result.text.casefold()
    assert result.diagnostics["validation_rejections"] == []


async def test_f219_mechanical_validator_does_not_parse_a_derived_amount():
    async def source(_args: dict, ctx: ToolContext) -> str:
        citation_id = ctx.citations.register(
            "https://dol.ny.gov/minimum-wage-tipped-workers",
            title="Minimum Wage for Tipped Workers",
            kind="WEB",
            snippet=(
                "Food Service Workers in NYC: $11.35 cash wage and $5.65 tip credit. "
                "The minimum wage is $17.00 per hour. Service Employees receive a "
                "$14.15 cash wage. Employers may not take a tip credit on days when "
                "workers spend more than two hours or twenty percent of a shift doing "
                "non-tipped work."
            ),
        )
        return f"Official tipped-worker rates. {{cite:{citation_id}}}"

    model_calls = 0

    async def model(
        _messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            return ModelResponse([ToolCallPart("guidance", {}, "guidance-1")])
        grounded_blocks = (
            [{
                "text": (
                    "The employer must pay at least $11.35 in cash wages and may "
                    "count no more than $5.65 in tips toward the $17.00 minimum, "
                    "so your $10 cash rate is $1.35 below that minimum. "
                    "If you are a service employee, the cash wage is $14.15."
                ),
                "citation_ids": ["S1"],
            }, {
                "text": (
                    "The tip credit cannot be used when you spend more than two "
                    "hours or 20% of a shift doing non-tipped work, and it has "
                    "other limits."
                ),
                "citation_ids": ["S1"],
            }]
            if model_calls == 2
            else [{
                "text": (
                    "The employer must pay at least $11.35 in cash wages and may "
                    "count no more than $5.65 in tips toward the $17.00 minimum. "
                    "If you are a service employee, the cash wage is $14.15."
                ),
                "citation_ids": ["S1"],
            }]
        )
        return ModelResponse([
            ToolCallPart(
                info.output_tools[0].name,
                {"grounded_blocks": grounded_blocks},
                "final-1",
            )
        ])

    result = await PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools={
            "guidance": Tool(
                name="guidance",
                description="Get current official guidance",
                parameters={"type": "object", "properties": {}},
                handler=source,
            )
        },
        structured_grounding=True,
        guard_grounding=True,
    ).run("I make $10 an hour plus tips")

    assert model_calls == 2
    assert "$11.35" in result.text
    assert "$5.65" in result.text
    assert "$17.00" in result.text
    assert "$14.15" in result.text
    assert "$1.35" in result.text
    assert "20%" in result.text
    assert result.diagnostics["validation_rejections"] == []


async def test_f219_recovery_leaves_a_fully_supported_answer_unchanged():
    expected = "The NYC cash wage is $11.35 and the total minimum is $17.00."

    async def source(_args: dict, ctx: ToolContext) -> str:
        citation_id = ctx.citations.register(
            "https://dol.ny.gov/minimum-wage-tipped-workers",
            title="Minimum Wage for Tipped Workers",
            kind="WEB",
            snippet="The NYC cash wage is $11.35 and the total minimum is $17.00.",
        )
        return f"Official tipped-worker rates. {{cite:{citation_id}}}"

    async def model(
        messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        if not any(getattr(part, "tool_name", "") == "guidance" for message in messages for part in message.parts):
            return ModelResponse([ToolCallPart("guidance", {}, "guidance-1")])
        return ModelResponse([
            ToolCallPart(
                info.output_tools[0].name,
                {
                    "grounded_blocks": [{
                        "text": expected,
                        "citation_ids": ["S1"],
                    }]
                },
                "final-1",
            )
        ])

    result = await PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools={
            "guidance": Tool(
                name="guidance",
                description="Get current official guidance",
                parameters={"type": "object", "properties": {}},
                handler=source,
            )
        },
        structured_grounding=True,
        guard_grounding=True,
    ).run("What is the tipped cash wage?")

    assert result.text == f"{expected} {{cite:S1}}"
    assert result.diagnostics["validation_rejections"] == []


async def test_mechanical_validator_does_not_semantically_parse_an_unsupported_date():
    async def source(_args: dict, ctx: ToolContext) -> str:
        citation_id = ctx.citations.register(
            "https://data.cityofnewyork.us/restrooms",
            title="NYC public restrooms",
            kind="WEB",
            snippet="The listing includes a year-round restroom at Flushing Library.",
        )
        return f"Official restroom listing. {{cite:{citation_id}}}"

    model_calls = 0

    async def model(
        messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            return ModelResponse([ToolCallPart("guidance", {}, "guidance-1")])
        text = (
            "The City’s restroom listing, dated June 27, 2025, includes a "
            "year-round restroom at Flushing Library."
        )
        return ModelResponse([
            ToolCallPart(
                info.output_tools[0].name,
                {
                    "grounded_blocks": [{
                        "text": text,
                        "citation_ids": ["S1"],
                    }]
                },
                "final-1",
            )
        ])

    result = await PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools={
            "guidance": Tool(
                name="guidance",
                description="Get the official restroom listing",
                parameters={"type": "object", "properties": {}},
                handler=source,
            )
        },
        structured_grounding=True,
        guard_grounding=True,
    ).run("Is there a public restroom at Flushing Library?")

    assert model_calls == 2
    assert result.text == (
        "The City’s restroom listing, dated June 27, 2025, includes a year-round restroom at "
        "Flushing Library. {cite:S1}"
    )
    assert result.diagnostics["validation_rejections"] == []


async def test_orphan_citation_fragments_require_a_clean_replacement():
    async def source(_args: dict, ctx: ToolContext) -> str:
        citation_id = ctx.citations.register(
            "https://www.nyc.gov/restrooms",
            title="NYC public restrooms",
            kind="WEB",
            snippet="NYC maintains public restrooms.",
        )
        return f"Official restroom guidance. {{cite:{citation_id}}}"

    model_calls = 0

    async def model(
        messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            return ModelResponse([ToolCallPart("guidance", {}, "guidance-1")])
        text = (
            "NYC maintains public restrooms. S1 S2} {cite:S1}"
            if model_calls == 2
            else "NYC maintains public restrooms."
        )
        return ModelResponse([
            ToolCallPart(
                info.output_tools[0].name,
                {
                    "grounded_blocks": [{
                        "text": text,
                        "citation_ids": ["S1"],
                    }]
                },
                f"final-{model_calls}",
            )
        ])

    result = await PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools={
            "guidance": Tool(
                name="guidance",
                description="Get official restroom guidance",
                parameters={"type": "object", "properties": {}},
                handler=source,
            )
        },
        structured_grounding=True,
        guard_grounding=True,
    ).run("Are all public restrooms closed?")

    assert model_calls == 3
    assert result.text == "NYC maintains public restrooms. {cite:S1}"
    assert [
        rejection["stage"]
        for rejection in result.diagnostics["validation_rejections"]
    ] == ["citation_marker"]


async def test_mechanical_validator_does_not_parse_event_times():
    async def source(_args: dict, ctx: ToolContext) -> str:
        first = ctx.citations.register(
            "https://example.com/first",
            title="First event",
            kind="WEB",
            snippet="First event starts at 10:30 AM.",
        )
        second = ctx.citations.register(
            "https://example.com/second",
            title="Second event",
            kind="WEB",
            snippet="Second event starts at 11:00 AM.",
        )
        return f"Official events. {{cite:{first}}} {{cite:{second}}}"

    model_calls = 0

    async def model(
        messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            return ModelResponse([ToolCallPart("guidance", {}, "guidance-1")])
        grounded_blocks = (
            [
                {
                    "text": (
                        "First event has listed start times of 10:30 AM and 10:45 AM. "
                        "[Official details](https://example.com/first)"
                    ),
                    "citation_ids": ["S1"],
                },
                {
                    "text": "Second event starts at 11:00 AM.",
                    "citation_ids": ["S2"],
                },
            ]
            if model_calls == 2
            else [{
                "text": "Second event starts at 11:00 AM.",
                "citation_ids": ["S2"],
            }]
        )
        return ModelResponse([
            ToolCallPart(
                info.output_tools[0].name,
                {
                    "grounded_blocks": grounded_blocks
                },
                "final-1",
            )
        ])

    result = await PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools={
            "guidance": Tool(
                name="guidance",
                description="Get official events",
                parameters={"type": "object", "properties": {}},
                handler=source,
            )
        },
        structured_grounding=True,
        guard_grounding=True,
    ).run("What can I do today?")

    assert model_calls == 2
    assert "10:45 AM" in result.text
    assert "Second event starts at 11:00 AM" in result.text
    assert result.diagnostics["validation_rejections"] == []


async def test_mechanical_validator_does_not_parse_map_coordinates():
    async def source(_args: dict, ctx: ToolContext) -> str:
        row = {
            "name": "RAICES Gowanus OAC Satellite",
            "address": "420 Baltic Street",
            "latitude": "40.68273",
            "longitude": "-73.98000",
        }
        citation_id = ctx.citations.register(
            "https://example.gov/cooling/1",
            title="NYC cooling option",
            kind="DATA",
            snippet="RAICES Gowanus OAC Satellite, 420 Baltic Street",
            provenance=data_provenance(row, record_id="1", field_pointer="/"),
        )
        return f"Official cooling option. {{cite:{citation_id}}}"

    model_calls = 0

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            return ModelResponse([ToolCallPart("guidance", {}, "guidance-1")])
        grounded_blocks = (
            [
                {
                    "text": (
                        "Directions: [Map](https://www.google.com/maps/dir/?api=1&"
                        "origin=40.67363,-73.95400&destination=40.68273,-73.98905). "
                        "The distance is only a rough estimate."
                    ),
                    "citation_ids": ["S1"],
                },
                {
                    "text": "RAICES Gowanus OAC Satellite is at 420 Baltic Street.",
                    "citation_ids": ["S1"],
                },
            ]
            if model_calls == 2
            else [{
                "text": "RAICES Gowanus OAC Satellite is at 420 Baltic Street.",
                "citation_ids": ["S1"],
            }]
        )
        return ModelResponse([
            ToolCallPart(
                info.output_tools[0].name,
                {
                    "grounded_blocks": grounded_blocks
                },
                "final-1",
            )
        ])

    result = await PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools={
            "guidance": Tool(
                name="guidance",
                description="Get current official guidance",
                parameters={"type": "object", "properties": {}},
                handler=source,
            )
        },
        structured_grounding=True,
        guard_grounding=True,
    ).run("Where can I cool down near Crown Heights?")

    assert model_calls == 2
    assert "RAICES Gowanus OAC Satellite" in result.text
    assert "-73.98905" in result.text
    assert result.diagnostics["validation_rejections"] == []


async def test_mechanical_validator_does_not_parse_a_location_question():
    async def source(_args: dict, ctx: ToolContext) -> str:
        citation_id = ctx.citations.register(
            "https://example.gov/accessibility",
            title="Official accessibility guidance",
            kind="WEB",
            snippet="Check elevator status before starting an accessible trip.",
        )
        return f"Check elevator status before starting. {{cite:{citation_id}}}"

    model_calls = 0

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        if not any(
            getattr(part, "tool_name", "") == "guidance"
            for message in messages
            for part in message.parts
        ):
            return ModelResponse([ToolCallPart("guidance", {}, "guidance-1")])
        question = (
            "What exact origin and destination should I use?"
            if model_calls == 3
            else "Do you mean the entrance at 1000 Fifth Avenue?"
        )
        return ModelResponse([
            ToolCallPart(
                info.output_tools[0].name,
                {
                    "grounded_blocks": [
                        {
                            "text": "Check elevator status before starting.",
                            "citation_ids": ["S1"],
                        },
                        {"text": question, "citation_ids": ["S1"]},
                    ]
                },
                f"final-{model_calls}",
            )
        ])

    result = await PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools={
            "guidance": Tool(
                name="guidance",
                description="Get current official guidance",
                parameters={"type": "object", "properties": {}},
                handler=source,
            )
        },
        structured_grounding=True,
        guard_grounding=True,
    ).run("Plan an accessible trip from Flushing to the museum")

    assert model_calls == 2
    assert "Do you mean the entrance at 1000 Fifth Avenue?" in result.text
    assert result.diagnostics["validation_rejections"] == []
