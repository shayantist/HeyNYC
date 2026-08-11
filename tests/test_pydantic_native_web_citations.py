from __future__ import annotations

import json

import httpx
import pytest
from pydantic_ai import Agent, WebSearchTool
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.messages import (
    ModelResponse,
    NativeToolCallPart,
    NativeToolReturnPart,
    TextPart,
)
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider


@pytest.mark.asyncio
async def test_openai_native_web_search_preserves_sources_annotations_and_usage() -> None:
    answer = "Jalen Brunson is the Knicks captain."
    source_url = (
        "https://www.nba.com/knicks/news/"
        "jalen-brunson-named-36th-captain-in-knicks-franchise-history"
    )
    annotation = {
        "end_index": len(answer),
        "start_index": 0,
        "title": "Jalen Brunson Named Captain",
        "type": "url_citation",
        "url": source_url,
    }
    response = {
        "id": "resp_test",
        "created_at": 0,
        "model": "gpt-5.6-luna",
        "object": "response",
        "output": [
            {
                "id": "ws_1",
                "type": "web_search_call",
                "status": "completed",
                "action": {
                    "type": "search",
                    "query": "Knicks captain",
                    "sources": [{"type": "url", "url": source_url}],
                },
            },
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": answer,
                        "annotations": [annotation],
                    }
                ],
            },
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": 12,
            "input_tokens_details": {
                "cache_write_tokens": 0,
                "cached_tokens": 3,
            },
            "output_tokens": 9,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 21,
        },
    }
    request_bodies: list[dict] = []

    def respond(request: httpx.Request) -> httpx.Response:
        request_bodies.append(json.loads(request.content))
        return httpx.Response(200, json=response)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        provider = OpenAIProvider(
            base_url="https://example.test/v1",
            api_key="test",
            http_client=client,
        )
        model = OpenAIResponsesModel(
            "gpt-5.6-luna",
            provider=provider,
            settings={
                "max_tokens": 500,
                "extra_body": {"max_tool_calls": 1},
                "openai_include_raw_annotations": True,
                "openai_include_web_search_sources": True,
            },
        )
        agent = Agent(
            model,
            capabilities=[
                NativeTool(WebSearchTool(search_context_size="low")),
            ],
        )

        result = await agent.run("Who is the Knicks captain?")

    request_body = request_bodies[0]
    assert request_body["tools"] == [
        {"type": "web_search", "search_context_size": "low"}
    ]
    assert request_body["max_tool_calls"] == 1
    assert request_body["max_output_tokens"] == 500
    assert "web_search_call.action.sources" in request_body["include"]

    model_response = next(
        message
        for message in result.all_messages()
        if isinstance(message, ModelResponse)
    )
    native_call = next(
        part
        for part in model_response.parts
        if isinstance(part, NativeToolCallPart)
    )
    native_return = next(
        part
        for part in model_response.parts
        if isinstance(part, NativeToolReturnPart)
    )
    text_part = next(
        part for part in model_response.parts if isinstance(part, TextPart)
    )

    assert native_call.args == {"type": "search", "query": "Knicks captain"}
    assert native_return.content["sources"] == [
        {"type": "url", "url": source_url}
    ]
    assert text_part.content == answer
    assert text_part.provider_details == {"annotations": [annotation]}
    assert result.usage.input_tokens == 12
    assert result.usage.cache_read_tokens == 3
    assert result.usage.output_tokens == 9
    assert result.usage.requests == 1
