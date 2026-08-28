from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from heynyc.core.pydantic_runtime import build_runtime
from heynyc.core.registry import Registry
from heynyc.core.tools.base import Tool


async def test_runtime_reuses_the_configured_index_embedder_in_tools(tmp_path):
    embedder = object()
    seen = []
    calls = 0

    class Index:
        pass

    index = Index()
    index.embedder = embedder

    async def probe(_args, ctx):
        seen.append((ctx.embedder, ctx.retrieval_cache_path))
        return "done"

    async def model(_messages, _info):
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse([ToolCallPart("probe", {}, "probe-1")])
        return ModelResponse([
            ToolCallPart("final_answer", {"answer": "No factual claim."}, "final-1")
        ])

    runtime = build_runtime(
        Registry([]),
        model=FunctionModel(model),
        tools={
            "probe": Tool(
                name="probe",
                description="Inspect runtime dependencies",
                handler=probe,
            )
        },
        index=index,
        retrieval_cache_path=tmp_path / "catalogs.lance",
        structured_grounding=True,
    )

    await runtime.run("hello")

    assert seen == [(embedder, tmp_path / "catalogs.lance")]
