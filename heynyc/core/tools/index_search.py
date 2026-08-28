"""Legacy index-search adapter retained outside the active model tool surface.

Returns passages with DOC citations so the agent can ground general questions
(how a service works, what to bring) that aren't location lookups.
"""
from __future__ import annotations

from pydantic import Field

from ..index import IndexRetriever
from .base import Tool, ToolContext, ToolInput


class IndexSearchInput(ToolInput):
    query: str = Field(description="Search text")
    k: int = Field(default=5, ge=1, description="Passages requested")


def index_search_tools(retriever: IndexRetriever) -> list[Tool]:
    async def _handler(args: IndexSearchInput, ctx: ToolContext) -> str:
        hits = retriever.search(args["query"], k=int(args.get("k", 5)))
        if not hits:
            return "No indexed NYC sources matched."
        blocks = []
        for doc, score in hits:
            cite = ctx.citations.register(doc.url, snippet=doc.text, title=doc.title, kind="DOC")
            blocks.append(f"[{cite}] {doc.title} ({doc.url})\n{doc.text}")
        return "\n\n".join(blocks)

    return [
        Tool(
            name="index_search",
            description=(
                "Search the curated index of official NYC pages. This legacy adapter is not "
                "registered in the active model tool surface."
            ),
            input_type=IndexSearchInput,
            handler=_handler,
        )
    ]
