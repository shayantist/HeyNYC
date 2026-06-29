"""index_search tool — retrieval over the curated NYC corpus (the "skeleton").

Returns passages with DOC citations so the agent can ground general questions
(how a service works, what to bring) that aren't location lookups. When nothing
matches, it says so, nudging the agent toward web_search or abstention.
"""
from __future__ import annotations

from ..index import IndexRetriever
from .base import Tool, ToolContext


def index_search_tools(retriever: IndexRetriever) -> list[Tool]:
    async def _handler(args: dict, ctx: ToolContext) -> str:
        hits = retriever.search(args["query"], k=int(args.get("k", 5)))
        if not hits:
            return "No indexed NYC sources matched. Use web_search for fresh/long-tail info, or abstain."
        blocks = []
        for doc, score in hits:
            cite = ctx.citations.register(doc.url, snippet=doc.text[:200], title=doc.title, kind="DOC")
            blocks.append(f"[{cite}] {doc.title} ({doc.url})\n{doc.text[:500]}")
        return "\n\n".join(blocks)

    return [
        Tool(
            name="index_search",
            description=(
                "Search the curated index of official NYC pages for how-to / eligibility / "
                "general info about services and events. Use for non-location questions; cite results."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to look up."},
                    "k": {"type": "integer", "description": "How many passages (default 5).", "default": 5},
                },
                "required": ["query"],
            },
            handler=_handler,
        )
    ]
