"""Local interactive REPL for the isolated PydanticAI candidate."""
from __future__ import annotations

import asyncio
import re

from dotenv import load_dotenv
from pydantic_ai.models import infer_model
from pydantic_ai.models.openai import OpenAIResponsesModel
from rich.console import Console
from rich.markdown import Markdown

load_dotenv()

from heynyc.__main__ import _default_reminders, _load_retriever
from heynyc.channels.store import ChannelStore
from heynyc.core import config
from heynyc.core.citations import text_fragment_url, used_citations
from heynyc.core.registry import Registry
from heynyc.core.tools import build_toolbox
from heynyc.modules.advisories.tools import current_awareness
from scripts.pydantic_ai_parity import (
    PydanticApprovalFlow,
    build_runtime,
)


async def _resolve_pending(console: Console, flow: PydanticApprovalFlow):
    console.print(flow.review_text())
    answer = console.input("[yellow]Reply YES or NO[/] [y/N] ")
    return await flow.resume(answer.strip().lower() in {"y", "yes"})


def _configured_model():
    if config.HEYNYC_MODEL.startswith("openai/"):
        settings = {
            key: value
            for key, value in {
                "openai_reasoning_effort": config.HEYNYC_REASONING_EFFORT,
                "openai_service_tier": config.HEYNYC_SERVICE_TIER,
            }.items()
            if value is not None
        }
        return OpenAIResponsesModel(
            config.HEYNYC_MODEL.removeprefix("openai/"),
            settings=settings,
        )
    return infer_model(config.HEYNYC_MODEL.replace("/", ":", 1))


async def main() -> None:
    console = Console()
    registry = Registry.discover(
        config.MODULES_DIR,
        config.BASE_ALLOWLIST,
        config.NEWS_ALLOWLIST,
    )
    runtime = build_runtime(
        registry,
        model=_configured_model(),
        answer_model_route=config.HEYNYC_MODEL,
        tools=build_toolbox(registry, index=_load_retriever(required=False)),
        use_module_capabilities=True,
        current_awareness=current_awareness,
    )
    approval_store = ChannelStore(
        config.HEYNYC_DATA_DIR / "pydantic-candidate.sqlite3",
        rate_limit=1,
        window_s=1,
        dedup_ttl_s=1,
    )
    flow = PydanticApprovalFlow(
        runtime,
        approval_store,
        "local-pydantic-repl",
        ttl_s=15 * 60,
    )
    console.print(
        "[bold]HeyNYC PydanticAI candidate[/]\n"
        "[dim]Local debug only. NEW clears model-visible history; EXIT quits. "
        "Nothing is sent through SMS or WhatsApp.[/]\n"
    )

    while True:
        if flow.conversation.pending_approvals:
            with console.status("restoring pending approval"):
                result = await _resolve_pending(console, flow)
        else:
            try:
                query = console.input("[bold green]you ▸ [/]").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not query:
                continue
            if query.upper() in {"EXIT", "QUIT"}:
                break
            if query.upper() == "NEW":
                approval_store.pop_pending_approval("local-pydantic-repl")
                flow = PydanticApprovalFlow(
                    runtime,
                    approval_store,
                    "local-pydantic-repl",
                    ttl_s=15 * 60,
                )
                console.print("[dim]Started a fresh candidate conversation.[/]\n")
                continue

            with console.status("thinking"):
                result = await flow.send(query, reminders=_default_reminders())
                if result.status == "approval_required":
                    result = await _resolve_pending(console, flow)

        body = re.sub(r"\{cite:(S\d+)\}", r"[\1]", result.text)
        console.print(Markdown(body))
        capabilities = result.usage.get("capabilities_used") or []
        tools = [
            name
            for name in result.tool_calls_made
            if name not in {"load_capability", "search_tools"}
        ]
        console.print(
            "[dim]"
            f"capabilities: {', '.join(capabilities) or '-'} · "
            f"tools: {', '.join(tools) or '-'} · "
            f"cache read: {result.usage.get('cached_input_tokens', 0)} tokens · "
            f"cost: {result.usage.get('cost_usd') if result.usage.get('cost_usd') is not None else 'unpriced'}"
            "[/]"
        )
        cited = used_citations(result.text, result.citations)
        if cited:
            console.print("[dim]Sources:[/]")
            for cite_id, citation in cited.items():
                url = text_fragment_url(
                    citation["url"],
                    citation.get("snippet", ""),
                    citation.get("kind", ""),
                )
                console.print(
                    f"  [dim][{cite_id}] {citation.get('title') or url} - {url}[/]"
                )
        console.print()


if __name__ == "__main__":
    asyncio.run(main())
