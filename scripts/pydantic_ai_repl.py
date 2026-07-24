"""Local interactive REPL for the isolated PydanticAI candidate."""
from __future__ import annotations

import asyncio
import json
import re

from dotenv import load_dotenv
from pydantic_ai.models import infer_model
from pydantic_ai.models.openai import OpenAIResponsesModel
from rich.console import Console
from rich.markdown import Markdown

load_dotenv()

from heynyc.__main__ import _default_reminders, _load_retriever
from heynyc.core import config
from heynyc.core.citations import text_fragment_url, used_citations
from heynyc.core.registry import Registry
from heynyc.core.tools import build_toolbox
from heynyc.modules.advisories.tools import current_awareness
from scripts.pydantic_ai_parity import build_runtime


def _approval_review(request: dict) -> str:
    return (
        f"{request['tool_name']}\n"
        f"{json.dumps(request['args'], ensure_ascii=False, indent=2, sort_keys=True)}"
    )


def _approval_copy(tool_name: str) -> tuple[str, str]:
    if tool_name.startswith("confirm_") and tool_name.endswith("_facts"):
        return (
            "Review the structured facts I understood:",
            "Are these facts accurate?",
        )
    return "Review the proposed action and exact values:", "Approve this action?"


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
    conversation = runtime.conversation()
    console.print(
        "[bold]HeyNYC PydanticAI candidate[/]\n"
        "[dim]Local debug only. NEW clears model-visible history; EXIT quits. "
        "Nothing is sent through SMS or WhatsApp.[/]\n"
    )

    while True:
        try:
            query = console.input("[bold green]you ▸ [/]").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query:
            continue
        if query.upper() in {"EXIT", "QUIT"}:
            break
        if query.upper() == "NEW":
            conversation = runtime.conversation()
            console.print("[dim]Started a fresh candidate conversation.[/]\n")
            continue

        with console.status("thinking"):
            result = await conversation.send(query, reminders=_default_reminders())
            if result.status == "approval_required":
                approvals = {}
                for call_id, request in conversation.pending_approvals.items():
                    heading, question = _approval_copy(request["tool_name"])
                    console.print(f"[yellow]{heading}[/]")
                    console.print(_approval_review(request))
                    answer = console.input(f"[yellow]{question}[/] [y/N] ")
                    approvals[call_id] = answer.strip().lower() in {"y", "yes"}
                result = await conversation.resume_approvals(approvals)

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
