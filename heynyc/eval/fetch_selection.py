"""Offline comparison of page passages presented to the answer model."""
from __future__ import annotations

import argparse
import asyncio
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Awaitable, Callable

import httpx

from ..core import config
from ..core.tools.web_fetch import _fetch_page_with_browser

ExtractFn = Callable[[str, str], Awaitable[list[str]]]


def _tokens(text: str, model: str) -> int:
    import litellm

    return int(litellm.token_counter(model=model, text=text)) if text else 0


def _duplicate_ratio(chunks: list[str]) -> float:
    if len(chunks) < 2:
        return 0.0
    return round(max(
        SequenceMatcher(None, left, right, autojunk=False).find_longest_match().size
        / min(len(left), len(right))
        for index, left in enumerate(chunks)
        for right in chunks[index + 1:]
        if left and right
    ), 4)


def _measure(
    chunks: list[str],
    required: list[str],
    model: str,
    *,
    eligible: bool = True,
    selection_ms: float = 0.0,
    external_cost_usd: float = 0.0,
) -> dict[str, Any]:
    text = "\n\n".join(chunks)
    folded = re.sub(r"\s+", " ", text.casefold())
    found = [
        passage
        for passage in required
        if re.sub(r"\s+", " ", passage.casefold()) in folded
    ]
    return {
        "eligible": eligible,
        "required_passage_recall": round(len(found) / len(required), 4) if required else 1.0,
        "required_text_density": round(
            min(1.0, sum(len(passage) for passage in found) / len(text)),
            4,
        ) if text else 0.0,
        "duplicate_ratio": _duplicate_ratio(chunks),
        "tokens": _tokens(text, model),
        "characters": len(text),
        "selection_ms": round(selection_ms, 4),
        "external_cost_usd": external_cost_usd,
        "chunks": chunks,
    }


def compare_case(
    case: dict[str, Any],
    *,
    model: str,
    full_token_budget: int,
) -> dict[str, Any]:
    text = str(case["text"])
    required = [str(passage) for passage in case.get("required_passages", ())]
    full_tokens = _tokens(text, model)
    full_eligible = full_tokens <= full_token_budget
    candidates = {
        "full": _measure(
            [text] if full_eligible else [],
            required,
            model,
            eligible=full_eligible,
        ),
    }
    if "tavily_chunks" in case:
        candidates["tavily_basic"] = _measure(
            [str(chunk) for chunk in case["tavily_chunks"]],
            required,
            model,
        )
    return {"case_id": str(case["id"]), "candidates": candidates}


def write_comparison(
    source: Path,
    output: Path,
    *,
    model: str,
    full_token_budget: int,
) -> None:
    cases = json.loads(source.read_text())
    report = {
        "model": model,
        "full_token_budget": full_token_budget,
        "cases": [
            compare_case(
                case,
                model=model,
                full_token_budget=full_token_budget,
            )
            for case in cases
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


async def collect_cases(source: Path, output: Path) -> None:
    specs = json.loads(source.read_text())

    async def collect(spec: dict[str, Any]) -> dict[str, Any]:
        page = await _fetch_page_with_browser(
            str(spec["url"]),
            None,
            None,
        )
        return {
            **spec,
            "url": page.final_url,
            "title": page.title,
            "text": page.text,
        }

    cases = await asyncio.gather(*(collect(spec) for spec in specs))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n")


async def _tavily_extract(url: str, query: str) -> list[str]:
    if not config.TAVILY_API_KEY:
        raise RuntimeError("TAVILY_API_KEY is required")
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            "https://api.tavily.com/extract",
            headers={"Authorization": f"Bearer {config.TAVILY_API_KEY}"},
            json={
                "urls": [url],
                "query": query,
                "extract_depth": "basic",
                "format": "markdown",
                "include_usage": True,
            },
        )
        response.raise_for_status()
    results = response.json().get("results") or []
    raw = results[0].get("raw_content") if results else ""
    if isinstance(raw, list):
        return [str(chunk) for chunk in raw if str(chunk).strip()]
    return [str(raw)] if str(raw).strip() else []


async def collect_tavily_extracts(
    source: Path,
    output: Path,
    *,
    extract_fn: ExtractFn | None = None,
) -> None:
    cases = json.loads(source.read_text())
    extract_page = extract_fn or _tavily_extract
    for case in cases:
        case["tavily_chunks"] = await extract_page(
            str(case["url"]),
            str(case["query"]),
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="openai/gpt-5.6-luna")
    parser.add_argument("--full-token-budget", type=int, default=3_000)
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--tavily-extract", action="store_true")
    args = parser.parse_args()
    if args.collect:
        asyncio.run(collect_cases(args.source, args.out))
        return
    if args.tavily_extract:
        asyncio.run(collect_tavily_extracts(args.source, args.out))
        return
    write_comparison(
        args.source,
        args.out,
        model=args.model,
        full_token_budget=args.full_token_budget,
    )


if __name__ == "__main__":
    main()
