"""Thin Ticketmaster Discovery API client, the structured events backbone (§16).

Verified live (2026-06-28): dmaId=345 (NYC metro) + startDateTime=<now Z> returns
date-sorted upcoming events; keyword=world cup surfaces the FIFA Final watch party.
Returns the raw events with the provider's page boundary; the events module normalizes
them into the common Event shape. The client is injectable so tests never touch the network.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

import httpx
from pydantic import BaseModel, ConfigDict, Field

from . import config

DISCOVERY_URL = "https://app.ticketmaster.com/discovery/v2/events.json"
NYC_DMA_ID = "345"  # New York metro Designated Market Area


class TicketmasterSearchResult(BaseModel):
    """One Discovery API page plus the provider's coverage boundary."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["complete", "partial", "unavailable"]
    events: list[dict] = Field(default_factory=list)
    page_number: int | None = None
    page_size: int | None = None
    total_elements: int | None = None
    total_pages: int | None = None
    next_page: int | None = None
    retrieved_at: str = ""


def _page_int(page: dict, key: str) -> int | None:
    value = page.get(key)
    return value if isinstance(value, int) and value >= 0 else None


async def ticketmaster_events(
    *,
    keyword: Optional[str] = None,
    classification: Optional[str] = None,
    start_datetime: Optional[str] = None,
    size: int = 20,
    client: Optional[httpx.AsyncClient] = None,
    api_key: Optional[str] = None,
) -> TicketmasterSearchResult:
    """Query one page of upcoming NYC-metro events with explicit coverage metadata."""
    key = api_key if api_key is not None else config.TICKETMASTER_API_KEY
    if not key:
        return TicketmasterSearchResult(status="unavailable")
    params: dict = {
        "apikey": key,
        "dmaId": NYC_DMA_ID,
        "sort": "date,asc",
        "size": size,
        "page": 0,
    }
    if start_datetime:
        params["startDateTime"] = start_datetime
    if keyword:
        params["keyword"] = keyword
    if classification:
        params["classificationName"] = classification

    own_client = client is None
    client = client or httpx.AsyncClient(timeout=20.0)
    try:
        response = await client.get(DISCOVERY_URL, params=params)
        response.raise_for_status()
        data = response.json()
    finally:
        if own_client:
            await client.aclose()
    events = data.get("_embedded", {}).get("events", []) or []
    page = data.get("page") if isinstance(data.get("page"), dict) else {}
    page_number = _page_int(page, "number")
    page_size = _page_int(page, "size")
    total_elements = _page_int(page, "totalElements")
    total_pages = _page_int(page, "totalPages")
    complete = (
        page_number is not None
        and total_pages is not None
        and page_number + 1 >= total_pages
    )
    return TicketmasterSearchResult(
        status="complete" if complete else "partial",
        events=events,
        page_number=page_number,
        page_size=page_size,
        total_elements=total_elements,
        total_pages=total_pages,
        next_page=(page_number + 1 if page_number is not None and not complete else None),
        retrieved_at=datetime.now(timezone.utc).isoformat(),
    )
