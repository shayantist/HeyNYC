"""Thin Ticketmaster Discovery API client — the structured events backbone (§16).

Verified live (2026-06-28): dmaId=345 (NYC metro) + startDateTime=<now Z> returns
date-sorted upcoming events; keyword=world cup surfaces the FIFA Final watch party.
Returns the raw `_embedded.events` list; the events module normalizes it into the
common Event shape. The client is injectable so tests never touch the network.
"""
from __future__ import annotations

from typing import Optional

import httpx

from . import config

DISCOVERY_URL = "https://app.ticketmaster.com/discovery/v2/events.json"
NYC_DMA_ID = "345"  # New York metro Designated Market Area


async def ticketmaster_events(
    *,
    keyword: Optional[str] = None,
    classification: Optional[str] = None,
    start_datetime: Optional[str] = None,
    size: int = 20,
    client: Optional[httpx.AsyncClient] = None,
    api_key: Optional[str] = None,
) -> list[dict]:
    """Query upcoming NYC-metro events. Returns raw TM event dicts, or [] when no
    key is configured (caller treats as 'unavailable' and falls back / abstains)."""
    key = api_key if api_key is not None else config.TICKETMASTER_API_KEY
    if not key:
        return []
    params: dict = {"apikey": key, "dmaId": NYC_DMA_ID, "sort": "date,asc", "size": size}
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
    return data.get("_embedded", {}).get("events", []) or []
