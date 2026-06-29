from __future__ import annotations

import httpx
import pytest

from heynyc.core.ticketmaster import DISCOVERY_URL, NYC_DMA_ID, ticketmaster_events


def _mock_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_builds_nyc_params_and_returns_events():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["params"] = dict(request.url.params)
        body = {"_embedded": {"events": [{"name": "FIFA Final Watch Party"}]}}
        return httpx.Response(200, json=body)

    async with _mock_client(handler) as client:
        events = await ticketmaster_events(
            keyword="world cup", start_datetime="2026-06-28T00:00:00Z",
            client=client, api_key="test-key",
        )

    assert seen["params"]["dmaId"] == NYC_DMA_ID
    assert seen["params"]["apikey"] == "test-key"
    assert seen["params"]["keyword"] == "world cup"
    assert seen["params"]["startDateTime"] == "2026-06-28T00:00:00Z"
    assert events[0]["name"] == "FIFA Final Watch Party"


async def test_no_key_returns_empty_without_network():
    # No api_key and no config key -> returns [] and never builds a client.
    events = await ticketmaster_events(keyword="anything", api_key="")
    assert events == []


async def test_missing_embedded_returns_empty():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})  # no _embedded

    async with _mock_client(handler) as client:
        events = await ticketmaster_events(client=client, api_key="k")
    assert events == []
