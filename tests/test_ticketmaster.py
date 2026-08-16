from __future__ import annotations

import httpx

from heynyc.core.ticketmaster import (
    NYC_DMA_ID,
    TicketmasterSearchResult,
    ticketmaster_events,
)


def _mock_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_builds_nyc_params_and_returns_events():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["params"] = dict(request.url.params)
        body = {
            "_embedded": {"events": [{"name": "FIFA Final Watch Party"}]},
            "page": {"size": 20, "totalElements": 1, "totalPages": 1, "number": 0},
        }
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
    assert isinstance(events, TicketmasterSearchResult)
    assert events.status == "complete"
    assert events.events[0]["name"] == "FIFA Final Watch Party"
    assert events.page_number == 0
    assert events.page_size == 20
    assert events.total_elements == 1
    assert events.total_pages == 1
    assert events.next_page is None
    assert events.retrieved_at


async def test_no_key_returns_empty_without_network():
    # No api_key and no config key is unavailable, not an empty successful search.
    events = await ticketmaster_events(keyword="anything", api_key="")
    assert events == TicketmasterSearchResult(status="unavailable")


async def test_missing_embedded_returns_empty():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})  # no _embedded

    async with _mock_client(handler) as client:
        events = await ticketmaster_events(client=client, api_key="k")
    assert events.status == "partial"
    assert events.events == []
    assert events.retrieved_at


async def test_preserves_partial_page_metadata():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["page"] == "0"
        return httpx.Response(200, json={
            "_embedded": {"events": [{"name": "First page event"}]},
            "page": {"size": 20, "totalElements": 35, "totalPages": 2, "number": 0},
        })

    async with _mock_client(handler) as client:
        result = await ticketmaster_events(client=client, api_key="k")

    assert result.status == "partial"
    assert result.next_page == 1
    assert result.total_elements == 35
