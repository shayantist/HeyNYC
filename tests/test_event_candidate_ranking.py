from heynyc.modules.events.tools import (
    Event,
    _from_web_citation,
    _from_web_citation_events,
)


def test_web_citation_normalizes_into_the_existing_event_record() -> None:
    event = _from_web_citation(
        "S4",
        {
            "url": "https://donyc.com/events/2026/08/21/show",
            "title": "A neighborhood concert",
            "snippet": "Friday night at Public Records in Brooklyn.",
            "provenance": {
                "source_tier": "editorial",
                "search": {"provider": "Tavily Search API", "score": 0.91},
            },
        },
        rank=2,
    )

    assert event == Event(
        name="A neighborhood concert",
        start_date="",
        start_time="",
        venue="",
        borough="",
        url="https://donyc.com/events/2026/08/21/show",
        source="Web discovery",
        tier="editorial",
        publishing_source="donyc.com",
        provider_id="https://donyc.com/events/2026/08/21/show",
        provider_record={
            "url": "https://donyc.com/events/2026/08/21/show",
            "title": "A neighborhood concert",
            "snippet": "Friday night at Public Records in Brooklyn.",
            "provenance": {
                "source_tier": "editorial",
                "search": {"provider": "Tavily Search API", "score": 0.91},
            },
        },
        evidence_excerpt="Friday night at Public Records in Brooklyn.",
        citation_id="S4",
        retrieval_rank=2,
    )


def test_web_citation_normalizes_structured_collection_events() -> None:
    normalized = _from_web_citation_events(
        "S4",
        {
            "url": "https://guide.example/events",
            "title": "NYC events this week",
            "snippet": "Two source-backed event records.",
            "provenance": {
                "source_tier": "editorial",
                "events": [
                    {
                        "name": "Indie night",
                        "start_date": "2026-08-27",
                        "start_time": "19:00",
                        "url": "https://venue.example/indie",
                        "venue": "Elsewhere",
                        "borough": "Brooklyn",
                    },
                    {
                        "name": "Jazz night",
                        "start_date": "2026-08-28",
                        "start_time": "20:00",
                        "url": "https://venue.example/jazz",
                        "venue": "Public Records",
                        "borough": "Brooklyn",
                    },
                ],
            },
        },
        rank=0,
    )

    assert [event.name for event in normalized] == ["Indie night", "Jazz night"]
    assert all(event.structured_source for event in normalized)
