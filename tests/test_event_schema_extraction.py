from heynyc.core.tools.web_fetch import _schema_events
from heynyc.modules.events.tools import _from_web_citation_events, _matches_keyword


def test_schema_event_microdata_extracts_direct_records() -> None:
    html = """
    <div class="listing event-category-music" itemscope itemtype="https://schema.org/Event">
      <a itemprop="url" href="/events/show"><span itemprop="name">Live show</span></a>
      <div itemprop="location" itemscope itemtype="https://schema.org/Place">
        <span itemprop="name">Public Records</span>
        <meta itemprop="addressLocality" content="Brooklyn" />
      </div>
      <meta itemprop="startDate" content="2026-08-21T19:30-04:00" />
      <meta itemprop="endDate" content="2026-08-21T21:00-04:00" />
    </div>
    """

    assert _schema_events(html, "https://events.example/calendar") == [{
        "name": "Live show",
        "url": "https://events.example/events/show",
        "venue": "Public Records",
        "borough": "Brooklyn",
        "category": "music",
        "start_date": "2026-08-21",
        "start_time": "19:30",
        "end_date": "2026-08-21",
        "end_time": "21:00",
    }]


def test_schema_event_microdata_handles_html_void_elements() -> None:
    html = """
    <div itemscope itemtype="http://schema.org/Event">
      <a itemprop="url" href="/events/show"><span itemprop="name">Live show</span></a>
      <div itemprop="location" itemscope itemtype="http://schema.org/Place">
        <span itemprop="name">Public Records</span>
      </div>
      <meta itemprop="startDate" content="2026-08-21T19:30-0400">
    </div>
    """

    assert _schema_events(html, "https://events.example/calendar")[0]["start_time"] == "19:30"


def test_schema_event_keeps_its_page_url_over_a_nested_ticket_offer() -> None:
    html = """
    <div itemscope itemtype="http://schema.org/Event">
      <a itemprop="url" href="/events/live-show"><span itemprop="name">Live show</span></a>
      <meta itemprop="startDate" content="2026-08-21T19:30-0400">
      <span itemprop="offers" itemscope itemtype="http://schema.org/Offer">
        <meta itemprop="name" content="VIP package">
        <meta itemprop="startDate" content="2099-01-01T00:00:00Z">
        <meta itemprop="url" content="https://tickets.example/live-show">
      </span>
    </div>
    """

    assert _schema_events(html, "https://events.example/calendar")[0] == {
        "name": "Live show",
        "url": "https://events.example/events/live-show",
        "venue": "",
        "borough": "",
        "category": "",
        "start_date": "2026-08-21",
        "start_time": "19:30",
        "end_date": "",
        "end_time": "",
    }


def test_schema_event_converts_offset_timestamps_to_new_york_time() -> None:
    html = """
    <div itemscope itemtype="https://schema.org/Event">
      <span itemprop="name">Late show</span>
      <meta itemprop="startDate" content="2026-08-21T23:00:00Z">
      <meta itemprop="endDate" content="2026-08-22T01:00:00Z">
    </div>
    """

    event = _schema_events(html, "https://events.example/late-show")[0]

    assert (event["start_date"], event["start_time"]) == ("2026-08-21", "19:00")
    assert (event["end_date"], event["end_time"]) == ("2026-08-21", "21:00")


def test_fetched_calendar_citation_expands_into_individual_event_records() -> None:
    citation = {
        "url": "https://events.example/calendar",
        "title": "Music today",
        "snippet": "A fetched calendar.",
        "provenance": {
            "evidence_grade": "fetched",
            "source_tier": "editorial",
            "events": [
                {
                    "name": "First show",
                    "url": "https://events.example/events/first",
                    "venue": "Venue One",
                    "borough": "Brooklyn",
                    "category": "music",
                    "start_date": "2026-08-21",
                    "start_time": "18:00",
                    "end_date": "",
                    "end_time": "",
                },
                {
                    "name": "Second show",
                    "url": "https://events.example/events/second",
                    "venue": "Venue Two",
                    "borough": "Queens",
                    "category": "music",
                    "start_date": "2026-08-21",
                    "start_time": "20:00",
                    "end_date": "",
                    "end_time": "",
                },
                {
                    "name": "Outside NYC",
                    "url": "https://events.example/events/outside",
                    "venue": "Suburban Venue",
                    "borough": "Port Chester",
                    "category": "music",
                    "start_date": "2026-08-21",
                    "start_time": "20:30",
                    "end_date": "",
                    "end_time": "",
                },
                {
                    "name": "Queens neighborhood",
                    "url": "https://events.example/events/maspeth",
                    "venue": "Local venue",
                    "borough": "Maspeth",
                    "category": "music",
                    "start_date": "2026-08-21",
                    "start_time": "21:00",
                    "end_date": "",
                    "end_time": "",
                },
                {
                    "name": "Unknown locality",
                    "url": "https://events.example/events/unknown",
                    "venue": "Unknown venue",
                    "borough": "",
                    "category": "music",
                    "start_date": "2026-08-21",
                    "start_time": "21:30",
                    "end_date": "",
                    "end_time": "",
                },
            ],
        },
    }

    events = _from_web_citation_events("S4", citation, rank=1)

    assert [event.name for event in events] == [
        "First show", "Second show", "Queens neighborhood",
    ]
    assert [event.url for event in events] == [
        "https://events.example/events/first",
        "https://events.example/events/second",
        "https://events.example/events/maspeth",
    ]
    assert [event.retrieval_rank for event in events] == [1, 2, 4]
    assert all(event.citation_id == "S4" for event in events)
    assert all("music" in event.evidence_excerpt for event in events)


def test_parent_calendar_topic_does_not_reclassify_each_child_event() -> None:
    citation = {
        "url": "https://events.example/calendar",
        "title": "Music and events today",
        "snippet": "A music festival appears somewhere on this mixed calendar.",
        "provenance": {
            "source_tier": "editorial",
            "events": [{
                "name": "Comedy workshop",
                "url": "https://events.example/comedy",
                "venue": "Workshop room",
                "borough": "Brooklyn",
                "category": "comedy",
                "start_date": "2026-08-21",
                "start_time": "18:00",
                "end_date": "",
                "end_time": "",
            }],
        },
    }

    event = _from_web_citation_events("S4", citation, rank=1)[0]

    assert not _matches_keyword(event, "Music")
