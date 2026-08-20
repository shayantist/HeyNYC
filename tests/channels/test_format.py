import signal
from dataclasses import dataclass, field

import heynyc.channels.format as channel_format
from heynyc.channels.format import TWILIO_TEXT_LIMIT, WA_LIMIT, _split, render
from heynyc.core.agent import ActionLink


@dataclass
class FakeResult:
    text: str
    citations: dict = field(default_factory=dict)
    action_links: tuple[ActionLink, ...] = ()


def test_sms_places_every_cited_source_inline_with_its_claim():
    r = FakeResult(
        "Cooling centers are open {cite:S1}. Bring ID {cite:S2}.",
        {"S1": {"url": "https://nyc.gov/cool", "title": "Cooling"},
         "S2": {"url": "https://nyc.gov/id", "title": ""}},
    )
    out = render(r, "sms_twilio")
    assert len(out) == 1
    body = out[0]
    assert "Cooling centers are open (Source: https://nyc.gov/cool)." in body
    assert "Bring ID (Source: https://nyc.gov/id)." in body
    assert "Sources:" not in body


def test_no_citations_no_footer():
    out = render(FakeResult("Just a plain reply."))
    assert out == ["Just a plain reply."]


def test_converts_common_markdown_to_whatsapp_markup():
    text = (
        "**🏠 Housing**\n"
        "## Map links\n"
        "- [Morningside Heights Library](https://maps.google.com/?q=40.806,-73.965)"
    )

    assert render(FakeResult(text)) == [
        "*🏠 Housing*\n"
        "*Map links*\n"
        "- Morningside Heights Library: https://maps.google.com/?q=40.806,-73.965"
    ]


def test_preserves_code_and_urls_while_converting_other_markup():
    text = (
        "`**literal**` and ```**block**```\n"
        "* item\n"
        "~~closed~~\n"
        "https://example.com/__keep__"
    )

    assert render(FakeResult(text)) == [
        "`**literal**` and ```**block**```\n"
        "- item\n"
        "~closed~\n"
        "https://example.com/__keep__"
    ]


def test_commonmark_parser_handles_balanced_parentheses_and_reference_links():
    text = (
        "[Map](https://example.com/a_(b)) and [Details][details]\n\n"
        "[details]: https://example.com/path_(one)"
    )

    assert render(FakeResult(text), "sms_twilio") == [
        "Map: https://example.com/a_(b) and "
        "Details: https://example.com/path_(one)"
    ]


def test_commonmark_parser_respects_escaped_delimiters_and_code_precedence():
    text = r"\*literal\* and `**not bold**` and **bold**"

    assert render(FakeResult(text), "sms_twilio") == [
        "*literal* and `**not bold**` and bold"
    ]


def test_commonmark_parser_preserves_nested_lists():
    text = "- Parent\n  - First child\n  - Second child"

    assert render(FakeResult(text), "sms_twilio") == [
        "- Parent\n  - First child\n  - Second child"
    ]


def test_commonmark_parser_preserves_each_blockquote_paragraph():
    text = "> First paragraph\n>\n> Second paragraph"

    assert render(FakeResult(text), "sms_twilio") == [
        "> First paragraph\n> Second paragraph"
    ]


def test_commonmark_parser_preserves_lists_inside_blockquotes():
    text = "> - First\n> - Second"

    assert render(FakeResult(text), "sms_twilio") == [
        "> - First\n> - Second"
    ]


def test_commonmark_parser_preserves_horizontal_rules():
    assert render(FakeResult("Before\n\n---\n\nAfter"), "sms_twilio") == [
        "Before\n---\nAfter"
    ]


def test_console_channel_keeps_raw_markdown_for_rich_to_render():
    """The console (REPL) channel is the SAME content as texters, only the typography differs:
    rich renders markdown, so render() must keep the raw markdown instead of collapsing it to the
    WhatsApp dialect (its default) or stripping it (SMS). Inline {cite:Sn} markers become
    numbered links on the console and exact source links on SMS and WhatsApp. The console footer
    lists one source per line."""
    text = "**Cooling centers** are open {cite:S1}.\n## Where"
    out = render(FakeResult(text, {"S1": {"url": "https://nyc.gov/cool", "title": "Cooling"}}),
                 "console")
    body = out[0]
    assert "**Cooling centers**" in body
    assert "{cite:S1}" not in body                  # marker replaced by a clickable tag
    assert "[\\[S1\\]](<https://nyc.gov/cool>)" in body  # OSC 8 via markdown link syntax
    assert "\nSources:\n- " in body and "Cooling - <https://nyc.gov/cool>" in body  # list items survive Markdown soft-break collapse      # markdown preserved (not *bold* WhatsApp, not stripped)
    assert "## Where" in body                  # heading markdown preserved
    assert "Sources:" in body and "https://nyc.gov/cool" in body


def test_heading_preserves_a_trailing_hash_character():
    assert render(FakeResult("# C#")) == ["*C#*"]


def test_replaces_model_em_dashes_for_channel_copy():
    assert render(FakeResult("Free Yoga \N{EM DASH} Saturday")) == ["Free Yoga - Saturday"]


def test_bold_heading_has_one_whatsapp_bold_delimiter():
    assert render(FakeResult("## **Housing**")) == ["*Housing*"]


def test_whatsapp_inline_source_uses_canonical_page_for_web_citation():
    r = FakeResult(
        "Help is available {cite:S1}.",
        {
            "S1": {
                "url": "https://nyc.gov/help",
                "title": "Help",
                "snippet": "The city-run service offers help to New Yorkers today.",
                "kind": "WEB",
            },
        },
    )

    rendered = render(r)[0]
    assert "Source: https://nyc.gov/help" in rendered
    assert "#:~:text=" not in rendered


def test_sms_deduplicates_an_existing_source_and_inlines_the_missing_one():
    r = FakeResult(
        (
            "Event details: https://nyc.gov/event {cite:S1}. "
            "Air quality warning {cite:S2}."
        ),
        {
            "S1": {"url": "https://nyc.gov/event", "title": "Event"},
            "S2": {"url": "https://nyc.gov/air", "title": "Air quality"},
        },
    )

    rendered = render(r, "sms_twilio")[0]

    assert rendered.count("https://nyc.gov/event") == 1
    assert "Air quality warning (Source: https://nyc.gov/air)" in rendered
    assert "Sources:" not in rendered


def test_inline_source_url_does_not_drop_its_structured_source_note():
    limitation = (
        "These are regular hours. Confirm holiday or temporary schedule exceptions before "
        "traveling."
    )
    result = FakeResult(
        "See https://nyc.gov/help {cite:S1}.",
        {
            "S1": {
                "id": "S1",
                "url": "https://nyc.gov/help",
                "title": "Help",
                "kind": "DATA",
                "provenance": {"derivation": {"limitations": limitation}},
            },
        },
    )

    rendered = render(result, "sms_twilio")[0]

    assert rendered.count("https://nyc.gov/help") == 1
    assert limitation in rendered
    assert "Sources:" not in rendered


def test_text_channels_keep_discovery_warning_inline_without_an_endnote():
    result = FakeResult(
        "This may be available {cite:S1}.",
        {
            "S1": {
                "id": "S1",
                "url": "https://nyc.gov/search-result",
                "title": "Official [click](<https://evil.example/phish>) result",
                "kind": "WEB",
                "provenance": {
                    "evidence_grade": "search_excerpt",
                    "source_tier": "unverified",
                },
            }
        },
    )

    rendered = render(result, "sms_twilio")[0]

    assert "Source: https://nyc.gov/search-result" in rendered
    assert "search-result excerpt" in rendered
    assert "evil.example" not in rendered
    assert "Sources:" not in rendered


def test_text_channels_percent_encode_source_urls_for_clickability():
    raw = (
        "https://data.cityofnewyork.us/resource/erm2-nwe9.json?"
        "$where=unique_key='70056272' AND status='Closed'"
    )
    result = FakeResult(
        "The request is closed {cite:S1}.",
        {"S1": {"id": "S1", "url": raw, "title": "311 request", "kind": "DATA"}},
    )

    rendered = render(result, "sms_twilio")[0]

    assert " " not in rendered.split("Source: ", 1)[1].split(")", 1)[0]
    assert "%27" in rendered
    assert "Sources:" not in rendered


def test_text_channels_render_a_repeated_source_note_once():
    limitation = (
        "These are regular hours. Confirm holiday or temporary schedule exceptions before "
        "traveling."
    )
    result = FakeResult(
        "First claim {cite:S1}. Second claim {cite:S1}.",
        {
            "S1": {
                "id": "S1",
                "url": "https://nyc.gov/help",
                "title": "Help",
                "kind": "DATA",
                "provenance": {"derivation": {"limitations": limitation}},
            }
        },
    )

    rendered = render(result, "sms_twilio")[0]

    assert rendered.count(limitation) == 1


def test_text_channels_render_one_link_for_two_citations_to_the_same_page():
    result = FakeResult(
        "One supported route {cite:S1} {cite:S2}.",
        {
            "S1": {"id": "S1", "url": "https://nyc.gov/help", "title": "Help"},
            "S2": {"id": "S2", "url": "https://nyc.gov/help", "title": "Help"},
        },
    )

    rendered = render(result, "sms_twilio")[0]

    assert rendered.count("https://nyc.gov/help") == 1


def test_inline_links_deduplicate_shared_action_urls():
    action_url = "https://www.google.com/maps/dir/?api=1&destination=40.7,-73.9"
    result = FakeResult(
        "One {cite:S1}. Two {cite:S2}.",
        {
            "S1": {"id": "S1", "url": "https://nyc.gov/one", "title": "One"},
            "S2": {"id": "S2", "url": "https://nyc.gov/two", "title": "Two"},
        },
        (
            ActionLink(citation_id="S1", url=action_url),
            ActionLink(citation_id="S2", url=action_url),
        ),
    )

    assert render(result, "sms_twilio")[0].count(action_url) == 1


def test_sms_inline_sources_preserve_socrata_row_links():
    r = FakeResult(
        "One {cite:S1}. Two {cite:S2}.",
        {
            "S1": {
                "url": "https://data.cityofnewyork.us/resource/i7jb-7jku/row-1.json",
                "title": "NYC Open Data - Public Restrooms",
                "kind": "DATA",
            },
            "S2": {
                "url": "https://data.cityofnewyork.us/resource/i7jb-7jku/row-2.json",
                "title": "NYC Open Data - Public Restrooms",
                "kind": "DATA",
            },
        },
    )

    rendered = render(r, "sms_twilio")[0]
    assert "row-1.json" in rendered and "row-2.json" in rendered
    assert "Sources:" not in rendered


def test_sms_inline_sources_preserve_arcgis_row_queries():
    layer = "https://services6.arcgis.com/example/FeatureServer/0"
    r = FakeResult(
        "One {cite:S1}. Two {cite:S2}.",
        {
            "S1": {"url": f"{layer}/query?where=id%3D1", "title": "NYC Finder", "kind": "DATA"},
            "S2": {"url": f"{layer}/query?where=id%3D2", "title": "NYC Finder", "kind": "DATA"},
        },
    )

    rendered = render(r, "sms_twilio")[0]
    assert f"{layer}/query?where=id%3D1" in rendered
    assert f"{layer}/query?where=id%3D2" in rendered
    assert "Sources:" not in rendered


def test_splits_long_text_with_inline_source_on_paragraph_boundaries():
    para = "x" * 3000
    r = FakeResult(f"{para}\n\n{para} {{cite:S1}}", {"S1": {"url": "https://nyc.gov", "title": "T"}})
    out = render(r, "sms_twilio")
    assert len(out) >= 2
    assert all(len(c) <= WA_LIMIT for c in out)
    assert "Source: https://nyc.gov" in out[-1]


def test_large_sources_footer_finishes_and_respects_the_limit():
    citations = {
        f"S{i}": {
            "url": f"https://example.com/{i}/" + "x" * 80,
            "title": f"Source {i}",
        }
        for i in range(100)
    }
    def timeout(_signum, _frame):
        raise TimeoutError("render did not finish")

    previous = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, 0.5)
    try:
        result = render(FakeResult("Answer " + " ".join(
            f"{{cite:S{i}}}" for i in range(100)
        ), citations))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)

    assert all(len(chunk) <= WA_LIMIT for chunk in result)


def test_delivery_chunks_match_twilio_numbering_and_meta_limit():
    result = FakeResult(("a" * 1200) + "\n\n" + ("b" * 1200))
    delivery_chunks = getattr(channel_format, "delivery_chunks", None)

    assert delivery_chunks is not None
    expected_twilio = [f"1/2 {'a' * 1200}", f"2/2 {'b' * 1200}"]
    assert delivery_chunks(result, "sms_twilio") == expected_twilio
    assert delivery_chunks(result, "whatsapp_twilio") == expected_twilio
    assert delivery_chunks(result, "whatsapp_meta") == render(result, "whatsapp_meta")


def test_twilio_split_keeps_a_final_limitation_sentence_together():
    supported = (("Verified detail " * 95).strip() + ".")
    limitation = (
        "Unverified: I could not confirm a direct phone number. "
        "Your notice should explain the next step."
    )

    chunks = channel_format.twilio_chunks(f"{supported} {limitation}")

    assert chunks[0].endswith(".")
    assert chunks[1].startswith("2/2 Unverified:")
    assert all(len(chunk) <= TWILIO_TEXT_LIMIT for chunk in chunks)


def test_natural_split_does_not_break_after_a_phone_area_code():
    text = (
        f"{'Context ' * 7}(800) 342-3334 is the phone number. "
        f"{'Next step ' * 6}."
    )

    chunks = _split(text, 100)

    assert chunks[0].endswith("number.")
    assert chunks[1].startswith("Next step")


def test_twilio_sized_split_keeps_each_source_url_intact():
    urls = [f"https://example.com/{i}/" + "x" * 80 for i in range(30)]
    footer = "Sources:\n" + "\n".join(f"• Source {i} - {url}" for i, url in enumerate(urls))

    chunks = _split(footer, 1600)

    assert all(len(chunk) <= 1600 for chunk in chunks)
    assert all(any(url in chunk for chunk in chunks) for url in urls)


# --- channel-aware rendering --------------------------------------------------------------------

def test_sms_channel_strips_markdown_to_plain_text():
    text = (
        "**🏠 Housing**\n"
        "## Map links\n"
        "- [Morningside Library](https://maps.google.com/?q=40.8,-73.9)\n"
        "~~old~~ info"
    )

    assert render(FakeResult(text), "sms_twilio") == [
        "🏠 Housing\n"
        "Map links\n"
        "- Morningside Library: https://maps.google.com/?q=40.8,-73.9\n"
        "old info"
    ]


def test_whatsapp_channel_keeps_native_bold_markup():
    # The explicit WhatsApp channel and the default both keep native *bold*.
    assert render(FakeResult("**Housing**"), "whatsapp_twilio") == ["*Housing*"]
    assert render(FakeResult("**Housing**")) == ["*Housing*"]


def test_whatsapp_places_every_cited_source_inline_with_its_claim():
    result = FakeResult(
        "- Art in the Park {cite:S1}\n- The Dancing Men {cite:S2}",
        {
            "S1": {"id": "S1", "url": "https://nyc.gov/art", "title": "Art"},
            "S2": {"id": "S2", "url": "https://nyc.gov/dancing", "title": "Dancing"},
        },
        (ActionLink(citation_id="S2", url="https://maps.google.com/dancing"),),
    )

    rendered = render(result, "whatsapp_meta")[0]

    assert "- Art in the Park (Source: https://nyc.gov/art)" in rendered
    assert (
        "- The Dancing Men (Source: https://nyc.gov/dancing; "
        "Directions: https://maps.google.com/dancing)"
    ) in rendered
    assert "Sources:" not in rendered


def test_whatsapp_does_not_confuse_a_longer_lookalike_url_for_the_cited_source():
    result = FakeResult(
        "- Event https://nyc.gov/event-extra {cite:S1}",
        {"S1": {"id": "S1", "url": "https://nyc.gov/event", "title": "Event"}},
    )

    rendered = render(result, "whatsapp_meta")[0]

    assert "https://nyc.gov/event-extra" in rendered
    assert "Source: https://nyc.gov/event" in rendered


def test_text_channels_replace_a_citation_used_as_a_markdown_link_target():
    result = FakeResult(
        "Event: [Details]({cite:S1})",
        {"S1": {"id": "S1", "url": "https://nyc.gov/event", "title": "Event"}},
    )

    for channel in ("sms_twilio", "whatsapp_meta"):
        rendered = render(result, channel)[0]
        assert "Details: https://nyc.gov/event" in rendered
        assert "Sources:" not in rendered


def test_text_channels_recognize_an_existing_url_with_balanced_parentheses():
    url = "https://example.gov/a_(b)"
    result = FakeResult(
        f"Event details: {url} {{cite:S1}}",
        {"S1": {"id": "S1", "url": url, "title": "Event"}},
    )

    for channel in ("sms_twilio", "whatsapp_meta"):
        rendered = render(result, channel)[0]
        assert rendered.count(url) == 1
        assert "Sources:" not in rendered


def test_text_channels_emit_an_identical_source_limit_once() -> None:
    limitation = (
        "These are regular hours. Confirm holiday or temporary schedule exceptions before "
        "traveling."
    )
    result = FakeResult(
        "First schedule {cite:S1}. Second schedule {cite:S2}.",
        {
            "S1": {
                "id": "S1",
                "url": "https://nyc.gov/hours",
                "title": "Hours",
                "provenance": {"derivation": {"limitations": limitation}},
            },
            "S2": {
                "id": "S2",
                "url": "https://nyc.gov/hours",
                "title": "Hours",
                "provenance": {"derivation": {"limitations": limitation}},
            },
        },
    )

    rendered = "\n".join(render(result, "sms_twilio"))

    assert rendered.count(limitation) == 1


def test_text_channels_do_not_turn_code_literal_markers_into_citations():
    result = FakeResult(
        "Use `{cite:S1}` literally.\n\n```\n{cite:S2}\n```",
        {
            "S1": {"id": "S1", "url": "https://nyc.gov/one", "title": "One"},
            "S2": {"id": "S2", "url": "https://nyc.gov/two", "title": "Two"},
        },
    )

    for channel in ("sms_twilio", "whatsapp_meta"):
        rendered = render(result, channel)[0]
        assert "{cite:" not in rendered
        assert "https://nyc.gov" not in rendered
        assert "Sources:" not in rendered


def test_body_preserves_exact_cited_socrata_row_url():
    permalink = "https://data.cityofnewyork.us/resource/tvpp-9vvx/abc123.json"
    r = FakeResult(
        f"- Street Fair, Saturday {{cite:S1}}\n  Details: {permalink}",
        {"S1": {"url": permalink, "title": "Street Fair", "kind": "DATA"}},
    )

    for channel in ("sms_twilio", "whatsapp_meta"):
        joined = "\n".join(render(r, channel))
        answer = joined.split("Sources:")[0]
        assert answer.count(permalink) == 1
        assert "/d/tvpp-9vvx" not in answer
    # the stored citation record is never rewritten
    assert r.citations["S1"]["url"] == permalink


def test_body_does_not_repair_malformed_brace_wrapped_link():
    permalink = "https://data.cityofnewyork.us/resource/tvpp-9vvx/abc.json"
    r = FakeResult(
        f"See [Details]({{{permalink}}}) {{cite:S1}}",  # [Details]({<permalink>}) observed live
        {"S1": {"url": permalink, "title": "T", "kind": "DATA"}},
    )

    answer = render(r, "sms_twilio")[0].split("Sources:")[0]
    assert "Details: %7Bhttps://data.cityofnewyork.us/resource/tvpp-9vvx/abc.json%7D" in answer


def test_body_does_not_invent_links_from_malformed_markdown():
    event_url = "https://www.nycgovparks.org/events/2026/07/26/open-run"
    map_url = "https://www.google.com/maps/search/?api=1&query=40.7,-73.8"
    source_url = "https://services.example.gov/FeatureServer/0/query?where=OBJECTID%3D1"
    r = FakeResult(
        "Event: [Details]({cite:S1}\n"
        f"  Details: {event_url})\n"
        "Place: [Map]({cite:S2}\n"
        f"  Directions: {map_url}\n"
        f"  Details: {source_url})",
        {
            "S1": {"url": event_url, "title": "Open Run", "kind": "WEB"},
            "S2": {"url": source_url, "title": "Cooling center", "kind": "DATA"},
        },
    )

    for channel in ("sms_twilio", "whatsapp_twilio"):
        answer = "\n".join(render(r, channel)).split("Sources:")[0]
        assert f"Details: {event_url}" in answer
        assert f"Directions: {map_url}" in answer
        assert f"Details: {source_url}" in answer
        assert "{cite:" not in answer
        assert "[Details](" in answer
        assert "[Map](" in answer

    console = "\n".join(render(r, "console"))
    assert "[Details](" in console
    assert "[Map](" in console
    assert source_url in console


def test_body_does_not_repair_unknown_or_mismatched_citation_target_links():
    safe_url = "https://www.nycgovparks.org/events/safe"
    attacker_url = "https://attacker.example/phish"
    r = FakeResult(
        "[Details]({cite:S1}\n"
        f"  Details: {attacker_url})\n"
        "[Map]({cite:S9}\n"
        f"  Directions: {attacker_url})",
        {"S1": {"url": safe_url, "title": "Safe event", "kind": "WEB"}},
    )

    answer = "\n".join(render(r, "sms_twilio")).split("Sources:")[0]

    assert "Details: https://attacker.example/phish" in answer
    assert "[Details](" in answer
    assert "[Map](" in answer


def test_render_is_presentation_only_and_preserves_the_audit_record():
    permalink = "https://data.cityofnewyork.us/resource/tvpp-9vvx/abc.json"
    r = FakeResult(
        f"**Street Fair** is Saturday {{cite:S1}}. Details: {permalink}",
        {"S1": {"url": permalink, "title": "Street Fair", "kind": "DATA"}},
    )
    original_text, original_citations = r.text, {k: dict(v) for k, v in r.citations.items()}

    render(r, "sms_twilio")
    render(r, "whatsapp_meta")

    assert r.text == original_text                 # raw generation untouched by rendering
    assert r.citations == original_citations       # citation mapping (permalinks) untouched
    assert "{cite:S1}" in r.text                   # audit markers survive
    assert r.citations["S1"]["url"] == permalink   # row-addressed permalink stays in the record
