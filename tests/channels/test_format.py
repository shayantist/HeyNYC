from dataclasses import dataclass, field
import signal
from heynyc.channels.format import _split, render, WA_LIMIT


@dataclass
class FakeResult:
    text: str
    citations: dict = field(default_factory=dict)


def test_strips_markers_and_appends_sources():
    r = FakeResult(
        "Cooling centers are open {cite:S1}. Bring ID {cite:S2}.",
        {"S1": {"url": "https://nyc.gov/cool", "title": "Cooling"},
         "S2": {"url": "https://nyc.gov/id", "title": ""}},
    )
    out = render(r)
    assert len(out) == 1
    body = out[0]
    assert "{cite:" not in body
    assert "Cooling centers are open. Bring ID." in body
    assert "Sources:" in body
    assert "https://nyc.gov/cool" in body and "https://nyc.gov/id" in body


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


def test_console_channel_keeps_raw_markdown_for_rich_to_render():
    """The console (REPL) channel is the SAME content as texters, only the typography differs:
    rich renders markdown, so render() must keep the raw markdown instead of collapsing it to the
    WhatsApp dialect (its default) or stripping it (SMS). Cite markers are still removed."""
    text = "**Cooling centers** are open {cite:S1}.\n## Where"
    out = render(FakeResult(text, {"S1": {"url": "https://nyc.gov/cool", "title": "Cooling"}}),
                 "console")
    body = out[0]
    assert "**Cooling centers**" in body      # markdown preserved (not *bold* WhatsApp, not stripped)
    assert "## Where" in body                  # heading markdown preserved
    assert "{cite:" not in body                # internal markers still stripped
    assert "Sources:" in body and "https://nyc.gov/cool" in body


def test_heading_preserves_a_trailing_hash_character():
    assert render(FakeResult("# C#")) == ["*C#*"]


def test_replaces_model_em_dashes_for_channel_copy():
    assert render(FakeResult("Free Yoga \N{EM DASH} Saturday")) == ["Free Yoga - Saturday"]


def test_bold_heading_has_one_whatsapp_bold_delimiter():
    assert render(FakeResult("## **Housing**")) == ["*Housing*"]


def test_sources_footer_uses_canonical_page_for_web_citation():
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

    footer = render(r)[0]
    assert "https://nyc.gov/help" in footer
    assert "#:~:text=" not in footer


def test_sources_footer_omits_links_already_shown_in_the_answer():
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

    rendered = render(r)[0]

    assert rendered.count("https://nyc.gov/event") == 1
    assert "Air quality - https://nyc.gov/air" in rendered


def test_sources_footer_preserves_socrata_row_links():
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

    footer = render(r)[0]
    assert footer.count("NYC Open Data - Public Restrooms") == 2
    assert "row-1.json" in footer and "row-2.json" in footer


def test_sources_footer_compacts_arcgis_row_queries_to_one_layer_link():
    layer = "https://services6.arcgis.com/example/FeatureServer/0"
    r = FakeResult(
        "One {cite:S1}. Two {cite:S2}.",
        {
            "S1": {"url": f"{layer}/query?where=id%3D1", "title": "NYC Finder", "kind": "DATA"},
            "S2": {"url": f"{layer}/query?where=id%3D2", "title": "NYC Finder", "kind": "DATA"},
        },
    )

    footer = render(r)[0]
    assert footer.count("NYC Finder") == 1
    assert f"NYC Finder - {layer}" in footer
    assert "/query?where=" not in footer


def test_splits_long_text_on_paragraph_boundaries_footer_last():
    para = "x" * 3000
    r = FakeResult(f"{para}\n\n{para} {{cite:S1}}", {"S1": {"url": "https://nyc.gov", "title": "T"}})
    out = render(r)
    assert len(out) >= 2
    assert all(len(c) <= WA_LIMIT for c in out)
    assert "Sources:" in out[-1] and "Sources:" not in out[0]


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


def test_body_replaces_cited_socrata_row_json_with_official_dataset_page():
    permalink = "https://data.cityofnewyork.us/resource/tvpp-9vvx/abc123.json"
    r = FakeResult(
        f"- Street Fair, Saturday {{cite:S1}}\n  Details: {permalink}",
        {"S1": {"url": permalink, "title": "Street Fair", "kind": "DATA"}},
    )

    for channel in ("sms_twilio", "whatsapp_meta"):
        joined = "\n".join(render(r, channel))
        answer = joined.split("Sources:")[0]
        assert "https://data.cityofnewyork.us/d/tvpp-9vvx" in answer  # official page in the body
        assert permalink not in answer                               # raw JSON permalink gone
        assert "abc123.json" in joined                               # footer keeps the row permalink
    # the stored citation record is never rewritten
    assert r.citations["S1"]["url"] == permalink


def test_body_debraces_stray_brace_wrapped_link():
    permalink = "https://data.cityofnewyork.us/resource/tvpp-9vvx/abc.json"
    r = FakeResult(
        f"See [Details]({{{permalink}}}) {{cite:S1}}",  # [Details]({<permalink>}) observed live
        {"S1": {"url": permalink, "title": "T", "kind": "DATA"}},
    )

    answer = render(r, "sms_twilio")[0].split("Sources:")[0]
    assert "{" not in answer and "}" not in answer
    assert "Details: https://data.cityofnewyork.us/d/tvpp-9vvx" in answer


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
