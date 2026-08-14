import json

from heynyc.eval.fetch_selection import (
    collect_cases,
    collect_tavily_extracts,
    compare_case,
    write_comparison,
)


def test_compare_case_reports_recall_noise_duplicates_and_tokens() -> None:
    case = {
        "id": "library-hours",
        "query": "hours address phone",
        "text": (
            "Navigation programs donate " * 90
            + "Sunset Park Library hours Wednesday 10 am to 6 pm. "
            + "Address 5108 Fourth Avenue. Phone 718-230-2255."
        ),
        "required_passages": [
            "Wednesday 10 am to 6 pm",
            "5108 Fourth Avenue",
            "718-230-2255",
        ],
        "tavily_chunks": [
            "Sunset Park Library hours Wednesday 10 am to 6 pm. "
            "Address 5108 Fourth Avenue. Phone 718-230-2255."
        ],
    }

    result = compare_case(case, model="openai/gpt-5.6-luna", full_token_budget=50)

    assert result["case_id"] == "library-hours"
    assert set(result["candidates"]) == {
        "current",
        "adaptive",
        "full",
        "tavily_basic",
    }
    assert result["candidates"]["adaptive"]["required_passage_recall"] == 1.0
    assert result["candidates"]["current"]["required_passage_recall"] == 1.0
    assert result["candidates"]["full"]["eligible"] is False
    assert result["candidates"]["tavily_basic"]["required_passage_recall"] == 1.0
    assert result["candidates"]["tavily_basic"]["required_text_density"] > 0.5
    assert result["candidates"]["tavily_basic"]["duplicate_ratio"] == 0.0
    assert result["candidates"]["tavily_basic"]["tokens"] > 0
    assert result["candidates"]["current"]["selection_ms"] >= 0
    assert result["candidates"]["current"]["external_cost_usd"] == 0.0


def test_compare_case_normalizes_extracted_line_breaks_for_recall() -> None:
    result = compare_case(
        {
            "id": "pdf-line-wrap",
            "query": "open door immigration",
            "text": "You do not have to open the door for ICE or\nimmigration.",
            "required_passages": [
                "You do not have to open the door for ICE or immigration",
            ],
        },
        model="openai/gpt-5.6-luna",
        full_token_budget=100,
    )

    assert result["candidates"]["full"]["required_passage_recall"] == 1.0


def test_write_comparison_preserves_the_complete_report(tmp_path) -> None:
    source = tmp_path / "cases.json"
    output = tmp_path / "report.json"
    source.write_text(json.dumps([{
        "id": "short-page",
        "query": "hours",
        "text": "Hours are Monday through Friday.",
        "required_passages": ["Monday through Friday"],
    }]))

    write_comparison(
        source,
        output,
        model="openai/gpt-5.6-luna",
        full_token_budget=100,
    )

    report = json.loads(output.read_text())
    assert report["model"] == "openai/gpt-5.6-luna"
    assert report["full_token_budget"] == 100
    assert report["cases"][0]["candidates"]["full"]["eligible"] is True


async def test_collect_cases_preserves_cleaned_preselection_text(
    tmp_path,
    monkeypatch,
) -> None:
    spec = tmp_path / "spec.json"
    output = tmp_path / "cases.json"
    spec.write_text(json.dumps([{
        "id": "library",
        "url": "https://example.com/library",
        "query": "hours",
        "required_passages": ["Open until 8 PM"],
    }]))

    async def fetch(url, client, query):
        assert client is None
        return url, "Library", "Navigation. Open until 8 PM."

    monkeypatch.setattr(
        "heynyc.eval.fetch_selection._fetch_page_with_browser",
        fetch,
    )

    await collect_cases(spec, output)

    cases = json.loads(output.read_text())
    assert cases[0]["title"] == "Library"
    assert cases[0]["text"] == "Navigation. Open until 8 PM."


async def test_collect_tavily_extracts_adds_query_focused_chunks(
    tmp_path,
) -> None:
    source = tmp_path / "cases.json"
    output = tmp_path / "tavily.json"
    source.write_text(json.dumps([{
        "id": "snap-spanish",
        "url": "https://example.com/snap",
        "query": "como solicito ayuda para comprar comida",
        "text": "Existing local extraction.",
    }]))

    async def extract(url, query):
        assert url == "https://example.com/snap"
        assert query == "como solicito ayuda para comprar comida"
        return ["SNAP applications are available through ACCESS HRA."]

    await collect_tavily_extracts(source, output, extract_fn=extract)

    cases = json.loads(output.read_text())
    assert cases[0]["text"] == "Existing local extraction."
    assert cases[0]["tavily_chunks"] == [
        "SNAP applications are available through ACCESS HRA."
    ]
