from heynyc.core.citations import CitationRegistry
from heynyc.core.prompts import BASE_SYSTEM_PROMPT
from heynyc.core.registry import Registry
from heynyc.core.tools import web_fetch as web_fetch_module
from heynyc.core.tools.base import ToolContext
from heynyc.core.tools.web_fetch import web_fetch_tools
from heynyc.core.tools.web_search import web_search_tools


async def test_web_fetch_labels_full_page_evidence(monkeypatch):
    url = "https://example.com/enrollment"

    async def fetched(_url, _client, *, render=False):
        assert render is False
        return web_fetch_module._FetchedPage(
            final_url=url,
            title="Enrollment",
            text="Call the enrollment office for an appointment.",
            acquisition=web_fetch_module._acquisition(url, url, "http"),
        )

    monkeypatch.setattr(web_fetch_module, "_fetch_page_with_browser", fetched)
    monkeypatch.setattr(web_fetch_module, "_text_tokens", lambda _text, _model=None: 20)
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]))

    result = await web_fetch_tools()[0].handler(
        {"url": url},
        ctx,
    )

    assert "CONTENT SCOPE: numbered extracted page" in result


def test_retrieval_tools_explain_constraint_and_refetch_boundaries():
    search = web_search_tools([])[0]
    fetch = web_fetch_tools()[0]

    query_description = search._input_schema()["properties"]["queries"]["description"]
    assert "independent focused searches" in query_description.lower()
    assert set(fetch._input_schema()["properties"]) == {"url", "find"}
    assert "numbered evidence" in fetch.description


def test_answer_contract_requires_hours_or_an_explicit_limit():
    assert "calling or visiting at a particular time" in BASE_SYSTEM_PROMPT
    assert "say that plainly" in BASE_SYSTEM_PROMPT
    assert "systemwide statement supports a location only" in BASE_SYSTEM_PROMPT
    assert "current high-stakes fact" in BASE_SYSTEM_PROMPT
