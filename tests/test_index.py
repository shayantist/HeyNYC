from __future__ import annotations

import sys
from types import SimpleNamespace

import httpx

from heynyc.core.citations import CitationRegistry
from heynyc.core.index import IndexRetriever
from heynyc.core.index.corpus import build_index, chunk_text, clean_html
from heynyc.core.index.embedder import FastEmbedEmbedder, HashEmbedder
from heynyc.core.index.store import IndexDoc, InMemoryVectorStore, LanceVectorStore
from heynyc.core.manifest import ServiceModule
from heynyc.core.registry import Registry
from heynyc.core.tools.base import ToolContext
from heynyc.core.tools.index_search import index_search_tools


def test_clean_html_extracts_title_and_strips_tags():
    html = "<html><head><title>Cooling Centers</title></head><body><script>x=1</script><p>Stay <b>cool</b> this summer.</p></body></html>"
    title, text = clean_html(html)
    assert title == "Cooling Centers"
    assert "Stay cool this summer." in text
    assert "x=1" not in text


def test_chunk_text_overlaps_and_covers():
    text = "word " * 1000  # ~5000 chars
    chunks = chunk_text(text, max_chars=1200, overlap=150)
    assert len(chunks) > 1
    assert all(len(c) <= 1200 for c in chunks)
    # short text returns a single chunk
    assert chunk_text("short") == ["short"]
    assert chunk_text("   ") == []


def test_hash_embedder_deterministic_and_normalized():
    e = HashEmbedder(dim=64)
    v1 = e.embed(["cooling center near me"])[0]
    v2 = e.embed(["cooling center near me"])[0]
    assert v1 == v2
    assert abs(sum(x * x for x in v1) ** 0.5 - 1.0) < 1e-9


def test_fastembed_uses_a_memory_bounded_batch(monkeypatch):
    seen = {}

    class FakeTextEmbedding:
        def __init__(self, **kwargs):
            pass

        def embed(self, texts, **kwargs):
            seen.update(kwargs)
            return [[1.0] for _ in texts]

    monkeypatch.setitem(sys.modules, "fastembed", SimpleNamespace(TextEmbedding=FakeTextEmbedding))

    assert FastEmbedEmbedder().embed(["one", "two"]) == [[1.0], [1.0]]
    assert seen == {"batch_size": 32}


def test_inmemory_store_ranks_relevant_first():
    e = HashEmbedder(dim=128)
    store = InMemoryVectorStore()
    docs = [
        IndexDoc(id="1", text="Cooling centers help you beat the summer heat.", title="Heat", url="u1"),
        IndexDoc(id="2", text="Apply for a reduced-fare MetroCard with Fair Fares.", title="Transit", url="u2"),
    ]
    for d in docs:
        d.vector = e.embed([f"{d.title} {d.text}"])[0]
    store.add(docs)
    qv = e.embed(["where can I cool off in the heat?"])[0]
    results = store.search(qv, "where can I cool off in the heat?", k=2)
    assert results[0][0].id == "1"  # cooling doc ranks above transit doc
    assert store.count() == 2


async def test_index_search_tool_cites_doc():
    e = HashEmbedder(dim=128)
    store = InMemoryVectorStore()
    d = IndexDoc(id="1", text="NYC has many free things to do in summer.", title="Free NYC", url="https://nyctourism.com/free")
    d.vector = e.embed([f"{d.title} {d.text}"])[0]
    store.add([d])
    retriever = IndexRetriever(store=store, embedder=e)
    tool = index_search_tools(retriever)[0]

    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]))
    out = await tool.handler({"query": "free things to do"}, ctx)
    assert "Free NYC" in out
    assert ctx.citations.mapping()["S1"]["kind"] == "DOC"
    assert ctx.citations.mapping()["S1"]["url"] == "https://nyctourism.com/free"


async def test_index_search_tool_empty():
    retriever = IndexRetriever(store=InMemoryVectorStore(), embedder=HashEmbedder(dim=64))
    tool = index_search_tools(retriever)[0]
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]))
    out = await tool.handler({"query": "anything"}, ctx)
    assert "No indexed" in out


async def test_build_index_reports_failures_without_publishing_a_partial_corpus():
    module = ServiceModule(name="things_to_do", category="tourism", seeds=["https://example.test/a", "https://dead.test/b"])
    registry = Registry([module])

    def handler(request: httpx.Request) -> httpx.Response:
        if "dead" in request.url.host:
            return httpx.Response(500)
        body = "<title>Things To Do</title><body>" + ("fun stuff in nyc " * 200) + "</body>"
        return httpx.Response(200, text=body)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    store = InMemoryVectorStore()
    summary = await build_index(registry, store, HashEmbedder(dim=64), client=client)
    await client.aclose()

    assert summary["ok"] == 1
    assert len(summary["failed"]) == 1  # the dead seed
    assert summary["chunks"] >= 1
    assert store.count() == 0


async def test_build_index_replaces_the_existing_persistent_corpus(tmp_path):
    url = "https://example.test/current"
    registry = Registry([ServiceModule(name="current", seeds=[url])])
    client = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda _request: httpx.Response(200, text="<title>Current</title><p>Current guidance.</p>")
    ))
    store = LanceVectorStore(tmp_path / "index.lance")

    first = await build_index(registry, store, HashEmbedder(dim=64), client=client)
    second = await build_index(registry, store, HashEmbedder(dim=64), client=client)
    await client.aclose()

    assert first["chunks"] == second["chunks"] == 1
    assert store.count() == 1


async def test_build_index_preserves_existing_corpus_when_any_seed_fails():
    current = "https://example.test/current"
    dead = "https://example.test/dead"
    registry = Registry([ServiceModule(name="current", seeds=[current, dead])])

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == dead:
            return httpx.Response(503)
        return httpx.Response(200, text="<title>Current</title><p>New guidance.</p>")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    embedder = HashEmbedder(dim=64)
    old = IndexDoc(id="old", text="Prior complete guidance", vector=embedder.embed(["prior"])[0])
    store = InMemoryVectorStore()
    store.add([old])

    summary = await build_index(registry, store, embedder, client=client)
    await client.aclose()

    assert len(summary["failed"]) == 1
    assert store.search(embedder.embed(["prior"])[0], "prior", k=1)[0][0].id == "old"


async def test_build_index_preserves_existing_corpus_when_every_seed_fails():
    url = "https://example.test/dead"
    registry = Registry([ServiceModule(name="current", seeds=[url])])
    client = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda _request: httpx.Response(503)
    ))
    store = InMemoryVectorStore()
    store.add([IndexDoc(id="old", text="Prior complete guidance", vector=[1.0])])

    summary = await build_index(registry, store, HashEmbedder(dim=64), client=client)
    await client.aclose()

    assert len(summary["failed"]) == 1
    assert store.count() == 1


async def test_build_index_preserves_existing_corpus_when_embedding_fails():
    class FailingEmbedder(HashEmbedder):
        def embed(self, texts):
            raise RuntimeError("embedding failed")

    url = "https://example.test/current"
    registry = Registry([ServiceModule(name="current", seeds=[url])])
    client = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda _request: httpx.Response(200, text="<p>Current guidance.</p>")
    ))
    store = InMemoryVectorStore()
    store.add([IndexDoc(id="old", text="Prior complete guidance", vector=[1.0])])

    try:
        await build_index(registry, store, FailingEmbedder(dim=64), client=client)
    except RuntimeError as exc:
        assert str(exc) == "embedding failed"
    else:
        raise AssertionError("embedding failure should propagate")
    finally:
        await client.aclose()

    assert store.count() == 1


async def test_build_index_rejects_truncated_embedding_output():
    class TruncatingEmbedder(HashEmbedder):
        def embed(self, texts):
            return super().embed(texts[:1])

    url = "https://example.test/current"
    registry = Registry([ServiceModule(name="current", seeds=[url])])
    body = "<p>" + ("Current detailed guidance. " * 200) + "</p>"
    client = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda _request: httpx.Response(200, text=body)
    ))
    store = InMemoryVectorStore()
    store.add([IndexDoc(id="old", text="Prior complete guidance", vector=[1.0])])

    try:
        await build_index(registry, store, TruncatingEmbedder(dim=64), client=client)
    except RuntimeError as exc:
        assert str(exc) == "embedder returned 1 vector(s) for 6 chunk(s)"
    else:
        raise AssertionError("truncated embedding output should fail closed")
    finally:
        await client.aclose()

    assert store.count() == 1


async def test_build_index_preserves_existing_corpus_when_replace_fails():
    class FailingReplaceStore(InMemoryVectorStore):
        def replace(self, docs):
            raise RuntimeError("replace failed")

    url = "https://example.test/current"
    registry = Registry([ServiceModule(name="current", seeds=[url])])
    client = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda _request: httpx.Response(200, text="<p>Current guidance.</p>")
    ))
    store = FailingReplaceStore()
    store.add([IndexDoc(id="old", text="Prior complete guidance", vector=[1.0])])

    try:
        await build_index(registry, store, HashEmbedder(dim=64), client=client)
    except RuntimeError as exc:
        assert str(exc) == "replace failed"
    else:
        raise AssertionError("replace failure should propagate")
    finally:
        await client.aclose()

    assert store.count() == 1


def test_embedders_expose_model_id():
    # The embed cache keys on the embedder identity (embeddings aren't content-addressed).
    emb = HashEmbedder(dim=256)
    assert emb.model_id == "hash:256"
    assert HashEmbedder(dim=128).model_id == "hash:128"


def test_index_search_can_emit_machine_readable_urls(monkeypatch, capsys):
    import heynyc.__main__ as cli

    docs = [
        IndexDoc(id="1", text="first", title="First", url="https://example.test/one"),
        IndexDoc(id="2", text="second", title="Second", url="https://example.test/two"),
    ]

    class Retriever:
        def search(self, query, k):
            assert query == "probe"
            assert k == 5
            return [(docs[0], 0.9), (docs[1], 0.8)]

    monkeypatch.setattr(cli, "_load_retriever", lambda required: Retriever())

    cli._cmd_index_search("probe", urls_only=True)

    assert capsys.readouterr().out.splitlines() == [
        "https://example.test/one",
        "https://example.test/two",
    ]


def test_embedded_store_memoizes_by_content_and_model():
    from heynyc.core.index import cache as idxcache

    idxcache.clear_cache()

    class CountingEmbedder(HashEmbedder):
        def __init__(self):
            super().__init__(dim=64)
            self.calls = 0

        def embed(self, texts):
            self.calls += 1
            return super().embed(texts)

    emb = CountingEmbedder()
    docs = lambda: [IndexDoc(id="1", text="food stamps snap"), IndexDoc(id="2", text="rent help drie")]

    store1 = idxcache.embedded_store(docs(), emb)
    store2 = idxcache.embedded_store(docs(), emb)  # same texts+model → cache hit
    assert emb.calls == 1                          # embedded only once
    assert store1 is store2
    assert store1.count() == 2

    idxcache.embedded_store([IndexDoc(id="3", text="something else")], emb)  # new text → re-embed
    assert emb.calls == 2


def test_embedded_store_is_searchable():
    from heynyc.core.index import cache as idxcache

    idxcache.clear_cache()
    emb = HashEmbedder(dim=64)
    docs = [IndexDoc(id="snap", text="food stamps snap grocery"),
            IndexDoc(id="rent", text="rent freeze drie housing")]
    store = idxcache.embedded_store(docs, emb)
    hits = store.search(emb.embed(["help buying food"])[0], "help buying food", k=2)
    assert hits[0][0].id == "snap"  # food query ranks the food doc first


def test_bm25_idf_prefers_rare_term_over_common():
    from heynyc.core.index.store import _bm25_scores

    docs = [
        ["apply", "for", "snap", "benefits"],          # has the rare term "snap"
        ["apply", "for", "the", "program", "today"],   # only common terms, no "snap"
        ["the", "weather", "outside", "is", "nice"],
    ]
    scores = _bm25_scores(["snap"], docs)
    assert scores[0] > scores[1]    # the doc with the query term wins
    assert scores[1] == 0.0         # no "snap" term → zero
    assert scores[2] == 0.0


def test_rrf_is_scale_agnostic():
    from heynyc.core.index.store import _rrf_fuse

    # dense favours doc0, sparse (BM25-scale) favours doc2; doc1 is second in both.
    dense = [0.91, 0.55, 0.10]
    sparse = [0.2, 1.0, 8.7]        # different magnitude entirely — RRF ignores it
    fused = _rrf_fuse(dense, sparse)
    assert fused[0] > fused[1]      # rank-0 in dense beats rank-1-in-both
    assert fused[2] > fused[1]      # rank-0 in sparse beats rank-1-in-both
