from heynyc.core.index import cache as index_cache
from heynyc.core.index.embedder import HashEmbedder
from heynyc.core.index.store import IndexDoc


def test_embedded_store_reuses_a_persistent_lance_table_after_restart(tmp_path):
    class CountingEmbedder(HashEmbedder):
        def __init__(self):
            super().__init__(dim=64)
            self.calls = 0

        def embed(self, texts):
            self.calls += 1
            return super().embed(texts)

    docs = [IndexDoc(id="snap", text="food stamps snap grocery")]
    first = CountingEmbedder()
    index_cache.embedded_store(docs, first, path=tmp_path / "catalogs.lance")
    assert first.calls == 1

    index_cache.clear_cache()
    second = CountingEmbedder()
    store = index_cache.embedded_store(
        [IndexDoc(id="snap", text="food stamps snap grocery")],
        second,
        path=tmp_path / "catalogs.lance",
    )

    assert second.calls == 0
    assert store.count() == 1


def test_embedded_store_does_not_reuse_memory_for_a_requested_lance_path(tmp_path):
    embedder = HashEmbedder(dim=64)
    docs = [IndexDoc(id="snap", text="food stamps snap grocery")]
    index_cache.clear_cache()

    memory = index_cache.embedded_store(docs, embedder)
    persistent = index_cache.embedded_store(
        [IndexDoc(id="snap", text="food stamps snap grocery")],
        embedder,
        path=tmp_path / "catalogs.lance",
    )

    assert persistent is not memory
    assert (tmp_path / "catalogs.lance").exists()
