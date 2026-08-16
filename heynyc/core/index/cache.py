"""Cache embedded catalog stores in memory and, when configured, LanceDB.

Embedding the same catalog on every call is wasteful; this memoizes the built
`InMemoryVectorStore` keyed by (hash of the embedded texts, embedder model id).

The KEY DESIGN is the established embedding-cache pattern, `hash(text)` namespaced
by the model id, used by LangChain's `CacheBackedEmbeddings` and Redis `EmbeddingsCache`:
embeddings are NOT content-addressed, so a model/tokenizer change yields different vectors
for identical text and the model id must be part of the key (spec §14). We reuse that
pattern plus our own `Embedder`/`InMemoryVectorStore` rather than pulling in LangChain or
Redis, disproportionate dependencies for a framework-light project and a tiny catalog.

Configured runtimes use the existing LanceDB store so unchanged catalogs survive restarts.
Tests and callers without a path retain the in-memory backend.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from .embedder import Embedder
from .store import IndexDoc, InMemoryVectorStore, LanceVectorStore, VectorStore

_CACHE: dict[tuple[str, str], VectorStore] = {}


def _key(docs: list[IndexDoc], model_id: str) -> str:
    h = hashlib.md5()
    for doc in docs:
        for value in (doc.id, doc.title, doc.text):
            h.update(value.encode("utf-8"))
            h.update(b"\x00")
    return f"{model_id}:{h.hexdigest()}"


def embedded_store(
    docs: list[IndexDoc],
    embedder: Embedder,
    *,
    path: Path | None = None,
) -> VectorStore:
    """Embed `docs` once per content hash and model, optionally persisted in LanceDB."""
    key = _key(docs, embedder.model_id)
    cache_key = (key, str(path.resolve()) if path is not None else "memory")
    cached = _CACHE.get(cache_key)
    if cached is not None:
        return cached
    if path is not None:
        table_name = f"catalog_{hashlib.sha256(key.encode()).hexdigest()[:24]}"
        store: VectorStore = LanceVectorStore(
            path,
            table_name=table_name,
            model_id=embedder.model_id,
        )
        if store.count():
            _CACHE[cache_key] = store
            return store
    else:
        store = InMemoryVectorStore()
    for doc, vector in zip(docs, embedder.embed([d.text for d in docs])):
        doc.vector = vector
    store.add(docs)
    _CACHE[cache_key] = store
    return store


def clear_cache() -> None:
    _CACHE.clear()
