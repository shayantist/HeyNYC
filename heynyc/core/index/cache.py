"""In-process cache of embedded vector stores.

Embedding the same catalog on every call is wasteful; this memoizes the built
`InMemoryVectorStore` keyed by (hash of the embedded texts, embedder model id).

The KEY DESIGN is the established embedding-cache pattern — `hash(text)` namespaced
by the model id — used by LangChain's `CacheBackedEmbeddings` and Redis `EmbeddingsCache`:
embeddings are NOT content-addressed, so a model/tokenizer change yields different vectors
for identical text and the model id must be part of the key (spec §14). We reuse that
pattern plus our own `Embedder`/`InMemoryVectorStore` rather than pulling in LangChain or
Redis — disproportionate dependencies for a framework-light project and a tiny catalog.

On-disk persistence (survive restarts) is the documented next step and should reuse our
existing LanceDB store (`core/index/store.py`); in-process covers the hot path (a
long-running process / an eval run) at MVP scale.
"""
from __future__ import annotations

import hashlib

from .embedder import Embedder
from .store import IndexDoc, InMemoryVectorStore

_CACHE: dict[str, InMemoryVectorStore] = {}


def _key(docs: list[IndexDoc], model_id: str) -> str:
    h = hashlib.md5()
    for doc in docs:
        h.update(doc.text.encode("utf-8"))
        h.update(b"\x00")
    return f"{model_id}:{h.hexdigest()}"


def embedded_store(docs: list[IndexDoc], embedder: Embedder) -> InMemoryVectorStore:
    """Embed `docs` into a searchable store, memoized by (doc-text hash, model id)."""
    key = _key(docs, embedder.model_id)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    for doc, vector in zip(docs, embedder.embed([d.text for d in docs])):
        doc.vector = vector
    store = InMemoryVectorStore()
    store.add(docs)
    _CACHE[key] = store
    return store


def clear_cache() -> None:
    _CACHE.clear()
