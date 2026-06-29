"""RAG index: embedder, vector store, corpus builder, retriever."""
from __future__ import annotations

from dataclasses import dataclass

from .embedder import Embedder, default_embedder
from .store import IndexDoc, InMemoryVectorStore, VectorStore, open_store


@dataclass
class IndexRetriever:
    """Bundles a store + embedder so tools can retrieve with one call."""

    store: VectorStore
    embedder: Embedder

    def search(self, query: str, k: int = 5):
        query_vec = self.embedder.embed([query])[0]
        return self.store.search(query_vec, query, k=k)


__all__ = [
    "Embedder",
    "default_embedder",
    "IndexDoc",
    "VectorStore",
    "InMemoryVectorStore",
    "open_store",
    "IndexRetriever",
]
