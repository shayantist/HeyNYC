"""Embedders. Default is keyless/local FastEmbed; deterministic hashing is for tests."""
from __future__ import annotations

import hashlib
import math
import re
from functools import lru_cache
from typing import Protocol, runtime_checkable

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@runtime_checkable
class Embedder(Protocol):
    dim: int
    model_id: str  # stable identity for cache keys (embeddings are not content-addressed)

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashEmbedder:
    """Deterministic bag-of-words hashing embedder. No deps, stable across runs.

    Not semantically rich, but exercises the full retrieval path and gives usable
    keyword-ish recall for fast, offline unit tests."""

    def __init__(self, dim: int = 256):
        self.dim = dim
        self.model_id = f"hash:{dim}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            for token in tokenize(text):
                idx = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim
                vec[idx] += 1.0
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            vectors.append([x / norm for x in vec])
        return vectors


class FastEmbedEmbedder:
    """Local ONNX embeddings via fastembed (no PyTorch, no API key)."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ):
        self._model_name = model_name
        self._model = None
        self.model_id = f"fastembed:{model_name}"

    def _load(self):
        if self._model is not None:
            return self._model
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=self._model_name)
        return self._model

    @property
    def dim(self) -> int:
        return int(self._load().embedding_size)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [
            list(map(float, vec))
            for vec in self._load().embed(texts, batch_size=32)
        ]


@lru_cache(maxsize=1)
def default_embedder() -> Embedder:
    """Return the production embedder, loading its model on first use."""
    return FastEmbedEmbedder()
