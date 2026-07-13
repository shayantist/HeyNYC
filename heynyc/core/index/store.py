"""Vector store with hybrid retrieval, dense cosine + BM25, fused by Reciprocal Rank Fusion.

Two backends behind one interface: InMemoryVectorStore (numpy, zero-config, the small-corpus
default + tests) and LanceVectorStore (persistent). Ranking follows the documented hybrid
standard, dense (semantic) + Okapi BM25 (lexical, IDF-weighted), combined with RRF, which fuses
by *rank* not score so the incompatible scales (cosine ∈ [-1,1] vs unbounded BM25) need no
normalization. See docs/superpowers/specs/2026-06-29-eval-grading-and-retrieval-amendment.md.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol

import numpy as np

from .embedder import tokenize

RRF_K = 60      # reciprocal-rank-fusion constant (the universal default, k=60)
BM25_K1 = 1.5   # BM25 term-frequency saturation
BM25_B = 0.75   # BM25 document-length normalization


@dataclass
class IndexDoc:
    id: str
    text: str
    url: str = ""
    title: str = ""
    module: str = ""
    vector: Optional[list[float]] = None
    terms: set[str] = field(default_factory=set)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return float(np.dot(a, b) / denom)


def _bm25_scores(query_terms: list[str], doc_tokens: list[list[str]],
                 k1: float = BM25_K1, b: float = BM25_B) -> list[float]:
    """Okapi BM25 of the query against each doc: IDF · saturated TF, length-normalized.

    The standard lexical signal, rare discriminative terms (SCRIE, IDNYC) outweigh common
    ones via IDF (which bare term-overlap lacked), with TF saturation and doc-length
    normalization. A doc sharing no query term scores 0.
    """
    n = len(doc_tokens)
    if n == 0:
        return []
    df: dict[str, int] = {}
    for toks in doc_tokens:
        for term in set(toks):
            df[term] = df.get(term, 0) + 1
    avgdl = (sum(len(toks) for toks in doc_tokens) / n) or 1.0
    query = set(query_terms)
    scores: list[float] = []
    for toks in doc_tokens:
        if not toks:
            scores.append(0.0)
            continue
        tf: dict[str, int] = {}
        for term in toks:
            tf[term] = tf.get(term, 0) + 1
        dl = len(toks)
        score = 0.0
        for term in query:
            freq = tf.get(term, 0)
            if not freq:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            score += idf * (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * dl / avgdl))
        scores.append(score)
    return scores


def _rrf_fuse(dense: list[float], sparse: list[float], k: int = RRF_K) -> list[float]:
    """Reciprocal Rank Fusion, combine two score lists by rank position, not magnitude,
    so cosine and BM25 (incompatible scales) fuse without per-retriever normalization."""
    def ranks(scores: list[float]) -> list[int]:
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out = [0] * len(scores)
        for rank, i in enumerate(order):
            out[i] = rank
        return out

    dr, sr = ranks(dense), ranks(sparse)
    return [1.0 / (k + dr[i]) + 1.0 / (k + sr[i]) for i in range(len(dense))]


def _hybrid_rank(docs: list[IndexDoc], query_vec: list[float], query_text: str,
                 k: int) -> list[tuple[IndexDoc, float]]:
    """Rank docs by RRF(dense cosine, BM25 lexical). Returns top-k [(doc, rrf_score)]."""
    if not docs:
        return []
    q = np.asarray(query_vec, dtype=float)
    dense = [_cosine(q, np.asarray(doc.vector, dtype=float)) for doc in docs]
    doc_tokens = [tokenize(f"{doc.title} {doc.text}") for doc in docs]
    sparse = _bm25_scores(tokenize(query_text), doc_tokens)
    fused = _rrf_fuse(dense, sparse)
    scored = sorted(zip(docs, fused), key=lambda pair: pair[1], reverse=True)
    return scored[:k]


class VectorStore(Protocol):
    def add(self, docs: list[IndexDoc]) -> None: ...
    def search(self, query_vec: list[float], query_text: str, k: int = 5) -> list[tuple[IndexDoc, float]]: ...
    def count(self) -> int: ...


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._docs: list[IndexDoc] = []

    def add(self, docs: list[IndexDoc]) -> None:
        self._docs.extend(docs)

    def count(self) -> int:
        return len(self._docs)

    def search(self, query_vec: list[float], query_text: str, k: int = 5) -> list[tuple[IndexDoc, float]]:
        docs = [doc for doc in self._docs if doc.vector is not None]
        return _hybrid_rank(docs, query_vec, query_text, k)


class LanceVectorStore:
    """Persistent backend. Dense top-N from LanceDB, then re-ranked by RRF(cosine, BM25)
    over that candidate set (BM25 IDF is over the candidates, an accepted approximation)."""

    def __init__(self, path: Path, table_name: str = "corpus"):
        import lancedb

        self._db = lancedb.connect(str(path))
        self._table_name = table_name
        self._table = None
        if table_name in self._db.table_names():
            self._table = self._db.open_table(table_name)

    def add(self, docs: list[IndexDoc]) -> None:
        rows = [
            {"id": d.id, "text": d.text, "url": d.url, "title": d.title,
             "module": d.module, "vector": d.vector}
            for d in docs if d.vector is not None
        ]
        if not rows:
            return
        if self._table is None:
            self._table = self._db.create_table(self._table_name, data=rows)
        else:
            self._table.add(rows)

    def count(self) -> int:
        return 0 if self._table is None else self._table.count_rows()

    def search(self, query_vec: list[float], query_text: str, k: int = 5) -> list[tuple[IndexDoc, float]]:
        if self._table is None:
            return []
        rows = self._table.search(query_vec).limit(max(k * 4, k)).to_list()
        docs = [
            IndexDoc(id=row["id"], text=row["text"], url=row["url"],
                     title=row["title"], module=row["module"], vector=list(row["vector"]))
            for row in rows
        ]
        return _hybrid_rank(docs, query_vec, query_text, k)


def open_store(path: Optional[Path] = None) -> VectorStore:
    """LanceVectorStore when a path is given and lancedb is importable, else in-memory."""
    if path is not None:
        try:
            return LanceVectorStore(path)
        except Exception:
            pass
    return InMemoryVectorStore()
