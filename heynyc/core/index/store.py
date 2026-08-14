"""Vector stores with hybrid dense and lexical retrieval fused by reciprocal rank fusion.

Two backends behind one interface: InMemoryVectorStore (numpy, zero-config, the small-corpus
default + tests) and LanceVectorStore (persistent). The in-memory backend retains the small
deterministic implementation for tests; the persistent backend uses LanceDB's native vector,
BM25, and RRF implementation. See
docs/internal/superpowers/specs/2026-06-29-eval-grading-and-retrieval-amendment.md.
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
    def replace(self, docs: list[IndexDoc]) -> None: ...
    def search(self, query_vec: list[float], query_text: str, k: int = 5) -> list[tuple[IndexDoc, float]]: ...
    def count(self) -> int: ...


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._docs: list[IndexDoc] = []

    def add(self, docs: list[IndexDoc]) -> None:
        self._docs.extend(docs)

    def replace(self, docs: list[IndexDoc]) -> None:
        self._docs = list(docs)

    def count(self) -> int:
        return len(self._docs)

    def search(self, query_vec: list[float], query_text: str, k: int = 5) -> list[tuple[IndexDoc, float]]:
        docs = [doc for doc in self._docs if doc.vector is not None]
        return _hybrid_rank(docs, query_vec, query_text, k)


class LanceVectorStore:
    """Persistent backend using LanceDB's native dense and full-text hybrid search."""

    def __init__(
        self,
        path: Path,
        table_name: str = "corpus",
        model_id: Optional[str] = None,
    ):
        import lancedb

        self._db = lancedb.connect(str(path))
        self._model_id = model_id
        self._model_id_validated = False
        self._table_name = table_name
        self._table = None
        if table_name in self._db.list_tables().tables:
            self._table = self._db.open_table(table_name)

    def _ensure_fts_indices(self, *, replace: bool = False) -> None:
        if self._table is None:
            return
        indexed = {
            column
            for index in self._table.list_indices()
            if index.index_type == "FTS"
            for column in index.columns
        }
        for column in ("title", "text"):
            if replace or column not in indexed:
                self._table.create_fts_index(column, replace=replace)

    def _assert_model_id(self) -> None:
        if self._model_id is None or self._table is None or self._model_id_validated:
            return
        if "embedding_model_id" not in self._table.schema.names:
            raise RuntimeError(
                "index does not record its embedding model; rebuild the index"
            )
        stored = set(
            self._table.search()
            .select(["embedding_model_id"])
            .to_arrow()
            .column("embedding_model_id")
            .to_pylist()
        )
        if not stored or stored == {None}:
            raise RuntimeError(
                "index does not record its embedding model; rebuild the index"
            )
        if len(stored) != 1:
            raise RuntimeError("index contains mixed embedding models; rebuild the index")
        actual = stored.pop()
        if actual != self._model_id:
            raise RuntimeError(
                f"index was built with {actual}, not {self._model_id}; rebuild the index"
            )
        self._model_id_validated = True

    def add(self, docs: list[IndexDoc]) -> None:
        rows = [
            {"id": d.id, "text": d.text, "url": d.url, "title": d.title,
             "module": d.module, "vector": d.vector,
             **({"embedding_model_id": self._model_id} if self._model_id else {})}
            for d in docs if d.vector is not None
        ]
        if not rows:
            return
        self._assert_model_id()
        if self._table is None:
            self._table = self._db.create_table(self._table_name, data=rows)
        else:
            self._table.add(rows)
        self._model_id_validated = self._model_id is not None
        self._ensure_fts_indices(replace=True)

    def replace(self, docs: list[IndexDoc]) -> None:
        rows = [
            {"id": d.id, "text": d.text, "url": d.url, "title": d.title,
             "module": d.module, "vector": d.vector,
             **({"embedding_model_id": self._model_id} if self._model_id else {})}
            for d in docs if d.vector is not None
        ]
        if rows:
            self._table = self._db.create_table(
                self._table_name, data=rows, mode="overwrite",
            )
            self._model_id_validated = self._model_id is not None
            self._ensure_fts_indices(replace=True)

    def count(self) -> int:
        return 0 if self._table is None else self._table.count_rows()

    def search(self, query_vec: list[float], query_text: str, k: int = 5) -> list[tuple[IndexDoc, float]]:
        if self._table is None or k <= 0:
            return []
        self._assert_model_id()
        from lancedb.rerankers import RRFReranker

        rows = (
            self._table.search(query_type="hybrid", fts_columns=["title", "text"])
            .vector(query_vec)
            .text(query_text)
            .rerank(RRFReranker())
            .limit(k)
            .to_list()
        )
        docs = [
            IndexDoc(id=row["id"], text=row["text"], url=row["url"],
                     title=row["title"], module=row["module"], vector=list(row["vector"]))
            for row in rows
        ]
        return [(doc, float(row["_relevance_score"])) for doc, row in zip(docs, rows)]


def open_store(path: Optional[Path] = None, model_id: Optional[str] = None) -> VectorStore:
    """LanceVectorStore when a path is given and lancedb is importable, else in-memory."""
    if path is not None:
        return LanceVectorStore(path, model_id=model_id)
    return InMemoryVectorStore()
