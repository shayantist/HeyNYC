"""Citation registry, every grounded claim links back to a real source.

Adapts DXA's `{cite:KB1}` model (api/state.py::get_or_register_citation). Tools
register the sources they return; the agent cites them inline as `{cite:S1}`.
Identical sources dedupe to the same id so answers don't accumulate duplicates.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Literal
from urllib.parse import quote

CiteKind = Literal["DATA", "DOC", "WEB"]


_USED_RE = re.compile(r"\{cite:(S\d+)\}")


def canonical_source_url(url: str) -> str:
    """Return the stored form of a source URL."""
    if url.startswith("https://www1.nyc.gov/"):
        return "https://www.nyc.gov/" + url[len("https://www1.nyc.gov/"):]
    return url


def text_fragment_url(url: str, snippet: str, kind: str) -> str:
    """Return a Chrome Text Fragment URL for a cited DOC or WEB source snippet."""
    normalized_snippet = " ".join(snippet.split())
    if kind not in {"DOC", "WEB"} or not normalized_snippet or "#" in url:
        return url
    phrase = normalized_snippet.split(" ")[:8]
    encoded = quote(" ".join(phrase)[:240], safe="").replace("-", "%2D")
    return f"{url}#:~:text={encoded}"


def used_citations(text: str, citations: dict) -> dict:
    """Only the citations the answer actually references via {cite:Sn}. A tool may register more
    sources than the answer ends up citing (a broad web_search, a tangential lookup); the Sources
    footer must show only what backs the answer, never an unused source (e.g. a World Cup link under
    a SNAP answer)."""
    used = set(_USED_RE.findall(text or ""))
    return {cid: c for cid, c in citations.items() if cid in used}


def used_discovery_citations(text: str, citations: dict) -> list[str]:
    """Discovery snippets can guide retrieval but cannot support a final answer."""
    used = set(_USED_RE.findall(text or ""))
    return [
        cid
        for cid, citation in citations.items()
        if cid in used
        and (citation.get("provenance") or {}).get("evidence_grade") == "discovery"
    ]


def used_unverified_citations(text: str, citations: dict) -> list[str]:
    """Unverified sources may support low-stakes excerpts, never restrictive turns."""
    used = set(_USED_RE.findall(text or ""))
    return [
        cid
        for cid, citation in citations.items()
        if cid in used
        and (citation.get("provenance") or {}).get("source_tier") == "unverified"
    ]


def content_hash(snapshot: dict) -> str:
    """Stable SHA-256 over a row payload (key-order-independent)."""
    canonical = json.dumps(snapshot, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def data_provenance(snapshot: dict, *, record_id: str, field_pointer: str,
                    derivation: dict | None = None) -> dict:
    """Provenance for a structured (DATA) citation: the exact row used, a content hash
    (reproducibility/integrity), the record id + JSON-Pointer field locator, and an optional
    `derivation` (the inputs to a computed value, e.g. distance)."""
    return {
        "record_id": record_id,
        "field_pointer": field_pointer,
        "content_hash": content_hash(snapshot),
        "snapshot": snapshot,
        "derivation": derivation or {},
    }


def api_provenance(endpoint: str, request_summary: dict, response: dict, *,
                   field_pointer: str = "", as_of: str = "") -> dict:
    """Provenance for an auditable-but-not-re-fetchable API exchange (a POST behind auth):
    the endpoint, a REDACTED request summary (never raw financials/PII), the exact response,
    a content hash over all three, a JSON-Pointer to the cited element, and an "as of" date."""
    payload = {"endpoint": endpoint, "request_summary": request_summary, "response": response}
    return {
        "endpoint": endpoint,
        "request_summary": request_summary,
        "response": response,
        "content_hash": content_hash(payload),
        "field_pointer": field_pointer,
        "as_of": as_of,
    }


@dataclass(frozen=True)
class Citation:
    id: str
    url: str
    title: str
    snippet: str
    kind: CiteKind
    valid_as_of: str = ""  # source "as of" date (temporal provenance, spec §11), never fetch time
    provenance: dict = field(default_factory=dict)  # structured DATA provenance (empty for DOC/WEB)


class CitationRegistry:
    def __init__(self) -> None:
        self._by_key: dict[tuple, Citation] = {}
        self._ordered: list[Citation] = []
        # Monotonic id counter: ids survive discards, so a marker already emitted into text
        # can never silently point at a different, later-registered source.
        self._counter = 0

    def register(
        self,
        url: str,
        *,
        snippet: str = "",
        title: str = "",
        kind: CiteKind = "WEB",
        valid_as_of: str = "",
        provenance: dict | None = None,
    ) -> str:
        """Register a source, returning its semantic id (S1, S2, ...).

        Dedupes on kind, URL, exact snippet, evidence grade, and DATA content hash so the same
        evidence reused across tool calls maps to one id without conflating changing snapshots.
        `provenance` carries structured DATA provenance or a WEB evidence grade.
        """
        # F056: the city 301s its legacy host to www.nyc.gov (verified live); store the
        # canonical host so replies and the liveness check never ride a deprecated hostname.
        url = canonical_source_url(url)
        evidence_grade = (provenance or {}).get("evidence_grade", "")
        data_hash = (provenance or {}).get("content_hash", "") if kind == "DATA" else ""
        key = (kind, url, snippet, evidence_grade, data_hash)
        existing = self._by_key.get(key)
        if existing is not None:
            return existing.id
        self._counter += 1
        cite_id = f"S{self._counter}"
        citation = Citation(
            id=cite_id, url=url, title=title, snippet=snippet, kind=kind,
            valid_as_of=valid_as_of, provenance=provenance or {},
        )
        self._by_key[key] = citation
        self._ordered.append(citation)
        return cite_id

    def discard(self, ids: set[str]) -> None:
        """Remove citations whose content never reached the model (F057: a coordinating tool
        may filter lane output after its sub-handlers registered). Discarded ids are never
        reissued."""
        if not ids:
            return
        self._ordered = [c for c in self._ordered if c.id not in ids]
        self._by_key = {key: c for key, c in self._by_key.items() if c.id not in ids}

    def mapping(self) -> dict[str, dict]:
        """{ "S1": {url, title, snippet, kind}, ... } in registration order."""
        return {c.id: asdict(c) for c in self._ordered}

    def dump_state(self) -> dict:
        """Serialize exact ids so a paused tool turn can resume without rebinding markers."""
        return {"citations": self.mapping(), "counter": self._counter}

    @classmethod
    def from_state(cls, state: dict) -> "CitationRegistry":
        """Restore a registry previously returned by `dump_state`."""
        registry = cls()
        raw_citations = state.get("citations", {})
        counter = int(state.get("counter", 0))
        for cite_id, raw in raw_citations.items():
            if raw.get("id") != cite_id or not re.fullmatch(r"S\d+", cite_id):
                raise ValueError(f"Invalid citation state id: {cite_id!r}")
            citation = Citation(**{**raw, "url": canonical_source_url(raw["url"])})
            evidence_grade = citation.provenance.get("evidence_grade", "")
            data_hash = (
                citation.provenance.get("content_hash", "")
                if citation.kind == "DATA"
                else ""
            )
            key = (citation.kind, citation.url, citation.snippet, evidence_grade, data_hash)
            if key in registry._by_key:
                raise ValueError(f"Duplicate citation state key: {cite_id!r}")
            registry._by_key[key] = citation
            registry._ordered.append(citation)
        highest = max(
            (int(citation.id.removeprefix("S")) for citation in registry._ordered),
            default=0,
        )
        if counter < highest:
            raise ValueError("Citation state counter is behind its registered ids")
        registry._counter = counter
        return registry

    def __len__(self) -> int:
        return len(self._ordered)
