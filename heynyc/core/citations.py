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

        Dedupes on (kind, url, snippet prefix) so the same source reused across
        tool calls maps to one id. `provenance` carries structured DATA provenance
        (snapshot + content hash + locator); empty for DOC/WEB.
        """
        key = (kind, url, snippet[:120])
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

    def __len__(self) -> int:
        return len(self._ordered)
