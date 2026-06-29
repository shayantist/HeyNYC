"""Citation registry — every grounded claim links back to a real source.

Adapts DXA's `{cite:KB1}` model (api/state.py::get_or_register_citation). Tools
register the sources they return; the agent cites them inline as `{cite:S1}`.
Identical sources dedupe to the same id so answers don't accumulate duplicates.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

CiteKind = Literal["DATA", "DOC", "WEB"]


@dataclass(frozen=True)
class Citation:
    id: str
    url: str
    title: str
    snippet: str
    kind: CiteKind
    valid_as_of: str = ""  # source "as of" date (temporal provenance, spec §11) — never fetch time


class CitationRegistry:
    def __init__(self) -> None:
        self._by_key: dict[tuple, Citation] = {}
        self._ordered: list[Citation] = []

    def register(
        self,
        url: str,
        *,
        snippet: str = "",
        title: str = "",
        kind: CiteKind = "WEB",
        valid_as_of: str = "",
    ) -> str:
        """Register a source, returning its semantic id (S1, S2, ...).

        Dedupes on (kind, url, snippet prefix) so the same source reused across
        tool calls maps to one id.
        """
        key = (kind, url, snippet[:120])
        existing = self._by_key.get(key)
        if existing is not None:
            return existing.id
        cite_id = f"S{len(self._ordered) + 1}"
        citation = Citation(
            id=cite_id, url=url, title=title, snippet=snippet, kind=kind, valid_as_of=valid_as_of,
        )
        self._by_key[key] = citation
        self._ordered.append(citation)
        return cite_id

    def mapping(self) -> dict[str, dict]:
        """{ "S1": {url, title, snippet, kind}, ... } in registration order."""
        return {c.id: asdict(c) for c in self._ordered}

    def __len__(self) -> int:
        return len(self._ordered)
