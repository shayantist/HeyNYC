"""Render an AgentResult into WhatsApp-ready message chunks: clean body (no inline
{cite:Sn} markers) + a compact Sources footer, split to WhatsApp's 4096-char limit."""
from __future__ import annotations

import re

WA_LIMIT = 4096
_CITE = re.compile(r"\s*\{cite:S\d+\}")


def _strip_markers(text: str) -> str:
    return _CITE.sub("", text or "").strip()


def _sources_footer(citations: dict) -> str:
    if not citations:
        return ""
    lines = ["Sources:"]
    for c in citations.values():
        title = c.get("title") or c.get("url", "")
        lines.append(f"• {title} — {c['url']}")
    return "\n".join(lines)


def _split(text: str, limit: int) -> list[str]:
    """Split on blank-line (paragraph) boundaries, never exceeding `limit`."""
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    current = ""
    for para in text.split("\n\n"):
        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) <= limit:
            current = candidate
            continue
        if current:
            chunks.append(current)
        # a single oversized paragraph: hard-wrap it
        while len(para) > limit:
            chunks.append(para[:limit])
            para = para[limit:]
        current = para
    if current:
        chunks.append(current)
    return chunks


def render(result) -> list[str]:
    body = _strip_markers(result.text)
    footer = _sources_footer(result.citations)
    if not footer:
        return _split(body, WA_LIMIT) or [""]
    # Reserve room for the footer on the last chunk.
    chunks = _split(body, WA_LIMIT - len(footer) - 2)
    chunks[-1] = f"{chunks[-1]}\n\n{footer}"
    return chunks
