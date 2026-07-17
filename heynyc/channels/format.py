"""Render an AgentResult into WhatsApp-ready message chunks: clean body (no inline
{cite:Sn} markers) + a compact Sources footer, split to WhatsApp's 4096-char limit."""
from __future__ import annotations

import re

from heynyc.core.citations import used_citations

WA_LIMIT = 4096
_CITE = re.compile(r"\s*\{cite:S\d+\}")
_ATTACH = re.compile(r"\s*\[attached:[^\]]*\]")   # delivered out-of-band; never shown in text
_CODE = re.compile(r"```.*?```|`[^`\n]*`", re.DOTALL)
_LINK = re.compile(r"\[([^\]\n]+)\]\((https?://[^\s)]+)\)")
_URL = re.compile(r"https?://\S+")
_HEADING = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+(.+?)(?:[ \t]+#+)?[ \t]*$", re.MULTILINE)


def _strip_markers(text: str) -> str:
    return _ATTACH.sub("", _CITE.sub("", text or "")).replace("\N{EM DASH}", "-").strip()


def _whatsapp_markup(text: str) -> str:
    """Convert common model Markdown to WhatsApp's smaller text dialect."""
    def convert(part: str) -> str:
        part = _LINK.sub(r"\1: \2", part)
        urls: list[str] = []

        def stash_url(match: re.Match) -> str:
            urls.append(match.group())
            return f"\x00{len(urls) - 1}\x00"

        part = _URL.sub(stash_url, part)
        part = re.sub(r"\*\*(\S(?:.*?\S)?)\*\*", r"*\1*", part)
        part = _HEADING.sub(
            lambda m: m.group(1).strip()
            if m.group(1).strip().startswith("*") and m.group(1).strip().endswith("*")
            else f"*{m.group(1).strip()}*",
            part,
        )
        part = re.sub(r"~~(\S(?:.*?\S)?)~~", r"~\1~", part)
        part = re.sub(r"^[ \t]*[+*][ \t]+", "- ", part, flags=re.MULTILINE)
        for index, url in enumerate(urls):
            part = part.replace(f"\x00{index}\x00", url)
        return part

    pieces: list[str] = []
    start = 0
    for match in _CODE.finditer(text):
        pieces.extend((convert(text[start:match.start()]), match.group()))
        start = match.end()
    pieces.append(convert(text[start:]))
    return "".join(pieces)


def _canonical_url(url: str) -> str:
    return url.rstrip(".,;:!?").split("#:~:text=", 1)[0].rstrip("/")


def _sources_footer(citations: dict, inline_urls: set[str] | None = None) -> str:
    if not citations:
        return ""
    inline_urls = inline_urls or set()
    lines = ["Sources:"]
    seen: set[tuple[str, str]] = set()
    for c in citations.values():
        title = c.get("title") or c.get("url", "")
        url = c["url"].split("#:~:text=", 1)[0]
        if c.get("kind") == "DATA" and "/FeatureServer/" in url and "/query?" in url:
            url = url.split("/query?", 1)[0]
        if _canonical_url(url) in inline_urls:
            continue
        if (title, url) in seen:
            continue
        seen.add((title, url))
        lines.append(f"• {title} - {url}")
    return "\n".join(lines) if len(lines) > 1 else ""


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
        current = ""
        lines = para.splitlines()
        if len(lines) > 1:
            for line in lines:
                candidate = f"{current}\n{line}" if current else line
                if len(candidate) <= limit:
                    current = candidate
                    continue
                if current:
                    chunks.append(current)
                while len(line) > limit:
                    chunks.append(line[:limit])
                    line = line[limit:]
                current = line
            continue
        # a single oversized paragraph: hard-wrap it
        while len(para) > limit:
            chunks.append(para[:limit])
            para = para[limit:]
        current = para
    if current:
        chunks.append(current)
    return chunks


def render(result) -> list[str]:
    body = _whatsapp_markup(_strip_markers(result.text))
    inline_urls = {_canonical_url(match.group()) for match in _URL.finditer(body)}
    footer = _sources_footer(used_citations(result.text, result.citations), inline_urls)
    if not footer:
        return _split(body, WA_LIMIT) or [""]
    return _split(f"{body}\n\n{footer}", WA_LIMIT)
