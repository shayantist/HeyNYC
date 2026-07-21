"""Render an AgentResult into channel-appropriate message chunks: clean body (no inline
{cite:Sn} markers) + a compact Sources footer, split to a 4096-char limit. SMS gets plain
text, WhatsApp its native dialect; both share one link policy. Presentation only, downstream
of the grounding guard: it never adds, drops, or alters a factual claim, and never mutates the
raw generation or the citation record that the session and telemetry persist."""
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
# A URL the model wrapped in braces, e.g. `[Details]({https://...})` seen live. Presentation-only
# cleanup: the braces are never part of the address, so strip them before any link handling.
_BRACED_URL = re.compile(r"\{(https?://[^\s{}]+)\}")
# A Socrata single-row API permalink: /resource/{4x4}/{:id}.json (see core.tools.datasets.row_url).
# Its resident-facing equivalent is the dataset's human landing page /d/{4x4}.
_SOCRATA_ROW = re.compile(r"(https?://[^/]+)/resource/([a-z0-9]{4}-[a-z0-9]{4})/[^/?#\s]+\.json", re.I)


def _strip_markers(text: str) -> str:
    return _ATTACH.sub("", _CITE.sub("", text or "")).replace("\N{EM DASH}", "-").strip()


def _official_page(url: str) -> str:
    """A Socrata row-JSON permalink -> its human dataset page; anything else unchanged."""
    match = _SOCRATA_ROW.fullmatch(url)
    return f"{match.group(1)}/d/{match.group(2)}" if match else url


def _clean_body_links(text: str, citations: dict) -> str:
    """Deterministic, presentation-only link fixes for the resident-facing body. Never touches the
    stored citation record: `citations` is read to decide WHICH links we vouch for, not mutated."""
    text = _BRACED_URL.sub(r"\1", text)                 # {https://...} -> https://...
    for citation in citations.values():                 # only rewrite links this answer cited
        url = citation.get("url", "")
        page = _official_page(url)
        if page != url:
            text = text.replace(url, page)
    return text


def _apply_outside_code(text: str, convert) -> str:
    """Run `convert` on every span except fenced/inline code, which passes through verbatim."""
    pieces: list[str] = []
    start = 0
    for match in _CODE.finditer(text):
        pieces.extend((convert(text[start:match.start()]), match.group()))
        start = match.end()
    pieces.append(convert(text[start:]))
    return "".join(pieces)


def _protect_urls(part: str, convert) -> str:
    """Stash URLs so markup substitutions can't corrupt them, run `convert`, restore them."""
    urls: list[str] = []

    def stash(match: re.Match) -> str:
        urls.append(match.group())
        return f"\x00{len(urls) - 1}\x00"

    part = _URL.sub(stash, _LINK.sub(r"\1: \2", part))
    part = convert(part)
    part = re.sub(r"^[ \t]*[+*][ \t]+", "- ", part, flags=re.MULTILINE)
    for index, url in enumerate(urls):
        part = part.replace(f"\x00{index}\x00", url)
    return part


def _whatsapp_markup(text: str) -> str:
    """Convert common model Markdown to WhatsApp's smaller text dialect (native *bold*)."""
    def convert(part: str) -> str:
        part = re.sub(r"\*\*(\S(?:.*?\S)?)\*\*", r"*\1*", part)
        part = _HEADING.sub(
            lambda m: m.group(1).strip()
            if m.group(1).strip().startswith("*") and m.group(1).strip().endswith("*")
            else f"*{m.group(1).strip()}*",
            part,
        )
        return re.sub(r"~~(\S(?:.*?\S)?)~~", r"~\1~", part)

    return _apply_outside_code(text, lambda part: _protect_urls(part, convert))


def _plain_markup(text: str) -> str:
    """Strip the same Markdown to plain text for SMS: no bold/heading/strike delimiters."""
    def convert(part: str) -> str:
        part = re.sub(r"\*\*(\S(?:.*?\S)?)\*\*", r"\1", part)
        part = _HEADING.sub(lambda m: m.group(1).strip(), part)
        return re.sub(r"~~(\S(?:.*?\S)?)~~", r"\1", part)

    return _apply_outside_code(text, lambda part: _protect_urls(part, convert))


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


def render(result, channel: str = "whatsapp") -> list[str]:
    """Render the grounded, guard-checked `result` for delivery on `channel`.

    Presentation only: the grounding guard has already run on `result.text` (the raw generation
    with its `{cite:Sn}` markers), and that raw text plus the full citation mapping are what the
    session and telemetry persist. This layer never adds, drops, or alters a factual claim, it only
    formats: SMS gets plain text, WhatsApp gets its native dialect, and both share one link policy.
    """
    if channel == "console":
        # The REPL is the rich surface: inline {cite:Sn} markers STAY visible (texters lose
        # them only because SMS/WhatsApp can't render them usefully), markdown stays raw for
        # rich, and the sources footer below goes one-per-line instead of the wrapped bullets.
        body = _clean_body_links(result.text, result.citations)
    else:
        text = _clean_body_links(_strip_markers(result.text), result.citations)
        body = _plain_markup(text) if channel.startswith("sms") else _whatsapp_markup(text)
    inline_urls = {_canonical_url(match.group()) for match in _URL.finditer(body)}
    # Sources footer is unchanged per channel: it keeps every cited source (row-addressed permalinks
    # included) so the audit record on screen matches the stored one. SMS length is bounded downstream
    # by the Twilio adapter's own per-message segment budget.
    cited = used_citations(result.text, result.citations)
    if channel == "console":
        rows = [
            f"  [{cid}] {c.get('title') or c.get('url', '')} - {c.get('url', '')}"
            for cid, c in cited.items()
            if _canonical_url(c.get("url", "")) not in inline_urls or True
        ]
        footer = "Sources:\n" + "\n".join(rows) if rows else ""
    else:
        footer = _sources_footer(cited, inline_urls)
    if not footer:
        return _split(body, WA_LIMIT) or [""]
    return _split(f"{body}\n\n{footer}", WA_LIMIT)
