"""Render an AgentResult into channel-appropriate message chunks: clean body (no inline
{cite:Sn} markers) + a compact Sources footer, split to a 4096-char limit. SMS gets plain
text, WhatsApp its native dialect; both share one link policy. Presentation only, downstream
of the grounding guard: it never adds, drops, or alters a factual claim, and never mutates the
raw generation or the citation record that the session and telemetry persist."""
from __future__ import annotations

import re

from markdown_it import MarkdownIt
from markdown_it.token import Token

from heynyc.core.citations import used_citations
from heynyc.core.localization import localize, localized_source_limit

WA_LIMIT = 4096
_CITE = re.compile(r"\s*\{cite:S\d+\}")
_ATTACH = re.compile(r"\s*\[attached:[^\]]*\]")   # delivered out-of-band; never shown in text
_MARKDOWN = MarkdownIt(
    "commonmark",
    {"linkify": True},
).enable(("strikethrough", "linkify"))


def _strip_markers(text: str) -> str:
    return _ATTACH.sub("", _CITE.sub("", text or "")).replace("\N{EM DASH}", "-").strip()


def _inline_text(tokens: list[Token], *, whatsapp: bool) -> str:
    output: list[str] = []
    links: list[tuple[str, int]] = []
    wrappers = {
        "strong_open": "*" if whatsapp else "",
        "strong_close": "*" if whatsapp else "",
        "em_open": "_" if whatsapp else "",
        "em_close": "_" if whatsapp else "",
        "s_open": "~" if whatsapp else "",
        "s_close": "~" if whatsapp else "",
    }
    for token in tokens:
        if token.type == "text":
            output.append(token.content)
        elif token.type in wrappers:
            output.append(wrappers[token.type])
        elif token.type in {"softbreak", "hardbreak"}:
            output.append("\n")
        elif token.type == "code_inline":
            markup = token.markup or "`"
            output.append(f"{markup}{token.content}{markup}")
        elif token.type == "link_open":
            links.append((token.attrGet("href") or "", len(output)))
        elif token.type == "link_close" and links:
            url, start = links.pop()
            label = "".join(output[start:]).strip()
            if url and _canonical_url(label) != _canonical_url(url):
                output.append(f": {url}")
        elif token.type == "image":
            label = token.content or token.attrGet("alt") or "Image"
            url = token.attrGet("src") or ""
            output.append(f"{label}: {url}" if url else label)
        elif token.type == "html_inline":
            output.append(token.content)
    return "".join(output)


def _render_markdown(text: str, *, whatsapp: bool) -> str:
    """Render CommonMark from parsed tokens, never by rewriting prose with regexes."""
    output: list[str] = []
    lists: list[dict[str, int | str]] = []
    heading = False
    quote_depth = 0
    for token in _MARKDOWN.parse(text):
        if token.type == "inline":
            content = _inline_text(token.children or [], whatsapp=whatsapp)
            if heading and whatsapp and not (
                content.startswith("*") and content.endswith("*")
            ):
                content = f"*{content}*"
            if quote_depth:
                prefix = "> " * quote_depth
                current_line = "".join(output).rsplit("\n", 1)[-1]
                content = content.replace("\n", f"\n{prefix}")
                if not current_line.startswith(prefix):
                    content = prefix + content
            output.append(content)
        elif token.type == "heading_open":
            heading = True
        elif token.type == "heading_close":
            output.append("\n")
            heading = False
        elif token.type == "paragraph_close" and not token.hidden and not heading:
            output.append("\n")
        elif token.type == "bullet_list_open":
            if lists and output and not output[-1].endswith("\n"):
                output.append("\n")
            lists.append({"kind": "bullet", "next": 0})
        elif token.type == "ordered_list_open":
            if lists and output and not output[-1].endswith("\n"):
                output.append("\n")
            lists.append({"kind": "ordered", "next": int(token.attrGet("start") or 1)})
        elif token.type in {"bullet_list_close", "ordered_list_close"}:
            lists.pop()
        elif token.type == "list_item_open":
            prefix = "- "
            if lists and lists[-1]["kind"] == "ordered":
                prefix = f"{lists[-1]['next']}. "
                lists[-1]["next"] = int(lists[-1]["next"]) + 1
            quote = "> " * quote_depth
            output.append(f"{quote}{'  ' * max(0, len(lists) - 1)}{prefix}")
        elif token.type == "list_item_close":
            if not output or not output[-1].endswith("\n"):
                output.append("\n")
        elif token.type == "blockquote_open":
            quote_depth += 1
        elif token.type == "blockquote_close":
            quote_depth -= 1
        elif token.type == "hr":
            output.append("---\n")
        elif token.type == "fence":
            markup = token.markup or "```"
            info = token.info.strip()
            output.append(f"{markup}{info}\n{token.content}{markup}\n")
        elif token.type == "code_block":
            output.append(f"```\n{token.content}```\n")
        elif token.type == "html_block":
            output.append(token.content)
    return "".join(output).strip()


def _canonical_url(url: str) -> str:
    return url.rstrip(".,;:!?").split("#:~:text=", 1)[0].rstrip("/")


def _sources_footer(
    citations: dict,
    inline_urls: set[str] | None = None,
    action_links: tuple = (),
    language: str | None = None,
) -> str:
    if not citations:
        return ""
    inline_urls = inline_urls or set()
    actions_by_citation: dict[str, list] = {}
    for action in action_links:
        actions_by_citation.setdefault(action.citation_id, []).append(action)
    lines = [f"{localize('Sources', language)}:"]
    seen: set[tuple[str, str]] = set()
    rendered_actions: set[str] = set()
    rendered_notes: set[str] = set()
    for c in citations.values():
        title = c.get("title") or c.get("url", "")
        url = c["url"].split("#:~:text=", 1)[0]
        if _canonical_url(url) not in inline_urls and (title, url) not in seen:
            seen.add((title, url))
            lines.append(f"• {title} - {url}")
        provenance = c.get("provenance") or {}
        if (
            provenance.get("evidence_grade") == "search_excerpt"
            and provenance.get("source_tier") == "unverified"
        ):
            lines.append(localize(
                "Verification note for {source}: this source is a search-result excerpt. "
                "I could not confirm it from the full page.",
                language,
            ).format(source=title))
        for action in actions_by_citation.get(c.get("id", ""), ()):
            canonical_action = _canonical_url(action.url)
            if (
                canonical_action not in inline_urls
                and canonical_action not in rendered_actions
            ):
                rendered_actions.add(canonical_action)
                lines.append(f"  {action.label} - {action.url}")
        limitation = str(
            (c.get("provenance") or {}).get("derivation", {}).get("limitations") or ""
        ).strip()
        source_note = localized_source_limit(limitation, language)
        if source_note and source_note not in rendered_notes:
            rendered_notes.add(source_note)
            lines.append(f"  {localize('Source note', language)} - {source_note}")
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



_CITE_ID = re.compile(r"\{cite:(S\d+)\}")


def _link_markers(text: str, citations: dict) -> str:
    """Console only: each inline {cite:Sn} becomes a clickable markdown link showing [Sn].
    Escaped brackets keep the visible tag; the angle-bracket destination survives gov and
    Socrata URLs full of ?, &, and #:~:text= fragments. Unknown ids pass through untouched."""
    def sub(m):
        cid = m.group(1)
        url = (citations.get(cid) or {}).get("url", "")
        return f"[\\[{cid}\\]](<{url}>)" if url else m.group(0)
    return _CITE_ID.sub(sub, text)

def render(result, channel: str = "whatsapp") -> list[str]:
    """Render the grounded, guard-checked `result` for delivery on `channel`.

    Presentation only: the grounding guard has already run on `result.text` (the raw generation
    with its `{cite:Sn}` markers), and that raw text plus the full citation mapping are what the
    session and telemetry persist. This layer never adds, drops, or alters a factual claim, it only
    formats: SMS gets plain text, WhatsApp gets its native dialect, and both share one link policy.
    """
    if getattr(result, "status", None) == "approval_required":
        body = result.text
    elif channel == "console":
        # The REPL is the rich surface: inline {cite:Sn} markers STAY visible (texters lose
        # them only because SMS/WhatsApp can't render them usefully), markdown stays raw for
        # rich, and the sources footer below goes one-per-line instead of the wrapped bullets.
        body = _link_markers(result.text, result.citations)
    else:
        text = _strip_markers(result.text)
        body = _render_markdown(text, whatsapp=not channel.startswith("sms"))
    action_links = tuple(getattr(result, "action_links", ()))
    candidate_urls = [
        str(citation.get("url") or "")
        for citation in result.citations.values()
    ] + [action.url for action in action_links]
    inline_urls = {
        _canonical_url(url)
        for url in candidate_urls
        if url and _canonical_url(url) in body
    }
    # Sources footer is unchanged per channel: it keeps every cited source (row-addressed permalinks
    # included) so the audit record on screen matches the stored one. SMS length is bounded downstream
    # by the Twilio adapter's own per-message segment budget.
    cited = used_citations(result.text, result.citations)
    if channel == "console":
        actions_by_citation = {}
        seen_action_urls: set[str] = set()
        for action in action_links:
            canonical_action = _canonical_url(action.url)
            if canonical_action in seen_action_urls:
                continue
            seen_action_urls.add(canonical_action)
            actions_by_citation[action.citation_id] = action
        rows = [
            # Markdown list items: single newlines are soft breaks that Markdown() collapses
            # into spaces (observed live as one wrapped blob), list items render one per line.
            f"- [\\[{cid}\\]](<{c.get('url', '')}>) {c.get('title') or c.get('url', '')} - <{c.get('url', '')}>"
            + (
                f" - [{actions_by_citation[cid].label}](<{actions_by_citation[cid].url}>)"
                if cid in actions_by_citation
                and _canonical_url(actions_by_citation[cid].url) not in inline_urls
                else ""
            )
            for cid, c in cited.items()
        ]
        footer = "Sources:\n" + "\n".join(rows) if rows else ""
    else:
        footer = _sources_footer(
            cited,
            inline_urls,
            action_links,
            (getattr(result, "diagnostics", {}) or {}).get("safety_language"),
        )
    if not footer:
        return _split(body, WA_LIMIT) or [""]
    return _split(f"{body}\n\n{footer}", WA_LIMIT)
