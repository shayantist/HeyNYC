"""Render an AgentResult into channel-appropriate message chunks. SMS and WhatsApp put each
cited source beside its claim and split to a 4096-character limit. Presentation stays downstream of
the grounding guard and never mutates the raw generation or citation record persisted by
sessions and telemetry."""
from __future__ import annotations

import re
from urllib.parse import quote

from markdown_it import MarkdownIt
from markdown_it.token import Token

from heynyc.core.citations import used_citations
from heynyc.core.localization import localize, localized_source_limit

WA_LIMIT = 4096
TWILIO_TEXT_LIMIT = 1600
TWILIO_PAGE_PREFIX_RESERVE = 16
_INLINE_CITE = re.compile(r"\s*\{cite:(S\d+)\}")
_MARKDOWN_CITE_LINK = re.compile(r"\[([^\]\n]+)\]\(\s*\{cite:(S\d+)\}\s*\)")
_REDUNDANT_LINK_LABEL = re.compile(
    r"\s*\[(?:Details|Tickets|Source)\](?!\s*[\[(:])", re.IGNORECASE,
)
_REDUNDANT_URL_LABEL = re.compile(
    r"(?<!\w)(?:Details|Tickets|Source):\s*(?=https?://)", re.IGNORECASE,
)
_CODE = re.compile(r"(`+|~{3,})(.*?)\1", re.DOTALL)
_ATTACH = re.compile(r"\s*\[attached:[^\]]*\]")   # delivered out-of-band; never shown in text
_MARKDOWN = MarkdownIt(
    "commonmark",
    {"linkify": True},
).enable(("strikethrough", "linkify"))


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
            if label.casefold() in {"details", "tickets", "source"}:
                del output[start:]
                output.append(_delivery_url(url))
            elif url and _canonical_url(label) != _canonical_url(url):
                output.append(f": {_delivery_url(url)}")
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


def _delivery_url(url: str) -> str:
    """Encode source links for text clients without changing the stored audit URL."""
    return quote(url, safe=":/?&=%#+-._~(),")


def _urls_in(text: str) -> set[str]:
    return {
        _canonical_url(token.attrGet("href") or "")
        for token in (_MARKDOWN.parseInline(text)[0].children or ())
        if token.type == "link_open" and token.attrGet("href")
    }


def _without_code_citations(text: str) -> str:
    return _CODE.sub(lambda match: _INLINE_CITE.sub("", match.group()), text)


def _plain_title(value: str) -> str:
    tokens = _MARKDOWN.parseInline(value)[0].children or ()
    return "".join(token.content for token in tokens if token.type in {"text", "code_inline"})


def _citation_notes(citation: dict, language: str | None) -> list[str]:
    notes: list[str] = []
    provenance = citation.get("provenance") or {}
    if (
        provenance.get("evidence_grade") == "search_excerpt"
        and provenance.get("source_tier") == "unverified"
    ):
        notes.append(localize(
            "Verification note for {source}: this source is a search-result excerpt. "
            "I could not confirm it from the full page.",
            language,
        ).format(source=_plain_title(str(citation.get("title") or "source"))))
    limitation = str(
        provenance.get("derivation", {}).get("limitations") or ""
    ).strip()
    if source_note := localized_source_limit(limitation, language):
        notes.append(source_note)
    return notes


def _inline_citation_links(
    text: str,
    citations: dict,
    action_links: tuple,
    language: str | None,
) -> str:
    actions: dict[str, list] = {}
    rendered_actions: set[str] = set()
    rendered_notes: set[str] = set()
    item_urls_by_text: dict[str, set[str]] = {}
    for action in action_links:
        actions.setdefault(action.citation_id, []).append(action)

    def replace_markdown_link(match: re.Match) -> str:
        label, citation_id = match.groups()
        citation = citations.get(citation_id) or {}
        source_url = str(citation.get("url") or "").split("#:~:text=", 1)[0]
        if not source_url:
            return match.group(0)
        rendered = f"[{label}](<{_delivery_url(source_url)}>)"
        for action in actions.get(citation_id, ()):
            action_url = _canonical_url(action.url)
            if action_url not in rendered_actions:
                rendered += f" ([{action.label}](<{_delivery_url(action.url)}>))"
                rendered_actions.add(action_url)
        notes = [
            note for note in _citation_notes(citation, language)
            if note not in rendered_notes
        ]
        rendered_notes.update(notes)
        if notes:
            rendered += f" ({' '.join(notes)})"
        return rendered

    def replace(match: re.Match) -> str:
        subject = match.string
        citation_id = match.group(1)
        citation = citations.get(citation_id) or {}
        line_start = subject.rfind("\n", 0, match.start()) + 1
        line_end = subject.find("\n", match.end())
        scope_end = len(subject) if line_end < 0 else line_end
        while scope_end < len(subject):
            continuation_start = scope_end + 1
            continuation_end = subject.find("\n", continuation_start)
            continuation_end = len(subject) if continuation_end < 0 else continuation_end
            if not subject[continuation_start:continuation_end].startswith((" ", "\t")):
                break
            scope_end = continuation_end
        item = subject[line_start:scope_end]
        item_urls = item_urls_by_text.get(item)
        if item_urls is None:
            item_urls = item_urls_by_text[item] = _urls_in(item)
        links: list[str] = []
        source_url = str(citation.get("url") or "").split("#:~:text=", 1)[0]
        source_key = _canonical_url(source_url)
        if source_url and source_key not in item_urls:
            links.append(f"[Source](<{_delivery_url(source_url)}>)")
            item_urls.add(source_key)
        for action in actions.get(citation_id, ()):
            action_url = _canonical_url(action.url)
            if action_url not in item_urls and action_url not in rendered_actions:
                links.append(f"[{action.label}](<{_delivery_url(action.url)}>)")
                rendered_actions.add(action_url)
        notes = [
            note for note in _citation_notes(citation, language)
            if note not in rendered_notes
        ]
        rendered_notes.update(notes)
        links.extend(notes)
        return f" ({'; '.join(links)})" if links else ""

    linked = _MARKDOWN_CITE_LINK.sub(replace_markdown_link, _without_code_citations(text))
    linked = _INLINE_CITE.sub(replace, linked)
    linked = _REDUNDANT_LINK_LABEL.sub("", linked)
    linked = _REDUNDANT_URL_LABEL.sub("", linked)
    return _ATTACH.sub("", linked).replace("\N{EM DASH}", "-").strip()


def _split(text: str, limit: int) -> list[str]:
    """Split at natural reading boundaries where possible, never exceeding `limit`."""
    def split_long(value: str) -> tuple[str, str]:
        boundaries = [
            index + 1
            for index in range(min(limit, len(value) - 1))
            if value[index] in ".!?)" and value[index + 1] == " "
        ]
        balanced = [boundary for boundary in boundaries if len(value) - boundary <= limit]
        boundary = (
            min(balanced, key=lambda boundary: abs(boundary - len(value) / 2))
            if balanced
            else max(boundaries, default=limit)
        )
        return value[:boundary].rstrip(), value[boundary:].lstrip()

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
                    chunk, line = split_long(line)
                    chunks.append(chunk)
                current = line
            continue
        # A single oversized paragraph: prefer a sentence boundary, then hard-wrap.
        while len(para) > limit:
            chunk, para = split_long(para)
            chunks.append(chunk)
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
    formats SMS as plain text and WhatsApp in its native dialect, both with cited links inline.
    """
    action_links = tuple(getattr(result, "action_links", ()))
    if getattr(result, "status", None) == "approval_required":
        body = result.text
    elif channel == "console":
        # The REPL is the rich surface: inline {cite:Sn} markers STAY visible (texters lose
        # them only because SMS/WhatsApp can't render them usefully), markdown stays raw for
        # rich, and the sources footer below goes one-per-line instead of the wrapped bullets.
        body = _link_markers(result.text, result.citations)
    else:
        language = (getattr(result, "diagnostics", {}) or {}).get("safety_language")
        text = _inline_citation_links(
            result.text,
            result.citations,
            action_links,
            language,
        )
        body = _render_markdown(text, whatsapp=not channel.startswith("sms"))
    candidate_urls = [
        str(citation.get("url") or "")
        for citation in result.citations.values()
    ] + [action.url for action in action_links]
    body_urls = _urls_in(body)
    inline_urls = {
        _canonical_url(url) for url in candidate_urls
        if url and _canonical_url(url) in body_urls
    }
    # SMS and WhatsApp already have exact cited links inline, so their footer is reserved for source
    # limitations and other notes.
    cited_text = result.text if channel == "console" else _without_code_citations(result.text)
    cited = used_citations(cited_text, result.citations)
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
        footer = ""
    if not footer:
        return _split(body, WA_LIMIT) or [""]
    return _split(f"{body}\n\n{footer}", WA_LIMIT)


def twilio_chunks(text: str) -> list[str]:
    chunks = _split(text, TWILIO_TEXT_LIMIT - TWILIO_PAGE_PREFIX_RESERVE)
    return chunks if len(chunks) < 2 else [
        f"{index}/{len(chunks)} {chunk}" for index, chunk in enumerate(chunks, 1)
    ]


def delivery_chunks(result, channel: str) -> list[str]:
    """Render the exact text parts a channel adapter will send."""
    chunks = render(result, channel)
    if channel not in {"sms_twilio", "whatsapp_twilio"}:
        return chunks
    return [part for chunk in chunks for part in twilio_chunks(chunk)]
