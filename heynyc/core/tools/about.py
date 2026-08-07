"""`about_heynyc`: serve HeyNYC's OWN shipped docs (the privacy notice + the README FAQ) as
citable DOC sources, so questions about HeyNYC itself, "what are you", "what do you do with my
data", "how do I delete my data", are answered from the running code's real files instead of the
model's memory. The deployed code and its docs ship together, so this answer can't drift from what
the service actually does the way a memorized description can."""
from __future__ import annotations

import subprocess
from pathlib import Path

from .. import config
from .base import Tool, ToolContext

# The running service's own files are the source of truth for self-description
_PRIVACY = config.PROJECT_ROOT / "PRIVACY.md"
_README = config.PROJECT_ROOT / "README.md"


def _git_revision() -> str:
    try:
        revision = subprocess.check_output(
            ("git", "-C", str(config.PROJECT_ROOT), "rev-parse", "HEAD"),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip().lower()
    except (OSError, subprocess.CalledProcessError):
        return ""
    valid = len(revision) == 40 and all(char in "0123456789abcdef" for char in revision)
    return revision if valid else ""


_DOCS_REVISION = _git_revision()
_DOCS_BASE = (
    f"https://github.com/shayantist/HeyNYC/blob/{_DOCS_REVISION}"
    if _DOCS_REVISION
    else ""
)


def _faq_section(readme_text: str) -> str:
    """The README's `## FAQ` block, up to the next `## ` heading (or EOF). Empty if there is none."""
    out: list[str] = []
    in_faq = False
    for line in readme_text.splitlines():
        if line.strip().lower() == "## faq":
            in_faq = True
            continue
        if in_faq and line.startswith("## "):
            break
        if in_faq:
            out.append(line)
    return "\n".join(out).strip()


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def about_tools() -> list[Tool]:
    async def _handler(args: dict, ctx: ToolContext) -> str:
        if not _DOCS_BASE:
            return "HeyNYC's own documentation could not be linked right now."

        sources = [
            ("HeyNYC Privacy Notice", f"{_DOCS_BASE}/PRIVACY.md", _read(_PRIVACY)),
            ("HeyNYC FAQ", f"{_DOCS_BASE}/README.md#faq", _faq_section(_read(_README))),
        ]
        blocks: list[str] = []
        for title, url, text in sources:
            if not text:
                continue
            cite = ctx.citations.register(url, snippet=text, title=title, kind="DOC")
            blocks.append(f"{title}\n{text} {{cite:{cite}}}")
        if not blocks:
            return "HeyNYC's own documentation could not be read right now."
        return (
            "HeyNYC's shipped documentation follows. Answer questions about HeyNYC itself only "
            "from this text, quote it closely, and cite each fact with its {cite:Sn}:\n\n"
            + "\n\n".join(blocks)
        )

    return [
        Tool(
            name="about_heynyc",
            description=(
                "Answer questions about HeyNYC ITSELF by returning HeyNYC's own shipped privacy "
                "notice and README FAQ as citable sources. Use it for 'what are you / what can you "
                "do / what do you do with my messages / what do you store / how do I delete my "
                "data' so the reply quotes the running service's real documents instead of memory. "
                "Returns document (DOC) sources to cite."
            ),
            parameters={"type": "object", "properties": {}},
            handler=_handler,
        )
    ]
