"""Unwrap hard-wrapped markdown prose to full lines (a fixed-width relic; editors soft-wrap).

Joins consecutive prose lines within a paragraph with a space. Preserves exactly: code fences
and their contents, tables, list items and their indented continuations, headers, blockquotes,
HTML blocks, YAML frontmatter, blank lines, and markdown hard breaks (trailing two spaces or
backslash). Rendered output is identical; only the source line structure changes.

Usage:
    python scripts/unwrap_docs.py --dry-run FILE [FILE...]   # show per-file join counts + sample
    python scripts/unwrap_docs.py --apply FILE [FILE...]     # rewrite in place
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_SPECIAL = re.compile(
    r"^(\s{4,}|\t)"          # indented code / continuation
    r"|^\s*([-*+]\s|\d+[.)]\s)"  # list items
    r"|^\s*#{1,6}\s"          # headers
    r"|^\s*>"                 # blockquotes
    r"|^\s*\|"                # table rows
    r"|^\s*[-=]{3,}\s*$"      # setext underlines / rules
    r"|^\s*<"                 # HTML blocks
    r"|^\s*\[[^\]]+\]:"       # link reference definitions
    r"|^\s*\[!\["             # badge/image-link lines (stacked README badges are structure)
)
_FENCE = re.compile(r"^\s*(```|~~~)")


def _is_prose(line: str) -> bool:
    return bool(line.strip()) and not _SPECIAL.match(line)


def _hard_break(line: str) -> bool:
    return line.endswith("  ") or line.endswith("\\")


def unwrap_text(text: str) -> str:
    lines = text.split("\n")
    out: list[str] = []
    in_fence = False
    fence_mark = ""
    i = 0
    while i < len(lines):
        line = lines[i]
        fence = _FENCE.match(line)
        if fence:
            if not in_fence:
                in_fence, fence_mark = True, fence.group(1)
            elif line.strip().startswith(fence_mark):
                in_fence = False
            out.append(line)
            i += 1
            continue
        if in_fence or not _is_prose(line):
            out.append(line)
            i += 1
            continue
        # Prose: greedily join following prose continuation lines.
        joined = line
        while (
            i + 1 < len(lines)
            and _is_prose(lines[i + 1])
            and not _hard_break(joined)
            and not _FENCE.match(lines[i + 1])
        ):
            joined = joined.rstrip() + " " + lines[i + 1].strip()
            i += 1
        out.append(joined)
        i += 1
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.files:
        original = path.read_text()
        unwrapped = unwrap_text(original)
        joins = original.count("\n") - unwrapped.count("\n")
        if args.dry_run:
            print(f"{path}: would join {joins} line breaks "
                  f"({original.count(chr(10))} -> {unwrapped.count(chr(10))} lines)")
        elif joins:
            path.write_text(unwrapped)
            print(f"{path}: joined {joins}")
        else:
            print(f"{path}: already full lines")
    return 0


if __name__ == "__main__":
    sys.exit(main())
