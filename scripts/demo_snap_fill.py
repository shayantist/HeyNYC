#!/usr/bin/env python
"""Deterministic end-to-end demo of the SNAP form-fill — no LLM, no phone needed.

    uv run python scripts/demo_snap_fill.py

Shows: the structured draft accumulating across turns (the model "forgets" the name on
turn 2 but the draft keeps it), the attestation review, and the filled official PDF.
"""
import asyncio
import tempfile
from pathlib import Path

from heynyc.core.citations import CitationRegistry
from heynyc.core.drafts import DraftStore
from heynyc.core.registry import Registry
from heynyc.core.tools.base import ToolContext
from heynyc.modules.benefits import tools as btools


async def main() -> None:
    work = Path(tempfile.mkdtemp(prefix="snap-demo-"))
    drafts = DraftStore(work / "drafts").for_user("demo-user")
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]),
                      http=None, output_dir=work, drafts=drafts)

    print("== Turn 1: the user gives only their name ==")
    print(await btools._prepare_application_handler({"slots": {"legal_name": "Ana Diaz"}}, ctx))
    print("   persisted draft:", drafts.load("snap"))

    print("\n== Turn 2: the user gives their address — the model does NOT re-send the name ==")
    print(await btools._prepare_application_handler({"slots": {
        "residence_street": "123 Grand Concourse Apt 4B",
        "residence_city": "Bronx", "residence_zip": "10453"}}, ctx))
    print("   persisted draft:", drafts.load("snap"), "  <- name retained from turn 1 (real state)")

    print("\n== Turn 3: the user confirms → the filled PDF is produced ==")
    print(await btools._prepare_application_handler({"slots": {}, "confirmed": True}, ctx))
    final = Path("snap-demo.pdf")
    final.write_bytes(next(work.glob("*.pdf")).read_bytes())
    print(f"\n📄 Filled LDSS-4826 written to: {final.resolve()}")
    print("   Open it — name + address are filled on the official form.")


if __name__ == "__main__":
    asyncio.run(main())
