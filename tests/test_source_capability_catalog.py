import re
from pathlib import Path

from heynyc.core.registry import Registry


def test_public_source_catalog_covers_every_registered_capability() -> None:
    catalog = Path("docs/source-capability-catalog.md").read_text()
    documented = re.findall(r"^\| `([^`]+)` \|", catalog, flags=re.MULTILINE)
    registered = [
        module.name
        for module in Registry.discover(Path("heynyc/modules")).modules
    ]

    assert documented
    assert len(documented) == len(set(documented))
    assert set(documented) == set(registered)


def test_public_source_catalog_attributes_and_limits_every_capability() -> None:
    catalog = Path("docs/source-capability-catalog.md").read_text()
    rows = [line for line in catalog.splitlines() if line.startswith("| `")]

    assert rows
    for row in rows:
        assert "http" in row
        assert row.count("|") == 8
