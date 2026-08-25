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


def test_public_source_catalog_records_notify_nyc_filtering_limit() -> None:
    catalog = Path("docs/source-capability-catalog.md").read_text()

    assert "RecentMessages?lang=en" in catalog
    assert "no documented topic-filtering API" in catalog
    assert "does not assume that CAP identifiers link translations" in catalog
    assert "2026-08-24" in catalog
