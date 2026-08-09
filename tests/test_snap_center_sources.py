from pathlib import Path

from heynyc.core.manifest import ServiceModule


def test_snap_center_handoffs_are_declared_retrieval_sources():
    module = ServiceModule.from_manifest(
        Path("heynyc/modules/snap_centers/manifest.yaml")
    )

    assert "https://a069-access.nyc.gov/accesshra/" in module.seeds
    assert any("snapfaq" in url for url in module.seeds)
    assert "official_sources" in module.prompt
    assert 'nearest(category="snap_center", near=<the user\'s address>, k=1)' in module.prompt
    assert "call 311" not in module.prompt.casefold()
    assert "include a phone handoff only when the retrieved page explicitly supports" in module.prompt.casefold()
    assert "phone handoff is for general snap help, never schedule confirmation" in module.prompt.casefold()
    assert "how to confirm current hours" not in module.prompt.casefold()
