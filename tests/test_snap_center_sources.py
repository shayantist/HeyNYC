from pathlib import Path

from heynyc.core.manifest import ServiceModule


def test_snap_center_handoffs_are_declared_retrieval_sources():
    module = ServiceModule.from_manifest(
        Path("heynyc/modules/snap_centers/manifest.yaml")
    )

    assert "https://www.nyc.gov/site/hra/help/snap-benefits-food-program.page" in module.seeds
    assert "https://www.nyc.gov/site/hra/help/access-hra-resources.page" in module.seeds
    assert "https://www.nyc.gov/site/hra/help/snap-interview-instructions.page" in module.seeds
    assert "https://www.nyc.gov/site/hra/about/contact.page" in module.seeds
    assert all(url in module.prompt for url in module.seeds)
    assert "web_fetch" in module.prompt
    assert (
        'nearest(category="snap_center", near=<the resident\'s supplied location>, max_results=1)'
        in module.prompt
    )
    assert "call 311" not in module.prompt.casefold()
    assert "include a phone handoff only when the retrieved page explicitly supports" in module.prompt.casefold()
    assert "phone handoff is for general snap help, never schedule confirmation" in module.prompt.casefold()
    assert "how to confirm current hours" not in module.prompt.casefold()
    assert "failed upload or near deadline" in module.prompt.casefold()
    assert "never promise a deadline extension" in module.prompt.casefold()
    assert "not a confirmed document-upload channel" in module.prompt.casefold()
    assert "dss onenumber" in module.prompt.casefold()
    assert "do not block urgent help" in module.prompt.casefold()
    assert "precise origin" in module.prompt.casefold()
    assert "in-person and online application options" in module.prompt.casefold()
    assert "runtime appends the structured dataset limitation" in module.prompt.casefold()
