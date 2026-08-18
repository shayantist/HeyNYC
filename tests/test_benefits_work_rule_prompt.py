from heynyc.core import config
from heynyc.core.manifest import ServiceModule


def test_snap_work_rule_prompt_does_not_force_an_unrequested_appeal():
    module = ServiceModule.from_manifest(config.MODULES_DIR / "benefits" / "manifest.yaml")
    situation = next(item for item in module.situations if item.name == "snap_work_rules")

    assert "Include a fair-hearing path only when the resident asks" in situation.reminder
    assert "label it unverified and preserve its URL" in situation.reminder
    assert "Do not add a conditional appeal branch to a reporting-only answer" in situation.reminder
    assert "fair hearing" not in situation.query.lower()
    assert "https://otda.ny.gov/oah/" not in situation.urls
