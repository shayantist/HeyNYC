from heynyc.core import config
from heynyc.core.manifest import ServiceModule


def test_snap_work_rule_prompt_does_not_force_an_unrequested_appeal():
    module = ServiceModule.from_manifest(config.MODULES_DIR / "benefits" / "manifest.yaml")
    situation = next(item for item in module.situations if item.name == "snap_work_rules")

    assert "Include a fair-hearing path only when the resident asks" in situation.reminder
    assert "one focused search for both a request route" not in situation.reminder
    assert "SNAP-specific deadline" in situation.reminder
    assert "Do not add a conditional appeal branch to a reporting-only answer" in situation.reminder
    assert "For immediate food, fetch https://finder.nyc.gov/foodhelp/" in situation.reminder
    assert "first actionable paragraph after one line of empathy" in situation.reminder
    assert "Use https://portal.311.nyc.gov/article/?kanumber=KA-02943 only for Fair Hearing" in situation.reminder
    assert "neighborhood or landmark, not a full address" in situation.reminder
    assert "Able-Bodied Adults Without Dependents (ABAWD)" in situation.reminder
    assert "alternatives, not cumulative requirements" in situation.reminder
    assert "Do not join them as if the resident must complete every path" in situation.reminder
    assert "required number of hours each month" in situation.reminder
    assert "Do not imply that any amount of participation is enough" in situation.reminder
    assert "Keep the notice condition and retrieved Fair Hearing routes in separate claim blocks" in situation.reminder
    assert "fair hearing" not in situation.query.lower()
    assert "https://otda.ny.gov/oah/" not in situation.urls
    assert "https://portal.311.nyc.gov/article/?kanumber=KA-02943" in situation.urls
    assert "https://otda.ny.gov/hearings/faq.asp" in situation.urls
    assert "https://otda.ny.gov/hearings/request/" not in situation.urls
    assert "https://finder.nyc.gov/foodhelp/" in situation.urls
    assert "https://access.nyc.gov/snap-work-requirements/" not in situation.urls
    assert "Fair Hearings line only for requesting or discussing the hearing" in situation.reminder
    assert "never for choosing a work activity or exemption" in situation.reminder
    assert "Use a retrieved PACE or HRA route" in situation.reminder
    assert "Required outcomes:" in situation.reminder
    assert "work 80 hours" in situation.reminder
    assert "denial notice" in situation.reminder
    assert "not a reduction notice" in situation.reminder
    assert "If you received a denial notice" in situation.reminder
    assert "separate claim block" in situation.reminder
    assert "Use the OTDA FAQ only for the SNAP-specific deadline" in situation.reminder
    assert "Do not use the NYC 311 page for a SNAP deadline" in situation.reminder
    assert "evidence_scope=\"online phone fax mail in-person Fair Hearing request methods\"" in situation.reminder
    assert "evidence_scope=\"SNAP benefits denied stopped reduced Fair Hearing time limit 60 90 days\"" in situation.reminder
