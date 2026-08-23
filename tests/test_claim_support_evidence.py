from heynyc.core.pydantic_runtime.projection import _claim_support_evidence


def test_data_claim_support_evidence_uses_public_snapshot_and_derivation() -> None:
    evidence = _claim_support_evidence({
        "kind": "DATA",
        "snippet": "Queens SNAP Center, Queens",
        "valid_as_of": "2026-07-01T17:24:46.099Z",
        "provenance": {
            "record_id": "row-e8rg_fvak_gvqu",
            "snapshot": {
                "street_address": "32-20 Northern Blvd., 2nd Fl.",
                "comments": "Monday - Friday 8:30am to 5:00pm",
            },
            "derivation": {"distance_mi": 2.4942654503383306},
        },
    })

    assert "32-20 Northern Blvd." in evidence
    assert "8:30am to 5:00pm" in evidence
    assert "2.4942654503383306" in evidence
    assert "2026-07-01T17:24:46.099Z" in evidence
    assert len(evidence) <= 1_200


def test_web_claim_support_evidence_does_not_expose_provenance() -> None:
    evidence = _claim_support_evidence({
        "kind": "WEB",
        "snippet": "Official guidance",
        "provenance": {"private_irrelevant_field": "must stay local"},
    })

    assert evidence == "Official guidance"
