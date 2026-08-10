from heynyc.core.pydantic_runtime.projection import GroundedAnswer


def test_f115_prohibition_cannot_authorize_a_positive_instruction():
    description = GroundedAnswer.model_json_schema()["$defs"]["GroundedBlock"][
        "properties"
    ]["text"]["description"]

    assert "Do not turn a cited prohibition into an unsupported positive instruction" in description
