from validate_stage4_v031 import validate


def test_synthetic_v031_pipeline():
    result = validate()
    assert result["status"] == "PASS"
    assert result["projection_coverage_at_t0"]["rate"] == 1.0
