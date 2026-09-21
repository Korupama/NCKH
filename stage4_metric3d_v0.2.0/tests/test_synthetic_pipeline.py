import json
from pathlib import Path
from benchmark_stage4 import synthetic_smoke, build_synthetic_case
from stage4_metric3d.processor import preflight_stage4
from stage4_metric3d.schemas import Stage4Config


def test_end_to_end_synthetic(tmp_path):
    result = synthetic_smoke(tmp_path / "smoke")
    m = result["summary"]["metrics"]
    assert result["stage4_metrics"]["MetricPoseCoverageAtT0_given_stage3_candidate"] == 1.0
    assert m["GlobalMPJPE_m"] < 0.20
    assert m["LongitudinalMAE_m"] < 0.05
    assert m["GALE_mean_m"] < 0.05


def test_preflight_rejects_resolution_mismatch(tmp_path):
    case = build_synthetic_case(tmp_path / "case")
    stage3 = json.loads(Path(case["stage3"]).read_text())
    stage3["replay_context"]["image_width"] = 1920
    Path(case["stage3"]).write_text(json.dumps(stage3))
    report = preflight_stage4(stage3_state=case["stage3"], camera_dir=case["camera_dir"], initializer_cache=case["initializer"], config=Stage4Config())
    assert not report["ready"]
    assert "stage3_camera_image_resolution_mismatch" in report["errors"]
