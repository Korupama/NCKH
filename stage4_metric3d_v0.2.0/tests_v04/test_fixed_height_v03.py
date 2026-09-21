import json

from stage4_metric3d.backends.fixed_height_v03 import run_fixed_height_v03
from .helpers import build_synthetic_stage3_and_cameras


def test_fixed_height_baseline_is_retained(tmp_path):
    stage3, camera_dir, _, _ = build_synthetic_stage3_and_cameras(tmp_path / "case")
    state = run_fixed_height_v03(
        stage3_state=stage3, camera_dir=camera_dir,
        output_dir=tmp_path / "out", selected_frame_only=True,
    )
    assert state["method"] == "FIXED_HEIGHT_HORIZONTAL_PLANE_BASELINE"
    assert state["research_accuracy_frozen"] is False
    assert (tmp_path / "out/metric_body_proxy_state_v03_baseline.json").is_file()
