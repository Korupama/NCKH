import numpy as np

from stage4_metric3d.camera import CameraTimelineLite
from stage4_metric3d.backends.field_converter.camera_adapter import (
    camera_projection_compatibility,
    camera_roundtrip_error_m,
    camera_domain_report,
)
from .helpers import build_synthetic_stage3_and_cameras


def test_camera_adapter_roundtrip_and_projection(tmp_path):
    _, camera_dir, frames, _ = build_synthetic_stage3_and_cameras(tmp_path / "case")
    timeline = CameraTimelineLite.load_dir(camera_dir)
    for f in frames:
        assert camera_roundtrip_error_m(timeline.by_frame(f)) < 1e-10
    report = camera_projection_compatibility(timeline, frames)
    assert report["p95_px"] < 1e-8
    domain = camera_domain_report(timeline, frames)
    assert domain["status"] in {"IN_DOMAIN", "NEAR_DOMAIN", "OUT_OF_DOMAIN"}
