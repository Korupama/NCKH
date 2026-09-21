import numpy as np

from stage4_metric3d.backends.field_converter.pipeline import preflight_v04
from stage4_metric3d.backends.field_converter.sam3d_cache import save_sam3d_cache
from stage4_metric3d.backends.field_converter.exporter import build_boxes_from_stage3
from stage4_metric3d.stage3_adapter import load_stage3_state
from .helpers import build_synthetic_stage3_and_cameras


def test_preflight_adapter_only_and_cache_validation(tmp_path):
    stage3, camera_dir, frames, tids = build_synthetic_stage3_and_cameras(tmp_path / "case")
    adapter_only = preflight_v04(
        stage3_state=stage3, camera_dir=camera_dir,
        sam3d_cache=None, bundle=None, probe_external_python=False,
    )
    assert adapter_only["camera_projection_compatibility_status"] == "PASS"
    assert adapter_only["camera_missing_or_invalid_frames"] == []

    s3 = load_stage3_state(stage3)
    boxes = build_boxes_from_stage3(s3, frames, tids)
    T, N = len(frames), len(tids)
    cache = save_sam3d_cache(
        tmp_path / "cache.npz", frame_indices=frames, track_ids=tids,
        boxes_xyxy=boxes,
        skel_2d_px=np.ones((T,N,25,2), np.float32),
        skel_3d_relative_m=np.ones((T,N,25,3), np.float32),
        valid_mask=np.ones((T,N), bool),
    )
    full_without_model = preflight_v04(
        stage3_state=stage3, camera_dir=camera_dir,
        sam3d_cache=cache, bundle=None, probe_external_python=False,
    )
    assert full_without_model["sam3d_cache"]["ready"] is True
    assert "sam3d_joint_semantics_not_validated_for_stage8" in full_without_model["warnings"]
