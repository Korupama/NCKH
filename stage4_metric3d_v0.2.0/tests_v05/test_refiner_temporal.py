from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from stage4_metric3d.backends.sam3d_pitch_refined.config import Sam3DPitchRefinedConfig
from stage4_metric3d.backends.sam3d_pitch_refined.refiner import (
    FrameEvidence,
    refine_translation_sequence,
    temporal_triplet_indices,
)
from stage4_metric3d.camera import CameraStateLite


def _camera(status: str = "VALID") -> CameraStateLite:
    return CameraStateLite.from_dict({
        "schema_version": "benchmark-camera-1.0",
        "frame_index": 0,
        "status": status,
        "image": {"width": 200, "height": 120},
        "intrinsics": {"K": [[100.0, 0.0, 100.0], [0.0, 100.0, 60.0], [0.0, 0.0, 1.0]]},
        "extrinsics": {
            "R_world_to_camera": np.diag([1.0, -1.0, -1.0]).tolist(),
            "camera_center_world_m": [0.0, 0.0, 10.0],
        },
        "distortion": {"radial": [0, 0, 0, 0, 0, 0], "tangential": [0, 0]},
        "pitch": {"origin": "center", "length_m": 105.0, "width_m": 68.0},
    })


def _frame(index: int, *, camera_status: str = "VALID") -> FrameEvidence:
    return FrameEvidence(
        frame_index=index, camera=_camera(camera_status),
        observation=SimpleNamespace(
            uv23=np.full((23, 2), np.nan, dtype=np.float64),
            state_weights23=np.zeros(23, dtype=np.float64),
        ),
        relative_joints_m=np.zeros((70, 3), dtype=np.float64),
        sam_prior_cam_m=np.asarray([0.0, 0.0, 5.0]),
        sam_2d_px=np.full((70, 2), np.nan, dtype=np.float64),
        ground_anchor=None,
    )


def test_temporal_triplets_require_exact_consecutive_frames():
    assert temporal_triplet_indices([_frame(10), _frame(11), _frame(12)]) == ((0, 1, 2),)
    assert temporal_triplet_indices([_frame(10), _frame(12), _frame(14)]) == ()


def test_temporal_triplet_rejects_non_valid_camera():
    assert temporal_triplet_indices([_frame(10), _frame(11, camera_status="DEGRADED"), _frame(12)]) == ()


def test_optimizer_exception_fails_closed_to_initial_sam_prior(monkeypatch):
    def fail_optimizer(*args, **kwargs):
        raise RuntimeError("synthetic solver failure")

    monkeypatch.setattr(
        "stage4_metric3d.backends.sam3d_pitch_refined.refiner.least_squares",
        fail_optimizer,
    )
    frames = [_frame(10), _frame(11), _frame(12)]
    result = refine_translation_sequence(frames, Sam3DPitchRefinedConfig())
    np.testing.assert_allclose(result.refined_cam_m, np.tile([0.0, 0.0, 5.0], (3, 1)))
    assert result.success is False
    assert result.fallback_to_initial is True
    assert result.status == -1
    assert "synthetic solver failure" in result.message
    assert result.temporal_triplet_count == 1
    assert result.reprojection_joint_count == 0
    assert result.invalid_reprojection_joint_count == 0


def test_gapped_sequence_records_skipped_temporal_triplet():
    result = refine_translation_sequence(
        [_frame(10), _frame(12), _frame(14)],
        Sam3DPitchRefinedConfig(),
    )
    assert result.temporal_triplet_count == 0
    assert result.skipped_temporal_triplet_count == 1
