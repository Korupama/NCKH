from __future__ import annotations

import numpy as np

from stage4_metric3d.camera import CameraStateLite
from stage4_metric3d.model_only_refinement import refine_model_only_translation
from stage4_metric3d.backends.sam3d_pitch_refined.joint_mapping import DEFAULT_MAPPING


def _camera() -> CameraStateLite:
    return CameraStateLite.from_dict({
        "schema_version": "benchmark-camera-1.0",
        "frame_index": 0,
        "status": "VALID",
        "image": {"width": 200, "height": 120},
        "intrinsics": {"K": [[100.0, 0.0, 100.0], [0.0, 100.0, 60.0], [0.0, 0.0, 1.0]]},
        "extrinsics": {"R_world_to_camera": np.eye(3).tolist(), "camera_center_world_m": [0.0, 0.0, 10.0]},
        "distortion": {"radial": [0, 0, 0, 0, 0, 0], "tangential": [0, 0]},
        "pitch": {"origin": "center", "length_m": 105.0, "width_m": 68.0},
    })


def _native_inputs(*names: str, uv=(100.0, 60.0), relative=(0.0, 0.0, 1.0)):
    native_2d = np.full((70, 2), np.nan, dtype=np.float64)
    native_3d = np.zeros((70, 3), dtype=np.float64)
    for canonical in names:
        entry = next(item for item in DEFAULT_MAPPING if item.canonical_name == canonical)
        native_2d[entry.sam3d_index] = uv
        native_3d[entry.sam3d_index] = relative
    return native_2d, native_3d


def _downward_camera(*, pitch=None) -> CameraStateLite:
    return CameraStateLite.from_dict({
        "schema_version": "benchmark-camera-1.0", "frame_index": 0, "status": "VALID",
        "image": {"width": 200, "height": 120},
        "intrinsics": {"K": [[100.0, 0.0, 100.0], [0.0, 100.0, 60.0], [0.0, 0.0, 1.0]]},
        "extrinsics": {"R_world_to_camera": np.diag([1.0, -1.0, -1.0]).tolist(), "camera_center_world_m": [0.0, 0.0, 10.0]},
        "distortion": {"radial": [0, 0, 0, 0, 0, 0], "tangential": [0, 0]},
        "pitch": pitch or {"origin": "center", "length_m": 105.0, "width_m": 68.0},
    })


def test_ground_first_returns_supported_anchor_without_stage3():
    native_2d = np.full((70, 2), np.nan, dtype=np.float64)
    native_3d = np.zeros((70, 3), dtype=np.float64)
    # A right-handed camera rotation looking down at the pitch plane.
    camera = CameraStateLite.from_dict({
        "schema_version": "benchmark-camera-1.0", "frame_index": 0, "status": "VALID",
        "image": {"width": 200, "height": 120},
        "intrinsics": {"K": [[100.0, 0.0, 100.0], [0.0, 100.0, 60.0], [0.0, 0.0, 1.0]]},
        "extrinsics": {"R_world_to_camera": np.diag([1.0, -1.0, -1.0]).tolist(), "camera_center_world_m": [0.0, 0.0, 10.0]},
        "distortion": {"radial": [0, 0, 0, 0, 0, 0], "tangential": [0, 0]},
        "pitch": {"origin": "center", "length_m": 105.0, "width_m": 68.0},
    })
    for canonical in ("left_big_toe", "left_small_toe", "left_heel", "right_big_toe", "right_small_toe", "right_heel"):
        entry = next(item for item in DEFAULT_MAPPING if item.canonical_name == canonical)
        native_2d[entry.sam3d_index] = [100.0, 60.0]
        native_3d[entry.sam3d_index] = [0.0, 0.0, 1.0]
    result = refine_model_only_translation(
        camera=camera, native_2d=native_2d, native_3d_relative_m=native_3d,
        sam_prior_cam_m=np.array([0.5, 0.0, 8.0]),
    )
    assert result.status == "CONTACT_SUPPORTED"
    assert result.candidate_count == 6
    assert result.consensus_count == 6
    assert result.source == "DISTAL_FOOT"
    assert result.consensus_fraction == 1.0
    assert result.correction_bounded is True
    assert np.isfinite(result.refined_root_cam_m).all()


def test_single_foot_does_not_support_ground():
    native_2d, native_3d = _native_inputs("left_big_toe")
    result = refine_model_only_translation(
        camera=_downward_camera(), native_2d=native_2d, native_3d_relative_m=native_3d,
        sam_prior_cam_m=np.array([0.0, 0.0, 9.0]),
    )
    assert result.status == "CONTACT_AMBIGUOUS"
    assert result.reason == "insufficient_ground_consensus"
    assert result.candidate_count == 1
    assert result.consensus_count == 1
    assert result.anchor_root_cam_m is None
    assert np.array_equal(result.refined_root_cam_m, np.array([0.0, 0.0, 9.0]))


def test_two_mismatched_feet_do_not_support_ground():
    native_2d, native_3d = _native_inputs("left_big_toe", "right_big_toe")
    right = next(item for item in DEFAULT_MAPPING if item.canonical_name == "right_big_toe")
    native_2d[right.sam3d_index] = [150.0, 60.0]
    result = refine_model_only_translation(
        camera=_downward_camera(), native_2d=native_2d, native_3d_relative_m=native_3d,
        sam_prior_cam_m=np.array([0.0, 0.0, 9.0]),
    )
    assert result.status == "CONTACT_AMBIGUOUS"
    assert result.reason == "insufficient_ground_consensus"
    assert result.consensus_count == 1
    assert result.anchor_root_cam_m is None


def test_consensus_ground_requires_multiple_candidates():
    native_2d, native_3d = _native_inputs("left_big_toe", "left_small_toe", "right_big_toe")
    result = refine_model_only_translation(
        camera=_downward_camera(), native_2d=native_2d, native_3d_relative_m=native_3d,
        sam_prior_cam_m=np.array([0.0, 0.0, 9.0]),
    )
    assert result.status == "CONTACT_SUPPORTED"
    assert result.candidate_count == 3
    assert result.consensus_count == 3
    assert result.anchor_root_cam_m is not None


def test_ankle_fallback_is_explicit_and_requires_both_ankles():
    native_2d, native_3d = _native_inputs("left_ankle", "right_ankle")
    result = refine_model_only_translation(
        camera=_downward_camera(), native_2d=native_2d, native_3d_relative_m=native_3d,
        sam_prior_cam_m=np.array([0.0, 0.0, 9.0]),
    )
    assert result.status == "CONTACT_SUPPORTED"
    assert result.source == "ANKLE_FALLBACK"
    assert result.candidate_count == 2
    assert result.consensus_count == 2


def test_invalid_pitch_fails_closed():
    native_2d, native_3d = _native_inputs("left_big_toe", "right_big_toe")
    result = refine_model_only_translation(
        camera=_downward_camera(pitch={"origin": "corner", "length_m": 105.0, "width_m": 68.0}),
        native_2d=native_2d, native_3d_relative_m=native_3d,
        sam_prior_cam_m=np.array([0.0, 0.0, 9.0]),
    )
    assert result.status == "GROUND_UNAVAILABLE"
    assert result.reason == "no_usable_ground_candidate"
    assert sum(reason.endswith("invalid_pitch_geometry") for reason in result.rejection_reasons) == 2


def test_invalid_camera_fails_closed():
    native_2d, native_3d = _native_inputs("left_big_toe", "right_big_toe")
    camera = _downward_camera()
    camera.status = "DEGRADED"
    result = refine_model_only_translation(
        camera=camera, native_2d=native_2d, native_3d_relative_m=native_3d,
        sam_prior_cam_m=np.array([0.0, 0.0, 9.0]),
    )
    assert result.status == "GROUND_UNAVAILABLE"
    assert result.reason == "invalid_camera_state"


def test_correction_is_bounded_and_recorded():
    native_2d, native_3d = _native_inputs("left_big_toe", "left_small_toe")
    prior = np.array([0.0, 0.0, 7.0])
    result = refine_model_only_translation(
        camera=_downward_camera(), native_2d=native_2d, native_3d_relative_m=native_3d,
        sam_prior_cam_m=prior, max_correction_m=0.5,
    )
    assert result.status == "CONTACT_SUPPORTED"
    assert result.correction_bounded is True
    np.testing.assert_allclose(np.linalg.norm(result.refined_root_cam_m - prior), 0.5, atol=1e-9)
