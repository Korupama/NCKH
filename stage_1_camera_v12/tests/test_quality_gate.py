import numpy as np

from stage_1_camera.contracts import CameraStatus
from stage_1_camera.pnlcalib_adapter import PnLCalibAdapter
from stage_1_camera.quality_gate import CameraQualityGate


def make_result(rep_err=3.0, mode="full", use_ransac=0, n_kp=12, n_lines=8, coverage=True):
    evidence = {
        "num_keypoints_used": n_kp,
        "num_lines_used": n_lines,
        "kp_threshold": 0.3434,
        "line_threshold": 0.7867,
    }
    if coverage:
        evidence.update({
            "keypoint_image_convex_hull_area_ratio": 0.10,
            "keypoint_image_bbox_area_ratio": 0.18,
            "keypoint_image_x_span_ratio": 0.70,
            "keypoint_image_y_span_ratio": 0.25,
            "keypoint_image_quadrants_occupied": 4,
            "keypoint_world_x_span_m": 70.0,
            "keypoint_world_y_span_m": 35.0,
        })
    return {
        "mode": mode,
        "use_ransac": use_ransac,
        "rep_err": rep_err,
        "evidence": evidence,
        "cam_params": {
            "x_focal_length": 1200.0,
            "y_focal_length": 1200.0,
            "principal_point": [640.0, 360.0],
            "position_meters": [0.0, 60.0, -20.0],
            "rotation_matrix": np.eye(3).tolist(),
        },
    }


def test_preferred_pnl_solution_can_be_valid():
    cam = PnLCalibAdapter.camera_state_from_result(
        make_result(), frame_index=0, image_width=1280, image_height=720, pnl_refine=True
    )
    status = CameraQualityGate().evaluate(cam)
    assert status == CameraStatus.VALID
    assert cam.source["pnl_refine"] is True
    assert cam.diagnostics["line_refinement"]["effective"] is True


def test_no_line_evidence_is_warning_not_degradation_when_keypoint_camera_is_strong():
    cam = PnLCalibAdapter.camera_state_from_result(
        make_result(n_lines=0), frame_index=0, image_width=1280, image_height=720, pnl_refine=True
    )
    status = CameraQualityGate().evaluate(cam)
    assert status == CameraStatus.VALID
    q = cam.diagnostics["quality_gate"]
    assert q["degraded_reasons"] == []
    assert any("no line evidence" in w for w in q["warnings"])
    lr = cam.diagnostics["line_refinement"]
    assert lr["requested"] is True
    assert lr["effective"] is False
    assert lr["reason"] == "no_line_evidence_above_threshold"


def test_fallback_solution_is_degraded_not_fake_valid():
    cam = PnLCalibAdapter.camera_state_from_result(
        make_result(rep_err=8.0, mode="ground_plane", use_ransac=10),
        frame_index=0,
        image_width=1280,
        image_height=720,
        pnl_refine=True,
    )
    status = CameraQualityGate().evaluate(cam)
    assert status == CameraStatus.DEGRADED
    reasons = cam.diagnostics["quality_gate"]["degraded_reasons"]
    assert any("rep_err" in r for r in reasons)
    assert any("fallback" in r for r in reasons)


def test_high_reprojection_error_is_invalid():
    cam = PnLCalibAdapter.camera_state_from_result(
        make_result(rep_err=50.0), frame_index=0, image_width=1280, image_height=720, pnl_refine=True
    )
    assert CameraQualityGate().evaluate(cam) == CameraStatus.INVALID


def test_missing_spatial_coverage_is_degraded():
    cam = PnLCalibAdapter.camera_state_from_result(
        make_result(coverage=False), frame_index=0, image_width=1280, image_height=720, pnl_refine=False
    )
    assert CameraQualityGate().evaluate(cam) == CameraStatus.DEGRADED
    reasons = cam.diagnostics["quality_gate"]["degraded_reasons"]
    assert any("spatial-coverage" in r for r in reasons)


def test_spatially_clustered_keypoints_are_degraded():
    result = make_result()
    result["evidence"].update({
        "keypoint_image_convex_hull_area_ratio": 0.001,
        "keypoint_image_x_span_ratio": 0.05,
        "keypoint_image_y_span_ratio": 0.03,
        "keypoint_image_quadrants_occupied": 1,
    })
    cam = PnLCalibAdapter.camera_state_from_result(
        result, frame_index=0, image_width=1280, image_height=720, pnl_refine=False
    )
    assert CameraQualityGate().evaluate(cam) == CameraStatus.DEGRADED
    reasons = cam.diagnostics["quality_gate"]["degraded_reasons"]
    assert any("hull ratio" in r for r in reasons)
    assert any("x-span" in r for r in reasons)
