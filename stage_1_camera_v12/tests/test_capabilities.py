import numpy as np

from stage_1_camera.contracts import CameraState, CameraStatus
from stage_1_camera.quality_gate import CameraQualityGate


def make_cam(mode='full', ransac=0, rep=3.0):
    cam = CameraState(
        frame_index=0, image_width=1920, image_height=1080,
        K=np.array([[1800.,0,960.],[0,1800.,540.],[0,0,1.]]),
        R_world_to_camera=np.eye(3), camera_center_world_m=np.array([0.,-55.,10.]),
        status=CameraStatus.DEGRADED,
        source={'backend':'pnlcalib','pnl_refine':True},
        evidence={
            'rep_err_px':rep, 'mode':mode, 'use_ransac':ransac,
            'num_keypoints_used':16, 'num_lines_used':1,
            'keypoint_image_convex_hull_area_ratio':0.09,
            'keypoint_image_x_span_ratio':0.60,
            'keypoint_image_y_span_ratio':0.21,
            'keypoint_image_quadrants_occupied':3,
        },
        diagnostics={'solver_status':'SOLVED'},
    )
    return cam


def test_full_no_ransac_is_vertical_ready():
    cam = make_cam('full', 0, 3.2)
    CameraQualityGate().evaluate(cam)
    assert cam.status == CameraStatus.VALID
    caps = cam.diagnostics['capabilities']
    assert caps['ground_geometry']['status'] == 'VALID'
    assert caps['vertical_3d']['status'] == 'VALID'
    assert caps['offside_3d_ready'] is True


def test_ground_fallback_can_be_ground_valid_but_vertical_degraded():
    cam = make_cam('ground_plane', 5, 2.0)
    CameraQualityGate().evaluate(cam)
    assert cam.status == CameraStatus.DEGRADED
    caps = cam.diagnostics['capabilities']
    assert caps['ground_geometry']['status'] == 'VALID'
    assert caps['ground_geometry']['ground_unprojection_ready'] is True
    assert caps['vertical_3d']['status'] == 'DEGRADED'
    assert caps['offside_3d_ready'] is False


def test_vertical_policy_requires_eight_keypoints_even_when_general_camera_is_valid():
    cam = make_cam('full', 0, 3.0)
    cam.evidence['num_keypoints_used'] = 7
    CameraQualityGate().evaluate(cam)
    assert cam.status == CameraStatus.VALID  # general/ground gate intentionally remains permissive
    caps = cam.diagnostics['capabilities']
    assert caps['ground_geometry']['status'] == 'VALID'
    assert caps['vertical_3d']['status'] == 'DEGRADED'
    assert any('(<8)' in r for r in caps['vertical_3d']['reasons'])
    assert caps['offside_3d_ready'] is False


def test_vertical_policy_requires_three_quadrants_even_when_general_camera_is_valid():
    cam = make_cam('full', 0, 3.0)
    cam.evidence['keypoint_image_quadrants_occupied'] = 2
    CameraQualityGate().evaluate(cam)
    assert cam.status == CameraStatus.VALID
    caps = cam.diagnostics['capabilities']
    assert caps['ground_geometry']['status'] == 'VALID'
    assert caps['vertical_3d']['status'] == 'DEGRADED'
    assert any('(<3)' in r for r in caps['vertical_3d']['reasons'])


def test_frozen_policy_provenance_is_exported():
    cam = make_cam('full', 0, 3.0)
    CameraQualityGate().evaluate(cam)
    policies = cam.diagnostics['capabilities']['policies']
    assert policies['vertical_3d']['thresholds_frozen'] is True
    assert policies['vertical_3d']['threshold_source'] == 'SoccerNet Calibration-2023 valid'
    assert policies['vertical_3d']['min_keypoints'] == 8
    assert policies['vertical_3d']['min_image_quadrants'] == 3
    assert policies['calibration_gate_is_not_offside_accuracy'] is True
