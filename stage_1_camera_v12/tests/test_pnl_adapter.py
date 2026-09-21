import numpy as np
from stage_1_camera.pnlcalib_adapter import PnLCalibAdapter
from stage_1_camera.contracts import CameraStatus


def test_adapter_preserves_pnl_projection_after_world_conversion():
    # PnLCalib/SoccerNet world: pitch-centred, height is negative Z.
    R_pnl=np.eye(3)
    C_pnl=np.array([1.,2.,-10.])
    result={"cam_params":{"x_focal_length":1000.,"y_focal_length":900.,"principal_point":[640.,360.],
                          "position_meters":C_pnl.tolist(),"rotation_matrix":R_pnl.tolist()}}
    cam=PnLCalibAdapter.camera_state_from_result(result,frame_index=5,image_width=1280,image_height=720)

    K=np.array([[1000,0,640],[0,900,360],[0,0,1]],float)
    It=np.eye(4)[:-1]; It[:,-1]=-C_pnl
    P_pnl=K@(R_pnl@It)

    # Compare a canonical Z-up point with the corresponding PnL point.
    X_can=np.array([5.,-3.,2.,1.])
    X_pnl=np.r_[PnLCalibAdapter.canonical_to_pnl_world(X_can[:3]),1.]
    q_pnl=P_pnl@X_pnl; q_pnl=q_pnl[:2]/q_pnl[2]
    q_can=cam.P@X_can; q_can=q_can[:2]/q_can[2]

    assert np.allclose(q_can,q_pnl)
    assert np.isclose(np.linalg.det(cam.R_world_to_camera),1.0)
    assert cam.camera_center_world_m[2] > 0
    assert cam.status==CameraStatus.DEGRADED


def test_adapter_records_explicit_refinement_and_evidence():
    result = {
        "mode": "full",
        "use_ransac": 0,
        "rep_err": 3.2,
        "evidence": {"num_keypoints_used": 9, "num_lines_used": 5},
        "cam_params": {
            "x_focal_length": 1000.0,
            "y_focal_length": 1000.0,
            "principal_point": [640.0, 360.0],
            "position_meters": [0.0, 55.0, -12.0],
            "rotation_matrix": np.eye(3).tolist(),
        },
    }
    cam = PnLCalibAdapter.camera_state_from_result(
        result, frame_index=0, image_width=1280, image_height=720, pnl_refine=True
    )
    assert cam.source["pnl_refine"] is True
    assert cam.evidence["rep_err_px"] == 3.2
    assert cam.evidence["num_keypoints_used"] == 9
    assert cam.evidence["num_lines_used"] == 5
    assert cam.diagnostics["solver_status"] == "SOLVED"
