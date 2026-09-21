import numpy as np
from stage_1_camera.contracts import CameraState, CameraStatus
from stage_1_camera.evaluation.offside_geometry import ground_longitudinal_error, vertical_plane_projection_error
from test_geometry import look_at_camera


def make_cam(C=np.array([0.,-60.,20.]),f=1200.):
    R=look_at_camera(C); K=np.array([[f,0,640],[0,f,360],[0,0,1]],float)
    return CameraState(0,1280,720,K,R,C,status=CameraStatus.VALID)


def test_zero_error_metrics():
    gt=make_cam(); pred=make_cam()
    pts=np.array([[-40,0,0],[-10,5,0],[20,-7,0]],float)
    gle=ground_longitudinal_error(pred,gt,pts); vppe=vertical_plane_projection_error(pred,gt,[-20,0,20])
    assert gle['p95_m']<1e-8
    assert vppe['p95_px']<1e-8
