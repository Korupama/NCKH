import numpy as np
from stage_1_camera.contracts import CameraState, CameraStatus
from stage_1_camera.geometry import roundtrip_ground_error


def look_at_camera(C, target=np.zeros(3), up=np.array([0.,0.,1.])):
    # Camera convention: +Z forward, +X right, +Y down-ish; construct orthonormal world->camera.
    f=(target-C); f=f/np.linalg.norm(f)
    r=np.cross(f,up); r=r/np.linalg.norm(r)
    d=np.cross(f,r); d=d/np.linalg.norm(d)
    return np.stack([r,d,f],axis=0)


def make_cam():
    C=np.array([0.,-60.,20.]); R=look_at_camera(C)
    K=np.array([[1200.,0.,640.],[0.,1200.,360.],[0.,0.,1.]])
    return CameraState(0,1280,720,K,R,C,status=CameraStatus.VALID)


def test_projection_ground_roundtrip():
    cam=make_cam(); pts=np.array([[-30,-10,0],[0,0,0],[25,15,0]],float)
    e=roundtrip_ground_error(cam,pts)
    assert np.max(e)<1e-7


def test_projection_matrix_consistency():
    cam=make_cam(); X=np.array([10.,5.,0.,1.]); q=cam.P@X; q=q[:2]/q[2]
    uv=cam.project_world(X[:3],distort=False)[0]
    assert np.linalg.norm(q-uv)<1e-7
