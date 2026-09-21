import cv2
import numpy as np

from ball_localization.camera import camera_from_soccernet_calibration
from ball_localization.coordinates import (
    SOCCERNET_TO_STAGE6_WORLD,
    soccernet_rotation_to_stage6,
    soccernet_xyz_to_stage6,
)


def _project_source(K, R, C, xyz):
    p=R @ (np.asarray(xyz,float)-np.asarray(C,float))
    q=K @ p
    return q[:2]/q[2]


def test_soccernet_crossbar_z_becomes_stage6_up_positive():
    p=soccernet_xyz_to_stage6([-52.5, 3.66, -2.44])
    assert np.allclose(p,[-52.5,-3.66,2.44])
    assert np.linalg.det(SOCCERNET_TO_STAGE6_WORLD) > 0.999999


def test_soccernet_camera_parser_canonicalizes_rotation_and_center():
    calib={
      'x_focal_length':1000.0,'y_focal_length':1000.0,'principal_point':[960.0,540.0],
      'position_meters':[0.0,4.0,-10.0],'rotation_matrix':np.eye(3).tolist(),
      'radial_distortion':[0,0,0,0,0,0],'tangential_distortion':[0,0],'thin_prism_distortion':[0,0,0,0]
    }
    c=camera_from_soccernet_calibration(calib,image_width=1920,image_height=1080)
    assert c.image_width==1920 and c.K[0,0]==1000
    assert np.allclose(c.camera_center_world_m,[0.0,-4.0,10.0])
    assert np.allclose(c.R_world_to_camera,SOCCERNET_TO_STAGE6_WORLD)
    assert np.isclose(np.linalg.det(c.R_world_to_camera),1.0)


def test_projection_is_invariant_under_soccernet_to_stage6_world_change():
    fx=1300.0; fy=1275.0; cx=960.0; cy=540.0
    K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1]],float)
    rvec=np.array([0.15,-0.08,0.22],float)
    R_sn,_=cv2.Rodrigues(rvec)
    C_sn=np.array([3.0,65.0,-14.0])
    X_sn=np.array([18.0,12.0,-2.0])
    # Ensure a forward-facing test point; if this arbitrary point is behind the
    # generated camera, flip the rotation 180 degrees around Y.
    if (R_sn @ (X_sn-C_sn))[2] <= 0:
        Ry=np.diag([-1.0,1.0,-1.0])
        R_sn=Ry@R_sn
    calib={
      'x_focal_length':fx,'y_focal_length':fy,'principal_point':[cx,cy],
      'position_meters':C_sn.tolist(),'rotation_matrix':R_sn.tolist(),
      'radial_distortion':[0,0,0,0,0,0],'tangential_distortion':[0,0],'thin_prism_distortion':[0,0,0,0]
    }
    uv_sn=_project_source(K,R_sn,C_sn,X_sn)
    c=camera_from_soccernet_calibration(calib,image_width=1920,image_height=1080)
    X_can=soccernet_xyz_to_stage6(X_sn)
    uv_can=c.project_world(X_can)[0]
    assert np.linalg.det(soccernet_rotation_to_stage6(R_sn)) > 0.999999
    assert np.allclose(uv_sn,uv_can,atol=1e-7)
