import numpy as np
from stage_1_camera.contracts import CameraState, CameraStatus
from stage_1_camera.geometry import camera_view_metadata, centre_ray_pitch_intersection


def look_at_camera(C, target, up=np.array([0.0, 0.0, 1.0])):
    f = target - C
    f = f / np.linalg.norm(f)
    r = np.cross(f, up)
    r = r / np.linalg.norm(r)
    d = np.cross(f, r)
    d = d / np.linalg.norm(d)
    return np.stack([r, d, f], axis=0)


def make_cam(target):
    C = np.array([0.0, -60.0, 20.0])
    R = look_at_camera(C, np.asarray(target, dtype=float))
    K = np.array([[1200.0, 0.0, 640.0], [0.0, 1200.0, 360.0], [0.0, 0.0, 1.0]])
    return CameraState(104, 1280, 720, K, R, C, status=CameraStatus.DEGRADED)


def test_right_half_view_contract_serialized():
    cam = make_cam([20.0, 0.0, 0.0])
    hit = centre_ray_pitch_intersection(cam)
    assert np.linalg.norm(hit - np.array([20.0, 0.0, 0.0])) < 1e-7
    view = camera_view_metadata(cam)
    assert view["centre_ray_pitch_hit_valid"] is True
    assert view["view_pitch_half"] == "RIGHT"
    payload = cam.to_dict()
    assert payload["view"]["principal_point_px"] == [640.0, 360.0]
    assert payload["view"]["view_pitch_half"] == "RIGHT"
    assert "attack_direction" not in payload["view"]


def test_left_half_view_contract():
    cam = make_cam([-18.0, 3.0, 0.0])
    view = camera_view_metadata(cam)
    assert view["centre_ray_pitch_hit_valid"] is True
    assert view["view_pitch_half"] == "LEFT"
    assert abs(view["centre_ray_pitch_hit_m"][0] + 18.0) < 1e-7


def test_midfield_is_geometry_not_attack_semantics():
    cam = make_cam([0.0, 0.0, 0.0])
    view = camera_view_metadata(cam)
    assert view["centre_ray_pitch_hit_valid"] is True
    assert view["view_pitch_half"] == "MIDFIELD"
