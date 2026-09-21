import numpy as np
from stage4_metric3d.camera import CameraStateLite


def make_camera():
    return CameraStateLite.from_dict({
        "schema_version":"1.2", "frame_index":0, "status":"VALID",
        "image":{"width":1280,"height":720},
        "intrinsics":{"K":[[1000,0,640],[0,1000,360],[0,0,1]]},
        "extrinsics":{"R_world_to_camera":[[1,0,0],[0,1,0],[0,0,1]],"camera_center_world_m":[0,0,-10]},
        "distortion":{"radial":[0,0,0,0,0,0],"tangential":[0,0],"thin_prism":[0,0,0,0]},
        "pitch":{}
    })


def test_project_ray_roundtrip():
    cam = make_camera()
    p = np.array([1.5, -0.5, 2.0])
    uv = cam.project_world(p)
    o, d = cam.world_ray(uv)
    v = p-o
    v /= np.linalg.norm(v)
    assert np.dot(v,d) > 1 - 1e-10


def test_pitch_intersection():
    cam = make_camera()
    p = np.array([2.0, 1.0, 0.0])
    uv = cam.project_world(p)
    q = cam.intersect_pitch(uv)
    assert np.allclose(p,q,atol=1e-8)
