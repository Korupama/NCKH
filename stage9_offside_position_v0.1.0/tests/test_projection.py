import numpy as np
from stage9_offside_position.projection import project_world_points, intersect_pixel_with_pitch


def camera():
    return {
        "image":{"width":1280,"height":720},
        "intrinsics":{"K":[[1000,0,640],[0,1000,360],[0,0,1]]},
        "extrinsics":{"R_world_to_camera":[[1,0,0],[0,1,0],[0,0,-1]],"camera_center_world_m":[0,0,10]},
        "distortion":{"radial":[],"tangential":[]},
    }


def test_projection_finite():
    uv=project_world_points(camera(),[[0,0,0]])
    assert np.isfinite(uv).all()
    assert abs(uv[0,0]-640)<1e-6 and abs(uv[0,1]-360)<1e-6


def test_intersect_principal_point_hits_origin():
    hit=intersect_pixel_with_pitch(camera(),[640,360])
    assert hit is not None
    assert np.linalg.norm(np.asarray(hit))<1e-6
