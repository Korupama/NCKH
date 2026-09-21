import numpy as np

from stage4_metric3d.camera import CameraStateLite


def test_ray_height_plane_roundtrip_is_exact():
    camera = CameraStateLite.from_dict({
        "schema_version": "1.2",
        "frame_index": 86,
        "status": "VALID",
        "image": {"width": 1280, "height": 720},
        "intrinsics": {"K": [[1000, 0, 640], [0, 1000, 360], [0, 0, 1]]},
        "extrinsics": {"R_world_to_camera": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "camera_center_world_m": [0, 0, -10]},
        "distortion": {"radial": [0] * 6, "tangential": [0] * 2, "thin_prism": [0] * 4},
        "pitch": {"length_m": 105, "width_m": 68},
    })
    point = np.asarray([3.5, -1.2, 1.476])
    uv = camera.project_world(point)
    reconstructed = camera.intersect_z_plane(uv, point[2])
    assert np.allclose(point, reconstructed, atol=1e-10)
