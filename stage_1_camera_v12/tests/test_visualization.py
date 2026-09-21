from pathlib import Path
import numpy as np
import cv2

from stage_1_camera.contracts import CameraState, CameraStatus
from stage_1_camera.visualization import render_pitch_overlay, save_camera_3d_view
from test_geometry import look_at_camera


def make_cam():
    C = np.array([0.0, -60.0, 20.0])
    R = look_at_camera(C)
    K = np.array([[1200.0, 0.0, 640.0], [0.0, 1200.0, 360.0], [0.0, 0.0, 1.0]])
    return CameraState(0, 1280, 720, K, R, C, status=CameraStatus.VALID)


def test_pitch_overlay_changes_image():
    cam = make_cam()
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    out = render_pitch_overlay(image, cam)
    assert out.shape == image.shape
    assert int(out.sum()) > 0


def test_3d_view_writes_file(tmp_path: Path):
    p = tmp_path / "qa.png"
    save_camera_3d_view(make_cam(), str(p))
    assert p.exists() and p.stat().st_size > 0


def test_near_touchline_crossing_camera_plane_is_clipped_and_drawn():
    """Regression for the panned-camera near-touchline rendering bug.

    One endpoint of the 105 m near touchline is behind the camera, but a
    substantial sub-segment is still visible in the image. The renderer must
    keep that visible piece without drawing a projective jump through Zc=0.
    """
    C = np.array([0.0, -54.0, 11.0])
    R = look_at_camera(C, target=np.array([-30.0, 0.0, 0.0]))
    K = np.array([[1200.0, 0.0, 640.0], [0.0, 1200.0, 360.0], [0.0, 0.0, 1.0]])
    cam = CameraState(0, 1280, 720, K, R, C, status=CameraStatus.VALID)

    x = np.linspace(-52.5, 52.5, 211)
    near_touchline = np.column_stack([x, np.full_like(x, -34.0), np.zeros_like(x)])

    from stage_1_camera.visualization import _camera_depth, _draw_projected_polyline
    endpoint_depth = _camera_depth(cam, near_touchline[[0, -1]])
    assert endpoint_depth[0] > 0.0 and endpoint_depth[1] < 0.0

    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    _draw_projected_polyline(canvas, cam, near_touchline, (255, 255, 255), 2)

    ys, xs = np.where(canvas.max(axis=2) > 0)
    assert len(xs) > 100
    # The visible near touchline is in the lower half of the image; a sky jump
    # would create pixels high in the frame.
    assert ys.min() > 450
    assert xs.min() <= 5
    assert ys.max() >= 715
