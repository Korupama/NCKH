from __future__ import annotations

import cv2
import numpy as np

from ...camera import CameraStateLite


def project_camera_points(camera: CameraStateLite, xyz_cam: np.ndarray, *, distort: bool) -> np.ndarray:
    pts = np.asarray(xyz_cam, dtype=np.float64)
    one = pts.ndim == 1
    pts = pts.reshape(-1, 3)
    dist = camera.distortion if distort else np.zeros_like(camera.distortion)
    uv, _ = cv2.projectPoints(
        pts,
        np.zeros((3, 1), dtype=np.float64),
        np.zeros((3, 1), dtype=np.float64),
        camera.K,
        dist,
    )
    uv = uv.reshape(-1, 2)
    invalid = (~np.isfinite(pts).all(axis=1)) | (pts[:, 2] <= 1e-8)
    uv[invalid] = np.nan
    return uv[0] if one else uv


def camera_to_world(camera: CameraStateLite, xyz_cam: np.ndarray) -> np.ndarray:
    pts = np.asarray(xyz_cam, dtype=np.float64)
    one = pts.ndim == 1
    pts = pts.reshape(-1, 3)
    out = (camera.R_world_to_camera.T @ (pts - camera.t_world_to_camera).T).T
    return out[0] if one else out
