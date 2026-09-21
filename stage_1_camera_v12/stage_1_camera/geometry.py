from __future__ import annotations
from typing import Tuple
import cv2
import numpy as np
from .contracts import CameraState


def _as_points2(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 1:
        a = a[None, :]
    if a.shape[1] != 2:
        raise ValueError("expected Nx2 points")
    return a


def _as_points3(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 1:
        a = a[None, :]
    if a.shape[1] != 3:
        raise ValueError("expected Nx3 points")
    return a


def project_world(cam: CameraState, xyz: np.ndarray, distort: bool = True) -> np.ndarray:
    pts = _as_points3(xyz)
    rvec, _ = cv2.Rodrigues(cam.R_world_to_camera)
    tvec = cam.t_world_to_camera.reshape(3, 1)
    dist = cam.distortion.opencv_vector() if distort else np.zeros(12, dtype=np.float64)
    uv, _ = cv2.projectPoints(pts, rvec, tvec, cam.K, dist)
    return uv.reshape(-1, 2)


def undistort_pixels(cam: CameraState, uv: np.ndarray) -> np.ndarray:
    pts = _as_points2(uv).reshape(-1, 1, 2)
    und = cv2.undistortPoints(pts, cam.K, cam.distortion.opencv_vector(), P=cam.K)
    return und.reshape(-1, 2)


def pixel_to_world_ray(cam: CameraState, uv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return origins Nx3 and normalized directions Nx3 in world coordinates."""
    pts = undistort_pixels(cam, uv)
    invK = np.linalg.inv(cam.K)
    dirs_cam = (invK @ np.column_stack([pts, np.ones(len(pts))]).T).T
    dirs_world = (cam.R_world_to_camera.T @ dirs_cam.T).T
    dirs_world /= np.linalg.norm(dirs_world, axis=1, keepdims=True).clip(min=1e-12)
    origins = np.repeat(cam.camera_center_world_m[None, :], len(pts), axis=0)
    return origins, dirs_world


def intersect_rays_with_plane(origins: np.ndarray, directions: np.ndarray,
                              plane_normal: np.ndarray, plane_offset: float,
                              require_forward: bool = True) -> np.ndarray:
    """Plane: n dot X + d = 0. Returns Nx3, NaN for parallel/behind intersections."""
    o = _as_points3(origins)
    d = _as_points3(directions)
    n = np.asarray(plane_normal, dtype=np.float64).reshape(3)
    denom = d @ n
    numer = -(o @ n + float(plane_offset))
    lam = np.full(len(o), np.nan, dtype=np.float64)
    good = np.abs(denom) > 1e-10
    lam[good] = numer[good] / denom[good]
    if require_forward:
        good &= lam > 0
    out = o + lam[:, None] * d
    out[~good] = np.nan
    return out


def intersect_pixel_rays_with_pitch(cam: CameraState, uv: np.ndarray) -> np.ndarray:
    origins, dirs = pixel_to_world_ray(cam, uv)
    return intersect_rays_with_plane(origins, dirs, np.array([0., 0., 1.]), 0.0)


def roundtrip_ground_error(cam: CameraState, xyz_ground: np.ndarray) -> np.ndarray:
    pts = _as_points3(xyz_ground)
    if np.max(np.abs(pts[:, 2])) > 1e-8:
        raise ValueError("roundtrip_ground_error expects Z=0 points")
    uv = project_world(cam, pts)
    rec = intersect_pixel_rays_with_pitch(cam, uv)
    return np.linalg.norm(rec - pts, axis=1)
