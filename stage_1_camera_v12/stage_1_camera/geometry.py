from __future__ import annotations
from typing import Any, Dict, Tuple
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


def principal_point_px(cam: CameraState) -> np.ndarray:
    """Return the calibrated principal point [cx, cy] in original-image pixels."""
    return np.asarray([float(cam.K[0, 2]), float(cam.K[1, 2])], dtype=np.float64)


def centre_ray_pitch_intersection(cam: CameraState) -> np.ndarray:
    """Intersect the optical/centre ray with the world pitch plane Z=0.

    The optical ray is evaluated at the calibrated principal point rather than the
    image centre. A non-finite vector is returned when the ray is parallel to, or
    intersects behind, the camera according to the existing Stage-1 geometry
    convention.
    """
    uv = principal_point_px(cam)[None, :]
    return intersect_pixel_rays_with_pitch(cam, uv)[0]


def camera_view_metadata(
    cam: CameraState,
    *,
    pitch_bounds_tolerance_m: float = 1.0,
    midfield_epsilon_m: float = 1e-6,
) -> Dict[str, Any]:
    """Build the non-semantic camera-view handoff consumed by Stage 7.

    This is deliberately camera geometry only. It does *not* emit an attacking
    team or attack direction. The latter remains Stage 7 responsibility.
    """
    pp = principal_point_px(cam)
    hit = centre_ray_pitch_intersection(cam)
    finite = bool(np.all(np.isfinite(hit)))

    reason = None
    in_pitch_bounds = False
    view_pitch_half = None
    hit_out = None

    if finite:
        hit = np.asarray(hit, dtype=np.float64).reshape(3)
        hit[2] = 0.0
        hit_out = hit.tolist()
        xmin, xmax = cam.pitch.x_limits
        ymin, ymax = cam.pitch.y_limits
        tol = float(max(0.0, pitch_bounds_tolerance_m))
        in_pitch_bounds = bool(
            (xmin - tol) <= hit[0] <= (xmax + tol)
            and (ymin - tol) <= hit[1] <= (ymax + tol)
        )
        if not in_pitch_bounds:
            reason = "CENTRE_RAY_PITCH_HIT_OUT_OF_BOUNDS"
        elif hit[0] < -abs(float(midfield_epsilon_m)):
            view_pitch_half = "LEFT"
        elif hit[0] > abs(float(midfield_epsilon_m)):
            view_pitch_half = "RIGHT"
        else:
            view_pitch_half = "MIDFIELD"
    else:
        reason = "CENTRE_RAY_PITCH_INTERSECTION_INVALID"

    valid = bool(finite and in_pitch_bounds)
    return {
        "principal_point_px": pp.tolist(),
        "centre_ray_pitch_hit_m": hit_out,
        "centre_ray_pitch_hit_valid": valid,
        "centre_ray_pitch_hit_in_bounds": bool(in_pitch_bounds),
        "view_pitch_half": view_pitch_half,
        "reason": reason,
        "source": "stage1.camera_geometry",
    }


def roundtrip_ground_error(cam: CameraState, xyz_ground: np.ndarray) -> np.ndarray:
    pts = _as_points3(xyz_ground)
    if np.max(np.abs(pts[:, 2])) > 1e-8:
        raise ValueError("roundtrip_ground_error expects Z=0 points")
    uv = project_world(cam, pts)
    rec = intersect_pixel_rays_with_pitch(cam, uv)
    return np.linalg.norm(rec - pts, axis=1)
