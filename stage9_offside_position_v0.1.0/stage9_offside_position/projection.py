from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import math
import numpy as np


def _array(value, shape):
    a = np.asarray(value, dtype=np.float64)
    if a.size != int(np.prod(shape)):
        raise ValueError(f"expected shape {shape}, got {a.shape}")
    return a.reshape(shape)


def parse_camera(stage1: Mapping[str, Any]) -> Dict[str, Any]:
    intr = stage1.get("intrinsics") or {}
    extr = stage1.get("extrinsics") or {}
    K = _array(intr.get("K"), (3, 3))
    R = _array(extr.get("R_world_to_camera"), (3, 3))
    C = _array(extr.get("camera_center_world_m"), (3,))
    t = -R @ C
    dist = stage1.get("distortion") or {}
    radial = list(dist.get("radial") or [])
    tangential = list(dist.get("tangential") or [])
    k = radial + [0.0] * (6 - len(radial))
    p = tangential + [0.0] * (2 - len(tangential))
    coeffs = np.asarray([k[0], k[1], p[0], p[1], k[2], k[3], k[4], k[5]], dtype=np.float64)
    image = stage1.get("image") or {}
    return {"K": K, "R": R, "C": C, "t": t, "dist": coeffs, "width": image.get("width"), "height": image.get("height")}


def project_world_points(stage1: Mapping[str, Any], xyz: Sequence[Sequence[float]], distort: bool = True) -> np.ndarray:
    cam = parse_camera(stage1)
    pts = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    try:
        import cv2
        rvec, _ = cv2.Rodrigues(cam["R"])
        dist = cam["dist"] if distort else np.zeros_like(cam["dist"])
        uv, _ = cv2.projectPoints(pts, rvec, cam["t"].reshape(3, 1), cam["K"], dist)
        return uv.reshape(-1, 2)
    except Exception:
        xc = (cam["R"] @ pts.T).T + cam["t"][None, :]
        uvw = (cam["K"] @ xc.T).T
        out = np.full((len(pts), 2), np.nan, dtype=np.float64)
        good = np.abs(uvw[:, 2]) > 1e-12
        out[good] = uvw[good, :2] / uvw[good, 2:3]
        return out


def intersect_pixel_with_pitch(stage1: Mapping[str, Any], uv: Sequence[float]) -> Optional[List[float]]:
    cam = parse_camera(stage1)
    p = np.asarray([float(uv[0]), float(uv[1]), 1.0], dtype=np.float64)
    d_cam = np.linalg.inv(cam["K"]) @ p
    d_world = cam["R"].T @ d_cam
    if abs(float(d_world[2])) < 1e-12:
        return None
    lam = -float(cam["C"][2]) / float(d_world[2])
    if not math.isfinite(lam) or lam <= 0:
        return None
    hit = cam["C"] + lam * d_world
    if not np.isfinite(hit).all():
        return None
    hit[2] = 0.0
    return hit.tolist()
