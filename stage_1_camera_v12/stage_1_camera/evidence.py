from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import cv2
import numpy as np


def summarize_keypoint_coverage(
    keypoints: Optional[Mapping],
    image_width: int,
    image_height: int,
    *,
    world_xy_lookup: Optional[Mapping[int, Sequence[float]]] = None,
) -> Dict[str, Any]:
    """Summarize how broadly calibration keypoints cover the observed pitch.

    The primary metrics live in *original raw image coordinates*, matching the
    Stage-1 CameraState contract.  Optional world-coordinate spans can be added
    when the backend exposes a keypoint-id -> pitch-XY lookup; these spans are
    diagnostic only until empirical SoccerNet calibration thresholds are set.

    Returns a JSON-serializable dictionary.  No synthetic points are created:
    only finite ``x``/``y`` entries actually present in ``keypoints`` are used.
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image_width and image_height must be positive")

    pts = []
    ids = []
    for key, value in (keypoints or {}).items():
        if not isinstance(value, Mapping) or "x" not in value or "y" not in value:
            continue
        try:
            x, y = float(value["x"]), float(value["y"])
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        pts.append([x, y])
        try:
            ids.append(int(key))
        except (TypeError, ValueError):
            ids.append(None)

    out: Dict[str, Any] = {
        "num_valid_keypoints": int(len(pts)),
        "image_convex_hull_area_ratio": None,
        "image_bbox_area_ratio": None,
        "image_x_span_ratio": None,
        "image_y_span_ratio": None,
        "image_quadrants_occupied": 0,
        "image_centroid_normalized": None,
        "world_x_span_m": None,
        "world_y_span_m": None,
    }
    if not pts:
        return out

    arr = np.asarray(pts, dtype=np.float64)
    w, h = float(image_width), float(image_height)

    xmin, ymin = arr.min(axis=0)
    xmax, ymax = arr.max(axis=0)
    xspan = max(0.0, xmax - xmin)
    yspan = max(0.0, ymax - ymin)
    out["image_x_span_ratio"] = float(xspan / w)
    out["image_y_span_ratio"] = float(yspan / h)
    out["image_bbox_area_ratio"] = float((xspan * yspan) / (w * h))
    out["image_centroid_normalized"] = [
        float(arr[:, 0].mean() / w),
        float(arr[:, 1].mean() / h),
    ]

    if len(arr) >= 3:
        hull = cv2.convexHull(arr.astype(np.float32))
        hull_area = float(cv2.contourArea(hull))
        out["image_convex_hull_area_ratio"] = float(hull_area / (w * h))
    else:
        out["image_convex_hull_area_ratio"] = 0.0

    cx, cy = w / 2.0, h / 2.0
    quadrants = set()
    for x, y in arr:
        quadrants.add((0 if x < cx else 1, 0 if y < cy else 1))
    out["image_quadrants_occupied"] = int(len(quadrants))

    if world_xy_lookup:
        world_pts = []
        for kp_id in ids:
            if kp_id is None or kp_id not in world_xy_lookup:
                continue
            xy = np.asarray(world_xy_lookup[kp_id], dtype=np.float64).reshape(-1)
            if len(xy) >= 2 and np.isfinite(xy[:2]).all():
                world_pts.append(xy[:2])
        if world_pts:
            warr = np.asarray(world_pts, dtype=np.float64)
            out["world_x_span_m"] = float(np.ptp(warr[:, 0]))
            out["world_y_span_m"] = float(np.ptp(warr[:, 1]))
            out["num_world_mapped_keypoints"] = int(len(warr))

    return out


def build_keypoint_correspondences(
    keypoints: Optional[Mapping],
    world_xyz_lookup: Mapping[int, Sequence[float]],
) -> list:
    """Serialize observed pixel ↔ canonical world calibration correspondences."""
    out = []
    for raw_id, value in (keypoints or {}).items():
        if not isinstance(value, Mapping) or "x" not in value or "y" not in value:
            continue
        try:
            kp_id = int(raw_id)
            uv = [float(value["x"]), float(value["y"])]
        except (TypeError, ValueError):
            continue
        xyz = world_xyz_lookup.get(kp_id)
        if xyz is None:
            continue
        xyz = np.asarray(xyz, dtype=np.float64).reshape(-1)
        if len(xyz) < 3 or not np.isfinite(xyz[:3]).all() or not np.isfinite(uv).all():
            continue
        out.append({"id": kp_id, "uv_px": uv, "world_xyz_m": xyz[:3].tolist()})
    return out


def reprojection_error_from_correspondences(camera, correspondences: Sequence[Mapping]) -> Dict[str, Any]:
    """Evaluate a camera against stored calibration correspondences."""
    if not correspondences:
        return {
            "count": 0, "non_ground_count": 0, "mean_px": None,
            "median_px": None, "p95_px": None, "max_px": None,
        }
    xyz, uv = [], []
    non_ground = 0
    for c in correspondences:
        try:
            X = np.asarray(c["world_xyz_m"], dtype=np.float64).reshape(3)
            u = np.asarray(c["uv_px"], dtype=np.float64).reshape(2)
        except Exception:
            continue
        if not (np.isfinite(X).all() and np.isfinite(u).all()):
            continue
        xyz.append(X); uv.append(u)
        if abs(float(X[2])) > 1e-6:
            non_ground += 1
    if not xyz:
        return {
            "count": 0, "non_ground_count": 0, "mean_px": None,
            "median_px": None, "p95_px": None, "max_px": None,
        }
    pred = camera.project_world(np.asarray(xyz, dtype=np.float64))
    err = np.linalg.norm(pred - np.asarray(uv, dtype=np.float64), axis=1)
    err = err[np.isfinite(err)]
    if len(err) == 0:
        return {
            "count": 0, "non_ground_count": non_ground, "mean_px": None,
            "median_px": None, "p95_px": None, "max_px": None,
        }
    return {
        "count": int(len(err)),
        "non_ground_count": int(non_ground),
        "mean_px": float(np.mean(err)),
        "median_px": float(np.median(err)),
        "p95_px": float(np.percentile(err, 95)),
        "max_px": float(np.max(err)),
    }
