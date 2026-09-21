from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import cv2
import numpy as np

TORSO_NAMES = ("left_shoulder", "right_shoulder", "right_hip", "left_hip")
LOWER_NAMES = ("left_hip", "right_hip", "right_knee", "left_knee")
VALID_KP_STATES = {"VALID", "LOW_MODEL_EVIDENCE", "TEMPORAL_IMPUTED"}


def kp_map(observation: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    return {str(k["name"]): k for k in observation.get("keypoints_133", [])}


def _point(k: Mapping[str, Any] | None) -> Optional[Tuple[float, float]]:
    if not k:
        return None
    if k.get("state") not in VALID_KP_STATES:
        return None
    x, y = k.get("x"), k.get("y")
    if x is None or y is None:
        return None
    if not np.isfinite(float(x)) or not np.isfinite(float(y)):
        return None
    return float(x), float(y)


def polygon_from_pose(observation: Mapping[str, Any], names: Sequence[str]) -> Optional[np.ndarray]:
    m = kp_map(observation)
    pts = [_point(m.get(name)) for name in names]
    if any(p is None for p in pts):
        return None
    arr = np.asarray(pts, dtype=np.float32)
    if abs(cv2.contourArea(arr)) < 4.0:
        return None
    return arr


def _bbox_polygon(bbox: Sequence[float], region: str) -> Optional[np.ndarray]:
    if len(bbox) != 4:
        return None
    x1, y1, x2, y2 = map(float, bbox)
    if x2 <= x1 or y2 <= y1:
        return None
    w, h = x2 - x1, y2 - y1
    if region == "torso":
        return np.asarray([
            [x1 + 0.18*w, y1 + 0.20*h],
            [x2 - 0.18*w, y1 + 0.20*h],
            [x2 - 0.24*w, y1 + 0.58*h],
            [x1 + 0.24*w, y1 + 0.58*h],
        ], dtype=np.float32)
    if region == "lower":
        return np.asarray([
            [x1 + 0.22*w, y1 + 0.50*h],
            [x2 - 0.22*w, y1 + 0.50*h],
            [x2 - 0.28*w, y1 + 0.82*h],
            [x1 + 0.28*w, y1 + 0.82*h],
        ], dtype=np.float32)
    raise ValueError(region)


def region_polygon(observation: Mapping[str, Any], region: str, allow_bbox_fallback: bool = True) -> Tuple[Optional[np.ndarray], str]:
    if region == "torso":
        poly = polygon_from_pose(observation, TORSO_NAMES)
    elif region == "lower":
        poly = polygon_from_pose(observation, LOWER_NAMES)
    else:
        raise ValueError(region)
    if poly is not None:
        return poly, "POSE_TORSO" if region == "torso" else "POSE_LOWER_BODY"
    if allow_bbox_fallback:
        poly = _bbox_polygon(observation.get("source_bbox_xyxy") or [], region)
        if poly is not None:
            return poly, "BBOX_TORSO_FALLBACK" if region == "torso" else "BBOX_LOWER_BODY_FALLBACK"
    return None, "UNAVAILABLE"


def rasterize_polygon(frame_shape: Sequence[int], polygon: np.ndarray, erode_fraction: float = 0.0) -> np.ndarray:
    h, w = int(frame_shape[0]), int(frame_shape[1])
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.round(polygon).astype(np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    cv2.fillConvexPoly(mask, pts, 255)
    if erode_fraction > 0:
        x, y, ww, hh = cv2.boundingRect(pts)
        k = max(1, int(round(min(ww, hh) * erode_fraction)))
        if k % 2 == 0:
            k += 1
        kernel = np.ones((k, k), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
    return mask
