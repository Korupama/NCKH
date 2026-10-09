"""Geometric, spatial, and appearance filters to suppress goal net and white object false positives.

Prevents common ball detection failure modes:
1. Goal net mesh: white cord intersections in the goal volume or elevated netting.
2. Pitch lines: high-contrast directional white lines (touchlines, goal lines, penalty markings).
3. Non-spherical objects: elongated player cleats, socks, goal posts, crossbars, and advertising boards.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Optional, Tuple
import cv2
import numpy as np

from .camera import CameraStateLite
from .contracts import BallCandidate2D


def compute_aspect_ratio(bbox_xyxy: list[float] | tuple[float, ...]) -> float:
    """Return max(w, h) / min(w, h). Value is >= 1.0 for valid boxes."""
    x1, y1, x2, y2 = bbox_xyxy[:4]
    w = max(0.0, float(x2) - float(x1))
    h = max(0.0, float(y2) - float(y1))
    mn = min(w, h)
    if mn <= 1e-6:
        return float("inf")
    return float(max(w, h) / mn)


def shape_aspect_ratio_score(
    bbox_xyxy: list[float] | tuple[float, ...],
    *,
    max_aspect_ratio: float = 1.45,
    ideal_threshold: float = 1.15,
    sigma_ar: float = 0.15,
) -> float:
    """Calculate geometric circularity score based on bounding box aspect ratio.

    A soccer ball is a 3D sphere whose 2D projection is an ellipse very close
    to a circle (AR in [1.0, 1.15] under broadcast lenses, up to 1.35 under motion blur).
    Elongated net mesh lines, goal posts, white line stripes, and cleats have AR > 1.40.
    """
    ar = compute_aspect_ratio(bbox_xyxy)
    if not math.isfinite(ar) or ar > max_aspect_ratio:
        return 0.0
    if ar <= ideal_threshold:
        return 1.0
    diff = ar - ideal_threshold
    return float(math.exp(-(diff * diff) / (sigma_ar * sigma_ar)))


def diameter_validity(
    diameter_px: float,
    *,
    min_diameter_px: float = 2.5,
    max_diameter_px: float = 55.0,
) -> float:
    """Ensure candidate diameter is within broadcast physics limits."""
    d = float(diameter_px)
    if not math.isfinite(d) or d < min_diameter_px or d > max_diameter_px:
        return 0.0
    if d < min_diameter_px + 1.0:
        return float((d - min_diameter_px) / 1.0)
    if d > max_diameter_px - 5.0:
        return float(max(0.0, (max_diameter_px - d) / 5.0))
    return 1.0


def ray_intersects_aabb_3d(
    origin: np.ndarray,
    direction: np.ndarray,
    box_min: np.ndarray,
    box_max: np.ndarray,
) -> Tuple[bool, float, float]:
    """Test ray-AABB intersection using slab method.

    Returns (hit, t_min, t_max) where t is the distance along the ray.
    """
    t_min = 0.0
    t_max = float("inf")
    for i in range(3):
        d_i = float(direction[i])
        o_i = float(origin[i])
        b_min_i = float(box_min[i])
        b_max_i = float(box_max[i])
        if abs(d_i) < 1e-9:
            if o_i < b_min_i or o_i > b_max_i:
                return False, 0.0, 0.0
        else:
            t1 = (b_min_i - o_i) / d_i
            t2 = (b_max_i - o_i) / d_i
            if t1 > t2:
                t1, t2 = t2, t1
            t_min = max(t_min, t1)
            t_max = min(t_max, t2)
            if t_min > t_max:
                return False, 0.0, 0.0
    return t_max >= 0.0, float(t_min), float(t_max)


def goal_net_ray_check(
    camera: CameraStateLite,
    center_uv: Iterable[float],
    *,
    net_depth_m: float = 2.7,
    goal_half_width_m: float = 4.0,
    goal_height_m: float = 2.6,
) -> Dict[str, Any]:
    """Check if the optical ray through center_uv passes through either 3D goal net structure."""
    pitch = camera.pitch or {}
    L = float(pitch.get("length_m", 105.0))
    half_L = L / 2.0

    # Left / Away Goal net: X in [-half_L - net_depth, -half_L + 0.1]
    left_min = np.array([-half_L - net_depth_m, -goal_half_width_m, 0.0], dtype=float)
    left_max = np.array([-half_L + 0.1, goal_half_width_m, goal_height_m], dtype=float)

    # Right / Home Goal net: X in [half_L - 0.1, half_L + net_depth]
    right_min = np.array([half_L - 0.1, -goal_half_width_m, 0.0], dtype=float)
    right_max = np.array([half_L + net_depth_m, goal_half_width_m, goal_height_m], dtype=float)

    uv = np.asarray(center_uv, dtype=float).reshape(2)
    origins, dirs = camera.world_ray(uv)
    o = origins[0]
    d = dirs[0]

    hit_left, t_min_l, t_max_l = ray_intersects_aabb_3d(o, d, left_min, left_max)
    hit_right, t_min_r, t_max_r = ray_intersects_aabb_3d(o, d, right_min, right_max)

    in_net = bool(hit_left or hit_right)
    hit_side = "left" if hit_left else ("right" if hit_right else None)
    t_hit = t_min_l if hit_left else (t_min_r if hit_right else None)

    return {
        "in_goal_net": in_net,
        "goal_side": hit_side,
        "ray_distance_m": t_hit,
    }


def structure_tensor_coherence(
    crop: np.ndarray,
    *,
    min_pixels: int = 4,
) -> float:
    """Compute local structure tensor orientation coherence.

    A straight white line on the pitch or a net wire cord has high directional
    coherence (coherence near 1.0).
    A soccer ball has radial/isotropic gradients in all directions (coherence near 0.0).
    """
    if crop is None or crop.shape[0] < min_pixels or crop.shape[1] < min_pixels:
        return 0.0
    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        gray = crop.astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    jxx = float(np.sum(gx * gx))
    jyy = float(np.sum(gy * gy))
    jxy = float(np.sum(gx * gy))
    trace = jxx + jyy
    det = jxx * jyy - jxy * jxy
    disc = max(0.0, trace * trace - 4.0 * det)
    l1 = (trace + math.sqrt(disc)) / 2.0
    l2 = (trace - math.sqrt(disc)) / 2.0
    coherence = (l1 - l2) / (l1 + l2 + 1e-6)
    return float(coherence)


def grid_structure_score(
    crop: np.ndarray,
    *,
    min_line_fraction: float = 0.25,
) -> float:
    """Score rectangular/grid-like structure using orthogonal Hough segments.

    A ball can have a strong circular edge, but a goal frame or net grid
    produces several long horizontal and vertical segments in the same crop.
    """
    if crop is None or crop.shape[0] < 8 or crop.shape[1] < 8:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    edges = cv2.Canny(gray, 50, 150)
    min_line_length = max(6, int(min(crop.shape[:2]) * min_line_fraction))
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180.0,
        threshold=max(8, min_line_length // 2),
        minLineLength=min_line_length,
        maxLineGap=max(2, min_line_length // 5),
    )
    if lines is None:
        return 0.0

    horizontal = 0
    vertical = 0
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        length = math.hypot(dx, dy)
        if length < min_line_length:
            continue
        angle = abs(math.atan2(dy, dx))
        if angle <= math.radians(12) or angle >= math.pi - math.radians(12):
            horizontal += 1
        elif abs(angle - math.pi / 2.0) <= math.radians(12):
            vertical += 1

    if not horizontal or not vertical:
        return 0.0
    return float(min(1.0, (horizontal + vertical) / 4.0))
