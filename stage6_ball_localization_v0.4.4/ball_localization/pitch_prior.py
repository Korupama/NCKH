from __future__ import annotations
from typing import Any, Iterable
import numpy as np

from .camera import CameraStateLite
from .candidate_filter import (
    compute_aspect_ratio,
    diameter_validity,
    goal_net_ray_check,
    grid_structure_score,
    shape_aspect_ratio_score,
    structure_tensor_coherence,
)
from .contracts import BallCandidate2D


def pitch_soft_prior(
    camera: CameraStateLite,
    center_uv: Iterable[float],
    *,
    margin_m: float = 12.0,
    far_prior: float = 0.20,
    goal_net_prior: float = 0.15,
) -> float:
    """Calculate soft spatial prior on the pitch, penalizing points far outside or inside the goal net."""
    xyz = camera.intersect_z_plane(np.asarray(center_uv, float), 0.0)[0]
    if not np.all(np.isfinite(xyz)):
        return float(far_prior)

    pitch = camera.pitch or {}
    L = float(pitch.get("length_m", 105.0))
    W = float(pitch.get("width_m", 68.0))

    # Goal Net suppression:
    # 1. 3D ray intersection with goal net box
    net_info = goal_net_ray_check(camera, center_uv)
    if net_info["in_goal_net"]:
        return float(goal_net_prior)

    # 2. Ground intersection behind goal line within goal mouth width
    if abs(float(xyz[0])) > L / 2.0 and abs(float(xyz[1])) <= 4.2:
        return float(goal_net_prior)

    dx = max(0.0, abs(float(xyz[0])) - L / 2.0)
    dy = max(0.0, abs(float(xyz[1])) - W / 2.0)
    d = float(np.hypot(dx, dy))
    if d <= 0:
        return 1.0
    if d >= margin_m:
        return float(far_prior)
    return float(1.0 - (1.0 - far_prior) * (d / margin_m))


def apply_pitch_prior(
    candidates: list[BallCandidate2D],
    camera: CameraStateLite,
    *,
    margin_m: float = 12.0,
    far_prior: float = 0.20,
    goal_net_prior: float = 0.15,
    max_aspect_ratio: float = 1.45,
    image_bgr: np.ndarray | None = None,
) -> list[BallCandidate2D]:
    """Apply spatial pitch, goal net rejection, geometric shape, and diameter priors to candidates."""
    for c in candidates:
        net_info = goal_net_ray_check(camera, c.center_uv)
        in_net = bool(net_info["in_goal_net"])
        c.metadata["in_goal_net"] = in_net
        if in_net:
            c.metadata["goal_side"] = net_info.get("goal_side")

        spatial = pitch_soft_prior(
            camera,
            c.center_uv,
            margin_m=margin_m,
            far_prior=far_prior,
            goal_net_prior=goal_net_prior,
        )

        shape = shape_aspect_ratio_score(c.bbox_xyxy, max_aspect_ratio=max_aspect_ratio)
        ar = compute_aspect_ratio(c.bbox_xyxy)
        c.metadata["aspect_ratio"] = ar
        c.metadata["shape_score"] = shape

        diam = diameter_validity(c.diameter_px)
        c.metadata["diameter_validity"] = diam

        # Linearity check on crop if image is provided
        lin_factor = 1.0
        if image_bgr is not None and shape > 0.0:
            x1, y1, x2, y2 = map(int, [round(v) for v in c.bbox_xyxy])
            h_img, w_img = image_bgr.shape[:2]
            if 0 <= x1 < x2 <= w_img and 0 <= y1 < y2 <= h_img:
                crop = image_bgr[y1:y2, x1:x2]
                coherence = structure_tensor_coherence(crop)
                c.metadata["line_coherence"] = coherence
                if coherence > 0.70:
                    lin_factor = max(0.10, 1.0 - coherence)
                grid_score = grid_structure_score(crop)
                c.metadata["grid_structure_score"] = grid_score
                if grid_score >= 0.75:
                    lin_factor = 0.0

        combined = spatial * shape * diam * lin_factor
        c.pitch_prior = float(combined)
        c.ranking_score = float(c.detector_score) * c.pitch_prior
        c.metadata["pitch_soft_prior"] = spatial
        c.metadata["combined_prior"] = combined

    return candidates
