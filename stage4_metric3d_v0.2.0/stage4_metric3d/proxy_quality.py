from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import numpy as np

from .proxy_schemas import Stage4ProjectionConfig


def proxy_points(observation: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    """Return v1.1 raw proxies, while still accepting v1.0 output."""

    return observation.get("raw_height_plane_proxies_23") or observation.get("body_proxies_23") or []


def _xy(point: Mapping[str, Any]) -> np.ndarray | None:
    xyz = point.get("xyz_proxy_world_m")
    if xyz is None:
        return None
    value = np.asarray(xyz, dtype=np.float64)
    if value.shape != (3,) or not np.isfinite(value).all():
        return None
    return value[:2]


def _max_pairwise_distance(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    delta = points[:, None, :] - points[None, :, :]
    return float(np.max(np.linalg.norm(delta, axis=2)))


def compactness_diagnostics(
    points: Sequence[Mapping[str, Any]],
    config: Stage4ProjectionConfig,
) -> Dict[str, Any]:
    projected = [(point, _xy(point)) for point in points]
    projected = [(point, xy) for point, xy in projected if xy is not None]
    if not projected:
        return {
            "status": "MISSING",
            "reasons": ["NO_PROJECTED_BODY_PROXY"],
            "projected_points": 0,
            "longitudinal_span_m": None,
            "transverse_span_m": None,
            "planar_diameter_m": None,
            "foot_to_upper_center_m": None,
            "thresholds_are_provisional": True,
        }

    array = np.stack([xy for _, xy in projected])
    x_span = float(np.ptp(array[:, 0]))
    y_span = float(np.ptp(array[:, 1]))
    diameter = _max_pairwise_distance(array)
    lower = np.stack([
        xy for point, xy in projected
        if point.get("anatomical_group") in {"ankle", "foot"}
    ]) if any(point.get("anatomical_group") in {"ankle", "foot"} for point, _ in projected) else np.empty((0, 2))
    upper = np.stack([
        xy for point, xy in projected
        if point.get("anatomical_group") in {"head", "shoulder", "hip"}
    ]) if any(point.get("anatomical_group") in {"head", "shoulder", "hip"} for point, _ in projected) else np.empty((0, 2))
    foot_to_upper = (
        float(np.linalg.norm(np.mean(lower, axis=0) - np.mean(upper, axis=0)))
        if len(lower) and len(upper)
        else None
    )

    reject_reasons = []
    warning_reasons = []
    if x_span > config.compactness_reject_axis_span_m:
        reject_reasons.append("LONGITUDINAL_SPAN_EXCEEDS_REJECT_LIMIT")
    elif x_span > config.compactness_warn_axis_span_m:
        warning_reasons.append("LONGITUDINAL_SPAN_EXCEEDS_WARNING_LIMIT")
    if y_span > config.compactness_reject_axis_span_m:
        reject_reasons.append("TRANSVERSE_SPAN_EXCEEDS_REJECT_LIMIT")
    elif y_span > config.compactness_warn_axis_span_m:
        warning_reasons.append("TRANSVERSE_SPAN_EXCEEDS_WARNING_LIMIT")
    if diameter > config.compactness_reject_diameter_m:
        reject_reasons.append("PLANAR_DIAMETER_EXCEEDS_REJECT_LIMIT")
    elif diameter > config.compactness_warn_diameter_m:
        warning_reasons.append("PLANAR_DIAMETER_EXCEEDS_WARNING_LIMIT")
    if foot_to_upper is not None:
        if foot_to_upper > config.compactness_reject_foot_to_upper_m:
            reject_reasons.append("FOOT_TO_UPPER_OFFSET_EXCEEDS_REJECT_LIMIT")
        elif foot_to_upper > config.compactness_warn_foot_to_upper_m:
            warning_reasons.append("FOOT_TO_UPPER_OFFSET_EXCEEDS_WARNING_LIMIT")

    status = "REJECTED" if reject_reasons else "DEGRADED" if warning_reasons else "VALID"
    return {
        "status": status,
        "reasons": reject_reasons + warning_reasons,
        "projected_points": len(projected),
        "longitudinal_span_m": x_span,
        "transverse_span_m": y_span,
        "planar_diameter_m": diameter,
        "foot_to_upper_center_m": foot_to_upper,
        "thresholds": {
            "warn_axis_span_m": config.compactness_warn_axis_span_m,
            "reject_axis_span_m": config.compactness_reject_axis_span_m,
            "warn_planar_diameter_m": config.compactness_warn_diameter_m,
            "reject_planar_diameter_m": config.compactness_reject_diameter_m,
            "warn_foot_to_upper_m": config.compactness_warn_foot_to_upper_m,
            "reject_foot_to_upper_m": config.compactness_reject_foot_to_upper_m,
        },
        "thresholds_are_provisional": True,
    }


def robust_ground_anchor(
    points: Sequence[Mapping[str, Any]],
    config: Stage4ProjectionConfig,
) -> Dict[str, Any]:
    candidates = []
    names = []
    for point in points:
        if point.get("anatomical_group") not in {"ankle", "foot"}:
            continue
        xy = _xy(point)
        if xy is not None:
            candidates.append(xy)
            names.append(str(point.get("name")))
    if not candidates:
        return {
            "status": "MISSING",
            "reason": "NO_PROJECTED_ANKLE_OR_FOOT_PROXY",
            "xyz_ground_m": None,
            "candidate_count": 0,
            "inlier_count": 0,
        }

    array = np.stack(candidates)
    distances = np.linalg.norm(array[:, None, :] - array[None, :, :], axis=2)
    neighbor_counts = np.sum(distances <= config.ground_anchor_cluster_radius_m, axis=1)
    best_count = int(np.max(neighbor_counts))
    choices = np.flatnonzero(neighbor_counts == best_count)
    if len(choices) > 1:
        distance_sums = np.sum(distances[choices], axis=1)
        medoid_index = int(choices[int(np.argmin(distance_sums))])
    else:
        medoid_index = int(choices[0])
    inlier_mask = distances[medoid_index] <= config.ground_anchor_cluster_radius_m
    inliers = array[inlier_mask]
    anchor = np.median(inliers, axis=0)
    radii = np.linalg.norm(inliers - anchor[None, :], axis=1)
    max_radius = float(np.max(radii)) if len(radii) else None
    inlier_count = int(np.count_nonzero(inlier_mask))

    if inlier_count < config.ground_anchor_min_inliers:
        status = "REJECTED"
        reason = "INSUFFICIENT_COHERENT_FOOT_INLIERS"
    elif max_radius is not None and max_radius > config.ground_anchor_max_inlier_radius_m:
        status = "DEGRADED"
        reason = "FOOT_CLUSTER_RADIUS_EXCEEDS_WARNING_LIMIT"
    else:
        status = "VALID"
        reason = None
    return {
        "status": status,
        "reason": reason,
        "xyz_ground_m": [float(anchor[0]), float(anchor[1]), 0.0],
        "method": "DENSEST_ANKLE_FOOT_CLUSTER_MEDIAN",
        "candidate_count": len(candidates),
        "inlier_count": inlier_count,
        "inlier_fraction": float(inlier_count / len(candidates)),
        "cluster_radius_m": float(config.ground_anchor_cluster_radius_m),
        "max_inlier_radius_m": max_radius,
        "inlier_keypoints": [name for name, keep in zip(names, inlier_mask) if bool(keep)],
        "semantics": "player ground-state proxy only; not a legal-body boundary",
    }
