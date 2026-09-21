from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence
import math
import numpy as np

from ...camera import CameraStateLite, CameraTimelineLite
from .contracts import (
    FC_AUTO_MIN_SCORE_IMPROVEMENT,
    FC_AUTO_OUTLIER_Z_THRESHOLD,
    FC_TRAIN_CAMERA_CENTER_MEAN,
    FC_TRAIN_CAMERA_CENTER_STD,
)


WORLD_ROTATIONS = {
    "identity": np.eye(3, dtype=np.float64),
    "rotate_x_180": np.diag([1.0, -1.0, -1.0]),
    "rotate_y_180": np.diag([-1.0, 1.0, -1.0]),
    "rotate_z_180": np.diag([-1.0, -1.0, 1.0]),
}


def stage1_to_fc_camera(cam: CameraStateLite) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    K = np.asarray(cam.K, dtype=np.float64).reshape(3, 3)
    R = np.asarray(cam.R_world_to_camera, dtype=np.float64).reshape(3, 3)
    C = np.asarray(cam.camera_center_world_m, dtype=np.float64).reshape(3)
    t = (-R @ C).astype(np.float64)
    # Field Converter inference intentionally consumes only radial k1/k2.
    k = np.asarray(cam.distortion[:2], dtype=np.float64).reshape(2)
    return K, R, t, k


def camera_roundtrip_error_m(cam: CameraStateLite) -> float:
    _, R, t, _ = stage1_to_fc_camera(cam)
    C2 = -R.T @ t
    return float(np.linalg.norm(C2 - np.asarray(cam.camera_center_world_m, dtype=np.float64)))


def _fc_project_world(cam: CameraStateLite, xyz_world: np.ndarray) -> np.ndarray:
    K, R, t, k = stage1_to_fc_camera(cam)
    pts = np.asarray(xyz_world, dtype=np.float64).reshape(-1, 3)
    Xc = pts @ R.T + t[None, :]
    z = Xc[:, 2]
    out = np.full((len(pts), 2), np.nan, dtype=np.float64)
    good = np.isfinite(Xc).all(axis=1) & (z > 1e-8)
    if not np.any(good):
        return out
    x = Xc[good, 0] / z[good]
    y = Xc[good, 1] / z[good]
    r2 = x * x + y * y
    factor = 1.0 + k[0] * r2 + k[1] * r2 * r2
    out[good, 0] = K[0, 0] * x * factor + K[0, 2]
    out[good, 1] = K[1, 1] * y * factor + K[1, 2]
    return out


def _sample_world_points() -> np.ndarray:
    xs = np.linspace(-52.5, 52.5, 9)
    ys = np.linspace(-34.0, 34.0, 7)
    zs = (0.0, 0.5, 1.0, 1.8, 2.2)
    return np.asarray([(x, y, z) for z in zs for y in ys for x in xs], dtype=np.float64)


def camera_projection_compatibility(
    timeline: CameraTimelineLite,
    frame_indices: Sequence[int],
    *,
    max_sample_frames: int = 7,
) -> dict:
    frames = [int(f) for f in frame_indices if timeline.by_frame(int(f)) is not None]
    if not frames:
        return {"status": "NO_CAMERA", "count": 0, "median_px": None, "p95_px": None, "max_px": None}
    if len(frames) > max_sample_frames:
        positions = np.linspace(0, len(frames) - 1, max_sample_frames).round().astype(int)
        frames = [frames[i] for i in sorted(set(positions.tolist()))]
    points = _sample_world_points()
    errors: list[float] = []
    per_frame = []
    for frame in frames:
        cam = timeline.by_frame(frame)
        if cam is None:
            continue
        uv_stage1 = np.asarray(cam.project_world(points, distort=True), dtype=np.float64)
        uv_fc = _fc_project_world(cam, points)
        valid = np.isfinite(uv_stage1).all(axis=1) & np.isfinite(uv_fc).all(axis=1)
        if np.any(valid):
            err = np.linalg.norm(uv_stage1[valid] - uv_fc[valid], axis=1)
            errors.extend(err.tolist())
            per_frame.append({
                "frame_index": frame,
                "count": int(err.size),
                "median_px": float(np.median(err)),
                "p95_px": float(np.percentile(err, 95.0)),
                "max_px": float(np.max(err)),
                "roundtrip_camera_center_error_m": camera_roundtrip_error_m(cam),
            })
    if not errors:
        return {"status": "NO_VALID_POINTS", "count": 0, "median_px": None, "p95_px": None, "max_px": None, "per_frame": per_frame}
    arr = np.asarray(errors, dtype=np.float64)
    return {
        "status": "EVALUATED",
        "count": int(arr.size),
        "median_px": float(np.median(arr)),
        "p95_px": float(np.percentile(arr, 95.0)),
        "max_px": float(np.max(arr)),
        "per_frame": per_frame,
        "note": "Compares Stage1 full OpenCV distortion with Field Converter k1/k2 radial projection only.",
    }


def _zscore(center: np.ndarray) -> np.ndarray:
    mean = np.asarray(FC_TRAIN_CAMERA_CENTER_MEAN, dtype=np.float64)
    std = np.maximum(np.asarray(FC_TRAIN_CAMERA_CENTER_STD, dtype=np.float64), 1e-8)
    return (np.asarray(center, dtype=np.float64) - mean) / std


def _rms(z: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(z))))


def camera_domain_report(timeline: CameraTimelineLite, frame_indices: Sequence[int]) -> dict:
    centers = []
    for frame in frame_indices:
        cam = timeline.by_frame(int(frame))
        if cam is not None and np.isfinite(cam.camera_center_world_m).all():
            centers.append(np.asarray(cam.camera_center_world_m, dtype=np.float64))
    if not centers:
        return {"status": "UNKNOWN", "reason": "no_finite_camera_centers"}
    source = np.median(np.stack(centers), axis=0)
    source_z = _zscore(source)
    source_score = _rms(source_z)
    candidates = {}
    for name, Q in WORLD_ROTATIONS.items():
        aligned = source @ Q.T
        z = _zscore(aligned)
        candidates[name] = {
            "camera_center_median": aligned.tolist(),
            "zscore": z.tolist(),
            "score_rms": _rms(z),
        }
    best_name = min(candidates, key=lambda k: candidates[k]["score_rms"])
    best_score = float(candidates[best_name]["score_rms"])
    improvement = (source_score - best_score) / max(source_score, 1e-8)
    source_outlier = bool(np.max(np.abs(source_z)) >= FC_AUTO_OUTLIER_Z_THRESHOLD)
    auto_selected = best_name if (
        best_name != "identity" and source_outlier and improvement >= FC_AUTO_MIN_SCORE_IMPROVEMENT
    ) else "identity"
    aligned_z = np.asarray(candidates[auto_selected]["zscore"], dtype=np.float64)
    max_abs = float(np.max(np.abs(aligned_z)))
    if max_abs < 3.0:
        status = "IN_DOMAIN"
    elif max_abs < FC_AUTO_OUTLIER_Z_THRESHOLD:
        status = "NEAR_DOMAIN"
    else:
        status = "OUT_OF_DOMAIN"
    return {
        "status": status,
        "camera_center_source_median": source.tolist(),
        "source_zscore": source_z.tolist(),
        "source_score_rms": source_score,
        "best_candidate": best_name,
        "auto_selected_transform_expected": auto_selected,
        "aligned_zscore": aligned_z.tolist(),
        "aligned_max_abs_zscore": max_abs,
        "score_improvement_fraction": float(improvement),
        "candidates": candidates,
        "upstream_auto_outlier_z_threshold": FC_AUTO_OUTLIER_Z_THRESHOLD,
        "upstream_auto_min_score_improvement": FC_AUTO_MIN_SCORE_IMPROVEMENT,
        "note": "Diagnostic only. OUT_OF_DOMAIN does not block inference, but pretrained accuracy is not assumed to transfer unchanged.",
    }
