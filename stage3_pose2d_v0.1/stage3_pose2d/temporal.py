from __future__ import annotations

from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple
import copy
import numpy as np

from .schemas import Stage3Config
from .wholebody133 import SYMMETRIC_PAIRS


def _norm_xy(obs: Mapping[str, Any]) -> np.ndarray:
    bbox = np.asarray(obs["source_bbox_xyxy"], dtype=float)
    x1, y1, x2, y2 = bbox
    w, h = max(x2 - x1, 1e-6), max(y2 - y1, 1e-6)
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    arr = np.full((133, 2), np.nan, dtype=float)
    for kp in obs.get("keypoints_133", []):
        i = int(kp["index"])
        x, y = kp.get("x"), kp.get("y")
        if x is not None and y is not None:
            arr[i] = ((float(x) - cx) / w, (float(y) - cy) / h)
    return arr


def _swap_cost(prev: np.ndarray, cur: np.ndarray) -> Tuple[float, float]:
    normal_vals, swap_vals = [], []
    for a, b in SYMMETRIC_PAIRS:
        if np.isfinite(prev[[a, b]]).all() and np.isfinite(cur[[a, b]]).all():
            normal_vals.extend([np.linalg.norm(cur[a] - prev[a]), np.linalg.norm(cur[b] - prev[b])])
            swap_vals.extend([np.linalg.norm(cur[b] - prev[a]), np.linalg.norm(cur[a] - prev[b])])
    if not normal_vals:
        return float("nan"), float("nan")
    return float(np.mean(normal_vals)), float(np.mean(swap_vals))


def annotate_temporal(track_observations: Sequence[Mapping[str, Any]], config: Stage3Config) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    obs = [copy.deepcopy(dict(x)) for x in sorted(track_observations, key=lambda r: int(r["frame_index"]))]
    if not obs:
        return [], {"normalized_temporal_jitter": None, "temporal_outlier_fraction": 0.0, "left_right_swap_suspected_frames": []}
    norms = [_norm_xy(o) for o in obs]
    temporal_outliers = 0
    temporal_checks = 0
    swap_frames: List[int] = []
    accelerations: List[float] = []

    # Per-frame left/right swap suspicion against the previous observation.
    for i in range(1, len(obs)):
        if int(obs[i]["frame_index"]) - int(obs[i - 1]["frame_index"]) != 1:
            continue
        normal_cost, swapped_cost = _swap_cost(norms[i - 1], norms[i])
        suspected = bool(
            np.isfinite(normal_cost)
            and np.isfinite(swapped_cost)
            and normal_cost >= config.temporal_swap_min_normal_cost
            and swapped_cost < config.temporal_swap_ratio * normal_cost
        )
        obs[i].setdefault("temporal_qa", {})["normal_pair_cost"] = None if not np.isfinite(normal_cost) else normal_cost
        obs[i]["temporal_qa"]["swapped_pair_cost"] = None if not np.isfinite(swapped_cost) else swapped_cost
        obs[i]["temporal_qa"]["left_right_swap_suspected"] = suspected
        if suspected:
            swap_frames.append(int(obs[i]["frame_index"]))
            for a, b in SYMMETRIC_PAIRS:
                for idx in (a, b):
                    if idx < len(obs[i].get("keypoints_133", [])) and obs[i]["keypoints_133"][idx]["state"] == "VALID":
                        obs[i]["keypoints_133"][idx]["state"] = "LEFT_RIGHT_SUSPECT"

    # Second-difference jitter/outlier diagnostic in bbox-normalized coordinates.
    for i in range(1, len(obs) - 1):
        f0, f1, f2 = (int(obs[j]["frame_index"]) for j in (i - 1, i, i + 1))
        if not (f1 - f0 == 1 and f2 - f1 == 1):
            continue
        valid = np.isfinite(norms[i - 1]).all(axis=1) & np.isfinite(norms[i]).all(axis=1) & np.isfinite(norms[i + 1]).all(axis=1)
        if not np.any(valid):
            continue
        accel = norms[i + 1] - 2.0 * norms[i] + norms[i - 1]
        mag = np.linalg.norm(accel, axis=1)
        for k in np.where(valid)[0]:
            temporal_checks += 1
            accelerations.append(float(mag[k]))
            if mag[k] > config.temporal_accel_threshold:
                temporal_outliers += 1
                kp = obs[i]["keypoints_133"][int(k)]
                if kp["state"] in ("VALID", "LOW_MODEL_EVIDENCE"):
                    kp["state"] = "TEMPORAL_OUTLIER"
        obs[i].setdefault("temporal_qa", {})["normalized_acceleration_median"] = float(np.median(mag[valid]))
        obs[i]["temporal_qa"]["normalized_acceleration_p90"] = float(np.quantile(mag[valid], 0.90))

    if config.emit_temporal_estimates:
        for i in range(1, len(obs) - 1):
            if not (
                int(obs[i]["frame_index"]) - int(obs[i - 1]["frame_index"]) == 1
                and int(obs[i + 1]["frame_index"]) - int(obs[i]["frame_index"]) == 1
            ):
                continue
            for k, kp in enumerate(obs[i].get("keypoints_133", [])):
                if kp.get("x") is not None and kp.get("y") is not None:
                    continue
                p0, p2 = norms[i - 1][k], norms[i + 1][k]
                if not (np.isfinite(p0).all() and np.isfinite(p2).all()):
                    continue
                bbox = np.asarray(obs[i]["source_bbox_xyxy"], dtype=float)
                x1, y1, x2, y2 = bbox
                w, h = max(x2 - x1, 1e-6), max(y2 - y1, 1e-6)
                cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
                est = (p0 + p2) * 0.5
                kp["temporal_estimate_xy"] = [float(est[0] * w + cx), float(est[1] * h + cy)]
                kp["coordinate_evidence_kind"] = "TEMPORAL_ESTIMATE_ONLY"
                # Raw x/y stay missing by design.

    summary = {
        "normalized_temporal_jitter": None if not accelerations else float(np.median(accelerations)),
        "normalized_temporal_jitter_p90": None if not accelerations else float(np.quantile(accelerations, 0.90)),
        "temporal_outlier_fraction": float(temporal_outliers / max(1, temporal_checks)),
        "left_right_swap_suspected_frames": swap_frames,
        "silent_imputation": False,
    }
    return obs, summary
