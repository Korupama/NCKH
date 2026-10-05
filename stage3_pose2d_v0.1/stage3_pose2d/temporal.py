from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple
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


def _empty_summary(config: Stage3Config) -> Dict[str, Any]:
    enabled = bool(config.emit_temporal_estimates)
    return {
        "normalized_temporal_jitter": None,
        "normalized_temporal_jitter_p90": None,
        "temporal_outlier_fraction": 0.0,
        "left_right_swap_suspected_frames": [],
        "ownership_switch_suspected_frames": [],
        "temporal_downgraded_frames": [],
        "temporal_estimates": {
            "enabled": enabled,
            "created": 0,
            "insufficient_context": 0,
            "insufficient_context_reasons": {},
            "disabled_reason": None if enabled else "disabled_by_config",
        },
        "raw_coordinates_modified": False,
        "silent_imputation": False,
    }


def annotate_temporal(track_observations: Sequence[Mapping[str, Any]], config: Stage3Config) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    obs = [copy.deepcopy(dict(x)) for x in sorted(track_observations, key=lambda r: int(r["frame_index"]))]
    if not obs:
        return [], _empty_summary(config)
    # Clear derived fields first so rerunning with a different context/config
    # cannot leak stale estimates into the current result.
    for item in obs:
        for kp in item.get("keypoints_133", []):
            kp.pop("temporal_estimate_xy", None)
            kp.pop("temporal_estimate_status", None)
            kp.pop("temporal_estimate_reason", None)
        (item.get("temporal_qa") or {}).pop("temporal_estimates", None)
    norms = [_norm_xy(o) for o in obs]
    temporal_outliers = 0
    temporal_checks = 0
    swap_frames: List[int] = []
    ownership_switch_frames: List[int] = []
    temporal_downgraded_frames: List[int] = []
    accelerations: List[float] = []

    def ownership_status(item: Mapping[str, Any]) -> str:
        qa = item.get("qa") or {}
        ownership = qa.get("ownership") or {}
        status = ownership.get("ownership_status")
        if status is None:
            status = (item.get("crop_diagnostics") or {}).get("ownership_status")
        return str(status or "UNKNOWN")

    def downgrade_for_temporal(item: Dict[str, Any], reason: str) -> None:
        qa = item.setdefault("qa", {})
        reasons = qa.setdefault("status_reasons", [])
        if reason not in reasons:
            reasons.append(reason)
        item.setdefault("temporal_qa", {})[reason] = True
        if item.get("pose_status") == "VALID":
            item["pose_status"] = "DEGRADED"
            qa["pose_status"] = "DEGRADED"
            frame_index = int(item["frame_index"])
            if frame_index not in temporal_downgraded_frames:
                temporal_downgraded_frames.append(frame_index)

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
            if config.temporal_downgrade_on_swap:
                downgrade_for_temporal(obs[i], "temporal_left_right_swap_suspected")

    # Ownership evidence is derived only from the pose QA already computed
    # against the fixed Stage-2 bbox and neighboring tracks.  A transition from
    # supported ownership to a strong failure state is useful evidence of a
    # crop/identity switch, but UNKNOWN is never treated as a failure.
    strong_ownership_failure = {"NEIGHBOR_DOMINANT", "OUTSIDE_SOURCE", "CENTER_MISMATCH"}
    weak_ownership = {"WEAK_SUPPORT"}
    for i in range(1, len(obs)):
        if int(obs[i]["frame_index"]) - int(obs[i - 1]["frame_index"]) != 1:
            continue
        previous = ownership_status(obs[i - 1])
        current = ownership_status(obs[i])
        switched = (
            current in strong_ownership_failure
            and previous in {"SUPPORTED", "WEAK_SUPPORT"}
        ) or (current == "NEIGHBOR_DOMINANT")
        if not switched:
            continue
        frame = int(obs[i]["frame_index"])
        ownership_switch_frames.append(frame)
        obs[i].setdefault("temporal_qa", {})["ownership_switch_suspected"] = True
        obs[i]["temporal_qa"]["previous_ownership_status"] = previous
        obs[i]["temporal_qa"]["current_ownership_status"] = current
        if config.temporal_downgrade_on_ownership_switch:
            downgrade_for_temporal(obs[i], "temporal_ownership_switch_suspected")

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

    estimate_created = 0
    estimate_insufficient = 0
    estimate_reasons: Dict[str, int] = {}
    if config.emit_temporal_estimates:
        for i, current in enumerate(obs):
            frame_estimates = 0
            frame_reasons: Dict[str, int] = {}
            for k, kp in enumerate(current.get("keypoints_133", [])):
                if kp.get("x") is not None and kp.get("y") is not None:
                    continue

                reason = None
                if i == 0 or i == len(obs) - 1:
                    reason = "track_boundary"
                elif not (
                    int(current["frame_index"]) - int(obs[i - 1]["frame_index"]) == 1
                    and int(obs[i + 1]["frame_index"]) - int(current["frame_index"]) == 1
                ):
                    reason = "nonconsecutive_frame_context"
                else:
                    p0, p2 = norms[i - 1][k], norms[i + 1][k]
                    if not (np.isfinite(p0).all() and np.isfinite(p2).all()):
                        reason = "neighbor_keypoint_missing"

                if reason is not None:
                    kp["temporal_estimate_status"] = "INSUFFICIENT_CONTEXT"
                    kp["temporal_estimate_reason"] = reason
                    frame_reasons[reason] = frame_reasons.get(reason, 0) + 1
                    estimate_reasons[reason] = estimate_reasons.get(reason, 0) + 1
                    estimate_insufficient += 1
                    continue

                bbox = np.asarray(current["source_bbox_xyxy"], dtype=float)
                x1, y1, x2, y2 = bbox
                w, h = max(x2 - x1, 1e-6), max(y2 - y1, 1e-6)
                cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
                est = (p0 + p2) * 0.5
                kp["temporal_estimate_xy"] = [float(est[0] * w + cx), float(est[1] * h + cy)]
                kp["temporal_estimate_status"] = "ESTIMATED"
                frame_estimates += 1
                estimate_created += 1

            current.setdefault("temporal_qa", {})["temporal_estimates"] = {
                "created": frame_estimates,
                "insufficient_context": frame_reasons,
            }

    summary = {
        "normalized_temporal_jitter": None if not accelerations else float(np.median(accelerations)),
        "normalized_temporal_jitter_p90": None if not accelerations else float(np.quantile(accelerations, 0.90)),
        "temporal_outlier_fraction": float(temporal_outliers / max(1, temporal_checks)),
        "left_right_swap_suspected_frames": swap_frames,
        "ownership_switch_suspected_frames": ownership_switch_frames,
        "temporal_downgraded_frames": temporal_downgraded_frames,
        "temporal_estimates": {
            "enabled": bool(config.emit_temporal_estimates),
            "created": estimate_created,
            "insufficient_context": estimate_insufficient,
            "insufficient_context_reasons": estimate_reasons,
            "disabled_reason": None if config.emit_temporal_estimates else "disabled_by_config",
        },
        "raw_coordinates_modified": False,
        "silent_imputation": False,
    }
    return obs, summary
