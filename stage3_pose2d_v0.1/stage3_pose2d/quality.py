from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple
import numpy as np

from .schemas import Stage3Config
from .wholebody133 import (
    BODY17,
    FEET6,
    CORE_OFFSIDE_ANATOMY,
    BODY_SKELETON_EDGES,
    WHOLEBODY_KEYPOINT_NAMES,
)


# These groups describe image-space evidence needed by current World-State
# consumers.  They are deliberately not an IFAB legal-body mask: downstream
# owns legal-body construction and the eventual offside decision.
T0_ANATOMY_EVIDENCE_GROUPS = {
    "head": ("nose", "left_eye", "right_eye", "left_ear", "right_ear"),
    "torso": ("left_shoulder", "right_shoulder", "left_hip", "right_hip"),
    "left_leg": ("left_hip", "left_knee", "left_ankle"),
    "right_leg": ("right_hip", "right_knee", "right_ankle"),
    "left_foot": ("left_ankle", "left_big_toe", "left_small_toe", "left_heel"),
    "right_foot": ("right_ankle", "right_big_toe", "right_small_toe", "right_heel"),
}

# The legacy metric-pose Stage-4 bridge checks this exact anchor set before
# optimization.  Keeping it explicit here lets a Stage-3 report communicate
# the same readiness without importing Stage-4 implementation code.
STAGE4_CORE_METRIC_ANCHORS = (
    "left_shoulder", "right_shoulder", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)


def coordinate_evidence_kind(record: Mapping[str, Any]) -> str:
    """Classify whether a keypoint has an observed raw coordinate.

    ``temporal_estimate_xy`` is diagnostic-only and must never be consumed as
    an observed pixel.  The fallback behaviour keeps old v0.1 JSON readable
    while all newly emitted records store the classification explicitly.
    """

    explicit = record.get("coordinate_evidence_kind")
    if explicit in {"RAW_OBSERVED", "TEMPORAL_ESTIMATE_ONLY", "RAW_MISSING"}:
        return str(explicit)
    x, y = record.get("x"), record.get("y")
    try:
        if x is not None and y is not None and np.isfinite(float(x)) and np.isfinite(float(y)):
            return "RAW_OBSERVED"
    except (TypeError, ValueError):
        pass
    return "TEMPORAL_ESTIMATE_ONLY" if record.get("temporal_estimate_xy") is not None else "RAW_MISSING"


def summarize_t0_anatomy_evidence(
    keypoints: Sequence[Mapping[str, Any]],
    *,
    pose_status: str,
) -> Dict[str, Any]:
    """Create a compact, non-decisional summary of selected-frame evidence."""

    by_name = {str(record.get("name")): record for record in keypoints}

    def summarize_group(names: Sequence[str]) -> Dict[str, Any]:
        raw_observed_names: List[str] = []
        missing_raw_names: List[str] = []
        temporal_estimate_only_names: List[str] = []
        state_counts: Dict[str, int] = {}
        for name in names:
            record = by_name.get(name, {})
            state = str(record.get("state", "MISSING"))
            state_counts[state] = state_counts.get(state, 0) + 1
            kind = coordinate_evidence_kind(record)
            # A raw coordinate marked MISSING is invalid by contract even if a
            # malformed third-party file supplied x/y values.
            raw_observed = kind == "RAW_OBSERVED" and state != "MISSING"
            if raw_observed:
                raw_observed_names.append(name)
            else:
                missing_raw_names.append(name)
            if kind == "TEMPORAL_ESTIMATE_ONLY":
                temporal_estimate_only_names.append(name)
        total = len(names)
        return {
            "keypoint_names": list(names),
            "raw_observed_count": len(raw_observed_names),
            "keypoint_count": total,
            "raw_observed_fraction": float(len(raw_observed_names) / max(1, total)),
            "all_raw_observed": len(raw_observed_names) == total,
            "raw_observed_names": raw_observed_names,
            "missing_raw_names": missing_raw_names,
            "temporal_estimate_only_names": temporal_estimate_only_names,
            "keypoint_state_counts": state_counts,
        }

    groups = {
        name: summarize_group(names)
        for name, names in T0_ANATOMY_EVIDENCE_GROUPS.items()
    }
    core = summarize_group(STAGE4_CORE_METRIC_ANCHORS)
    temporal_estimates = [
        str(record.get("name"))
        for record in keypoints
        if coordinate_evidence_kind(record) == "TEMPORAL_ESTIMATE_ONLY"
    ]
    diagnostic_states = {
        "GEOMETRIC_OUTLIER",
        "TEMPORAL_OUTLIER",
        "LEFT_RIGHT_SUSPECT",
    }
    diagnostics = [
        str(record.get("name"))
        for record in keypoints
        if str(record.get("state", "MISSING")) in diagnostic_states
    ]
    review_flags: List[str] = []
    if str(pose_status) != "VALID":
        review_flags.append("POSE_NOT_VALID_AT_T0")
    if not core["all_raw_observed"]:
        review_flags.append("STAGE4_CORE_METRIC_ANCHORS_INCOMPLETE")
    if temporal_estimates:
        review_flags.append("TEMPORAL_ESTIMATE_ONLY_AT_T0")
    if diagnostics:
        review_flags.append("KEYPOINT_QA_DIAGNOSTIC_AT_T0")

    return {
        "schema_version": "stage3-t0-anatomy-evidence-1.0",
        "scope": "raw image-space evidence for World-State consumers; not Law-11 legal-body membership or an offside decision",
        "observed_coordinate_policy": "RAW_OBSERVED requires finite raw x/y and state != MISSING; temporal_estimate_xy never counts as observed raw evidence",
        "temporal_estimate_policy": "TEMPORAL_ESTIMATE_ONLY is diagnostic context; consumers must not replace raw x/y with it",
        "groups": groups,
        "stage4_core_metric_anchors": core,
        "temporal_estimate_only_keypoint_names": temporal_estimates,
        "qa_diagnostic_keypoint_names": diagnostics,
        "consumer_review_flags": review_flags,
    }


def _bbox_dims(bbox: Sequence[float]) -> Tuple[float, float, float, float, float, float]:
    x1, y1, x2, y2 = map(float, bbox)
    w, h = max(x2 - x1, 1e-6), max(y2 - y1, 1e-6)
    return x1, y1, x2, y2, w, h


def _available(xy: np.ndarray, scores: np.ndarray) -> np.ndarray:
    return np.isfinite(xy).all(axis=1) & np.isfinite(scores) & (scores > 0.0)


def _fraction(mask: np.ndarray, indices: Sequence[int]) -> float:
    if not indices:
        return 0.0
    return float(np.mean(mask[np.asarray(indices, dtype=int)]))


def _inside_margin(xy: np.ndarray, bbox: Sequence[float], margin: float) -> np.ndarray:
    x1, y1, x2, y2, w, h = _bbox_dims(bbox)
    return (
        (xy[:, 0] >= x1 - margin * w)
        & (xy[:, 0] <= x2 + margin * w)
        & (xy[:, 1] >= y1 - margin * h)
        & (xy[:, 1] <= y2 + margin * h)
    )


def _bone_length_outliers(xy: np.ndarray, available: np.ndarray, bbox: Sequence[float]) -> Tuple[float, List[Tuple[int, int, float]]]:
    _, _, _, _, w, h = _bbox_dims(bbox)
    diag = max(float(np.hypot(w, h)), 1e-6)
    lengths: List[Tuple[int, int, float]] = []
    outliers: List[Tuple[int, int, float]] = []
    for a, b in BODY_SKELETON_EDGES:
        if a >= 23 or b >= 23 or not (available[a] and available[b]):
            continue
        length = float(np.linalg.norm(xy[a] - xy[b]) / diag)
        lengths.append((a, b, length))
        # Deliberately broad: catches impossible cross-person/crop failures, not unusual athletic pose.
        if length < 0.004 or length > 0.80:
            outliers.append((a, b, length))
    frac = float(len(outliers) / max(1, len(lengths)))
    return frac, outliers


def evaluate_pose(
    xy: np.ndarray,
    scores: np.ndarray,
    bbox_xyxy: Sequence[float],
    config: Stage3Config,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    xy = np.asarray(xy, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if xy.shape != (133, 2) or scores.shape != (133,):
        return ({
            "pose_status": "REJECTED",
            "reason": f"invalid_shape xy={xy.shape}, scores={scores.shape}",
            "body_completeness": 0.0,
            "feet_completeness": 0.0,
            "core_completeness": 0.0,
            "inside_fraction": 0.0,
            "geometry_valid": False,
        }, [])

    available = _available(xy, scores)
    inside = _inside_margin(xy, bbox_xyxy, config.bbox_margin_for_qa) & available
    body_comp = _fraction(available, BODY17)
    feet_comp = _fraction(available, FEET6)
    core_comp = _fraction(available, CORE_OFFSIDE_ANATOMY)
    inside_frac = float(np.sum(inside) / max(1, np.sum(available)))

    positive_scores = scores[available]
    median_score = float(np.median(positive_scores)) if len(positive_scores) else 0.0
    low_floor = median_score * float(config.low_score_ratio_to_median)
    low_evidence = available & (scores < low_floor) if median_score > 0 else np.zeros(133, dtype=bool)

    bone_outlier_fraction, bone_outliers = _bone_length_outliers(xy, available, bbox_xyxy)
    geometry_valid = bool(inside_frac >= 0.60 and bone_outlier_fraction <= 0.35)

    if body_comp < config.min_body_completeness_reject or core_comp < config.min_core_completeness_reject:
        status = "REJECTED"
    elif (
        body_comp >= config.min_body_completeness_valid
        and core_comp >= config.min_core_completeness_valid
        and feet_comp >= config.min_feet_completeness_valid
        and inside_frac >= config.min_inside_fraction_valid
        and geometry_valid
    ):
        status = "VALID"
    else:
        status = "DEGRADED"

    records: List[Dict[str, Any]] = []
    outlier_indices = {i for a, b, _ in bone_outliers for i in (a, b)}
    for i, name in enumerate(WHOLEBODY_KEYPOINT_NAMES):
        if not available[i]:
            state = "MISSING"
        elif not _inside_margin(xy[i : i + 1], bbox_xyxy, config.bbox_margin_for_qa)[0] or i in outlier_indices:
            state = "GEOMETRIC_OUTLIER"
        elif low_evidence[i]:
            state = "LOW_MODEL_EVIDENCE"
        else:
            state = "VALID"
        records.append({
            "index": int(i),
            "name": name,
            "x": None if not np.isfinite(xy[i, 0]) else float(xy[i, 0]),
            "y": None if not np.isfinite(xy[i, 1]) else float(xy[i, 1]),
            "raw_model_score": float(scores[i]) if np.isfinite(scores[i]) else None,
            "state": state,
            "source": "RTMW_CACHE",
            "coordinate_evidence_kind": "RAW_OBSERVED" if state != "MISSING" else "RAW_MISSING",
            "temporal_estimate_xy": None,
        })

    qa = {
        "pose_status": status,
        "body_completeness": body_comp,
        "feet_completeness": feet_comp,
        "core_completeness": core_comp,
        "inside_fraction": inside_frac,
        "median_positive_raw_score": median_score,
        "relative_low_evidence_floor": low_floor,
        "low_model_evidence_fraction": float(np.mean(low_evidence[available])) if np.any(available) else 1.0,
        "bone_outlier_fraction": bone_outlier_fraction,
        "bone_outliers": [
            {"a": WHOLEBODY_KEYPOINT_NAMES[a], "b": WHOLEBODY_KEYPOINT_NAMES[b], "normalized_length": length}
            for a, b, length in bone_outliers
        ],
        "geometry_valid": geometry_valid,
        "note": "raw_model_score is RTMW SimCC evidence, not a calibrated probability",
    }
    return qa, records


def status_rank(status: str) -> int:
    return {"REJECTED": 0, "MISSING": 0, "DEGRADED": 1, "VALID": 2}.get(str(status), -1)


def pose_selection_score(qa: Mapping[str, Any]) -> float:
    """Ranking only, not a probability. Used to choose among controlled fallback crops."""
    return float(
        2.0 * float(qa.get("core_completeness", 0.0))
        + 1.5 * float(qa.get("feet_completeness", 0.0))
        + 1.0 * float(qa.get("body_completeness", 0.0))
        + 0.5 * float(qa.get("inside_fraction", 0.0))
        - 1.0 * float(qa.get("bone_outlier_fraction", 1.0))
    )
