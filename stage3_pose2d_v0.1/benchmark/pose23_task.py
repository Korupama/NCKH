"""Task-aware Pose23@t0 metrics for Stage 3.

This evaluator is intentionally limited to image-space Stage-3 evidence.  It
does not compute camera, 3D, team, ball or offside-decision metrics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple
import json
import math

import numpy as np

from stage3_pose2d.task_pose23 import load_and_validate_manifest, wholebody133_to_pose23
from stage3_pose2d.wholebody133 import WHOLEBODY_KEYPOINT_NAMES


POSE23_NAMES: Tuple[str, ...] = tuple(WHOLEBODY_KEYPOINT_NAMES[:23])
# Engineering OKS sigmas for the 17 COCO body points plus 6 foot points. The
# body values follow the COCO keypoint convention; foot values are explicit
# project defaults and are reported as an optional diagnostic, not as an
# official COCO-WholeBody score.
POSE23_OKS_SIGMAS: Tuple[float, ...] = (
    0.026, 0.025, 0.025, 0.035, 0.035,
    0.079, 0.079, 0.072, 0.072, 0.062, 0.062,
    0.107, 0.107, 0.087, 0.087, 0.089, 0.089,
    0.068, 0.066, 0.089, 0.068, 0.066, 0.089,
)
POSE23_GROUPS: Dict[str, Tuple[int, ...]] = {
    "head": (0, 1, 2, 3, 4),
    "torso_hip": (5, 6, 11, 12),
    "arms": (7, 8, 9, 10),
    "knees": (13, 14),
    "ankles": (15, 16),
    "toes": (17, 18, 20, 21),
    "heels": (19, 22),
    "feet": (17, 18, 19, 20, 21, 22),
}
ANATOMY_EVIDENCE_GROUPS: Dict[str, Tuple[int, ...]] = {
    "head": (0, 1, 2, 3, 4),
    "torso_hip": (5, 6, 11, 12),
    "left_leg": (11, 13, 15),
    "right_leg": (12, 14, 16),
    "left_foot": (17, 18, 19),
    "right_foot": (20, 21, 22),
}
VISIBILITY_VALUES = ("VISIBLE", "OCCLUDED", "TRUNCATED", "OUT_OF_FRAME", "NOT_ANNOTATED")
DIFFICULTY_FIELDS = (
    "scale_bin", "occlusion_level", "motion_blur", "view", "border_truncated", "crowd_level"
)


def _json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _items(payload: Any, *, ground_truth: bool) -> list[Mapping[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, Mapping)]
    if not isinstance(payload, Mapping):
        return []
    keys = ("samples", "annotations") if ground_truth else ("predictions", "samples", "annotations")
    for key in keys:
        if isinstance(payload.get(key), list):
            return [x for x in payload[key] if isinstance(x, Mapping)]
    return []


def _id(item: Mapping[str, Any]) -> str:
    value = item.get("sample_id", item.get("id"))
    if value is None:
        raise ValueError("Pose23 item is missing sample_id/id")
    return str(value)


def _point(value: Any) -> tuple[float, float] | None:
    if isinstance(value, Mapping):
        value = (value.get("x"), value.get("y"))
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        x, y = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    return (x, y) if math.isfinite(x) and math.isfinite(y) else None


def _records(records: Any, *, prediction: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xy = np.full((23, 2), np.nan, dtype=float)
    visibility = np.full(23, "NOT_ANNOTATED", dtype=object)
    raw_observed = np.zeros(23, dtype=bool)
    temporal_only = np.zeros(23, dtype=bool)
    if isinstance(records, Mapping):
        iterator: Iterable[tuple[int, Any]] = ((int(k), v) for k, v in records.items())
    elif isinstance(records, list):
        iterator = enumerate(records[:23])
    else:
        iterator = ()
    for index, value in iterator:
        if not 0 <= index < 23 or not isinstance(value, Mapping):
            continue
        xy_value = _point(value)
        if xy_value is not None:
            xy[index] = xy_value
        if prediction:
            kind = value.get("coordinate_evidence_kind")
            estimate = _point(value.get("temporal_estimate_xy"))
            is_temporal = kind == "TEMPORAL_ESTIMATE_ONLY" or (
                estimate is not None and kind != "RAW_OBSERVED"
            )
            temporal_only[index] = is_temporal
            if is_temporal and xy_value is None and estimate is not None:
                # Keep this coordinate available for the explicitly requested
                # temporal diagnostic. Primary scoring still masks it out via
                # raw_observed=False.
                xy[index] = estimate
            # Legacy fixtures have no provenance field and are treated as raw
            # observed. New Stage-3 temporal estimates are never raw points.
            raw_observed[index] = xy_value is not None and not is_temporal and (
                kind is None or kind == "RAW_OBSERVED"
            )
        else:
            visibility[index] = str(value.get("visibility", "VISIBLE")).upper()
    return xy, visibility, raw_observed, temporal_only


def _prediction_records(item: Mapping[str, Any]) -> Any:
    if "keypoints_23" in item:
        return item["keypoints_23"]
    if "keypoints_133" in item:
        return wholebody133_to_pose23(item["keypoints_133"])
    return item.get("keypoints", [])


def _bbox(item: Mapping[str, Any]) -> np.ndarray:
    value = item.get("bbox_xyxy") or item.get("bbox")
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"Pose23 item {_id(item)!r} is missing bbox_xyxy")
    box = np.asarray(value, dtype=float)
    if not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f"Pose23 item {_id(item)!r} has invalid bbox_xyxy")
    return box


def _scale(item: Mapping[str, Any]) -> float:
    box = _bbox(item)
    return float(max(box[2] - box[0], box[3] - box[1]))


def _policy(payload: Any) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if isinstance(payload, Mapping) and isinstance(payload.get("evaluation_policy"), Mapping):
        policy = payload["evaluation_policy"]
        primary = tuple(str(x).upper() for x in policy.get("primary", ["VISIBLE"]))
        secondary = tuple(str(x).upper() for x in policy.get("secondary", ["OCCLUDED"]))
        return primary or ("VISIBLE",), secondary
    return ("VISIBLE",), ("OCCLUDED",)


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


def _metric(rows: Sequence[Mapping[str, Any]], visibility: Sequence[str], thresholds: Sequence[float], groups: Mapping[str, Sequence[int]]) -> Dict[str, Any]:
    if not rows:
        return {
            "gt_count": 0, "raw_observed_count": 0, "temporal_estimate_only_count": 0,
            "raw_missing_count": 0, "raw_coverage": None, "median_normalized_error": None,
            **{f"PCK@{float(t):.2f}": None for t in thresholds},
            "per_keypoint": {}, "groups": {},
        }
    gt = np.stack([row["gt_xy"] for row in rows])
    pred = np.stack([row["pred_xy"] for row in rows])
    vis = np.stack([row["visibility"] for row in rows])
    raw = np.stack([row["raw_observed"] for row in rows])
    temporal = np.stack([row["temporal_only"] for row in rows])
    pixel_errors = np.linalg.norm(pred - gt, axis=2)
    errors = pixel_errors / np.asarray([row["scale"] for row in rows])[:, None]
    errors[~np.isfinite(pred).all(axis=2)] = np.nan

    # Bbox area is used only for the optional OKS diagnostic. PCK remains
    # normalized by max(bbox width, bbox height), as frozen in Phase 2.
    boxes = np.stack([row["bbox"] for row in rows])
    areas = np.maximum((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), 1e-9)
    sigmas = np.asarray(POSE23_OKS_SIGMAS, dtype=float)
    oks = np.exp(-(pixel_errors ** 2) / (2.0 * (sigmas[None, :] ** 2) * areas[:, None]))
    oks[~np.isfinite(errors)] = np.nan

    def one(indices: Sequence[int]) -> Dict[str, Any]:
        idx = np.asarray(tuple(indices), dtype=int)
        eligible = np.isin(vis[:, idx], np.asarray(tuple(visibility), dtype=object))
        eligible &= np.isfinite(gt[:, idx]).all(axis=2)
        observed = eligible & raw[:, idx]
        temporal_points = eligible & temporal[:, idx]
        valid_error = observed & np.isfinite(errors[:, idx])
        result: Dict[str, Any] = {
            "gt_count": int(np.sum(eligible)),
            "raw_observed_count": int(np.sum(observed)),
            "temporal_estimate_only_count": int(np.sum(temporal_points)),
            "raw_missing_count": int(np.sum(eligible & ~observed & ~temporal_points)),
        }
        for threshold in thresholds:
            result[f"PCK@{float(threshold):.2f}"] = _ratio(
                int(np.sum(valid_error & (errors[:, idx] <= float(threshold)))), result["gt_count"]
            )
        result["raw_coverage"] = _ratio(result["raw_observed_count"], result["gt_count"])
        finite = errors[:, idx][valid_error]
        result["median_normalized_error"] = None if finite.size == 0 else float(np.median(finite))
        oks_valid = observed & np.isfinite(oks[:, idx])
        result["OKS"] = None if not np.any(oks_valid) else float(np.mean(oks[:, idx][oks_valid]))
        result["oks_observed_count"] = int(np.sum(oks_valid))
        return result

    result = one(tuple(range(23)))
    result["per_keypoint"] = {
        name: {"index": index, **one((index,))}
        for index, name in enumerate(POSE23_NAMES)
    }
    result["groups"] = {name: one(indices) for name, indices in groups.items()}
    return result


def _anatomy_coverage(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    groups: Dict[str, Any] = {}
    for name, indices in ANATOMY_EVIDENCE_GROUPS.items():
        idx = np.asarray(indices, dtype=int)
        raw_complete = sum(bool(np.all(row["raw_observed"][idx])) for row in rows)
        usable_complete = sum(
            bool(np.all(row["raw_observed"][idx]))
            and str(row.get("pose_status", "")).upper() not in {"MISSING", "REJECTED"}
            for row in rows
        )
        temporal_present = sum(bool(np.any(row["temporal_only"][idx])) for row in rows)
        groups[name] = {
            "raw_observed_complete_count": raw_complete,
            "raw_observed_complete_coverage": _ratio(raw_complete, len(rows)),
            "usable_raw_complete_count": usable_complete,
            "usable_raw_complete_coverage": _ratio(usable_complete, len(rows)),
            "temporal_estimate_present_count": temporal_present,
        }
    return {
        "sample_count": len(rows),
        "groups": groups,
        "note": "Image-space evidence only; not legal-body or offside semantics.",
    }


def _tag_value(item: Mapping[str, Any], field: str) -> str:
    tags = item.get("tags") if isinstance(item.get("tags"), Mapping) else {}
    value = tags.get(field, "UNKNOWN")
    if field == "border_truncated" and isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    return str(value).upper() if value is not None else "UNKNOWN"


def _rows_for_slice(rows: Sequence[Mapping[str, Any]], field: str, value: str) -> list[Mapping[str, Any]]:
    return [row for row in rows if _tag_value(row["gt_item"], field) == value]


def _slices(rows: Sequence[Mapping[str, Any]], primary: Sequence[str], secondary: Sequence[str], thresholds: Sequence[float]) -> Dict[str, Any]:
    report: Dict[str, Any] = {}
    for field in DIFFICULTY_FIELDS:
        values = sorted({_tag_value(row["gt_item"], field) for row in rows})
        report[field] = {}
        for value in values:
            subset = _rows_for_slice(rows, field, value)
            report[field][value] = {
                "sample_count": len(subset),
                "primary": _metric(subset, primary, thresholds, POSE23_GROUPS),
                "secondary": _metric(subset, secondary, thresholds, POSE23_GROUPS),
                "anatomy_coverage": _anatomy_coverage(subset),
            }
    return report


def evaluate_pose23_task(
    gt_path: str | Path,
    pred_path: str | Path,
    *,
    thresholds: Sequence[float] = (0.05, 0.10),
    score_temporal_estimates: bool = False,
    include_oks: bool = False,
) -> Dict[str, Any]:
    """Evaluate Pose23 with per-point, group, slice and provenance reports."""
    gt_payload = _json(gt_path)
    pred_payload = _json(pred_path)
    gt_items = _items(gt_payload, ground_truth=True)
    pred_items = _items(pred_payload, ground_truth=False)
    if not gt_items:
        raise RuntimeError("No Pose23 ground-truth annotations")
    if isinstance(gt_payload, Mapping) and gt_payload.get("samples"):
        load_and_validate_manifest(gt_path, allow_empty=False)
    primary, secondary = _policy(gt_payload)

    predictions: Dict[str, Mapping[str, Any]] = {}
    for item in pred_items:
        key = _id(item)
        if key in predictions:
            raise ValueError(f"Duplicate prediction id {key!r}")
        predictions[key] = item

    rows: list[Dict[str, Any]] = []
    missing: list[str] = []
    for gt_item in gt_items:
        key = _id(gt_item)
        pred_item = predictions.get(key, {})
        if key not in predictions:
            missing.append(key)
        gt_xy, visibility, _, _ = _records(gt_item.get("keypoints_23", gt_item.get("keypoints", [])), prediction=False)
        pred_xy, _, raw, temporal = _records(_prediction_records(pred_item), prediction=True)
        rows.append({
            "id": key, "gt_item": gt_item, "gt_xy": gt_xy, "visibility": visibility,
            "pred_xy": pred_xy, "raw_observed": raw, "temporal_only": temporal,
            "scale": _scale(gt_item), "bbox": _bbox(gt_item),
            "pose_status": pred_item.get("pose_status", ""),
        })

    primary_metric = _metric(rows, primary, thresholds, POSE23_GROUPS)
    secondary_metric = _metric(rows, secondary, thresholds, POSE23_GROUPS)
    primary_report = {
        "overall": {key: value for key, value in primary_metric.items()
                    if key not in {"per_keypoint", "groups"}},
        "per_keypoint": primary_metric["per_keypoint"],
        "groups": primary_metric["groups"],
    }
    secondary_report = {
        "overall": {key: value for key, value in secondary_metric.items()
                     if key not in {"per_keypoint", "groups"}},
        "per_keypoint": secondary_metric["per_keypoint"],
        "groups": secondary_metric["groups"],
    }
    metrics: Dict[str, Any] = {
        # Keep top-level aliases for callers of the previous evaluator.
        **primary_metric,
        "primary": primary_report,
        "secondary": secondary_report,
        "anatomy_coverage": _anatomy_coverage(rows),
        "difficulty_slices": _slices(rows, primary, secondary, thresholds),
        "provenance": {
            "raw_observed_points": int(sum(np.sum(row["raw_observed"]) for row in rows)),
            "temporal_estimate_only_points": int(sum(np.sum(row["temporal_only"]) for row in rows)),
            "primary_metric_coordinate_source": "RAW_OBSERVED_ONLY",
            "temporal_estimates_scored": False,
        },
    }
    if not include_oks:
        def strip_oks(value: Any) -> None:
            if isinstance(value, dict):
                value.pop("OKS", None)
                value.pop("oks_observed_count", None)
                for child in value.values():
                    strip_oks(child)
            elif isinstance(value, list):
                for child in value:
                    strip_oks(child)
        strip_oks(metrics)
    else:
        metrics["oks_protocol"] = {
            "enabled": True,
            "normalization": "bbox_area",
            "formula": "exp(-pixel_error^2 / (2 * sigma^2 * bbox_area))",
            "sigmas": list(POSE23_OKS_SIGMAS),
            "foot_sigma_policy": "project_engineering_defaults_not_official_wholebody_score",
        }
    if score_temporal_estimates:
        # Explicit diagnostic only; never overwrite the primary score.
        diagnostic_rows = []
        for row in rows:
            copy = dict(row)
            copy["raw_observed"] = np.logical_or(row["raw_observed"], row["temporal_only"])
            diagnostic_rows.append(copy)
        metrics["temporal_estimate_diagnostic"] = _metric(
            diagnostic_rows, primary, thresholds, POSE23_GROUPS
        )
        metrics["provenance"]["temporal_estimates_scored"] = True

    return {
        "schema_version": "stage3-pose23-eval-1.1",
        "protocol": {
            "thresholds": [float(x) for x in thresholds],
            "normalization": "max(bbox_width,bbox_height)",
            "primary_visibility": list(primary),
            "secondary_visibility": list(secondary),
            "coordinate_space": "RAW_DISTORTED_PIXEL",
            "primary_coordinate_source": "RAW_OBSERVED_ONLY",
            "temporal_estimates_are_diagnostic_only": True,
            "oks_enabled": bool(include_oks),
        },
        "samples": len(rows),
        "matched_predictions": len(rows) - len(missing),
        "missing_prediction_ids": missing,
        "extra_prediction_ids": sorted(set(predictions) - {row["id"] for row in rows}),
        "matched_ids": [row["id"] for row in rows if row["id"] not in missing],
        "metrics": metrics,
    }
