from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, Mapping, Sequence
import math

from .metrics2d import summarize_2d
from .metrics3d import summarize_3d


def diameter_bin(value: float | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return "unknown"
    x = float(value)
    if x < 5:
        return "<5px"
    if x < 10:
        return "5-10px"
    if x < 15:
        return "10-15px"
    return ">=15px"


def height_bin(z: float | None) -> str:
    if z is None or not math.isfinite(float(z)):
        return "unknown"
    z = float(z)
    if z <= 0.35:
        return "ground<=0.35m"
    if z <= 1.0:
        return "0.35-1m"
    if z <= 3.0:
        return "1-3m"
    return ">3m"


def camera_distance_bin(value: float | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return "unknown"
    d = float(value)
    if d < 30:
        return "<30m"
    if d < 50:
        return "30-50m"
    if d < 70:
        return "50-70m"
    return ">=70m"


def detector_score_bin(value: float | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return "missing"
    s = float(value)
    if s < 0.25:
        return "<0.25"
    if s < 0.50:
        return "0.25-0.50"
    if s < 0.75:
        return "0.50-0.75"
    return ">=0.75"


def iou_bin(value: float | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return "missing"
    x = float(value)
    if x < 0.25:
        return "<0.25"
    if x < 0.50:
        return "0.25-0.50"
    if x < 0.75:
        return "0.50-0.75"
    return ">=0.75"


def _group(rows: Sequence[Mapping[str, Any]], key_fn: Callable[[Mapping[str, Any]], str]) -> Dict[str, list[Mapping[str, Any]]]:
    out: Dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        out[key_fn(row)].append(row)
    return dict(out)


def stratify_2d(rows: Sequence[Mapping[str, Any]], *, candidate_k: int = 5) -> Dict[str, Dict[str, Any]]:
    specs = {
        "gt_diameter_bin": lambda r: diameter_bin(r.get("gt_diameter_px")),
        "view_type": lambda r: "action" if bool(r.get("action")) else "replay",
        "ball_presence": lambda r: "ball" if r.get("gt_bbox") is not None else "no_ball",
    }
    out: Dict[str, Dict[str, Any]] = {}
    for name, fn in specs.items():
        groups = _group(rows, fn)
        out[name] = {key: summarize_2d(group, candidate_k=candidate_k) for key, group in sorted(groups.items())}
    return out


def stratify_3d(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    specs = {
        "gt_diameter_bin": lambda r: diameter_bin(r.get("gt_diameter_px")),
        "gt_height_bin": lambda r: height_bin((r.get("gt_xyz") or [None, None, None])[2]),
        "camera_distance_bin": lambda r: camera_distance_bin(r.get("camera_to_gt_distance_m")),
        "camera_status": lambda r: str(r.get("camera_status", "unknown")),
        "localization_status": lambda r: str(r.get("status", r.get("localization_status", "unknown"))),
        "view_type": lambda r: "action" if bool(r.get("action")) else "replay",
        "observation_type": lambda r: str(r.get("observation_type", "direct")),
        "detector_score_bin": lambda r: detector_score_bin(r.get("detector_score")),
        "top1_iou_bin": lambda r: iou_bin(r.get("top1_iou")),
    }
    out: Dict[str, Dict[str, Any]] = {}
    for name, fn in specs.items():
        groups = _group(rows, fn)
        out[name] = {key: summarize_3d(group) for key, group in sorted(groups.items())}
    return out


def flatten_stratification(stratified: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dimension, groups in stratified.items():
        for group, metrics in groups.items():
            row = {"dimension": dimension, "group": group}
            row.update(metrics)
            rows.append(row)
    return rows
