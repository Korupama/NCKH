from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Sequence
import cv2
import numpy as np

from ..datasets import read_image_reference
from .metrics2d import evaluate_record_2d, iou
from .metrics3d import evaluate_record_3d


def classify_2d_failure(
    row: Mapping[str, Any],
    *,
    iou_threshold: float = 0.5,
    candidate_k: int = 5,
    center_error_px_threshold: float = 15.0,
    diameter_relative_error_threshold: float = 0.30,
) -> Dict[str, Any] | None:
    m = evaluate_record_2d(row, iou_threshold=iou_threshold, candidate_k=candidate_k)
    types: list[str] = []
    severity = 0.0
    gt = row.get("gt_bbox")
    preds = list(row.get("predictions") or [])
    if gt is not None:
        if not preds:
            types.append("MISS_NO_DETECTION")
            severity = max(severity, 3.0)
        elif not m["top1_match"]:
            if m["candidate_hit"]:
                types.append("WRONG_TOP1_BUT_GT_IN_TOPK")
                severity = max(severity, 2.0)
            else:
                types.append("MISS_GT_NOT_IN_TOPK")
                severity = max(severity, 3.0)
        if m["center_error_px"] is not None and m["center_error_px"] > center_error_px_threshold:
            types.append("LARGE_CENTER_ERROR")
            severity = max(severity, float(m["center_error_px"]) / max(center_error_px_threshold, 1e-6))
        if m["diameter_relative_error"] is not None and m["diameter_relative_error"] > diameter_relative_error_threshold:
            types.append("LARGE_DIAMETER_ERROR")
            severity = max(severity, float(m["diameter_relative_error"]) / max(diameter_relative_error_threshold, 1e-6))
    elif preds:
        types.append("FALSE_POSITIVE_NO_BALL_GT")
        severity = max(severity, 2.0 + float(preds[0].get("score", 0.0)))
    if not types:
        return None
    return {
        "record_id": row.get("record_id"),
        "failure_types": types,
        "severity": float(severity),
        "image_path": row.get("image_path"),
        "gt_bbox": gt,
        "predictions": preds[: max(candidate_k, 3)],
        "metrics": m,
    }


def classify_3d_failure(
    row: Mapping[str, Any],
    *,
    error_3d_threshold_m: float = 2.0,
    blex_threshold_m: float = 1.0,
) -> Dict[str, Any] | None:
    m = evaluate_record_3d(row)
    types: list[str] = []
    severity = 0.0
    if row.get("gt_xyz") is None:
        return None
    if row.get("pred_xyz") is None:
        types.append("NO_VALID_3D_PREDICTION")
        severity = 4.0
    else:
        if m["error_3d_m"] is not None and m["error_3d_m"] > error_3d_threshold_m:
            types.append("LARGE_3D_ERROR")
            severity = max(severity, float(m["error_3d_m"]) / max(error_3d_threshold_m, 1e-6))
        if m["abs_dx_m"] is not None and m["abs_dx_m"] > blex_threshold_m:
            types.append("LARGE_BLE_X")
            severity = max(severity, float(m["abs_dx_m"]) / max(blex_threshold_m, 1e-6))
    if row.get("pred_bbox") is None and row.get("detector_required", False):
        types.append("NO_BALL_DETECTION")
        severity = max(severity, 4.0)
    elif row.get("top1_iou") is not None and float(row.get("top1_iou")) < 0.5:
        types.append("WRONG_TOP1_2D")
        severity = max(severity, 2.0)
    if not types:
        return None
    return {
        "record_id": row.get("record_id"),
        "failure_types": types,
        "severity": float(severity),
        "image_path": row.get("image_path"),
        "gt_bbox": row.get("gt_bbox"),
        "pred_bbox": row.get("pred_bbox"),
        "gt_xyz": row.get("gt_xyz"),
        "pred_xyz": row.get("pred_xyz"),
        "status": row.get("status"),
        "metrics": m,
    }


def render_failure_overlays(failures: Sequence[Mapping[str, Any]], out_dir: str | Path, *, max_images: int = 20) -> list[str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rendered: list[str] = []
    ranked = sorted(failures, key=lambda x: float(x.get("severity", 0.0)), reverse=True)
    for idx, failure in enumerate(ranked[: max(0, int(max_images))], start=1):
        image_path = failure.get("image_path")
        if not image_path:
            continue
        image = read_image_reference(str(image_path))
        if image is None:
            continue
        gt = failure.get("gt_bbox")
        if gt is not None:
            x1, y1, x2, y2 = [int(round(float(x))) for x in gt]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, "GT", (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        pred_bbox = failure.get("pred_bbox")
        if pred_bbox is not None:
            x1, y1, x2, y2 = [int(round(float(x))) for x in pred_bbox]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(image, "PR", (x1, min(image.shape[0] - 8, y2 + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            for rank, pred in enumerate(failure.get("predictions") or [], start=1):
                box = pred.get("bbox_xyxy")
                if box is None:
                    continue
                x1, y1, x2, y2 = [int(round(float(x))) for x in box]
                color = (0, 0, 255) if rank == 1 else (0, 165, 255)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, f"P{rank}", (x1, min(image.shape[0] - 8, y2 + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        labels = ",".join(str(x) for x in failure.get("failure_types") or [])
        cv2.rectangle(image, (0, 0), (image.shape[1], 42), (0, 0, 0), -1)
        cv2.putText(image, labels[:120], (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        path = out / f"{idx:03d}_{str(failure.get('record_id','record')).replace('/','_')}.jpg"
        cv2.imwrite(str(path), image)
        rendered.append(str(path.resolve()))
    return rendered
