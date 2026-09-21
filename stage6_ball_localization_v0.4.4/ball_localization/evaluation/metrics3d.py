from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Mapping, Sequence
import numpy as np


def evaluate_record_3d(row: Mapping[str, Any]) -> Dict[str, Any]:
    gt = row.get("gt_xyz")
    pred = row.get("pred_xyz")
    out: Dict[str, Any] = {
        "record_id": row.get("record_id"),
        "has_gt_3d": gt is not None,
        "has_pred_3d": pred is not None,
        "error_3d_m": None,
        "abs_dx_m": None,
        "abs_dy_m": None,
        "abs_dz_m": None,
        "relative_error": None,
    }
    if gt is None or pred is None:
        return out
    g = np.asarray(gt, float)
    p = np.asarray(pred, float)
    diff = np.abs(p - g)
    err = float(np.linalg.norm(p - g))
    out.update({
        "error_3d_m": err,
        "abs_dx_m": float(diff[0]),
        "abs_dy_m": float(diff[1]),
        "abs_dz_m": float(diff[2]),
    })
    denom = float(row.get("camera_to_gt_distance_m") or np.linalg.norm(g))
    if denom > 1e-9:
        out["relative_error"] = err / denom
    return out


def summarize_validity(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    counts: Counter[str] = Counter()
    for row in rows:
        diag = row.get("geometry_diagnostics") or {}
        reason = diag.get("size_prior_validity")
        if reason is None:
            if row.get("pred_xyz") is not None:
                reason = "VALID"
            else:
                reason = str(row.get("status") or "UNKNOWN")
        counts[str(reason)] += 1
    total = len(rows)
    ordered = [
        "VALID",
        "INVALID_ANGULAR_SIZE",
        "INVALID_RAY_OR_DEPTH",
        "INVALID_PITCH_XY",
        "INVALID_HEIGHT",
        "INVALID_CAMERA",
        "NO_BALL_OBSERVATION",
    ]
    for key in ordered:
        counts.setdefault(key, 0)
    return {
        "total": total,
        "counts": dict(sorted(counts.items())),
        "rates": {k: (v / max(1, total)) for k, v in sorted(counts.items())},
    }


def summarize_3d(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    gt_total = sum(r.get("gt_xyz") is not None for r in rows)
    details = [evaluate_record_3d(r) for r in rows]
    valid = [d for d in details if d["has_gt_3d"] and d["has_pred_3d"]]
    base = {
        "gt_rows": gt_total,
        "valid_predictions": len(valid),
        "coverage": len(valid) / max(1, gt_total),
        "validity_breakdown": summarize_validity(rows),
    }
    if not valid:
        return {
            **base,
            "MAE_3D_m": None,
            "BLE_X_MAE_m": None,
            "SelfReprojectionErrorPx_mean": None,
            "GT3DReprojectionErrorPx_mean": None,
        }
    d = np.asarray([x["error_3d_m"] for x in valid], dtype=float)
    dx = np.asarray([x["abs_dx_m"] for x in valid], dtype=float)
    dy = np.asarray([x["abs_dy_m"] for x in valid], dtype=float)
    dz = np.asarray([x["abs_dz_m"] for x in valid], dtype=float)
    rel = [float(x["relative_error"]) for x in valid if x["relative_error"] is not None]
    self_reproj = [float(r["self_reprojection_error_px"]) for r in rows if r.get("pred_xyz") is not None and r.get("self_reprojection_error_px") is not None]
    gt_reproj = [float(r["gt_reprojection_error_px"]) for r in rows if r.get("gt_reprojection_error_px") is not None]
    return {
        **base,
        "MAE_3D_m": float(np.mean(d)),
        "RMSE_3D_m": float(np.sqrt(np.mean(d ** 2))),
        "Median3D_m": float(np.median(d)),
        "P90_3D_m": float(np.percentile(d, 90)),
        "P95_3D_m": float(np.percentile(d, 95)),
        "Precision_at_1m": float(np.mean(d <= 1.0)),
        "Precision_at_2m": float(np.mean(d <= 2.0)),
        "Precision_at_5m": float(np.mean(d <= 5.0)),
        "RelativeMAE": float(np.mean(rel)) if rel else None,
        "BLE_X_MAE_m": float(np.mean(dx)),
        "BLE_X_RMSE_m": float(np.sqrt(np.mean(dx ** 2))),
        "BLE_X_median_m": float(np.median(dx)),
        "BLE_X_P90_m": float(np.percentile(dx, 90)),
        "BLE_X_P95_m": float(np.percentile(dx, 95)),
        "MAE_Y_m": float(np.mean(dy)),
        "MAE_Z_m": float(np.mean(dz)),
        # This is intentionally named SELF reprojection: the size-prior point is
        # constructed on the camera ray through the observed center, so this
        # value is expected to be near numerical zero and is not an accuracy metric.
        "SelfReprojectionErrorPx_mean": float(np.mean(self_reproj)) if self_reproj else None,
        "GT3DReprojectionErrorPx_mean": float(np.mean(gt_reproj)) if gt_reproj else None,
    }
