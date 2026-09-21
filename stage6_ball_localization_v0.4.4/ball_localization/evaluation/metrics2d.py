from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence
import math
import numpy as np


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + bb - inter
    return inter / union if union > 0 else 0.0


def center(box: Sequence[float]) -> np.ndarray:
    x1, y1, x2, y2 = map(float, box)
    return np.asarray([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=float)


def diameter(box: Sequence[float]) -> float:
    x1, y1, x2, y2 = map(float, box)
    return 0.5 * (max(0.0, x2 - x1) + max(0.0, y2 - y1))


def ap101(scores: Sequence[float], tp: Sequence[int], fp: Sequence[int], num_gt: int) -> float:
    if num_gt <= 0 or not scores:
        return 0.0
    order = np.argsort(-np.asarray(scores, float))
    t = np.asarray(tp, float)[order]
    f = np.asarray(fp, float)[order]
    ct = np.cumsum(t)
    cf = np.cumsum(f)
    rec = ct / max(1, num_gt)
    prec = ct / np.maximum(ct + cf, 1e-12)
    vals = []
    for r in np.linspace(0, 1, 101):
        mask = rec >= r
        vals.append(float(np.max(prec[mask])) if np.any(mask) else 0.0)
    return float(np.mean(vals))


def evaluate_record_2d(rec: Mapping[str, Any], *, iou_threshold: float = 0.5, candidate_k: int = 5) -> Dict[str, Any]:
    gt = rec.get("gt_bbox")
    preds = list(rec.get("predictions") or [])
    top = preds[0] if preds else None
    has_gt = gt is not None
    top_iou = None if top is None or not has_gt else float(iou(top["bbox_xyxy"], gt))
    top_match = bool(has_gt and top is not None and top_iou is not None and top_iou >= iou_threshold)
    candidate_hit = bool(
        has_gt and any(iou(p["bbox_xyxy"], gt) >= iou_threshold for p in preds[:candidate_k])
    )
    gt_d = None if not has_gt else float(diameter(gt))
    pred_d = None if top is None else float(diameter(top["bbox_xyxy"]))
    center_error = None
    center_error_norm = None
    diameter_relative_error = None
    bbox_diag_error_pct = None
    if has_gt and top is not None:
        center_error = float(np.linalg.norm(center(top["bbox_xyxy"]) - center(gt)))
        diag = max(math.hypot(float(rec.get("img_w", 1)), float(rec.get("img_h", 1))), 1.0)
        center_error_norm = center_error / diag
        if gt_d and gt_d > 1e-9:
            diameter_relative_error = abs(float(pred_d) - gt_d) / gt_d
        gx1, gy1, gx2, gy2 = map(float, gt)
        px1, py1, px2, py2 = map(float, top["bbox_xyxy"])
        gdiag = math.hypot(gx2 - gx1, gy2 - gy1)
        pdiag = math.hypot(px2 - px1, py2 - py1)
        if gdiag > 1e-9:
            bbox_diag_error_pct = 100.0 * abs(pdiag - gdiag) / gdiag
    return {
        "record_id": rec.get("record_id"),
        "has_gt": has_gt,
        "num_predictions": len(preds),
        "top1_exists": top is not None,
        "top1_score": None if top is None else float(top.get("score", 0.0)),
        "top1_iou": top_iou,
        "top1_match": top_match,
        "candidate_hit": candidate_hit,
        "center_error_px": center_error,
        "center_error_norm": center_error_norm,
        "gt_diameter_px": gt_d,
        "pred_diameter_px": pred_d,
        "diameter_relative_error": diameter_relative_error,
        "bbox_diagonal_size_error_pct": bbox_diag_error_pct,
    }


def _ap_at_iou(records: Sequence[Mapping[str, Any]], threshold: float) -> float:
    total_gt = 0
    scores: list[float] = []
    flags_tp: list[int] = []
    flags_fp: list[int] = []
    for rec in records:
        gt = rec.get("gt_bbox")
        preds = list(rec.get("predictions") or [])
        has_gt = gt is not None
        total_gt += int(has_gt)
        matched = False
        for pred in preds:
            ok = bool(has_gt and not matched and iou(pred["bbox_xyxy"], gt) >= threshold)
            matched = matched or ok
            scores.append(float(pred.get("score", 0.0)))
            flags_tp.append(int(ok))
            flags_fp.append(int(not ok))
    return ap101(scores, flags_tp, flags_fp, total_gt)


def summarize_2d(
    records: Sequence[Mapping[str, Any]],
    *,
    iou_threshold: float = 0.5,
    candidate_k: int = 5,
    candidate_ks: Sequence[int] | None = None,
) -> Dict[str, Any]:
    details = [evaluate_record_2d(r, iou_threshold=iou_threshold, candidate_k=candidate_k) for r in records]
    num_gt = sum(int(d["has_gt"]) for d in details)
    top1_tp = sum(int(d["top1_match"]) for d in details)
    top1_fn = sum(int(d["has_gt"] and not d["top1_match"]) for d in details)
    top1_fp = sum(int(d["top1_exists"] and not d["top1_match"]) for d in details)
    # Legacy error summaries remain conditioned on IoU-matched top1 detections for
    # backward compatibility.  v0.3.6 also reports all-top1 errors so original
    # vs optimized GT comparisons do not silently change the evaluated subset.
    cerr = [float(d["center_error_px"]) for d in details if d["top1_match"] and d["center_error_px"] is not None]
    cnorm = [float(d["center_error_norm"]) for d in details if d["top1_match"] and d["center_error_norm"] is not None]
    derr = [float(d["diameter_relative_error"]) for d in details if d["top1_match"] and d["diameter_relative_error"] is not None]
    diagerr = [float(d["bbox_diagonal_size_error_pct"]) for d in details if d["top1_match"] and d["bbox_diagonal_size_error_pct"] is not None]
    topious_all = [float(d["top1_iou"]) for d in details if d["has_gt"] and d["top1_iou"] is not None]
    cerr_all = [float(d["center_error_px"]) for d in details if d["has_gt"] and d["top1_exists"] and d["center_error_px"] is not None]
    derr_all = [float(d["diameter_relative_error"]) for d in details if d["has_gt"] and d["top1_exists"] and d["diameter_relative_error"] is not None]
    diagerr_all = [float(d["bbox_diagonal_size_error_pct"]) for d in details if d["has_gt"] and d["top1_exists"] and d["bbox_diagonal_size_error_pct"] is not None]
    thresholds = [round(0.50 + 0.05 * i, 2) for i in range(10)]
    per_ap = {f"AP@{t:.2f}": _ap_at_iou(records, t) for t in thresholds}

    ks = sorted({max(1, int(k)) for k in (candidate_ks or [candidate_k])})
    candidate_recalls: Dict[str, float] = {}
    for k in ks:
        hits = 0
        for rec in records:
            gt = rec.get("gt_bbox")
            if gt is None:
                continue
            preds = list(rec.get("predictions") or [])
            hits += int(any(iou(p["bbox_xyxy"], gt) >= iou_threshold for p in preds[:k]))
        candidate_recalls[f"CandidateRecall@{k}"] = hits / max(1, num_gt)

    return {
        "images": len(records),
        "num_gt": num_gt,
        "top1_tp": top1_tp,
        "top1_fp": top1_fp,
        "top1_fn": top1_fn,
        "precision": top1_tp / max(1, top1_tp + top1_fp),
        "recall": top1_tp / max(1, top1_tp + top1_fn),
        "AP50": per_ap["AP@0.50"],
        "AP75": per_ap["AP@0.75"],
        "mAP50_95": float(np.mean(list(per_ap.values()))),
        "per_threshold_AP": per_ap,
        **candidate_recalls,
        "CenterErrorPx_mean": float(np.mean(cerr)) if cerr else None,
        "CenterErrorPx_median": float(np.median(cerr)) if cerr else None,
        "CenterErrorPx_P90": float(np.percentile(cerr, 90)) if cerr else None,
        "CenterErrorNorm_mean": float(np.mean(cnorm)) if cnorm else None,
        "DiameterRelativeError_mean": float(np.mean(derr)) if derr else None,
        "DiameterRelativeError_median": float(np.median(derr)) if derr else None,
        "DiameterRelativeError_P90": float(np.percentile(derr, 90)) if derr else None,
        "BBoxDiagonalSizeErrorPct_mean": float(np.mean(diagerr)) if diagerr else None,
        "BBoxDiagonalSizeErrorPct_median": float(np.median(diagerr)) if diagerr else None,
        "Top1IoUAllGT_mean": float(np.mean(topious_all)) if topious_all else None,
        "Top1IoUAllGT_median": float(np.median(topious_all)) if topious_all else None,
        "Top1IoUAllGT_P90": float(np.percentile(topious_all, 90)) if topious_all else None,
        "CenterErrorPxAllTop1_mean": float(np.mean(cerr_all)) if cerr_all else None,
        "CenterErrorPxAllTop1_median": float(np.median(cerr_all)) if cerr_all else None,
        "CenterErrorPxAllTop1_P90": float(np.percentile(cerr_all, 90)) if cerr_all else None,
        "DiameterRelativeErrorAllTop1_mean": float(np.mean(derr_all)) if derr_all else None,
        "DiameterRelativeErrorAllTop1_median": float(np.median(derr_all)) if derr_all else None,
        "DiameterRelativeErrorAllTop1_P90": float(np.percentile(derr_all, 90)) if derr_all else None,
        "BBoxDiagonalSizeErrorPctAllTop1_mean": float(np.mean(diagerr_all)) if diagerr_all else None,
        "BBoxDiagonalSizeErrorPctAllTop1_median": float(np.median(diagerr_all)) if diagerr_all else None,
    }
