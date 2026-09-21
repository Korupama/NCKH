from __future__ import annotations

from pathlib import Path
from ..version import PACKAGE_VERSION, runtime_provenance
from typing import Any, Mapping, Sequence

from ..datasets import SoccerNetV3DCSV, gt_bbox_for_record, optimized_bbox_from_record
from .common import atomic_json, flatten_mapping, load_json, record_key, sha256_file, write_csv
from .failure_analysis import classify_2d_failure, render_failure_overlays
from .metrics2d import diameter, evaluate_record_2d, summarize_2d
from .stratification import flatten_stratification, stratify_2d


def _load_source_checkpoint(benchmark_2d_dir: str | Path) -> dict[str, Any]:
    root = Path(benchmark_2d_dir).expanduser().resolve()
    checkpoint = root / "checkpoint.json"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return load_json(checkpoint)


def _load_prediction(benchmark_2d_dir: str | Path, record_id: str) -> dict[str, Any] | None:
    path = Path(benchmark_2d_dir).expanduser().resolve() / "predictions" / f"{record_key(record_id)}.json"
    if not path.is_file():
        return None
    payload = load_json(path)
    if str(payload.get("record_id")) != str(record_id):
        raise RuntimeError(f"Stale/colliding cached prediction: {path}")
    return payload


def _row_for_mode(record, payload: Mapping[str, Any], mode: str) -> dict[str, Any]:
    gt = gt_bbox_for_record(record, mode)
    row = dict(payload)
    row["record_id"] = record.record_id
    row["row_index"] = record.row_index
    row["img_w"] = record.img_w
    row["img_h"] = record.img_h
    row["action"] = record.action
    row["replay"] = record.replay
    row["gt_box_mode"] = mode
    row["gt_bbox"] = gt
    row["gt_diameter_px"] = None if gt is None else diameter(gt)
    row["gt_bbox_original"] = gt_bbox_for_record(record, "original")
    row["gt_bbox_optimized"] = optimized_bbox_from_record(record) if record.is_ball else None
    row["gt_diameter_original_px"] = None if record.ball_bbox is None else diameter(record.ball_bbox)
    row["gt_diameter_optimized_px"] = record.optimized_d if record.is_ball else None
    return row


def _safe_delta(a: Any, b: Any) -> float | None:
    if a is None or b is None:
        return None
    return float(a) - float(b)


def _relative_reduction(original: Any, optimized: Any) -> float | None:
    if original is None or optimized is None:
        return None
    original = float(original)
    optimized = float(optimized)
    if abs(original) <= 1e-12:
        return None
    return 100.0 * (original - optimized) / abs(original)


def _comparison_deltas(original: Mapping[str, Any], optimized: Mapping[str, Any]) -> dict[str, Any]:
    direct = [
        "precision", "recall", "AP50", "AP75", "mAP50_95",
        "CandidateRecall@1", "CandidateRecall@5", "CandidateRecall@10",
        "CenterErrorPx_mean", "CenterErrorPx_median", "CenterErrorPx_P90",
        "DiameterRelativeError_mean", "DiameterRelativeError_median", "DiameterRelativeError_P90",
        "BBoxDiagonalSizeErrorPct_mean", "BBoxDiagonalSizeErrorPct_median",
        "Top1IoUAllGT_mean", "Top1IoUAllGT_median", "Top1IoUAllGT_P90",
        "CenterErrorPxAllTop1_mean", "CenterErrorPxAllTop1_median", "CenterErrorPxAllTop1_P90",
        "DiameterRelativeErrorAllTop1_mean", "DiameterRelativeErrorAllTop1_median", "DiameterRelativeErrorAllTop1_P90",
        "BBoxDiagonalSizeErrorPctAllTop1_mean", "BBoxDiagonalSizeErrorPctAllTop1_median",
    ]
    out = {f"{k}_optimized_minus_original": _safe_delta(optimized.get(k), original.get(k)) for k in direct}
    for k in (
        "CenterErrorPx_mean", "CenterErrorPx_median", "CenterErrorPx_P90",
        "DiameterRelativeError_mean", "DiameterRelativeError_median", "DiameterRelativeError_P90",
        "BBoxDiagonalSizeErrorPct_mean", "BBoxDiagonalSizeErrorPct_median",
        "CenterErrorPxAllTop1_mean", "CenterErrorPxAllTop1_median", "CenterErrorPxAllTop1_P90",
        "DiameterRelativeErrorAllTop1_mean", "DiameterRelativeErrorAllTop1_median", "DiameterRelativeErrorAllTop1_P90",
        "BBoxDiagonalSizeErrorPctAllTop1_mean", "BBoxDiagonalSizeErrorPctAllTop1_median",
    ):
        out[f"{k}_relative_reduction_pct"] = _relative_reduction(original.get(k), optimized.get(k))
    return out


def run_cached_2d_gt_comparison(
    *,
    csv_path: str | Path,
    benchmark_2d_dir: str | Path,
    split: str,
    output_dir: str | Path,
    candidate_ks: Sequence[int] = (1, 5, 10),
    iou_threshold: float = 0.5,
    max_images: int | None = None,
    failure_images: int = 0,
    allow_partial_2d: bool = False,
) -> dict[str, Any]:
    dataset = SoccerNetV3DCSV(csv_path)
    records = dataset.split(split)
    if max_images is not None:
        records = records[: int(max_images)]
    if not records:
        raise RuntimeError(f"No records found in split {split!r}")
    missing_optimized_gt = [r.record_id for r in records if r.is_ball and r.ball_bbox is not None and optimized_bbox_from_record(r) is None]
    if missing_optimized_gt:
        raise RuntimeError(
            f"Cannot compare optimized GT: optimized_d is unavailable/invalid for {len(missing_optimized_gt)} ball rows; "
            f"example: {missing_optimized_gt[0]}"
        )

    source_root = Path(benchmark_2d_dir).expanduser().resolve()
    source_state = _load_source_checkpoint(source_root)
    if not allow_partial_2d and str(source_state.get("status")) != "COMPLETE":
        raise RuntimeError(
            "Source 2D benchmark is not COMPLETE. Resume the detector run first or pass --allow-partial-2d."
        )
    source_identity = dict(source_state.get("run_identity") or {})
    csv_hash = sha256_file(csv_path)
    if source_identity.get("csv_sha256") and str(source_identity["csv_sha256"]).lower() != csv_hash.lower():
        raise RuntimeError("Source 2D cache was produced from a different SNv3D.csv")
    if source_identity.get("split") and str(source_identity["split"]) != str(split):
        raise RuntimeError("Source 2D cache split differs from requested split")

    ks = sorted({max(1, int(k)) for k in candidate_ks})
    if not ks:
        ks = [1, 5, 10]
    primary_k = 5 if 5 in ks else ks[-1]

    rows_by_mode: dict[str, list[dict[str, Any]]] = {"original": [], "optimized": []}
    missing_cache: list[str] = []
    for record in records:
        payload = _load_prediction(source_root, record.record_id)
        if payload is None:
            missing_cache.append(record.record_id)
            continue
        expected_signature = source_state.get("run_signature")
        payload_signature = payload.get("run_signature")
        if expected_signature and payload_signature and str(payload_signature) != str(expected_signature):
            raise RuntimeError(f"Cached prediction for {record.record_id} does not belong to the source checkpoint")
        for mode in ("original", "optimized"):
            rows_by_mode[mode].append(_row_for_mode(record, payload, mode))

    if missing_cache and not allow_partial_2d:
        raise RuntimeError(
            f"Source 2D cache is missing {len(missing_cache)} requested records. "
            "Resume run-2d or pass --allow-partial-2d for a provisional comparison."
        )

    evaluable_by_mode = {
        mode: [r for r in rows if not bool(r.get("missing_image"))]
        for mode, rows in rows_by_mode.items()
    }
    metrics = {
        mode: summarize_2d(
            rows,
            iou_threshold=iou_threshold,
            candidate_k=primary_k,
            candidate_ks=ks,
        )
        for mode, rows in evaluable_by_mode.items()
    }

    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    # Per-frame side-by-side diagnostics. Predictions are identical in both columns;
    # only the GT convention changes.
    original_map = {r["record_id"]: r for r in evaluable_by_mode["original"]}
    optimized_map = {r["record_id"]: r for r in evaluable_by_mode["optimized"]}
    frame_rows: list[dict[str, Any]] = []
    for record in records:
        ro = original_map.get(record.record_id)
        rp = optimized_map.get(record.record_id)
        if ro is None or rp is None:
            continue
        mo = evaluate_record_2d(ro, iou_threshold=iou_threshold, candidate_k=primary_k)
        mp = evaluate_record_2d(rp, iou_threshold=iou_threshold, candidate_k=primary_k)
        top = list(ro.get("predictions") or [])
        frame = {
            "record_id": record.record_id,
            "image_path": ro.get("image_path"),
            "action": record.action,
            "replay": record.replay,
            "top1_score": mo.get("top1_score"),
            "pred_diameter_px": mo.get("pred_diameter_px"),
            "gt_diameter_original_px": ro.get("gt_diameter_original_px"),
            "gt_diameter_optimized_px": rp.get("gt_diameter_optimized_px"),
            "top1_iou_original": mo.get("top1_iou"),
            "top1_iou_optimized": mp.get("top1_iou"),
            "top1_match_original": mo.get("top1_match"),
            "top1_match_optimized": mp.get("top1_match"),
            "center_error_px_original": mo.get("center_error_px"),
            "center_error_px_optimized": mp.get("center_error_px"),
            "diameter_relative_error_original": mo.get("diameter_relative_error"),
            "diameter_relative_error_optimized": mp.get("diameter_relative_error"),
            "bbox_diagonal_size_error_pct_original": mo.get("bbox_diagonal_size_error_pct"),
            "bbox_diagonal_size_error_pct_optimized": mp.get("bbox_diagonal_size_error_pct"),
            "num_predictions": len(top),
        }
        for k in ks:
            frame[f"candidate_hit_original_at_{k}"] = evaluate_record_2d(ro, iou_threshold=iou_threshold, candidate_k=k)["candidate_hit"]
            frame[f"candidate_hit_optimized_at_{k}"] = evaluate_record_2d(rp, iou_threshold=iou_threshold, candidate_k=k)["candidate_hit"]
        frame_rows.append(frame)

    failures_by_mode: dict[str, list[dict[str, Any]]] = {}
    stratification_by_mode: dict[str, Any] = {}
    for mode in ("original", "optimized"):
        rows = evaluable_by_mode[mode]
        failures = [f for r in rows if (f := classify_2d_failure(r, iou_threshold=iou_threshold, candidate_k=primary_k)) is not None]
        failures.sort(key=lambda x: float(x.get("severity", 0.0)), reverse=True)
        failures_by_mode[mode] = failures
        atomic_json(out / f"failure_cases_{mode}.json", failures)
        if failure_images > 0:
            render_failure_overlays(failures, out / f"failure_cases_{mode}", max_images=failure_images)
        strat = stratify_2d(rows, candidate_k=primary_k)
        stratification_by_mode[mode] = strat
        atomic_json(out / f"stratification_{mode}.json", strat)
        write_csv(out / f"stratification_{mode}.csv", flatten_stratification(strat))

    deltas = _comparison_deltas(metrics["original"], metrics["optimized"])
    report = {
        "schema_version": "stage6-ball-2d-gt-comparison-1.0",
        "stage6_version": PACKAGE_VERSION,
        "runtime_provenance": runtime_provenance(),
        "status": "PARTIAL" if missing_cache else "COMPLETE",
        "dataset": dataset.summary(),
        "source_benchmark_2d_dir": str(source_root),
        "source_checkpoint_status": source_state.get("status"),
        "source_run_identity": source_identity,
        "csv": str(Path(csv_path).expanduser().resolve()),
        "csv_sha256": csv_hash,
        "split": split,
        "records_requested": len(records),
        "cached_records_found": len(records) - len(missing_cache),
        "missing_cached_records": len(missing_cache),
        "missing_cached_examples": missing_cache[:10],
        "iou_threshold": float(iou_threshold),
        "candidate_ks": ks,
        "gt_conventions": {
            "original": "Original SoccerNet-v3 ball_bbox parsed from (x_top,y_top,width,height).",
            "optimized": "Square box centered at the original ball_bbox center with side length equal to SoccerNet-v3D optimized_d (pixels).",
            "prediction_reuse": "Identical cached detector predictions are evaluated against both GT conventions; YOLO is not rerun.",
        },
        "metrics": metrics,
        "delta_optimized_minus_original": deltas,
        "failure_cases": {mode: len(items) for mode, items in failures_by_mode.items()},
        "note": (
            "Use optimized GT for yolo-sn-ball-opt.pt alignment. This comparison is an internal 101-point AP implementation, "
            "not a claim of byte-for-byte equivalence with the authors' training/evaluation code."
        ),
    }
    atomic_json(out / "benchmark_2d_gt_comparison.json", report)
    write_csv(out / "benchmark_2d_gt_comparison.csv", [flatten_mapping(report)])
    write_csv(out / "frame_metrics_gt_comparison.csv", frame_rows)
    return report
