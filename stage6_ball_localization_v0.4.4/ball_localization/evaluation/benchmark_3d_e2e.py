from __future__ import annotations

from pathlib import Path
from ..version import PACKAGE_VERSION, runtime_provenance
from typing import Any
import time

import numpy as np

from ..camera import camera_from_soccernet_calibration
from ..contracts import BallCandidate2D
from ..coordinates import coordinate_transform_metadata
from ..datasets import SoccerNetV3DCSV, gt_bbox_for_record
from ..datasets.soccernet_v3d import parse_literal
from ..geometry import estimate_frame
from .checkpointing import RecordCheckpointStore
from .common import atomic_json, load_json, record_key, sha256_file
from .failure_analysis import classify_3d_failure, render_failure_overlays
from .metrics2d import center, diameter, iou
from .metrics3d import evaluate_record_3d, summarize_3d
from .reporting import write_summary_bundle
from .stratification import stratify_3d


def _load_2d_checkpoint(benchmark_2d_dir: str | Path) -> dict[str, Any]:
    root = Path(benchmark_2d_dir).resolve()
    checkpoint = root / "checkpoint.json"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return load_json(checkpoint)


def _load_prediction(benchmark_2d_dir: str | Path, record_id: str) -> dict[str, Any] | None:
    path = Path(benchmark_2d_dir) / "predictions" / f"{record_key(record_id)}.json"
    if not path.is_file():
        return None
    payload = load_json(path)
    if str(payload.get("record_id")) != str(record_id):
        raise RuntimeError(f"Stale/colliding 2D prediction checkpoint: {path}")
    return payload


def _select_prediction(preds: list[dict[str, Any]], gt_bbox: list[float], selection: str) -> tuple[dict[str, Any] | None, int | None]:
    if not preds:
        return None, None
    if selection == "top1":
        return preds[0], 1
    if selection == "best-iou":
        best_idx = max(range(len(preds)), key=lambda idx: iou(preds[idx]["bbox_xyxy"], gt_bbox))
        return preds[best_idx], best_idx + 1
    raise ValueError("selection must be top1 or best-iou")


def _evaluate_e2e_record(record, pred_payload, *, selection: str, gt_box_mode: str, ball_radius_m: float, localization_mode: str, pitch_margin_m: float, max_height_m: float) -> dict[str, Any]:
    gt_bbox = gt_bbox_for_record(record, gt_box_mode)
    gt_xyz = record.ball_3d
    base: dict[str, Any] = {
        "row_index": record.row_index,
        "image_path": None if pred_payload is None else pred_payload.get("image_path"),
        "action": record.action,
        "replay": record.replay,
        "gt_bbox": gt_bbox,
        "gt_box_mode": gt_box_mode,
        "gt_diameter_px": None if gt_bbox is None else diameter(gt_bbox),
        "gt_xyz": gt_xyz,
        "gt_xyz_soccernet": record.ball_3d_soccernet,
        "pred_bbox": None,
        "pred_xyz": None,
        "detector_required": True,
        "observation_type": "direct",
        "detector_score": None,
        "top1_iou": None,
        "selected_prediction_rank": None,
        "status": "NO_2D_PREDICTION_CHECKPOINT" if pred_payload is None else "NO_BALL_DETECTION",
        "camera_status": "INVALID",
        "evaluation_eligible": pred_payload is not None and not bool((pred_payload or {}).get("missing_image")),
    }
    if pred_payload is None:
        return base
    if bool(pred_payload.get("missing_image")):
        base["status"] = "MISSING_IMAGE"
        return base
    predictions = list(pred_payload.get("predictions") or [])
    selected, rank = _select_prediction(predictions, gt_bbox, selection)
    if selected is None:
        return base
    calib = parse_literal(record.raw.get("calibration"))
    if not isinstance(calib, dict):
        base["status"] = "MISSING_CALIBRATION"
        return base
    camera = camera_from_soccernet_calibration(
        calib,
        frame_index=record.row_index,
        image_width=record.img_w,
        image_height=record.img_h,
    )
    box = [float(x) for x in selected["bbox_xyxy"]]
    c = center(box)
    cand = BallCandidate2D(
        record.row_index,
        str(selected.get("candidate_id") or f"pred_rank_{rank}"),
        box,
        c.astype(float).tolist(),
        float(selected.get("score", 0.0)),
        str(selected.get("source", "cached-2d-detector")),
        float(selected.get("diameter_px", diameter(box))),
    )
    state = estimate_frame(
        camera,
        cand,
        fps=25.0,
        ball_radius_m=ball_radius_m,
        mode=localization_mode,
        pitch_margin_m=pitch_margin_m,
        max_size_prior_height_m=max_height_m,
    )
    gt = np.asarray(gt_xyz, float)
    gt_center = center(gt_bbox)
    gt_proj = camera.project_world(gt)[0]
    pred = state.selected_center_xyz_world_m
    pred_reproj = None
    if pred is not None:
        pred_reproj = float(np.linalg.norm(camera.project_world(np.asarray(pred, float))[0] - gt_center))
    base.update({
        "pred_bbox": box,
        "pred_xyz": pred,
        "detector_score": float(selected.get("score", 0.0)),
        "top1_iou": float(iou(box, gt_bbox)),
        "selected_prediction_rank": int(rank),
        "status": state.localization_status,
        "camera_status": camera.status,
        "camera_to_gt_distance_m": float(np.linalg.norm(gt - camera.camera_center_world_m)),
        "self_reprojection_error_px": pred_reproj,
        "gt_reprojection_error_px": float(np.linalg.norm(gt_proj - gt_center)),
        "geometry_diagnostics": state.diagnostics,
    })
    return base


def _collect(store, records) -> list[dict[str, Any]]:
    return [x for r in records if (x := store.load_result(r.record_id)) is not None]


def finalize_e2e(*, store, records, dataset, benchmark_2d_dir: str | Path, selection: str, gt_box_mode: str, failure_images: int, partial: bool) -> dict[str, Any]:
    rows = _collect(store, records)
    evaluable_rows = [r for r in rows if bool(r.get("evaluation_eligible", True))]
    metrics = summarize_3d(evaluable_rows)
    frame_rows = []
    failures = []
    for row in evaluable_rows:
        d = evaluate_record_3d(row)
        frame_rows.append({
            "record_id": row.get("record_id"),
            "image_path": row.get("image_path"),
            "action": row.get("action"),
            "replay": row.get("replay"),
            "gt_box_mode": row.get("gt_box_mode"),
            "gt_diameter_px": row.get("gt_diameter_px"),
            "gt_z_m": None if row.get("gt_xyz") is None else row["gt_xyz"][2],
            "gt_z_soccernet_m": None if row.get("gt_xyz_soccernet") is None else row["gt_xyz_soccernet"][2],
            "size_prior_validity": (row.get("geometry_diagnostics") or {}).get("size_prior_validity"),
            "camera_to_gt_distance_m": row.get("camera_to_gt_distance_m"),
            "detector_score": row.get("detector_score"),
            "top1_iou": row.get("top1_iou"),
            "selected_prediction_rank": row.get("selected_prediction_rank"),
            "status": row.get("status"),
            **d,
        })
        failure = classify_3d_failure(row)
        if failure is not None:
            failures.append(failure)
    failures.sort(key=lambda x: float(x.get("severity", 0.0)), reverse=True)
    rendered = render_failure_overlays(failures, store.root / "failure_cases", max_images=failure_images)
    stratification = stratify_3d(evaluable_rows)
    report = {
        "schema_version": "stage6-ball-3d-e2e-benchmark-1.2",
        "stage6_version": PACKAGE_VERSION,
        "runtime_provenance": runtime_provenance(),
        "status": "PARTIAL" if partial else "COMPLETE",
        "dataset": dataset.summary(),
        "run_identity": store.run_identity,
        "checkpoint": store.status(),
        "benchmark_2d_dir": str(Path(benchmark_2d_dir).resolve()),
        "selection": selection,
        "gt_box_mode": gt_box_mode,
        "checkpointed_rows": len(rows),
        "rows": len(evaluable_rows),
        "excluded_rows": len(rows) - len(evaluable_rows),
        "metrics": metrics,
        "failure_cases": {"count": len(failures), "rendered_examples": len(rendered)},
        "coordinate_transform": coordinate_transform_metadata(),
        "note": "End-to-end single-image 3D benchmark using cached detector boxes and canonicalized SoccerNet-v3D calibration. Temporal 3D refinement is intentionally not part of v0.3.",
    }
    if selection != "top1":
        report["warning"] = "best-iou selection uses GT bbox to choose among detector candidates and is diagnostic, not end-to-end production behavior."
    if partial:
        atomic_json(store.root / "partial_summary.json", report)
    else:
        write_summary_bundle(
            store.root,
            summary_name="benchmark_3d_e2e_summary.json",
            summary=report,
            frame_rows=frame_rows,
            failures=failures,
            stratification=stratification,
        )
    return report


def run_3d_e2e_benchmark(
    *,
    csv_path: str | Path,
    benchmark_2d_dir: str | Path,
    split: str,
    output_dir: str | Path,
    selection: str = "top1",
    localization_mode: str = "size-prior",
    ball_radius_m: float = 0.11,
    pitch_margin_m: float = 6.0,
    max_height_m: float = 30.0,
    max_rows: int | None = None,
    progress_every: int = 100,
    failure_images: int = 20,
    allow_partial_2d: bool = False,
    gt_box_mode: str = "optimized",
) -> dict[str, Any]:
    dataset = SoccerNetV3DCSV(csv_path)
    gt_box_mode = str(gt_box_mode).strip().lower()
    if gt_box_mode not in {"original", "optimized"}:
        raise ValueError("gt_box_mode must be original or optimized")
    split_records = dataset.split(split)
    parsed_3d = sum(r.ball_3d is not None for r in split_records)
    raw_3d = sum(
        bool(str(r.raw.get("ball_3D", "")).strip())
        and str(r.raw.get("ball_3D", "")).strip().lower() not in {"nan", "none", "null"}
        for r in split_records
    )
    if split_records and parsed_3d == 0:
        raise RuntimeError(
            f"SoccerNet-v3D split {split!r} contains {len(split_records)} rows but zero parsed ball_3D points "
            f"({raw_3d} rows contain a non-empty raw ball_3D field). "
            "A zero-unit 3D benchmark is invalid."
        )
    records = [r for r in split_records if gt_bbox_for_record(r, gt_box_mode) is not None and r.ball_3d is not None]
    if not records:
        raise RuntimeError(f"No 3D-evaluable rows found in split {split!r}.")
    if max_rows is not None:
        records = records[: int(max_rows)]
    two_d_state = _load_2d_checkpoint(benchmark_2d_dir)
    if not allow_partial_2d and str(two_d_state.get("status")) != "COMPLETE":
        raise RuntimeError("2D benchmark is not COMPLETE. Resume run-2d first or pass --allow-partial-2d for a provisional diagnostic.")
    two_d_identity = dict(two_d_state.get("run_identity") or {})
    if two_d_identity.get("csv_sha256") and str(two_d_identity["csv_sha256"]).lower() != sha256_file(csv_path).lower():
        raise RuntimeError("2D prediction cache was produced from a different SNv3D.csv")
    if two_d_identity.get("split") and str(two_d_identity["split"]) != str(split):
        raise RuntimeError("2D prediction cache split differs from requested 3D split")
    missing_prediction_checkpoints = [
        r.record_id for r in records
        if not (Path(benchmark_2d_dir) / "predictions" / f"{record_key(r.record_id)}.json").is_file()
    ]
    if missing_prediction_checkpoints and not allow_partial_2d:
        raise RuntimeError(
            f"2D cache is missing {len(missing_prediction_checkpoints)} required 3D rows. "
            "Resume run-2d or pass --allow-partial-2d for a provisional diagnostic."
        )
    run_identity = {
        "benchmark": "3d-e2e",
        "stage6_version": PACKAGE_VERSION,
        "runtime_provenance": runtime_provenance(),
        "coordinate_transform": coordinate_transform_metadata(),
        "csv": str(Path(csv_path).expanduser().resolve()),
        "csv_sha256": sha256_file(csv_path),
        "split": split,
        "records": len(records),
        "benchmark_2d_dir": str(Path(benchmark_2d_dir).resolve()),
        "benchmark_2d_run_signature": two_d_state.get("run_signature"),
        "selection": selection,
        "gt_box_mode": gt_box_mode,
        "localization_mode": localization_mode,
        "ball_radius_m": float(ball_radius_m),
        "pitch_margin_m": float(pitch_margin_m),
        "max_height_m": float(max_height_m),
        "allow_partial_2d": bool(allow_partial_2d),
    }
    store = RecordCheckpointStore(Path(output_dir), "stage6-ball-record-checkpoint-1.0", run_identity, len(records), result_subdir="rows")
    store.mark_running()
    started = time.perf_counter()
    new_count = 0
    try:
        for record in records:
            if store.load_result(record.record_id) is not None:
                continue
            pred_payload = _load_prediction(benchmark_2d_dir, record.record_id)
            row = _evaluate_e2e_record(
                record,
                pred_payload,
                selection=selection,
                gt_box_mode=gt_box_mode,
                ball_radius_m=ball_radius_m,
                localization_mode=localization_mode,
                pitch_margin_m=pitch_margin_m,
                max_height_m=max_height_m,
            )
            store.commit(record.record_id, row)
            new_count += 1
            completed = int(store.state.get("completed_units", 0))
            if progress_every > 0 and (completed == 1 or completed % progress_every == 0 or completed == len(records)):
                elapsed = time.perf_counter() - started
                eta = (elapsed / max(new_count, 1)) * max(0, len(records) - completed)
                print(f"[BALL 3D E2E {completed}/{len(records)}] ETA={eta/60:.1f} min | CHECKPOINT", flush=True)
    except KeyboardInterrupt:
        store.mark_interrupted()
        finalize_e2e(store=store, records=records, dataset=dataset, benchmark_2d_dir=benchmark_2d_dir, selection=selection, gt_box_mode=gt_box_mode, failure_images=0, partial=True)
        raise
    except Exception as exc:
        store.mark_failed(exc)
        finalize_e2e(store=store, records=records, dataset=dataset, benchmark_2d_dir=benchmark_2d_dir, selection=selection, gt_box_mode=gt_box_mode, failure_images=0, partial=True)
        raise
    store.mark_complete()
    return finalize_e2e(
        store=store,
        records=records,
        dataset=dataset,
        benchmark_2d_dir=benchmark_2d_dir,
        selection=selection,
        gt_box_mode=gt_box_mode,
        failure_images=failure_images,
        partial=False,
    )
