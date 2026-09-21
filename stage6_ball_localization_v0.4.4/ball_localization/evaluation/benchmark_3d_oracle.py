from __future__ import annotations

from pathlib import Path
from ..version import PACKAGE_VERSION, runtime_provenance
from typing import Any
import time

import numpy as np

from ..camera import camera_from_soccernet_calibration
from ..contracts import BallCandidate2D
from ..coordinates import coordinate_transform_metadata
from ..datasets import SoccerNetV3DCSV
from ..datasets.soccernet_v3d import parse_literal
from ..geometry import estimate_frame
from .checkpointing import RecordCheckpointStore
from .common import atomic_json, sha256_file
from .failure_analysis import classify_3d_failure
from .metrics2d import center, diameter
from .metrics3d import evaluate_record_3d, summarize_3d
from .reporting import write_summary_bundle
from .stratification import stratify_3d


def _effective_bbox(record, diameter_source: str) -> list[float]:
    box = list(record.ball_bbox)
    if diameter_source == "optimized" and record.optimized_d is not None:
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        d = float(record.optimized_d)
        return [cx - d / 2.0, cy - d / 2.0, cx + d / 2.0, cy + d / 2.0]
    return box


def _evaluate_oracle_record(record, *, diameter_source: str, ball_radius_m: float, pitch_margin_m: float, max_height_m: float) -> dict[str, Any]:
    calib = parse_literal(record.raw.get("calibration"))
    base = {
        "row_index": record.row_index,
        "action": record.action,
        "replay": record.replay,
        "gt_bbox": record.ball_bbox,
        "gt_diameter_px": None if record.ball_bbox is None else diameter(record.ball_bbox),
        "optimized_d_px": record.optimized_d,
        "gt_xyz": record.ball_3d,
        "gt_xyz_soccernet": record.ball_3d_soccernet,
        "pred_xyz": None,
        "status": "MISSING_CALIBRATION",
        "camera_status": "INVALID",
        "observation_type": "oracle",
        "detector_score": None,
        "top1_iou": 1.0,
    }
    if not isinstance(calib, dict):
        return base
    camera = camera_from_soccernet_calibration(
        calib,
        frame_index=record.row_index,
        image_width=record.img_w,
        image_height=record.img_h,
    )
    box = _effective_bbox(record, diameter_source)
    c = center(box)
    candidate = BallCandidate2D(
        record.row_index,
        "oracle",
        list(map(float, box)),
        c.astype(float).tolist(),
        1.0,
        f"oracle-{diameter_source}",
        float(diameter(box)),
    )
    state = estimate_frame(
        camera,
        candidate,
        fps=25.0,
        ball_radius_m=ball_radius_m,
        mode="size-prior",
        pitch_margin_m=pitch_margin_m,
        max_size_prior_height_m=max_height_m,
    )
    pred = state.selected_center_xyz_world_m
    gt = np.asarray(record.ball_3d, float)
    gt_center = center(record.ball_bbox)
    gt_proj = camera.project_world(gt)[0]
    pred_reproj = None
    if pred is not None:
        pred_reproj = float(np.linalg.norm(camera.project_world(np.asarray(pred, float))[0] - gt_center))
    base.update({
        "effective_bbox": list(map(float, box)),
        "pred_xyz": pred,
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


def finalize_oracle(*, store, records, dataset, diameter_source: str, failure_images: int, partial: bool) -> dict[str, Any]:
    rows = _collect(store, records)
    metrics = summarize_3d(rows)
    frame_rows = []
    failures = []
    for row in rows:
        d = evaluate_record_3d(row)
        frame_rows.append({
            "record_id": row.get("record_id"),
            "action": row.get("action"),
            "replay": row.get("replay"),
            "gt_diameter_px": row.get("gt_diameter_px"),
            "gt_z_m": None if row.get("gt_xyz") is None else row["gt_xyz"][2],
            "gt_z_soccernet_m": None if row.get("gt_xyz_soccernet") is None else row["gt_xyz_soccernet"][2],
            "size_prior_validity": (row.get("geometry_diagnostics") or {}).get("size_prior_validity"),
            "camera_to_gt_distance_m": row.get("camera_to_gt_distance_m"),
            "status": row.get("status"),
            **d,
        })
        failure = classify_3d_failure(row)
        if failure is not None:
            failures.append(failure)
    failures.sort(key=lambda x: float(x.get("severity", 0.0)), reverse=True)
    stratification = stratify_3d(rows)
    report = {
        "schema_version": "stage6-ball-3d-oracle-benchmark-1.2",
        "stage6_version": PACKAGE_VERSION,
        "runtime_provenance": runtime_provenance(),
        "status": "PARTIAL" if partial else "COMPLETE",
        "dataset": dataset.summary(),
        "run_identity": store.run_identity,
        "checkpoint": store.status(),
        "diameter_source": diameter_source,
        "rows": len(rows),
        "metrics": metrics,
        "failure_cases": {"count": len(failures)},
        "coordinate_transform": coordinate_transform_metadata(),
        "note": "Oracle 2D center/bbox geometry diagnostic in canonical Stage-6 XYZ. Detector errors are intentionally excluded. SelfReprojectionErrorPx is a ray-consistency diagnostic, not an accuracy metric.",
    }
    if partial:
        atomic_json(store.root / "partial_summary.json", report)
    else:
        write_summary_bundle(
            store.root,
            summary_name="benchmark_3d_oracle_summary.json",
            summary=report,
            frame_rows=frame_rows,
            failures=failures,
            stratification=stratification,
        )
    return report


def run_3d_oracle_benchmark(
    *,
    csv_path: str | Path,
    split: str,
    output_dir: str | Path,
    diameter_source: str = "optimized",
    ball_radius_m: float = 0.11,
    pitch_margin_m: float = 6.0,
    max_height_m: float = 30.0,
    max_rows: int | None = None,
    progress_every: int = 100,
    failure_images: int = 0,
) -> dict[str, Any]:
    dataset = SoccerNetV3DCSV(csv_path)
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
            "Run `python benchmark_ball.py inspect --csv <SNv3D.csv>` and update the CSV parser/package; "
            "a zero-unit oracle benchmark is invalid and will no longer be marked COMPLETE."
        )
    records = [r for r in split_records if r.ball_bbox is not None and r.ball_3d is not None]
    if not records:
        raise RuntimeError(
            f"No oracle-evaluable rows found in split {split!r}: requires both ball_bbox and ball_3D."
        )
    if max_rows is not None:
        records = records[: int(max_rows)]
    run_identity = {
        "benchmark": "3d-oracle",
        "stage6_version": PACKAGE_VERSION,
        "runtime_provenance": runtime_provenance(),
        "coordinate_transform": coordinate_transform_metadata(),
        "csv": str(Path(csv_path).expanduser().resolve()),
        "csv_sha256": sha256_file(csv_path),
        "split": split,
        "records": len(records),
        "diameter_source": diameter_source,
        "ball_radius_m": float(ball_radius_m),
        "pitch_margin_m": float(pitch_margin_m),
        "max_height_m": float(max_height_m),
    }
    store = RecordCheckpointStore(Path(output_dir), "stage6-ball-record-checkpoint-1.0", run_identity, len(records), result_subdir="rows")
    store.mark_running()
    started = time.perf_counter()
    new_count = 0
    try:
        for record in records:
            if store.load_result(record.record_id) is not None:
                continue
            row = _evaluate_oracle_record(
                record,
                diameter_source=diameter_source,
                ball_radius_m=ball_radius_m,
                pitch_margin_m=pitch_margin_m,
                max_height_m=max_height_m,
            )
            store.commit(record.record_id, row)
            new_count += 1
            completed = int(store.state.get("completed_units", 0))
            if progress_every > 0 and (completed == 1 or completed % progress_every == 0 or completed == len(records)):
                elapsed = time.perf_counter() - started
                eta = (elapsed / max(new_count, 1)) * max(0, len(records) - completed)
                print(f"[BALL 3D ORACLE {completed}/{len(records)}] ETA={eta/60:.1f} min | CHECKPOINT", flush=True)
    except KeyboardInterrupt:
        store.mark_interrupted()
        finalize_oracle(store=store, records=records, dataset=dataset, diameter_source=diameter_source, failure_images=0, partial=True)
        raise
    except Exception as exc:
        store.mark_failed(exc)
        finalize_oracle(store=store, records=records, dataset=dataset, diameter_source=diameter_source, failure_images=0, partial=True)
        raise
    store.mark_complete()
    return finalize_oracle(store=store, records=records, dataset=dataset, diameter_source=diameter_source, failure_images=failure_images, partial=False)
