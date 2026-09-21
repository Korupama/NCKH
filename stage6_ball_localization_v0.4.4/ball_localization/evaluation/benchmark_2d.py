from __future__ import annotations

from pathlib import Path
from ..version import PACKAGE_VERSION, runtime_provenance
from typing import Any, Dict
from collections import Counter
import time

from ..datasets import SoccerNetImageResolver, SoccerNetV3DCSV, gt_bbox_for_record, optimized_bbox_from_record
from ..providers import SoccerNetV3DYOLOProvider
from .checkpointing import RecordCheckpointStore
from .common import atomic_json, load_json, sha256_file
from .failure_analysis import classify_2d_failure, render_failure_overlays
from .metrics2d import diameter, evaluate_record_2d, summarize_2d
from .reporting import write_summary_bundle
from .stratification import stratify_2d


def _collect_rows(store: RecordCheckpointStore, records) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        payload = store.load_result(record.record_id)
        if payload is not None:
            rows.append(payload)
    return rows


def finalize_2d(
    *,
    store: RecordCheckpointStore,
    records,
    dataset: SoccerNetV3DCSV,
    candidate_k: int,
    failure_images: int,
    partial: bool,
    gt_box_mode: str = "optimized",
) -> dict[str, Any]:
    all_rows = _collect_rows(store, records)
    evaluable = [r for r in all_rows if not r.get("missing_image")]
    top_k = int((store.run_identity.get("detector") or {}).get("top_k", 10))
    metrics = summarize_2d(
        evaluable,
        iou_threshold=0.5,
        candidate_k=candidate_k,
        candidate_ks=sorted({1, int(candidate_k), max(1, top_k)}),
    )
    frame_rows = []
    failures = []
    for row in evaluable:
        m = evaluate_record_2d(row, candidate_k=candidate_k)
        frame_rows.append({
            "record_id": row.get("record_id"),
            "image_path": row.get("image_path"),
            "action": row.get("action"),
            "replay": row.get("replay"),
            "gt_box_mode": gt_box_mode,
            **m,
        })
        failure = classify_2d_failure(row, candidate_k=candidate_k)
        if failure is not None:
            failures.append(failure)
    failures.sort(key=lambda x: float(x.get("severity", 0.0)), reverse=True)
    rendered = render_failure_overlays(failures, store.root / "failure_cases", max_images=failure_images)
    stratification = stratify_2d(evaluable, candidate_k=candidate_k)
    state = store.status()
    report = {
        "schema_version": "stage6-ball-2d-benchmark-1.2",
        "stage6_version": PACKAGE_VERSION,
        "runtime_provenance": runtime_provenance(),
        "status": "PARTIAL" if partial else "COMPLETE",
        "dataset": dataset.summary(),
        "run_identity": store.run_identity,
        "checkpoint": state,
        "evaluated_images": len(evaluable),
        "checkpointed_rows": len(all_rows),
        "missing_images": sum(bool(r.get("missing_image")) for r in all_rows),
        "image_sources": dict(Counter(str(r.get("image_source_type") or ("missing" if r.get("missing_image") else "unknown")) for r in all_rows)),
        "gt_box_mode": gt_box_mode,
        "metrics": metrics,
        "failure_cases": {"count": len(failures), "rendered_examples": len(rendered)},
        "note": "2D detector benchmark only; Stage-1 camera and 3D localization are not evaluated here. Optimized GT is reconstructed from the original box center plus SoccerNet-v3D optimized_d.",
    }
    name = "partial_summary.json" if partial else "benchmark_2d_summary.json"
    if partial:
        atomic_json(store.root / name, report)
    else:
        write_summary_bundle(
            store.root,
            summary_name=name,
            summary=report,
            frame_rows=frame_rows,
            failures=failures,
            stratification=stratification,
        )
    return report


def run_2d_benchmark(
    *,
    csv_path: str | Path,
    image_root: str | Path,
    weights: str | Path,
    split: str,
    output_dir: str | Path,
    conf_floor: float = 0.05,
    top_k: int = 10,
    candidate_k: int = 5,
    imgsz: int = 1920,
    device: str = "cpu",
    nms_iou: float = 0.5,
    max_images: int | None = None,
    progress_every: int = 10,
    failure_images: int = 20,
    gt_box_mode: str = "optimized",
) -> dict[str, Any]:
    dataset = SoccerNetV3DCSV(csv_path)
    gt_box_mode = str(gt_box_mode).strip().lower()
    if gt_box_mode not in {"original", "optimized"}:
        raise ValueError("gt_box_mode must be original or optimized")
    records = dataset.split(split)
    if max_images is not None:
        records = records[: int(max_images)]
    if gt_box_mode == "optimized":
        missing_optimized = [r.record_id for r in records if r.is_ball and r.ball_bbox is not None and optimized_bbox_from_record(r) is None]
        if missing_optimized:
            raise RuntimeError(
                f"Optimized GT requested but optimized_d is unavailable/invalid for {len(missing_optimized)} ball rows; "
                f"example: {missing_optimized[0]}"
            )
    resolver = SoccerNetImageResolver(image_root)
    provider = SoccerNetV3DYOLOProvider(
        weights,
        conf_floor=conf_floor,
        top_k=top_k,
        imgsz=imgsz,
        device=device,
        iou=nms_iou,
    )
    run_identity = {
        "benchmark": "2d",
        "stage6_version": PACKAGE_VERSION,
        "runtime_provenance": runtime_provenance(),
        "csv": str(Path(csv_path).expanduser().resolve()),
        "csv_sha256": sha256_file(csv_path),
        "image_root": str(Path(image_root).expanduser().resolve()),
        "split": split,
        "records": len(records),
        "detector": provider.info(),
        "candidate_k": int(candidate_k),
        "gt_box_mode": gt_box_mode,
        "gt_box_semantics": (
            "original SoccerNet-v3 ball_bbox" if gt_box_mode == "original"
            else "square centered at original ball_bbox center with side length SoccerNet-v3D optimized_d"
        ),
        "image_access": {
            "loose_files": True,
            "frames_v3_zip": True,
            "huggingface_repo": "SoccerNet/SoccerNet_raw_HQ",
            "huggingface_revision": "frames-v3",
            "archive_layout": "<league>/<season>/<match>/Frames-v3.zip",
            "preference": "loose-then-Frames-v3.zip",
        },
    }
    store = RecordCheckpointStore(
        Path(output_dir),
        "stage6-ball-record-checkpoint-1.0",
        run_identity,
        len(records),
        result_subdir="predictions",
    )
    store.mark_running()
    started = time.perf_counter()
    processed_this_run = 0
    try:
        for idx, record in enumerate(records, start=1):
            if store.load_result(record.record_id) is not None:
                continue
            image, image_source = resolver.read(record)
            if image_source is None:
                payload = {
                    "row_index": record.row_index,
                    "image_path": None,
                    "image_source_type": None,
                    "img_w": record.img_w,
                    "img_h": record.img_h,
                    "action": record.action,
                    "replay": record.replay,
                    "gt_bbox": gt_bbox_for_record(record, gt_box_mode),
                    "gt_diameter_px": None if gt_bbox_for_record(record, gt_box_mode) is None else diameter(gt_bbox_for_record(record, gt_box_mode)),
                    "gt_bbox_original": gt_bbox_for_record(record, "original"),
                    "gt_bbox_optimized": optimized_bbox_from_record(record) if record.is_ball else None,
                    "gt_diameter_original_px": None if record.ball_bbox is None else diameter(record.ball_bbox),
                    "gt_diameter_optimized_px": record.optimized_d if record.is_ball else None,
                    "predictions": [],
                    "missing_image": True,
                }
            else:
                if image is None:
                    raise RuntimeError(f"Cannot decode image: {image_source.reference}")
                preds = provider.detect(image, record.row_index)
                payload = {
                    "row_index": record.row_index,
                    "image_path": image_source.reference,
                    "image_source_type": image_source.kind,
                    "img_w": record.img_w,
                    "img_h": record.img_h,
                    "action": record.action,
                    "replay": record.replay,
                    "gt_bbox": gt_bbox_for_record(record, gt_box_mode),
                    "gt_diameter_px": None if gt_bbox_for_record(record, gt_box_mode) is None else diameter(gt_bbox_for_record(record, gt_box_mode)),
                    "gt_bbox_original": gt_bbox_for_record(record, "original"),
                    "gt_bbox_optimized": optimized_bbox_from_record(record) if record.is_ball else None,
                    "gt_diameter_original_px": None if record.ball_bbox is None else diameter(record.ball_bbox),
                    "gt_diameter_optimized_px": record.optimized_d if record.is_ball else None,
                    "predictions": [
                        {
                            "bbox_xyxy": p.bbox_xyxy,
                            "score": p.detector_score,
                            "diameter_px": p.diameter_px,
                            "candidate_id": p.candidate_id,
                            "source": p.source,
                        }
                        for p in preds
                    ],
                    "missing_image": False,
                }
            store.commit(record.record_id, payload)
            processed_this_run += 1
            completed = int(store.state.get("completed_units", 0))
            if progress_every > 0 and (completed == 1 or completed % progress_every == 0 or completed == len(records)):
                elapsed = time.perf_counter() - started
                rate = elapsed / max(processed_this_run, 1)
                remaining_new = max(0, len(records) - completed)
                eta = rate * remaining_new
                print(
                    f"[BALL 2D {completed}/{len(records)} {100.0*completed/max(1,len(records)):.1f}%] "
                    f"new={processed_this_run} ETA={eta/60:.1f} min | CHECKPOINT",
                    flush=True,
                )
    except KeyboardInterrupt:
        store.mark_interrupted()
        finalize_2d(store=store, records=records, dataset=dataset, candidate_k=candidate_k, failure_images=0, partial=True, gt_box_mode=gt_box_mode)
        raise
    except Exception as exc:
        store.mark_failed(exc)
        finalize_2d(store=store, records=records, dataset=dataset, candidate_k=candidate_k, failure_images=0, partial=True, gt_box_mode=gt_box_mode)
        raise
    store.mark_complete()
    return finalize_2d(
        store=store,
        records=records,
        dataset=dataset,
        candidate_k=candidate_k,
        failure_images=failure_images,
        partial=False,
        gt_box_mode=gt_box_mode,
    )
