#!/usr/bin/env python3
"""Create resumable RTMW-L pre-annotations for Phase-8 tasks.

Model output is kept separate from ``keypoints_133``. This tool never turns
predictions into ground truth and never writes the source manifest in place.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stage3_pose2d.crop_qa import analyze_crop
from stage3_pose2d.quality import evaluate_pose
from stage3_pose2d.rtmw_onnx import RTMWOpenCVDNN
from stage3_pose2d.schemas import Stage3Config


CHECKPOINT_SCHEMA = "stage3-phase8-preannotation-checkpoint-1.0"
MODEL_FIELDS = (
    "model_preannotation_133",
    "model_preannotation_qa",
    "model_preannotation_hard_negative_flags",
    "model_preannotation_provenance",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _task_gt_hash(task: Dict[str, Any]) -> str:
    immutable = {key: value for key, value in task.items() if key not in MODEL_FIELDS}
    return _sha256_json(immutable)


def _task_order_hash(tasks: list[Dict[str, Any]]) -> str:
    return _sha256_json([str(task.get("task_id")) for task in tasks])


def _atomic_json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _checkpoint_paths(checkpoint: Path) -> tuple[Path, Path]:
    return checkpoint / "meta.json", checkpoint / "chunks"


def _new_meta(
    manifest: Path,
    model_path: Path,
    config: Stage3Config,
    tasks: list[Dict[str, Any]],
    chunk_size: int,
) -> Dict[str, Any]:
    config_dict = config.to_dict()
    return {
        "schema_version": CHECKPOINT_SCHEMA,
        "created_at_utc": _now(),
        "updated_at_utc": _now(),
        "manifest_path": str(manifest.resolve()),
        "manifest_sha256": _sha256(manifest),
        "model_path": str(model_path.resolve()),
        "model_sha256": _sha256(model_path),
        "config": config_dict,
        "config_sha256": _sha256_json(config_dict),
        "task_count": len(tasks),
        "task_order_sha256": _task_order_hash(tasks),
        "task_ground_truth_sha256": {
            str(task.get("task_id")): _task_gt_hash(task) for task in tasks
        },
        "chunk_size": int(chunk_size),
        "completed_task_ids": [],
        "chunk_files": [],
        "chunk_sha256": {},
    }


def _validate_meta(
    meta: Dict[str, Any],
    manifest: Path,
    model_path: Path,
    config: Stage3Config,
    tasks: list[Dict[str, Any]],
) -> None:
    if meta.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError("unsupported preannotation checkpoint schema")
    expected = {
        "manifest_sha256": _sha256(manifest),
        "model_sha256": _sha256(model_path),
        "config_sha256": _sha256_json(config.to_dict()),
        "task_count": len(tasks),
        "task_order_sha256": _task_order_hash(tasks),
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f"checkpoint provenance mismatch: {key}")
    if meta.get("config") != config.to_dict():
        raise ValueError("checkpoint provenance mismatch: config")
    expected_gt = {str(task.get("task_id")): _task_gt_hash(task) for task in tasks}
    if meta.get("task_ground_truth_sha256") != expected_gt:
        raise ValueError("checkpoint ground-truth fields changed since checkpoint creation")


def _load_checkpoint(
    checkpoint: Path,
    manifest: Path,
    model_path: Path,
    config: Stage3Config,
    tasks: list[Dict[str, Any]],
) -> tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    meta_path, chunks_dir = _checkpoint_paths(checkpoint)
    if not meta_path.is_file() or not chunks_dir.is_dir():
        raise ValueError(f"invalid checkpoint directory: {checkpoint}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    _validate_meta(meta, manifest, model_path, config, tasks)
    updates: Dict[str, Dict[str, Any]] = {}
    for chunk_name in meta.get("chunk_files") or []:
        chunk_path = chunks_dir / str(chunk_name)
        if not chunk_path.is_file():
            raise ValueError(f"checkpoint chunk is missing: {chunk_path}")
        expected_hash = (meta.get("chunk_sha256") or {}).get(str(chunk_name))
        if not expected_hash or _sha256(chunk_path) != expected_hash:
            raise ValueError(f"checkpoint chunk hash mismatch: {chunk_path}")
        chunk = json.loads(chunk_path.read_text(encoding="utf-8"))
        if chunk.get("schema_version") != CHECKPOINT_SCHEMA:
            raise ValueError(f"unsupported checkpoint chunk: {chunk_path}")
        for task_id, update in (chunk.get("updates") or {}).items():
            if task_id in updates:
                raise ValueError(f"duplicate task update in checkpoint: {task_id}")
            updates[str(task_id)] = update
    completed = {str(value) for value in meta.get("completed_task_ids") or []}
    if completed != set(updates):
        raise ValueError("checkpoint completed-task index does not match chunk contents")
    valid_ids = {str(task.get("task_id")) for task in tasks}
    if not set(updates).issubset(valid_ids):
        raise ValueError("checkpoint contains a task not present in the source manifest")
    return meta, updates


def _create_checkpoint(
    checkpoint: Path,
    manifest: Path,
    model_path: Path,
    config: Stage3Config,
    tasks: list[Dict[str, Any]],
    chunk_size: int,
) -> Dict[str, Any]:
    if checkpoint.exists():
        if not checkpoint.is_dir() or any(checkpoint.iterdir()):
            raise FileExistsError(f"checkpoint already exists; pass --resume: {checkpoint}")
    checkpoint.mkdir(parents=True, exist_ok=True)
    _checkpoint_paths(checkpoint)[1].mkdir(parents=True, exist_ok=True)
    meta = _new_meta(manifest, model_path, config, tasks, chunk_size)
    _atomic_json_write(checkpoint / "meta.json", meta)
    return meta


def _write_chunk(checkpoint: Path, meta: Dict[str, Any], updates: Dict[str, Dict[str, Any]]) -> None:
    if not updates:
        return
    chunk_number = len(meta.get("chunk_files") or []) + 1
    chunk_name = f"chunk-{chunk_number:06d}.json"
    _atomic_json_write(
        checkpoint / "chunks" / chunk_name,
        {"schema_version": CHECKPOINT_SCHEMA, "chunk_index": chunk_number, "updates": updates},
    )
    meta.setdefault("chunk_files", []).append(chunk_name)
    meta.setdefault("chunk_sha256", {})[chunk_name] = _sha256(checkpoint / "chunks" / chunk_name)
    meta.setdefault("completed_task_ids", []).extend(str(task_id) for task_id in updates)
    meta["updated_at_utc"] = _now()
    _atomic_json_write(checkpoint / "meta.json", meta)


def _apply_updates(document: Dict[str, Any], updates: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    output_document = copy.deepcopy(document)
    status_counts: Dict[str, int] = {}
    crop_status_counts: Dict[str, int] = {}
    ownership_status_counts: Dict[str, int] = {}
    hard_negative_counts: Dict[str, int] = {}
    processed = 0
    skipped = 0
    for task in output_document.get("tasks") or []:
        update = updates.get(str(task.get("task_id")))
        if not update:
            continue
        if update.get("state") == "SKIPPED":
            skipped += 1
            continue
        for key in MODEL_FIELDS:
            if key in update:
                task[key] = copy.deepcopy(update[key])
        processed += 1
        status = str(update.get("pose_status", "UNKNOWN"))
        crop_status = str(update.get("crop_status", "UNKNOWN"))
        ownership_status = str(update.get("ownership_status", "UNKNOWN"))
        status_counts[status] = status_counts.get(status, 0) + 1
        crop_status_counts[crop_status] = crop_status_counts.get(crop_status, 0) + 1
        ownership_status_counts[ownership_status] = ownership_status_counts.get(ownership_status, 0) + 1
        for flag, enabled in (update.get("hard_negative_flags") or {}).items():
            if enabled:
                hard_negative_counts[flag] = hard_negative_counts.get(flag, 0) + 1
    total = len(output_document.get("tasks") or [])
    output_document["preannotation"] = {
        "status": "MODEL_PREANNOTATION_ONLY",
        "processed_tasks": processed,
        "skipped_tasks": skipped,
        "completed_tasks": processed + skipped,
        "remaining_tasks": max(0, total - processed - skipped),
        "ground_truth_unchanged": True,
        "pose_status_counts": status_counts,
        "crop_status_counts": crop_status_counts,
        "ownership_status_counts": ownership_status_counts,
        "hard_negative_counts": hard_negative_counts,
        "pseudo_labels_are_not_training_ground_truth": True,
    }
    return output_document


def _write_output(output: Path, document: Dict[str, Any], updates: Dict[str, Dict[str, Any]]) -> None:
    _atomic_json_write(output, _apply_updates(document, updates))


def preannotate(
    manifest: Path,
    model_path: Path,
    output: Path,
    max_tasks: int | None = None,
    *,
    checkpoint: Path | None = None,
    resume: bool = False,
    chunk_size: int = 32,
) -> Dict[str, Any]:
    """Pre-annotate tasks and resume completed work from immutable chunks."""
    manifest = manifest.resolve()
    model_path = model_path.resolve()
    output = output.resolve()
    if max_tasks is not None and max_tasks < 0:
        raise ValueError("max_tasks must be non-negative")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    document = json.loads(manifest.read_text(encoding="utf-8"))
    config = Stage3Config()
    tasks = document.get("tasks") or []
    task_ids = [str(task.get("task_id")) for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("manifest contains duplicate task_id values")
    checkpoint = (output.with_suffix(".checkpoint") if checkpoint is None else checkpoint).resolve()
    if resume:
        meta, updates = _load_checkpoint(checkpoint, manifest, model_path, config, tasks)
    else:
        meta = _create_checkpoint(checkpoint, manifest, model_path, config, tasks, chunk_size)
        updates = {}

    tasks_by_image: Dict[str, list[Dict[str, Any]]] = {}
    for task in tasks:
        tasks_by_image.setdefault(str(task.get("image_path")), []).append(task)
    model = RTMWOpenCVDNN(
        model_path,
        input_width=config.rtmw_input_width,
        input_height=config.rtmw_input_height,
        bbox_padding=config.bbox_padding,
        device="cpu",
    )
    pending_chunk: Dict[str, Dict[str, Any]] = {}
    completed_this_run = 0
    for task in tasks:
        task_id = str(task.get("task_id"))
        if task_id in updates:
            continue
        if max_tasks is not None and completed_this_run >= max_tasks:
            break
        image = cv2.imread(str(task["image_path"]))
        if image is None:
            pending_chunk[task_id] = {"state": "SKIPPED", "skip_reason": "image_unreadable"}
        else:
            result = model.infer_one(image, task["bbox_xyxy"])
            neighbors = [
                {"track_id": str(other.get("track_id") or other.get("task_id")), "bbox_xyxy": other["bbox_xyxy"]}
                for other in tasks_by_image.get(str(task.get("image_path")), [])
                if other is not task and other.get("bbox_xyxy")
            ]
            crop_diagnostics = analyze_crop(
                task["bbox_xyxy"],
                (image.shape[1], image.shape[0]),
                neighbors=neighbors,
                pose_xy=result.keypoints_xy,
                pose_scores=result.scores,
                bbox_padding=config.bbox_padding,
                input_size_wh=(config.rtmw_input_width, config.rtmw_input_height),
            )
            qa, _records = evaluate_pose(
                result.keypoints_xy,
                result.scores,
                task["bbox_xyxy"],
                config,
                crop_diagnostics=crop_diagnostics,
            )
            flags = _hard_negative_flags(
                qa,
                crop_diagnostics,
                image_size_wh=(image.shape[1], image.shape[0]),
            )
            pending_chunk[task_id] = {
                "state": "DONE",
                "model_preannotation_133": [
                    {"index": int(i), "x": float(x), "y": float(y), "raw_model_score": float(score)}
                    for i, ((x, y), score) in enumerate(zip(result.keypoints_xy, result.scores))
                ],
                "model_preannotation_qa": qa,
                "model_preannotation_hard_negative_flags": flags,
                "model_preannotation_provenance": {
                    "model": "RTMW-L",
                    "model_path": str(model_path),
                    "model_sha256": meta["model_sha256"],
                    "input_size_wh": [config.rtmw_input_width, config.rtmw_input_height],
                    "bbox_padding": config.bbox_padding,
                    "is_ground_truth": False,
                    "review_required": True,
                },
                "pose_status": str(qa.get("pose_status", "UNKNOWN")),
                "crop_status": str(crop_diagnostics.get("crop_status", "UNKNOWN")),
                "ownership_status": str(crop_diagnostics.get("ownership_status", "UNKNOWN")),
                "hard_negative_flags": flags,
            }
        completed_this_run += 1
        if len(pending_chunk) >= int(meta.get("chunk_size", chunk_size)):
            _write_chunk(checkpoint, meta, pending_chunk)
            updates.update(pending_chunk)
            pending_chunk = {}
            _write_output(output, document, updates)
    if pending_chunk:
        _write_chunk(checkpoint, meta, pending_chunk)
        updates.update(pending_chunk)
    _write_output(output, document, updates)
    result = _apply_updates(document, updates)["preannotation"]
    result["checkpoint"] = str(checkpoint)
    result["resumed"] = bool(resume)
    result["completed_this_run"] = completed_this_run
    return result


def _hard_negative_flags(
    qa: Dict[str, Any],
    crop_diagnostics: Dict[str, Any],
    *,
    image_size_wh: tuple[int, int],
) -> Dict[str, bool]:
    """Mark review slices without promoting model output to a label."""
    crop_status = str(crop_diagnostics.get("crop_status", "UNKNOWN"))
    ownership_status = str(crop_diagnostics.get("ownership_status", "UNKNOWN"))
    bbox = crop_diagnostics.get("source_bbox_xyxy") or [0, 0, 0, 0]
    bbox_height = max(0.0, float(bbox[3]) - float(bbox[1]))
    bbox_area = max(0.0, float(bbox[2]) - float(bbox[0])) * bbox_height
    width, height = image_size_wh
    border_touching = bool(
        float(bbox[0]) <= 0.0 or float(bbox[1]) <= 0.0
        or float(bbox[2]) >= float(width) or float(bbox[3]) >= float(height)
    )
    return {
        "low_pose_evidence": float(qa.get("low_model_evidence_fraction", 0.0)) > 0.25,
        "pose_not_valid": str(qa.get("pose_status", "UNKNOWN")) != "VALID",
        "crop_not_ok": crop_status != "OK",
        "ownership_not_supported": ownership_status not in {"SUPPORTED", "UNKNOWN"},
        "neighbor_overlap": float(crop_diagnostics.get("neighbor_max_crop_iou", 0.0)) >= 0.30,
        "tiny_person": bbox_height < 24.0 or bbox_area < 512.0,
        "image_border": border_touching,
        "feet_incomplete": float(qa.get("feet_completeness", 0.0)) < 0.50,
        "geometry_sanity_failed": not bool(qa.get("geometry_valid", False)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--rtmw-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(json.dumps(preannotate(
        args.manifest,
        args.rtmw_model,
        args.output,
        args.max_tasks,
        checkpoint=args.checkpoint,
        resume=args.resume,
        chunk_size=args.chunk_size,
    ), indent=2))


if __name__ == "__main__":
    main()
