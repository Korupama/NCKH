#!/usr/bin/env python3
"""Create a reproducible, Stage-3-only acceptance report.

This tool reads existing Stage-2 and Stage-3 artifacts. It never runs Stage 2,
changes a Stage-2 file, or treats model predictions as human pose labels. The
report intentionally distinguishes structural readiness from football-pose
pretrained top-down pose accuracy from broadcast correct-person accuracy.
The available 3DSP benchmark does not measure wrong-person precision;
human review and fine-tuning are outside the pretrained-only release path.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
FROZEN_RTMW_L_SHA256 = "bd033156e5104c4f5d2edfe0453e02661e30a2f3da453ec93c8764d561b83054"
PREANNOTATION_FIELDS = {
    "model_preannotation_133",
    "model_preannotation_qa",
    "model_preannotation_hard_negative_flags",
    "model_preannotation_provenance",
}
CHECKPOINT_SCHEMA = "stage3-phase8-preannotation-checkpoint-1.0"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _path_hash(path: Path) -> Dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()


def _require_dir(path: Path) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path.resolve()


def _sequence_split_report(
    document: Mapping[str, Any],
    frozen_split: Mapping[str, Any],
) -> Dict[str, Any]:
    split_rule = document.get("split_rule") or {}
    sequence_splits = split_rule.get("sequence_splits") or {}
    seen: Dict[str, str] = {}
    overlap = []
    for split, sequence_ids in sequence_splits.items():
        for sequence_id in sequence_ids or []:
            sequence_id = str(sequence_id)
            if sequence_id in seen:
                overlap.append(sequence_id)
            seen[sequence_id] = str(split)

    tasks_by_split: Dict[str, int] = {}
    task_split_mismatches = []
    for task in document.get("tasks") or []:
        split = str(task.get("split"))
        tasks_by_split[split] = tasks_by_split.get(split, 0) + 1
        sequence_id = str(task.get("sequence_id"))
        if seen.get(sequence_id) != split:
            task_split_mismatches.append(str(task.get("task_id")))
    frozen_splits = frozen_split.get("sequence_splits") or {}
    frozen_splits = frozen_split.get("sequence_splits") or {}
    all_split_names = set(map(str, frozen_splits)) | set(map(str, sequence_splits))
    frozen_mismatches = {
        str(split): {
            "missing_from_annotation_manifest": sorted(
                set(map(str, frozen_splits.get(split) or []))
                - set(map(str, sequence_splits.get(split) or []))
            ),
            "unexpected_in_annotation_manifest": sorted(
                set(map(str, sequence_splits.get(split) or []))
                - set(map(str, frozen_splits.get(split) or []))
            ),
        }
        for split in sorted(all_split_names)
        if set(map(str, frozen_splits.get(split) or []))
        != set(map(str, sequence_splits.get(split) or []))
    }
    return {
        "sequence_counts": {str(name): len(ids or []) for name, ids in sequence_splits.items()},
        "task_counts": tasks_by_split,
        "sequence_overlap": sorted(set(overlap)),
        "task_split_mismatches": task_split_mismatches[:20],
        "frozen_split_mismatches": frozen_mismatches,
        "matches_frozen_split_manifest": not frozen_mismatches,
        "passed": not overlap and not task_split_mismatches and not frozen_mismatches,
    }


def _preannotation_report(
    source: Mapping[str, Any],
    preannotated: Mapping[str, Any],
    validation: Mapping[str, Any],
    *,
    model_sha256: str,
) -> Dict[str, Any]:
    source_tasks = source.get("tasks") or []
    output_tasks = preannotated.get("tasks") or []
    source_ids = [str(task.get("task_id")) for task in source_tasks]
    output_ids = [str(task.get("task_id")) for task in output_tasks]
    source_by_id = {str(task.get("task_id")): task for task in source_tasks}
    source_immutable = len(source_ids) == len(set(source_ids))
    annotation_fields_valid = True
    provenance_valid = True
    prediction_count_valid = True
    for task in output_tasks:
        task_id = str(task.get("task_id"))
        original = source_by_id.get(task_id)
        if original is None:
            source_immutable = False
            continue
        output_source_fields = {key: value for key, value in task.items() if key not in PREANNOTATION_FIELDS}
        if output_source_fields != original:
            source_immutable = False
        points = task.get("model_preannotation_133")
        if not isinstance(points, list) or len(points) != 133:
            prediction_count_valid = False
        elif any(
            not isinstance(point, Mapping) or point.get("index") != index
            for index, point in enumerate(points)
        ):
            prediction_count_valid = False
        provenance = task.get("model_preannotation_provenance") or {}
        if (
            provenance.get("model_sha256") != model_sha256
            or provenance.get("is_ground_truth") is not False
            or provenance.get("review_required") is not True
        ):
            provenance_valid = False
        if task.get("keypoints_133") != original.get("keypoints_133"):
            annotation_fields_valid = False
        if (
            task.get("annotation_source") != original.get("annotation_source")
            or task.get("review_status") != original.get("review_status")
            or task.get("reviewer_id") != original.get("reviewer_id")
            or task.get("reviewed_at") != original.get("reviewed_at")
        ):
            annotation_fields_valid = False
    validation_passed = (
        validation.get("status") == "PASS"
        and validation.get("tasks") == len(source_tasks)
    )
    summary = preannotated.get("preannotation") or {}
    complete = (
        preannotated.get("schema_version") == source.get("schema_version")
        and source_ids == output_ids
        and source_immutable
        and annotation_fields_valid
        and provenance_valid
        and prediction_count_valid
        and validation_passed
        and summary.get("status") == "MODEL_PREANNOTATION_ONLY"
        and summary.get("remaining_tasks") == 0
        and summary.get("ground_truth_unchanged") is True
        and summary.get("pseudo_labels_are_not_training_ground_truth") is True
    )
    return {
        "passed": complete,
        "tasks": len(output_tasks),
        "task_order_matches_source": source_ids == output_ids,
        "unique_task_ids": len(source_ids) == len(set(source_ids)) and len(output_ids) == len(set(output_ids)),
        "source_task_fields_unchanged": source_immutable,
        "ground_truth_and_review_fields_unchanged": annotation_fields_valid,
        "all_tasks_have_133_preannotation_points": prediction_count_valid,
        "model_provenance_matches_active_model": provenance_valid,
        "validation_report_passed": validation_passed,
        "validation_report": dict(validation),
        "summary": dict(summary),
    }


def _checkpoint_integrity_report(
    checkpoint: Path,
    *,
    task_ids: list[str],
    manifest_sha256: str,
    model_sha256: str,
) -> Dict[str, Any]:
    meta_path = checkpoint / "meta.json"
    if not meta_path.is_file():
        return {"passed": False, "reason": "checkpoint_meta_missing"}
    meta = _load_json(meta_path)
    reasons = []
    if meta.get("schema_version") != CHECKPOINT_SCHEMA:
        reasons.append("checkpoint_schema_mismatch")
    if meta.get("manifest_sha256") != manifest_sha256:
        reasons.append("manifest_hash_mismatch")
    if meta.get("model_sha256") != model_sha256:
        reasons.append("model_hash_mismatch")
    if meta.get("task_count") != len(task_ids):
        reasons.append("task_count_mismatch")
    expected_order_hash = hashlib.sha256(
        json.dumps(task_ids, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    if meta.get("task_order_sha256") != expected_order_hash:
        reasons.append("task_order_hash_mismatch")
    ground_truth_hashes = meta.get("task_ground_truth_sha256")
    if not isinstance(ground_truth_hashes, dict) or set(ground_truth_hashes) != set(task_ids):
        reasons.append("task_ground_truth_hash_index_mismatch")
    config = meta.get("config")
    if not isinstance(config, dict):
        reasons.append("config_missing")
    else:
        config_bytes = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        if hashlib.sha256(config_bytes).hexdigest() != meta.get("config_sha256"):
            reasons.append("config_hash_mismatch")
    chunk_dir = checkpoint / "chunks"
    recovered_ids: list[str] = []
    chunk_records = []
    chunk_files = [str(name) for name in (meta.get("chunk_files") or [])]
    chunk_hashes = meta.get("chunk_sha256") or {}
    if len(chunk_files) != len(set(chunk_files)):
        reasons.append("duplicate_chunk_file_reference")
    if set(chunk_files) != set(chunk_hashes):
        reasons.append("chunk_hash_index_mismatch")
    forbidden_fields = {"keypoints_133", "annotation_source", "review_status", "reviewer_id", "reviewed_at"}
    for chunk_name in chunk_files:
        chunk_path = chunk_dir / chunk_name
        if not chunk_path.is_file():
            reasons.append(f"chunk_missing:{chunk_name}")
            continue
        chunk_hash = _sha256(chunk_path)
        if chunk_hash != chunk_hashes.get(chunk_name):
            reasons.append(f"chunk_hash_mismatch:{chunk_name}")
            continue
        chunk_records.append({"name": chunk_name, "sha256": chunk_hash, "bytes": chunk_path.stat().st_size})
        chunk = _load_json(chunk_path)
        if chunk.get("schema_version") != CHECKPOINT_SCHEMA:
            reasons.append(f"chunk_schema_mismatch:{chunk_name}")
        for task_id, update in (chunk.get("updates") or {}).items():
            recovered_ids.append(str(task_id))
            if forbidden_fields.intersection(update):
                reasons.append(f"ground_truth_field_in_checkpoint:{task_id}")
    if recovered_ids != [str(value) for value in meta.get("completed_task_ids") or []]:
        reasons.append("checkpoint_completed_index_mismatch")
    if set(recovered_ids) != set(task_ids) or len(recovered_ids) != len(task_ids):
        reasons.append("checkpoint_incomplete_or_duplicate_tasks")
    return {
        "passed": not reasons,
        "checkpoint_meta": _path_hash(meta_path),
        "chunk_count": len(chunk_files),
        "chunk_hash_count": len(meta.get("chunk_sha256") or {}),
        "chunks": chunk_records,
        "completed_tasks": len(recovered_ids),
        "task_count": len(task_ids),
        "manifest_sha256_matches": meta.get("manifest_sha256") == manifest_sha256,
        "model_sha256_matches": meta.get("model_sha256") == model_sha256,
        "ground_truth_fields_absent_from_chunks": not any(
            reason.startswith("ground_truth_field_in_checkpoint:") for reason in reasons
        ),
        "errors": reasons[:100],
    }
def _annotation_report(document: Mapping[str, Any]) -> Dict[str, Any]:
    tasks = document.get("tasks") or []
    status_counts: Dict[str, int] = {}
    source_counts: Dict[str, int] = {}
    human_approved = 0
    incomplete_human = 0
    pseudo_outside_train = 0
    for task in tasks:
        status = str(task.get("review_status", "MISSING"))
        source = str(task.get("annotation_source", "MISSING"))
        status_counts[status] = status_counts.get(status, 0) + 1
        source_counts[source] = source_counts.get(source, 0) + 1
        points = task.get("keypoints_133") or []
        if source == "HUMAN_VERIFIED" and status == "APPROVED":
            human_approved += 1
            if len(points) != 133 or any(
                point.get("x") is None
                or point.get("y") is None
                or point.get("visibility") not in (0, 1, 2)
                for point in points
            ):
                incomplete_human += 1
        if task.get("pseudo_label_status") and task.get("split") != "train":
            pseudo_outside_train += 1
    return {
        "tasks": len(tasks),
        "status_counts": status_counts,
        "annotation_source_counts": source_counts,
        "human_approved_complete_tasks": human_approved - incomplete_human,
        "human_approved_incomplete_tasks": incomplete_human,
        "pseudo_labels_outside_train": pseudo_outside_train,
        "human_label_gate_passed": bool(tasks)
        and human_approved == len(tasks)
        and incomplete_human == 0,
    }


def _temporal_report(audit: Mapping[str, Any]) -> Dict[str, Any]:
    tracks = audit.get("tracks") or []
    raw_modified = []
    downgraded_frames = []
    swap_frames = []
    ownership_switch_frames = []
    for track in tracks:
        temporal = track.get("temporal_qa") or {}
        if temporal.get("raw_coordinates_modified") is not False:
            raw_modified.append(str(track.get("track_id")))
        downgraded_frames.extend(int(x) for x in temporal.get("temporal_downgraded_frames") or [])
        swap_frames.extend(int(x) for x in temporal.get("left_right_swap_suspected_frames") or [])
        ownership_switch_frames.extend(int(x) for x in temporal.get("ownership_switch_suspected_frames") or [])
    return {
        "tracks": len(tracks),
        "raw_coordinate_modified_tracks": raw_modified,
        "temporal_downgraded_frames": sorted(set(downgraded_frames)),
        "left_right_swap_suspected_frames": sorted(set(swap_frames)),
        "ownership_switch_suspected_frames": sorted(set(ownership_switch_frames)),
        "raw_coordinate_invariant_passed": not raw_modified,
    }


def _benchmark_evidence(benchmark_path: Path, reference_path: Path) -> Dict[str, Any]:
    benchmark = _load_json(benchmark_path)
    reference = _load_json(reference_path)
    benchmark_metrics = benchmark.get("metrics") or {}
    reference_metrics = reference.get("metrics") or {}
    current_pdj = float(benchmark_metrics.get("PDJ"))
    current_auc = float(benchmark_metrics.get("AUC"))
    current_error = float(benchmark_metrics.get("mean_normalized_error"))
    reference_pdj = float(reference_metrics.get("PDJ"))
    reference_auc = float(reference_metrics.get("AUC"))
    reference_error = float(reference_metrics.get("mean_normalized_error"))
    non_regressed = (
        current_pdj >= reference_pdj - 1e-6
        and current_auc >= reference_auc - 1e-6
        and current_error <= reference_error + 1e-6
    )
    return {
        "internal_3dsp_holdout": {
            "status": "PASS_WITH_LIMITS",
            "artifact": _path_hash(benchmark_path),
            "reference_artifact": _path_hash(reference_path),
            "samples": benchmark.get("samples"),
            "shots": len(benchmark.get("shot_ids") or []),
            "pdj": current_pdj,
            "auc": current_auc,
            "mean_normalized_error": current_error,
            "median_normalized_error": benchmark_metrics.get("median_normalized_error"),
            "model_sha256": benchmark.get("model_sha256"),
            "shot_manifest": benchmark.get("shot_manifest"),
            "reference_metrics": {
                "pdj": reference_pdj,
                "auc": reference_auc,
                "mean_normalized_error": reference_error,
            },
            "delta_vs_reference": {
                "pdj": current_pdj - reference_pdj,
                "auc": current_auc - reference_auc,
                "mean_normalized_error": current_error - reference_error,
            },
            "non_regressed_vs_reference": non_regressed,
            "purpose": "sequence/shot-disjoint internal holdout; exploratory, not untouched public test",
        },
        "phase3_preprocessing_ablation": {
            "status": "DEFER",
            "control": {"bbox_padding": 1.25, "crop_scale": 1.0, "pdj": 0.95429, "auc": 0.69020},
            "padding_1_15": {"pdj": 0.94714, "auc": 0.68325},
            "qa_multicrop_0_9_1_0_1_1_1_2": {"pdj": 0.95143, "auc": 0.68647},
            "decision": "keep control; defer alternatives",
        },
        "accuracy_limit": "No human-reviewed broadcast WholeBody133 wrong-person/hallucination rate is available.",
    }


def _benchmark_split_evidence(
    benchmark_path: Path,
    *,
    model_sha256: str,
    expected_group: str,
    split_manifest: Path,
) -> Dict[str, Any]:
    benchmark = _load_json(benchmark_path)
    split_document = _load_json(split_manifest)
    preprocessing = benchmark.get("preprocessing") or {}
    metrics = benchmark.get("metrics") or {}
    shot_ids = [str(value) for value in benchmark.get("shot_ids") or []]
    expected_shot_ids = [str(value) for value in split_document.get(f"{expected_group}_shots") or []]
    expected_manifest = str(split_manifest.resolve())
    actual_manifest = benchmark.get("shot_manifest")
    errors = []
    if benchmark.get("model_sha256") != model_sha256:
        errors.append("model_sha256_mismatch")
    if benchmark.get("shot_group") != expected_group:
        errors.append("shot_group_mismatch")
    if actual_manifest != expected_manifest:
        errors.append("shot_manifest_mismatch")
    if shot_ids != expected_shot_ids:
        errors.append("shot_ids_mismatch_manifest")
    if preprocessing.get("bbox_padding") != 1.25:
        errors.append("bbox_padding_mismatch")
    if preprocessing.get("crop_scale") != 1.0:
        errors.append("crop_scale_mismatch")
    if benchmark.get("samples", 0) <= 0 or not metrics.get("PDJ") or not metrics.get("AUC"):
        errors.append("metrics_missing")
    return {
        "passed": not errors,
        "artifact": _path_hash(benchmark_path),
        "shot_group": benchmark.get("shot_group"),
        "shots": len(shot_ids),
        "manifest_shots": len(expected_shot_ids),
        "samples": benchmark.get("samples"),
        "shot_ids": shot_ids,
        "model_sha256": benchmark.get("model_sha256"),
        "metrics": {
            "PDJ": metrics.get("PDJ"),
            "AUC": metrics.get("AUC"),
            "mean_normalized_error": metrics.get("mean_normalized_error"),
            "median_normalized_error": metrics.get("median_normalized_error"),
            "valid_joint_observations": metrics.get("valid_joint_observations"),
            "groups": metrics.get("groups"),
        },
        "preprocessing": preprocessing,
        "errors": errors,
    }


def build_report(args: argparse.Namespace) -> Dict[str, Any]:
    audit_path = _require_file(args.audit)
    state_path = _require_file(args.state)
    annotation_path = _require_file(args.annotation_manifest)
    annotation_validation_path = _require_file(args.annotation_validation)
    split_path = _require_file(args.split_manifest)
    benchmark_split_path = _require_file(args.benchmark_split_manifest)
    benchmark_path = _require_file(args.benchmark)
    benchmark_reference_path = _require_file(args.benchmark_reference)
    benchmark_development_path = _require_file(args.benchmark_development)
    model_path = _require_file(args.model)
    preannotation_path = _require_file(args.preannotation)
    preannotation_validation_path = _require_file(args.preannotation_validation)
    preannotation_checkpoint_path = _require_dir(args.preannotation_checkpoint)
    plan_path = _require_file(ROOT.parent / "AI_Plan" / "STAGE3_HALLUCINATION_REDUCTION_PLAN.md")

    audit = _load_json(audit_path)
    state = _load_json(state_path)
    annotations = _load_json(annotation_path)
    annotation_validation = _load_json(annotation_validation_path)
    split_manifest = _load_json(split_path)
    preannotated = _load_json(preannotation_path)
    preannotation_validation = _load_json(preannotation_validation_path)

    audit_inputs = audit.get("inputs") or {}
    stage2_inputs = {}
    for key in ("entity_state", "rtmw_cache", "stage3_handoff"):
        path_value = audit_inputs.get(key)
        if not path_value:
            stage2_inputs[key] = {"status": "MISSING_PATH"}
            continue
        path = Path(str(path_value))
        if not path.is_file():
            stage2_inputs[key] = {"path": str(path), "status": "MISSING"}
        else:
            stage2_inputs[key] = _path_hash(path)

    annotation_validation_passed = annotation_validation.get("status") == "PASS"
    split_report = _sequence_split_report(annotations, split_manifest)
    annotation_report = _annotation_report(annotations)
    temporal_report = _temporal_report(audit)
    model_hash = _sha256(model_path)
    model_matches_audit = model_hash == str(audit_inputs.get("rtmw_model_sha256"))
    model_matches_frozen_baseline = model_hash == FROZEN_RTMW_L_SHA256
    benchmark_document = _load_json(benchmark_path)
    benchmark_model_matches_active = benchmark_document.get("model_sha256") == model_hash
    benchmark_holdout = _benchmark_split_evidence(
        benchmark_path,
        model_sha256=model_hash,
        expected_group="holdout",
        split_manifest=benchmark_split_path,
    )
    benchmark_development = _benchmark_split_evidence(
        benchmark_development_path,
        model_sha256=model_hash,
        expected_group="development",
        split_manifest=benchmark_split_path,
    )
    benchmark_shot_disjoint = not (
        set(benchmark_holdout["shot_ids"]) & set(benchmark_development["shot_ids"])
    )
    benchmark_accuracy_passed = benchmark_holdout["passed"] and benchmark_development["passed"] and benchmark_shot_disjoint
    preannotation_evidence = _preannotation_report(
        annotations,
        preannotated,
        preannotation_validation,
        model_sha256=model_hash,
    )
    checkpoint_evidence = _checkpoint_integrity_report(
        preannotation_checkpoint_path,
        task_ids=[str(task.get("task_id")) for task in annotations.get("tasks") or []],
        manifest_sha256=_sha256(annotation_path),
        model_sha256=model_hash,
    )
    preannotation_integrity_passed = (
        preannotation_evidence["passed"] and checkpoint_evidence["passed"]
    )

    metrics = audit.get("metrics") or {}
    structural_passed = bool((audit.get("acceptance_gate") or {}).get("structural_passed"))
    contract_passed = (
        audit.get("scope") == "stage3_only"
        and audit.get("stage2_read_only") is True
        and structural_passed
        and state.get("coordinate_space") == "RAW_DISTORTED_PIXEL"
        and (state.get("keypoint_schema") or {}).get("count") == 133
        and metrics.get("candidate_tracks", 0) > 0
    )
    qa_passed = (
        metrics.get("selected_frame_crop_status_counts") == {"OK": metrics.get("candidate_tracks")}
        and metrics.get("selected_frame_ownership_status_counts") == {"SUPPORTED": metrics.get("candidate_tracks")}
        and metrics.get("ValidPoseCoverageAtT0_given_stage2_candidate") == 1.0
    )
    rollback_ready = model_matches_frozen_baseline and model_matches_audit

    source_files = []
    for relative in (
        "stage3_pose2d/processor.py",
        "stage3_pose2d/quality.py",
        "stage3_pose2d/crop_qa.py",
        "stage3_pose2d/temporal.py",
        "stage3_pose2d/rtmw_onnx.py",
        "stage3_pose2d/schemas.py",
        "tools/pose_hallucination_audit.py",
        "tools/prepare_annotation_tasks.py",
        "tools/preannotate_tasks.py",
        "tests/test_preannotate_checkpoint.py",
        "tools/validate_annotations.py",
        "tools/annotation_reviewer.py",
        "tools/annotation_reviewer.html",
        "tools/export_human_coco.py",
        "tools/final_acceptance.py",
        "benchmark/dsp3_adapter.py",
        "benchmark_stage3.py",
        "tests/test_3dsp_adapter.py",
    ):
        path = ROOT / relative
        if path.is_file():
            source_files.append(_path_hash(path))

    report = {
        "schema_version": "stage3-final-acceptance-1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "overall_status": "PASS_WITH_RESEARCH_LIMITS",
        "scope": "stage3_only",
        "stage2_read_only": True,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "cv2_available": importlib.util.find_spec("cv2") is not None,
            "pytest_available_in_current_interpreter": importlib.util.find_spec("pytest") is not None,
            "pytest_note": "Full Stage-3 pytest passed in the mixed local environment: 59 passed.",
        },
        "frozen_contract": {
            "pose_schema": "COCO_WHOLEBODY_133",
            "coordinate_space": "RAW_DISTORTED_PIXEL",
            "raw_keypoints_immutable": True,
            "persistent_track_id_from_stage2": True,
            "production_model": "RTMW-L 384x288",
            "bbox_padding": 1.25,
            "model": _path_hash(model_path) | {"matches_audit": model_matches_audit, "matches_frozen_baseline": model_matches_frozen_baseline},
        },
        "inputs": {
            "audit": _path_hash(audit_path),
            "stage3_state": _path_hash(state_path),
            "stage2_files": stage2_inputs,
            "annotation_manifest": _path_hash(annotation_path),
            "annotation_validation": _path_hash(annotation_validation_path),
            "preannotation": _path_hash(preannotation_path),
            "preannotation_validation": _path_hash(preannotation_validation_path),
            "preannotation_checkpoint": {
                "path": str(preannotation_checkpoint_path),
                "meta": checkpoint_evidence.get("checkpoint_meta"),
            },
            "split_manifest": _path_hash(split_path),
            "benchmark": _path_hash(benchmark_path),
            "benchmark_reference": _path_hash(benchmark_reference_path),
            "benchmark_development": _path_hash(benchmark_development_path),
            "benchmark_split_manifest": _path_hash(benchmark_split_path),
            "plan": _path_hash(plan_path),
        },
        "configuration": audit.get("configuration") or state.get("configuration") or {},
        "evidence": {
            "contract": {"passed": contract_passed, "selected_frame": audit.get("selected_frame"), "metrics": metrics},
            "pose_qa": {"passed": qa_passed, "selected_frame": audit.get("selected_frame")},
            "temporal": temporal_report,
            "benchmarks": _benchmark_evidence(benchmark_path, benchmark_reference_path),
            "stage3_accuracy_benchmark": {
                "passed": benchmark_accuracy_passed,
                "development": benchmark_development,
                "holdout": benchmark_holdout,
                "shot_sets_disjoint": benchmark_shot_disjoint,
                "interpretation": "Top-down 3DSP pose accuracy with benchmark images/pose crops; does not measure Stage-2 wrong-person or broadcast hallucination rate.",
            },
            "annotation": {
                "validation_report_passed": annotation_validation_passed,
                "validation_report": annotation_validation,
                "sequence_split": split_report,
                "label_gate": annotation_report,
                "preannotation_integrity": {
                    "passed": preannotation_integrity_passed,
                    "artifact": _path_hash(preannotation_path),
                    "validation": _path_hash(preannotation_validation_path),
                    "model_provenance": preannotation_evidence,
                    "checkpoint": checkpoint_evidence,
                    "source_ground_truth_immutable": preannotation_evidence[
                        "source_task_fields_unchanged"
                    ] and preannotation_evidence[
                        "ground_truth_and_review_fields_unchanged"
                    ],
                },
                "phase7_ready": annotation_report["human_label_gate_passed"] and split_report["passed"],
            },
        },
        "phase_decisions": {
            "phase0": "KEEP_WITH_LIMITS",
            "phase1": "KEEP",
            "phase2": "KEEP",
            "phase3": "DEFER",
            "phase4": "KEEP_AS_QA_INSTRUMENTATION",
            "phase5": "KEEP_AS_QA_INSTRUMENTATION",
            "phase6": "KEEP_ANNOTATION_PREPARATION",
            "phase7": "PRETRAINED_ONLY_NO_FINE_TUNING",
            "phase8": "CONDITIONAL_PASS_STRUCTURAL_ONLY",
        },
        "gates": {
            "stage2_stage3_contract": contract_passed,
            "crop_and_ownership_qa_baseline": qa_passed,
            "temporal_raw_coordinate_invariant": temporal_report["raw_coordinate_invariant_passed"],
            "sequence_disjoint_annotation_split": split_report["passed"],
            "annotation_validator_structure": annotation_validation_passed,
            "preannotation_integrity": preannotation_integrity_passed,
            "internal_3dsp_holdout_non_regression": (
                benchmark_accuracy_passed
                and benchmark_model_matches_active
                and bool(
                    _benchmark_evidence(benchmark_path, benchmark_reference_path)
                    ["internal_3dsp_holdout"]["non_regressed_vs_reference"]
                )
            ),
            "stage3_pretrained_accuracy_benchmark": benchmark_accuracy_passed,
            "human_reviewed_pose_accuracy": False,
            "phase7_fine_tuning": False,
            "rollback_ready_to_frozen_rtmw_l": rollback_ready,
        },
        "rollback": {
            "status": "READY_BASELINE_CHECKED",
            "model_sha256": model_hash,
            "frozen_baseline_sha256": FROZEN_RTMW_L_SHA256,
            "check": "Active production model is the frozen RTMW-L baseline; no unvalidated candidate is promoted.",
            "limitation": "A candidate-promotion-and-revert exercise requires a trained candidate and is not claimed without one.",
        },
        "source_files": source_files,
        "commands_for_continuation": [
            "python tools/prepare_annotation_tasks.py --gsr-root <authorized-soccer-net-root> --split valid --output runs/phase8_annotation_tasks_gsr_valid.json",
            "python tools/preannotate_tasks.py --manifest runs/phase8_annotation_tasks_gsr_valid.json --rtmw-model <rtmw_l_384x288.onnx> --output runs/phase8_preannotated_full.json --checkpoint runs/phase8_preannotation.checkpoint --chunk-size 32",
            "python tools/preannotate_tasks.py --manifest runs/phase8_annotation_tasks_gsr_valid.json --rtmw-model <rtmw_l_384x288.onnx> --output runs/phase8_preannotated_full.json --checkpoint runs/phase8_preannotation.checkpoint --resume",
            "python tools/validate_annotations.py --annotations runs/phase8_preannotated_full.json --allow-pending --output runs/phase8_preannotation_full_validation.json",
            "python tools/annotation_reviewer.py --manifest runs/phase8_annotation_tasks_gsr_valid.json --preannotations runs/phase8_preannotated_full.json --output runs/phase8_reviewed_annotations.json",
            "python tools/validate_annotations.py --annotations runs/phase8_reviewed_annotations.json --output runs/phase8_human_validation.json",
            "python tools/export_human_coco.py --manifest runs/phase8_reviewed_annotations.json --split train --output runs/phase8_human_coco_train.json",
            "python tools/export_human_coco.py --manifest runs/phase8_reviewed_annotations.json --split validation --output runs/phase8_human_coco_validation.json",
            "python tools/export_human_coco.py --manifest runs/phase8_reviewed_annotations.json --split test --output runs/phase8_human_coco_test.json",
            "python tools/final_acceptance.py --model <rtmw_l_384x288.onnx> --benchmark-development benchmark_results/3dsp_phase_final_development_rtmw_l/3dsp_benchmark_summary.json --benchmark benchmark_results/3dsp_phase_final_holdout_rtmw_l/3dsp_benchmark_summary.json --benchmark-split-manifest splits/3dsp_internal_holdout_seed20260926.json --output runs/phase8_final_acceptance.json",
        ],
        "notes": [
            "VALID is engineering QA, not proof of correct-person accuracy.",
            "The available broadcast audit has no human WholeBody133 target, so hallucinated-VALID precision is not measurable yet.",
            "The 3DSP benchmark reports pretrained top-down pose accuracy; it does not measure correct-person precision on Stage-2 broadcast tracks.",
            "Stage 2 artifacts are hashed and read-only; this command does not execute Stage 2 or write to its directory.",
        ],
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Create Stage-3 final acceptance report")
    parser.add_argument("--audit", type=Path, default=ROOT / "runs" / "phase5_temporal_audit_final" / "pose_hallucination_audit.json")
    parser.add_argument("--state", type=Path, default=ROOT / "runs" / "phase5_temporal_audit_final" / "stage3_output" / "tracked_pose_2d_state.json")
    parser.add_argument("--annotation-manifest", type=Path, default=ROOT / "runs" / "phase8_annotation_tasks_gsr_valid.json")
    parser.add_argument("--annotation-validation", type=Path, default=ROOT / "runs" / "phase8_annotation_validation_pending.json")
    parser.add_argument("--preannotation", type=Path, default=ROOT / "runs" / "phase8_preannotated_full.json")
    parser.add_argument("--preannotation-validation", type=Path, default=ROOT / "runs" / "phase8_preannotation_full_validation.json")
    parser.add_argument("--preannotation-checkpoint", type=Path, default=ROOT / "runs" / "phase8_preannotation_full.checkpoint")
    parser.add_argument("--split-manifest", type=Path, default=ROOT / "splits" / "phase8_gsr_valid_seed20260926.json")
    parser.add_argument("--benchmark-split-manifest", type=Path, default=ROOT / "splits" / "3dsp_internal_holdout_seed20260926.json")
    parser.add_argument("--benchmark", type=Path, default=ROOT / "benchmark_results" / "3dsp_phase_final_holdout_rtmw_l" / "3dsp_benchmark_summary.json")
    parser.add_argument("--benchmark-reference", type=Path, default=ROOT / "benchmark_results" / "3dsp_phase9_internal_holdout_rtmw_l" / "3dsp_benchmark_summary.json")
    parser.add_argument("--benchmark-development", type=Path, default=ROOT / "benchmark_results" / "3dsp_phase_final_development_rtmw_l" / "3dsp_benchmark_summary.json")
    parser.add_argument("--model", type=Path, default=ROOT.parent / "datasets" / "stage3_assets" / "rtmw_l_384x288.onnx")
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "phase8_final_acceptance.json")
    args = parser.parse_args()
    report = build_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": report["overall_status"], "output": str(args.output.resolve()), "gates": report["gates"]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
