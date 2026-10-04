from __future__ import annotations

"""Fail-closed evaluator for independent Stage-4 model-only outputs."""

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .backends.sam3d_pitch_refined.cache import MHR70_NAMES
from .backends.sam3d_pitch_refined.geometry import project_camera_points
from .evaluation import mpjpe, procrustes_mpjpe, root_aligned_mpjpe
from .model_only import ModelOnlyManifest, load_model_only_manifest, sha256_file


MODEL_ONLY_OUTPUT_SCHEMA = "stage4-model-only-sam3d-output-1.1"
SUPPORTED_MODEL_ONLY_OUTPUT_SCHEMAS = {
    "stage4-model-only-sam3d-output-1.0",
    MODEL_ONLY_OUTPUT_SCHEMA,
}
MODEL_ONLY_EVALUATION_SCHEMA = "stage4-model-only-evaluation-report-1.1"
_REQUIRED_ARRAYS = (
    "record_ids", "sequence_ids", "bboxes_xyxy", "skel_2d_px", "skel_3d_relative_m",
    "pred_cam_t_m", "focal_length_px", "pose23_world_m", "pose23_ground_first_world_m",
    "valid_mask", "joint_names", "metadata_json",
)


def _stats(values: list[float] | np.ndarray) -> dict[str, float | int | None]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "mean": None, "median": None, "p90": None, "p95": None}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
    }


def _array(data: Any, name: str) -> np.ndarray:
    if name not in data.files:
        raise ValueError(f"Model-only output is missing array: {name}")
    return np.asarray(data[name])


def _metric_record(pred: np.ndarray, gt: np.ndarray, visible: np.ndarray) -> dict[str, Any]:
    finite_pred = np.isfinite(pred).all(axis=1)
    finite_gt = np.isfinite(gt).all(axis=1)
    mask = np.asarray(visible, dtype=bool) & finite_pred & finite_gt
    count = int(mask.sum())
    if count < 3:
        return {"status": "NOT_EVALUATED_INSUFFICIENT_VALID_JOINTS", "valid_joint_count": count}
    pred_eval = np.asarray(pred, dtype=np.float64).copy()
    gt_eval = np.asarray(gt, dtype=np.float64).copy()
    pred_eval[~mask] = np.nan
    gt_eval[~mask] = np.nan
    x_error = np.abs(pred_eval[:, 0] - gt_eval[:, 0])
    return {
        "status": "EVALUATED",
        "valid_joint_count": count,
        "global_mpjpe_m": mpjpe(pred_eval, gt_eval),
        "root_aligned_mpjpe_m": root_aligned_mpjpe(pred_eval, gt_eval),
        "pa_mpjpe_m": procrustes_mpjpe(pred_eval, gt_eval),
        "longitudinal_abs_error_m": _stats(x_error[mask]),
    }


def _reprojection_errors(manifest: ModelOnlyManifest, index: int, native_2d: np.ndarray,
                         native_3d_relative: np.ndarray, translation: np.ndarray) -> np.ndarray:
    record = manifest.records[index]
    points = np.asarray(native_3d_relative, dtype=np.float64) + np.asarray(translation, dtype=np.float64)[None, :]
    projected = project_camera_points(record.camera, points, distort=False)
    observed = np.asarray(native_2d, dtype=np.float64)
    mask = np.isfinite(projected).all(axis=1) & np.isfinite(observed).all(axis=1)
    return np.linalg.norm(projected[mask] - observed[mask], axis=1)


def _scale_slice(bbox: np.ndarray) -> str:
    height = float(bbox[3] - bbox[1])
    if height < 80.0:
        return "small_bbox_height_lt_80px"
    if height < 160.0:
        return "medium_bbox_height_80_160px"
    return "large_bbox_height_ge_160px"


def _load_and_validate_output(path: Path, manifest: ModelOnlyManifest) -> tuple[dict[str, np.ndarray], dict[str, Any], str]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"Model-only output not found or empty: {path}")
    output_sha256 = sha256_file(path)
    with np.load(path, allow_pickle=False) as data:
        for name in _REQUIRED_ARRAYS:
            if name not in data.files:
                raise ValueError(f"Model-only output is missing array: {name}")
        schema = str(np.asarray(data["schema_version"]).item())
        if schema not in SUPPORTED_MODEL_ONLY_OUTPUT_SCHEMAS:
            raise ValueError(f"Unsupported model-only output schema: {schema}")
        arrays = {name: _array(data, name) for name in _REQUIRED_ARRAYS if name != "metadata_json"}
        try:
            metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("Model-only output metadata_json is not valid JSON") from exc
    if not isinstance(metadata, dict):
        raise ValueError("Model-only output metadata must be an object")
    if metadata.get("manifest_sha256") != manifest.sha256:
        raise ValueError("Model-only output manifest_sha256 does not match the supplied manifest")
    if metadata.get("stage1_stage3_inputs") is not None:
        raise ValueError("Model-only output must not contain Stage-1/Stage-3 inputs")

    expected_ids = tuple(record.record_id for record in manifest.records)
    expected_sequences = tuple(record.sequence_id for record in manifest.records)
    record_ids = tuple(str(value) for value in arrays["record_ids"].tolist())
    sequence_ids = tuple(str(value) for value in arrays["sequence_ids"].tolist())
    if record_ids != expected_ids:
        raise ValueError(f"Model-only output record_ids mismatch: expected {expected_ids}, got {record_ids}")
    if sequence_ids != expected_sequences:
        raise ValueError("Model-only output sequence_ids mismatch")

    n = len(expected_ids)
    expected_shapes = {
        "bboxes_xyxy": (n, 4), "skel_2d_px": (n, 70, 2), "skel_3d_relative_m": (n, 70, 3),
        "pred_cam_t_m": (n, 3), "focal_length_px": (n,), "pose23_world_m": (n, 23, 3),
        "pose23_ground_first_world_m": (n, 23, 3), "valid_mask": (n,),
    }
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise ValueError(f"Model-only output {name} shape mismatch: expected {shape}, got {arrays[name].shape}")
    names = tuple(str(value) for value in arrays["joint_names"].tolist())
    if names != MHR70_NAMES:
        raise ValueError("Model-only output joint_names do not match official MHR70 ordering")
    if arrays["valid_mask"].dtype.kind not in {"b", "i", "u"}:
        raise ValueError("Model-only output valid_mask must be boolean/integer")
    valid_values = np.asarray(arrays["valid_mask"], dtype=np.int64)
    if not np.isin(valid_values, [0, 1]).all():
        raise ValueError("Model-only output valid_mask must contain only 0/1 values")
    if not np.isfinite(arrays["bboxes_xyxy"]).all():
        raise ValueError("Model-only output bboxes_xyxy contains non-finite values")

    valid_mask = valid_values.astype(bool)
    finite_by_record = {
        "skel_2d_px": np.isfinite(arrays["skel_2d_px"]).all(axis=(1, 2)),
        "skel_3d_relative_m": np.isfinite(arrays["skel_3d_relative_m"]).all(axis=(1, 2)),
        "pred_cam_t_m": np.isfinite(arrays["pred_cam_t_m"]).all(axis=1),
        "focal_length_px": np.isfinite(arrays["focal_length_px"]),
        "pose23_world_m": np.isfinite(arrays["pose23_world_m"]).all(axis=(1, 2)),
        "pose23_ground_first_world_m": np.isfinite(arrays["pose23_ground_first_world_m"]).all(axis=(1, 2)),
    }
    finite_stack = np.column_stack(list(finite_by_record.values()))
    if np.any(valid_mask & ~finite_stack.all(axis=1)):
        bad = np.flatnonzero(valid_mask & ~finite_stack.all(axis=1)).tolist()
        raise ValueError(f"valid_mask marks records with incomplete finite output: indices={bad}")
    partial = (~valid_mask) & finite_stack.any(axis=1)
    if np.any(partial):
        bad = np.flatnonzero(partial).tolist()
        raise ValueError(f"invalid records must not contain partial finite output: indices={bad}")

    recorded_hashes = metadata.get("record_input_hashes")
    if not isinstance(recorded_hashes, dict):
        raise ValueError("Model-only output record_input_hashes are required")
    for record in manifest.records:
        expected = record.input_hashes
        actual = recorded_hashes.get(record.record_id)
        if actual != expected:
            raise ValueError(f"Input hash mismatch for record {record.record_id}")
    return arrays, metadata, output_sha256


def evaluate_model_only_output(*, manifest_path: str | Path, output_path: str | Path,
                               report_path: str | Path | None = None) -> dict[str, Any]:
    manifest = load_model_only_manifest(manifest_path)
    output_file = Path(output_path).expanduser().resolve()
    arrays, metadata, output_sha256 = _load_and_validate_output(output_file, manifest)

    per_record: list[dict[str, Any]] = []
    direct_preds: list[np.ndarray] = []
    refined_preds: list[np.ndarray] = []
    ground_truths: list[np.ndarray] = []
    direct_record_metrics: list[float] = []
    refined_record_metrics: list[float] = []
    x_direct: list[float] = []
    x_refined: list[float] = []
    reprojection: list[float] = []
    slice_values: dict[str, list[float]] = {}
    failures: list[dict[str, Any]] = []
    refinement_by_id = {
        str(item.get("record_id")): item
        for item in (metadata.get("ground_first_refinement") or [])
        if isinstance(item, dict) and item.get("record_id") is not None
    }
    refinement_status_counts: dict[str, int] = {}
    refinement_reason_counts: dict[str, int] = {}
    refinement_source_counts: dict[str, int] = {}
    refinement_bounded_count = 0
    missingness_counts: dict[str, int] = {}

    for index, record in enumerate(manifest.records):
        direct = arrays["pose23_world_m"][index].astype(np.float64)
        refined = arrays["pose23_ground_first_world_m"][index].astype(np.float64)
        gt = record.gt_xyz_world_m.astype(np.float64)
        direct_metric = _metric_record(direct, gt, record.gt_visible)
        refined_metric = _metric_record(refined, gt, record.gt_visible)
        refinement = refinement_by_id.get(record.record_id, {})
        refinement_status = str(refinement.get("status", "NOT_RECORDED"))
        refinement_status_counts[refinement_status] = refinement_status_counts.get(refinement_status, 0) + 1
        refinement_reason = refinement.get("reason")
        if refinement_reason:
            refinement_reason = str(refinement_reason)
            refinement_reason_counts[refinement_reason] = refinement_reason_counts.get(refinement_reason, 0) + 1
        refinement_source = str(refinement.get("source", "NOT_RECORDED"))
        refinement_source_counts[refinement_source] = refinement_source_counts.get(refinement_source, 0) + 1
        if bool(refinement.get("correction_bounded")):
            refinement_bounded_count += 1
        reproj = _reprojection_errors(
            manifest, index, arrays["skel_2d_px"][index], arrays["skel_3d_relative_m"][index], arrays["pred_cam_t_m"][index]
        ) if bool(arrays["valid_mask"][index]) else np.empty((0,), dtype=np.float64)
        reprojection.extend(reproj.tolist())

        diagnostics = {
            "record_id": record.record_id,
            "sequence_id": record.sequence_id,
            "split": record.split,
            "scale_slice": _scale_slice(arrays["bboxes_xyxy"][index]),
            "bbox_height_px": float(arrays["bboxes_xyxy"][index, 3] - arrays["bboxes_xyxy"][index, 1]),
            "model_valid": bool(arrays["valid_mask"][index]),
            "native_reprojection_px": _stats(reproj),
            "direct": direct_metric,
            "ground_first": refined_metric,
            "ground_first_diagnostics": refinement,
        }
        if not bool(arrays["valid_mask"][index]):
            failure_reason = next(
                (
                    item.get("reason") for item in (metadata.get("failures") or [])
                    if isinstance(item, dict) and item.get("record_id") == record.record_id
                ),
                None,
            )
            diagnostics["failure_attribution"] = "MODEL_OUTPUT_MISSING_OR_FAILED"
            diagnostics["missingness"] = {
                "status": "MISSING",
                "reason": failure_reason or "model_output_missing_or_failed",
                "finite_output": False,
            }
            missingness_counts["MISSING"] = missingness_counts.get("MISSING", 0) + 1
            failures.append({"record_id": record.record_id, "category": diagnostics["failure_attribution"]})
        elif direct_metric["status"] != "EVALUATED":
            diagnostics["failure_attribution"] = "GT_INSUFFICIENT_VISIBLE_JOINTS"
            diagnostics["missingness"] = {
                "status": "NOT_EVALUATED",
                "reason": "gt_insufficient_visible_joints",
                "finite_output": bool(np.isfinite(direct).all()),
            }
            missingness_counts["NOT_EVALUATED"] = missingness_counts.get("NOT_EVALUATED", 0) + 1
            failures.append({"record_id": record.record_id, "category": diagnostics["failure_attribution"]})
        elif refined_metric["status"] != "EVALUATED":
            diagnostics["failure_attribution"] = "GROUND_FIRST_OUTPUT_INVALID"
            diagnostics["missingness"] = {
                "status": "REJECTED",
                "reason": "ground_first_output_invalid",
                "finite_output": True,
            }
            missingness_counts["REJECTED"] = missingness_counts.get("REJECTED", 0) + 1
            failures.append({"record_id": record.record_id, "category": diagnostics["failure_attribution"]})
        else:
            direct_value = float(direct_metric["global_mpjpe_m"])
            refined_value = float(refined_metric["global_mpjpe_m"])
            delta = direct_value - refined_value
            diagnostics["ground_first_delta_global_mpjpe_m"] = delta
            diagnostics["failure_attribution"] = "GROUND_FIRST_IMPROVED" if delta > 0 else "GROUND_FIRST_REGRESSED" if delta < 0 else "NO_GLOBAL_CHANGE"
            if refinement_status == "CONTACT_AMBIGUOUS":
                missingness = {"status": "DEGRADED", "reason": refinement.get("reason") or "contact_ambiguous", "finite_output": True}
            elif refinement_status == "GROUND_UNAVAILABLE":
                missingness = {"status": "DEGRADED", "reason": refinement.get("reason") or "ground_unavailable", "finite_output": True}
            else:
                missingness = {"status": "VALID", "reason": "ground_first_evaluated", "finite_output": True}
            diagnostics["missingness"] = missingness
            missingness_counts[missingness["status"]] = missingness_counts.get(missingness["status"], 0) + 1
            direct_record_metrics.append(direct_value)
            refined_record_metrics.append(refined_value)
            direct_mask = record.gt_visible & np.isfinite(direct).all(axis=1) & np.isfinite(gt).all(axis=1)
            refined_mask = record.gt_visible & np.isfinite(refined).all(axis=1) & np.isfinite(gt).all(axis=1)
            x_direct.extend(np.abs(direct[direct_mask, 0] - gt[direct_mask, 0]).tolist())
            x_refined.extend(np.abs(refined[refined_mask, 0] - gt[refined_mask, 0]).tolist())
            direct_preds.append(direct)
            refined_preds.append(refined)
            ground_truths.append(gt)
            slice_values.setdefault(diagnostics["scale_slice"], []).append(delta)
        per_record.append(diagnostics)

    def cohort_metrics(predictions: list[np.ndarray], gts: list[np.ndarray]) -> dict[str, Any]:
        if not predictions:
            return {"status": "NOT_EVALUATED_NO_COMMON_COHORT"}
        pred = np.stack(predictions, axis=0)
        gt = np.stack(gts, axis=0)
        return {
            "status": "EVALUATED",
            "record_count": len(predictions),
            "global_mpjpe_m": mpjpe(pred, gt),
            "root_aligned_mpjpe_m": root_aligned_mpjpe(pred, gt),
            "pa_mpjpe_m": procrustes_mpjpe(pred, gt),
        }

    report: dict[str, Any] = {
        "schema_version": MODEL_ONLY_EVALUATION_SCHEMA,
        "manifest": str(manifest.path),
        "manifest_sha256": manifest.sha256,
        "output": str(output_file),
        "output_sha256": output_sha256,
        "dataset": manifest.dataset,
        "split": manifest.split,
        "model_provenance": {
            "checkpoint": metadata.get("checkpoint"),
            "checkpoint_sha256": metadata.get("checkpoint_sha256"),
            "mhr_path": metadata.get("mhr_path"),
            "mhr_model_sha256": metadata.get("mhr_model_sha256"),
        },
        "coverage": {
            "record_count": len(manifest.records),
            "valid_model_records": int(np.asarray(arrays["valid_mask"], dtype=bool).sum()),
            "valid_model_fraction": float(np.asarray(arrays["valid_mask"], dtype=bool).mean()) if manifest.records else None,
            "evaluated_record_count": len(direct_record_metrics),
            "failure_count": len(failures),
        },
        "metrics": {
            "direct_common_cohort": cohort_metrics(direct_preds, ground_truths),
            "ground_first_common_cohort": cohort_metrics(refined_preds, ground_truths),
            "direct_record_global_mpjpe_m": _stats(direct_record_metrics),
            "ground_first_record_global_mpjpe_m": _stats(refined_record_metrics),
            "direct_longitudinal_abs_error_m": _stats(x_direct),
            "ground_first_longitudinal_abs_error_m": _stats(x_refined),
            "ground_first_delta_global_mpjpe_m": _stats([a - b for a, b in zip(direct_record_metrics, refined_record_metrics)]),
        },
        "failure_attribution": {
            "categories": failures,
            "scale_slice_ground_first_delta_global_mpjpe_m": {key: _stats(value) for key, value in sorted(slice_values.items())},
            "native_reprojection_px": _stats(reprojection),
            "interpretation": "Diagnostics for attribution only; they do not establish causality or calibrated uncertainty.",
        },
        "ground_first_diagnostics_summary": {
            "status_counts": refinement_status_counts,
            "reason_counts": refinement_reason_counts,
            "source_counts": refinement_source_counts,
            "correction_bounded_count": refinement_bounded_count,
            "policy": next((item.get("policy") for item in refinement_by_id.values() if item.get("policy")), None),
        },
        "missingness_summary": {
            "status_counts": missingness_counts,
            "semantics": {
                "VALID": "finite output evaluated; no claim that uncertainty is calibrated",
                "DEGRADED": "finite output retained with ambiguous/unavailable ground evidence",
                "REJECTED": "output cannot be used for the requested evaluation lane",
                "MISSING": "model output absent or failed closed",
            },
        },
        "uncertainty": {
            "status": "NOT_CALIBRATED",
            "scope": "SENSITIVITY_ONLY",
            "components": {
                "pixel_perturbation": "NOT_RUN",
                "camera_sensitivity": "NOT_AVAILABLE",
                "ground_contact_ambiguity": "DIAGNOSTIC_ONLY",
                "pretrained_model_disagreement": "NOT_AVAILABLE_SINGLE_BACKEND",
                "temporal_support": "NOT_AVAILABLE_SINGLE_RECORD",
            },
            "note": "No interval coverage or confidence claim is allowed without a calibration split and locked metric-GT holdout.",
        },
        "per_record": per_record,
        "metric_computation_completed": bool(direct_record_metrics),
        "accuracy_claim_allowed": False,
        "research_accuracy_frozen": False,
        "upstream_dependencies": None,
    }
    if report_path is not None:
        target = Path(report_path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    return report
