from __future__ import annotations

"""Run official SAM 3D Body on an independent Stage-4 benchmark manifest.

The command consumes benchmark RGB/crop, bbox and benchmark camera inputs. It
does not load Stage 1/3 state, camera timelines, tracks or upstream adapters.
"""

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stage4_metric3d.backends.sam3d_pitch_refined.cache import MHR70_NAMES
from stage4_metric3d.backends.sam3d_pitch_refined.geometry import camera_to_world
from stage4_metric3d.backends.sam3d_pitch_refined.joint_mapping import DEFAULT_MAPPING
from stage4_metric3d.evaluation import mpjpe, procrustes_mpjpe, root_aligned_mpjpe
from stage4_metric3d.model_only import load_model_only_manifest, sha256_file
from stage4_metric3d.model_only_cache import (
    build_cache_contract,
    validate_cached_run,
    write_cache_manifest,
)
from stage4_metric3d.model_only_evaluation import evaluate_model_only_output
from stage4_metric3d.model_only_refinement import MODEL_ONLY_REFINEMENT_POLICY, refine_model_only_translation


def _build_estimator(checkpoint: str, mhr_path: str, *, device: str, sam3d_root: str | None):
    import torch

    if sam3d_root:
        root = Path(sam3d_root).expanduser().resolve()
        if not (root / "sam_3d_body").is_dir():
            raise FileNotFoundError(f"sam3d-root does not contain sam_3d_body: {root}")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
    resolved_device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
    if resolved_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("SAM3D_CUDA_UNAVAILABLE: CUDA was requested but torch.cuda.is_available() is false")
    if resolved_device == "cpu":
        print("[SAM3D-model-only] CPU compatibility mode enabled; inference can be very slow.", flush=True)
    from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body
    from sam_3d_body.metadata.mhr70 import mhr_names

    installed_names = tuple(str(x).strip().lower().replace("-", "_").replace(" ", "_") for x in mhr_names)
    if installed_names != MHR70_NAMES:
        raise RuntimeError("SAM3D_MHR70_SCHEMA_MISMATCH: installed metadata differs from verified MHR70 ordering")
    torch_device = torch.device(resolved_device)
    model, model_cfg = load_sam_3d_body(checkpoint, device=torch_device, mhr_path=mhr_path)
    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=None,
        human_segmentor=None,
        fov_estimator=None,
    )
    return estimator, resolved_device


def _resolve_device(device: str) -> str:
    import torch
    resolved = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
    if resolved == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("SAM3D_CUDA_UNAVAILABLE: CUDA was requested but torch.cuda.is_available() is false")
    return resolved


def _read_rgb(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Failed to decode benchmark image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def run(*, manifest_path: str | Path, output_path: str | Path, checkpoint: str, mhr_path: str,
        sam3d_root: str | None = None, device: str = "auto", inference_type: str = "body",
        cache_manifest_path: str | Path | None = None, resume: bool = False) -> dict:
    manifest = load_model_only_manifest(manifest_path)
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    mhr_model_path = Path(mhr_path).expanduser().resolve()
    if not checkpoint_path.is_file() or checkpoint_path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing SAM3D checkpoint: {checkpoint_path}")
    if not mhr_model_path.is_file() or mhr_model_path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing MHR model: {mhr_model_path}")
    if inference_type != "body":
        raise ValueError("Model-only runner currently supports --inference-type body")

    output_file = Path(output_path).expanduser().resolve()
    cache_file = Path(cache_manifest_path).expanduser().resolve() if cache_manifest_path else output_file.with_suffix(".cache.json")
    resolved_device = _resolve_device(device)
    cache_contract = build_cache_contract(
        project_root=PROJECT_ROOT,
        manifest_sha256=manifest.sha256,
        dataset=manifest.dataset,
        split=manifest.split,
        record_input_hashes={record.record_id: record.input_hashes for record in manifest.records},
        checkpoint_path=checkpoint_path,
        mhr_model_path=mhr_model_path,
        sam3d_root=sam3d_root,
        device=device,
        resolved_device=resolved_device,
        inference_type=inference_type,
        output_schema="stage4-model-only-sam3d-output-1.1",
    )
    if resume:
        validate_cached_run(cache_path=cache_file, output_path=output_file, expected_contract=cache_contract)
        report_path = output_file.with_suffix(".resume.metrics.json")
        report = evaluate_model_only_output(manifest_path=manifest_path, output_path=output_file, report_path=report_path)
        return {
            "output": str(output_file),
            "metrics_report": str(report_path),
            "cache_manifest": str(cache_file),
            "cache_key": cache_contract["cache_key"],
            "cache_hit": True,
            "schema": "stage4-model-only-sam3d-output-1.1",
            "records": len(manifest.records),
            "valid_records": int(report["coverage"]["valid_model_records"]),
            "metric_evaluation": report["metrics"],
            "failures": report["failure_attribution"]["categories"],
        }

    estimator, resolved_device = _build_estimator(str(checkpoint_path), str(mhr_model_path), device=resolved_device, sam3d_root=sam3d_root)
    record_ids = [record.record_id for record in manifest.records]
    bboxes = np.asarray([record.bbox_xyxy for record in manifest.records], dtype=np.float32)
    skel_2d = np.full((len(record_ids), 70, 2), np.nan, dtype=np.float32)
    skel_3d = np.full((len(record_ids), 70, 3), np.nan, dtype=np.float32)
    cam_t = np.full((len(record_ids), 3), np.nan, dtype=np.float32)
    focal = np.full((len(record_ids),), np.nan, dtype=np.float32)
    pose23_world = np.full((len(record_ids), 23, 3), np.nan, dtype=np.float32)
    pose23_refined_world = np.full((len(record_ids), 23, 3), np.nan, dtype=np.float32)
    valid = np.zeros((len(record_ids),), dtype=bool)
    refinement_records: list[dict] = []
    failures: list[dict] = []

    import torch
    for index, record in enumerate(manifest.records):
        try:
            image = _read_rgb(record.image_path)
            cam_int = torch.as_tensor(np.asarray(record.camera.K, dtype=np.float32), dtype=torch.float32, device=resolved_device).unsqueeze(0)
            outputs = estimator.process_one_image(
                image,
                bboxes=np.asarray([record.bbox_xyxy], dtype=np.float32),
                cam_int=cam_int,
                use_mask=False,
                inference_type=inference_type,
            )
            if len(outputs) != 1:
                raise RuntimeError(f"SAM3D output count mismatch: expected 1, got {len(outputs)}")
            output = outputs[0]
            p2d = np.asarray(output.get("pred_keypoints_2d"), dtype=np.float32)
            p3d = np.asarray(output.get("pred_keypoints_3d"), dtype=np.float32)
            translation = np.asarray(output.get("pred_cam_t"), dtype=np.float32).reshape(-1)
            focal_value = np.asarray(output.get("focal_length"), dtype=np.float32).reshape(-1)
            if p2d.shape != (70, 2) or p3d.shape != (70, 3) or translation.shape != (3,) or focal_value.size < 1:
                raise RuntimeError(f"SAM3D output schema mismatch: 2D={p2d.shape}, 3D={p3d.shape}, translation={translation.shape}")
            if not (np.isfinite(p2d).all() and np.isfinite(p3d).all() and np.isfinite(translation).all() and np.isfinite(focal_value[0])):
                raise RuntimeError("SAM3D returned non-finite output")
            skel_2d[index] = p2d
            skel_3d[index] = p3d
            cam_t[index] = translation
            focal[index] = float(focal_value[0])
            native_camera = p3d + translation[None, :]
            native_world = camera_to_world(record.camera, native_camera)
            for mapping in DEFAULT_MAPPING:
                pose23_world[index, mapping.rtmw_index] = native_world[mapping.sam3d_index]
            refinement = refine_model_only_translation(
                camera=record.camera,
                native_2d=p2d,
                native_3d_relative_m=p3d,
                sam_prior_cam_m=translation,
            )
            refined_camera = p3d + refinement.refined_root_cam_m[None, :]
            refined_world = camera_to_world(record.camera, refined_camera)
            for mapping in DEFAULT_MAPPING:
                pose23_refined_world[index, mapping.rtmw_index] = refined_world[mapping.sam3d_index]
            refinement_records.append({
                "record_id": record.record_id,
                "policy": MODEL_ONLY_REFINEMENT_POLICY,
                "status": refinement.status,
                "reason": refinement.reason,
                "source": refinement.source,
                "candidate_count": refinement.candidate_count,
                "consensus_count": refinement.consensus_count,
                "consensus_fraction": refinement.consensus_fraction,
                "candidate_names": list(refinement.candidate_names),
                "consensus_names": list(refinement.consensus_names),
                "candidate_spread_m": refinement.candidate_spread_m,
                "rejection_reasons": list(refinement.rejection_reasons),
                "correction_bounded": refinement.correction_bounded,
                "refinement_applied": refinement.status == "CONTACT_SUPPORTED",
                "sam_prior_cam_m": translation.tolist(),
                "refined_root_cam_m": refinement.refined_root_cam_m.tolist(),
                "root_correction_m": float(np.linalg.norm(refinement.refined_root_cam_m - translation)),
            })
            valid[index] = True
        except Exception as exc:
            failures.append({"record_id": record.record_id, "reason": f"{type(exc).__name__}: {exc}"})

    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_file.with_name(output_file.stem + ".tmp.npz")
    per_record_metrics = []
    all_direct_pred = []
    all_refined_pred = []
    all_gt = []
    for index, record in enumerate(manifest.records):
        pred = pose23_world[index].astype(np.float64)
        refined_pred = pose23_refined_world[index].astype(np.float64)
        gt = record.gt_xyz_world_m.copy()
        mask = record.gt_visible & np.isfinite(pred).all(axis=1) & np.isfinite(gt).all(axis=1)
        pred_eval = pred.copy()
        refined_eval = refined_pred.copy()
        gt_eval = gt.copy()
        pred_eval[~mask] = np.nan
        refined_eval[~(record.gt_visible & np.isfinite(refined_pred).all(axis=1) & np.isfinite(gt).all(axis=1))] = np.nan
        gt_eval[~mask] = np.nan
        if valid[index] and int(mask.sum()) >= 3:
            record_metrics = {
                "record_id": record.record_id,
                "visible_joint_count": int(mask.sum()),
                "direct": {
                    "global_mpjpe_m": mpjpe(pred_eval, gt_eval),
                    "root_aligned_mpjpe_m": root_aligned_mpjpe(pred_eval, gt_eval),
                    "pa_mpjpe_m": procrustes_mpjpe(pred_eval, gt_eval),
                },
                "ground_first": {
                    "global_mpjpe_m": mpjpe(refined_eval, gt_eval),
                    "root_aligned_mpjpe_m": root_aligned_mpjpe(refined_eval, gt_eval),
                    "pa_mpjpe_m": procrustes_mpjpe(refined_eval, gt_eval),
                },
            }
            all_direct_pred.append(pred_eval)
            all_refined_pred.append(refined_eval)
            all_gt.append(gt_eval)
        else:
            record_metrics = {"record_id": record.record_id, "visible_joint_count": int(mask.sum()), "status": "NOT_EVALUATED_INSUFFICIENT_VALID_OUTPUT"}
        per_record_metrics.append(record_metrics)
    def _cohort_metrics(predictions, ground_truth):
        cohort_pred = np.stack(predictions, axis=0)
        cohort_gt = np.stack(ground_truth, axis=0)
        return {
            "global_mpjpe_m": mpjpe(cohort_pred, cohort_gt),
            "root_aligned_mpjpe_m": root_aligned_mpjpe(cohort_pred, cohort_gt),
            "pa_mpjpe_m": procrustes_mpjpe(cohort_pred, cohort_gt),
        }

    if all_direct_pred:
        cohort_gt = np.stack(all_gt, axis=0)
        metric_evaluation = {
            "status": "EVALUATED",
            "scope": "PITCH_WORLD_METRIC",
            "unit": "m",
            "record_count": len(all_direct_pred),
            "direct": _cohort_metrics(all_direct_pred, all_gt),
            "ground_first": _cohort_metrics(all_refined_pred, all_gt),
            "note": "Single-view SAM3D direct baseline mapped to Pose23 and transformed with benchmark camera; this is a model-only baseline, not the refined pipeline.",
        }
    else:
        metric_evaluation = {"status": "NOT_EVALUATED_NO_VALID_PREDICTIONS", "scope": "PITCH_WORLD_METRIC", "unit": "m"}
    metadata = {
        "schema_version": "stage4-model-only-sam3d-output-1.1",
        "producer": "run_sam3d_model_only.py",
        "manifest": str(manifest.path),
        "manifest_sha256": manifest.sha256,
        "dataset": manifest.dataset,
        "split": manifest.split,
        "coordinate_frame": manifest.coordinate_frame,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "mhr_path": str(mhr_model_path),
        "mhr_model_sha256": sha256_file(mhr_model_path),
        "device": resolved_device,
        "inference_type": inference_type,
        "record_input_hashes": {record.record_id: record.input_hashes for record in manifest.records},
        "failures": failures,
        "metric_evaluation": metric_evaluation,
        "per_record_metrics": per_record_metrics,
        "ground_first_refinement": refinement_records,
        "stage1_stage3_inputs": None,
        "cache_manifest": str(cache_file),
        "cache_key": cache_contract["cache_key"],
        "uncertainty_contract": {
            "status": "NOT_CALIBRATED",
            "scope": "SENSITIVITY_ONLY",
            "camera_uncertainty": "NOT_AVAILABLE",
            "ground_contact_ambiguity": "DIAGNOSTIC_ONLY",
            "model_disagreement": "NOT_AVAILABLE_SINGLE_BACKEND",
            "temporal_support": "NOT_AVAILABLE_SINGLE_RECORD",
        },
    }
    np.savez_compressed(
        temporary_output,
        schema_version=np.array("stage4-model-only-sam3d-output-1.1"),
        record_ids=np.asarray(record_ids),
        sequence_ids=np.asarray([record.sequence_id for record in manifest.records]),
        bboxes_xyxy=bboxes,
        skel_2d_px=skel_2d,
        skel_3d_relative_m=skel_3d,
        pred_cam_t_m=cam_t,
        focal_length_px=focal,
        pose23_world_m=pose23_world,
        pose23_ground_first_world_m=pose23_refined_world,
        valid_mask=valid,
        joint_names=np.asarray(MHR70_NAMES),
        metadata_json=np.array(json.dumps(metadata, ensure_ascii=True)),
    )
    temporary_output.replace(output_file)
    output_sha256 = sha256_file(output_file)
    write_cache_manifest(cache_file, cache_contract, output_sha256=output_sha256)
    report_file = output_file.with_suffix(".metrics.json")
    report_file.write_text(json.dumps({
        "schema_version": "stage4-model-only-evaluation-report-1.1",
        "manifest": str(manifest.path),
        "manifest_sha256": manifest.sha256,
        "dataset": manifest.dataset,
        "split": manifest.split,
        "model": {"name": "SAM3D-Body", "checkpoint_sha256": metadata["checkpoint_sha256"], "mhr_model_sha256": metadata["mhr_model_sha256"]},
        "metric_evaluation": metric_evaluation,
        "per_record_metrics": per_record_metrics,
        "ground_first_refinement": refinement_records,
        "coverage": {"record_count": len(record_ids), "valid_count": int(valid.sum()), "fraction": float(valid.mean()) if valid.size else None},
        "failures": failures,
        "metric_computation_completed": bool(metric_evaluation["status"] == "EVALUATED"),
        "accuracy_claim_allowed": False,
        "research_accuracy_frozen": False,
        "development_metric_is_final_accuracy": False,
        "refinement": "SAM3D_DIRECT_VS_MODEL_ONLY_GROUND_FIRST_ABLATION",
        "refinement_policy": MODEL_ONLY_REFINEMENT_POLICY,
        "cache_manifest": str(cache_file),
        "cache_key": cache_contract["cache_key"],
        "uncertainty": {
            "status": "NOT_CALIBRATED",
            "scope": "SENSITIVITY_ONLY",
            "note": "Model-only output contains no calibrated intervals.",
        },
    }, indent=2), encoding="utf-8")
    return {"output": str(output_file), "metrics_report": str(report_file), "schema": metadata["schema_version"], "records": len(record_ids), "valid_records": int(valid.sum()), "metric_evaluation": metric_evaluation, "failures": failures}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SAM3D on an independent Stage-4 model-only benchmark manifest")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mhr-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sam3d-root", default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--inference-type", choices=("body",), default="body")
    parser.add_argument("--cache-manifest", default=None, help="Sidecar cache provenance manifest; defaults beside --output")
    parser.add_argument("--resume", action="store_true", help="Reuse output only when the complete provenance contract matches")
    args = parser.parse_args()
    print(json.dumps(run(manifest_path=args.manifest, output_path=args.output, checkpoint=args.checkpoint, mhr_path=args.mhr_path, sam3d_root=args.sam3d_root, device=args.device, inference_type=args.inference_type, cache_manifest_path=args.cache_manifest, resume=args.resume), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
