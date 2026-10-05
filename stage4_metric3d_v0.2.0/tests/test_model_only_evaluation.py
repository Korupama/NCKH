from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from stage4_metric3d.backends.sam3d_pitch_refined.cache import MHR70_NAMES
from stage4_metric3d.model_only import load_model_only_manifest, sha256_file
from stage4_metric3d.model_only_evaluation import evaluate_model_only_output
from stage4_metric3d.wholebody import POSE23_NAMES


def _write_fixture(root: Path) -> Path:
    image = np.zeros((120, 200, 3), dtype=np.uint8)
    assert cv2.imwrite(str(root / "frame.png"), image)
    camera = {
        "schema_version": "benchmark-camera-1.0",
        "coordinate_frame": "BENCHMARK_PITCH_WORLD",
        "frame_index": 0,
        "status": "VALID",
        "image": {"width": 200, "height": 120},
        "intrinsics": {"K": [[100.0, 0.0, 100.0], [0.0, 100.0, 60.0], [0.0, 0.0, 1.0]]},
        "extrinsics": {"R_world_to_camera": np.eye(3).tolist(), "camera_center_world_m": [0.0, 0.0, -10.0]},
        "distortion": {"radial": [0, 0, 0, 0, 0, 0], "tangential": [0, 0], "thin_prism": [0, 0, 0, 0]},
        "pitch": {"origin": "center", "length_m": 105.0, "width_m": 68.0},
    }
    (root / "camera.json").write_text(json.dumps(camera), encoding="utf-8")
    record_id = "seq_a_frame_0000_person_00"
    gt = {
        "schema_version": "metric-pose23-gt-1.0",
        "poses": [{
            "record_id": record_id,
            "joint_names": list(POSE23_NAMES),
            "xyz23_world_m": [[1.0, 2.0, 0.1] for _ in range(23)],
            "visible23": [True] * 23,
        }],
    }
    (root / "gt.json").write_text(json.dumps(gt), encoding="utf-8")
    manifest = {
        "schema_version": "stage4-model-only-benchmark-manifest-1.0",
        "dataset": {"name": "licensed-fixture", "release": "fixture-1.0", "license_id": "TEST-ONLY", "license_url": "https://example.invalid/license"},
        "split": "development",
        "coordinate_frame": {
            "name": "BENCHMARK_PITCH_WORLD", "origin": "pitch_center", "scope": "PITCH_WORLD_METRIC", "units": "m",
            "axes": {"x": "goal-to-goal", "y": "touchline-to-touchline", "z": "up"},
        },
        "records": [{
            "record_id": record_id, "sequence_id": "seq_a", "split": "development",
            "image": "frame.png", "bbox_xyxy": [10, 10, 100, 110], "camera": "camera.json", "gt": "gt.json",
        }],
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _write_output(root: Path, *, manifest_path: Path, mutate_metadata: dict | None = None, record_ids=None) -> Path:
    manifest = load_model_only_manifest(manifest_path)
    record = manifest.records[0]
    xyz = record.gt_xyz_world_m.astype(np.float32)
    native_2d = np.full((1, 70, 2), [100.0, 60.0], dtype=np.float32)
    native_3d = np.zeros((1, 70, 3), dtype=np.float32)
    native_translation = np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32)
    metadata = {
        "schema_version": "stage4-model-only-sam3d-output-1.0",
        "manifest": str(manifest.path),
        "manifest_sha256": manifest.sha256,
        "checkpoint": "real-checkpoint-path",
        "checkpoint_sha256": "checkpoint-hash",
        "mhr_path": "real-mhr-path",
        "mhr_model_sha256": "mhr-hash",
        "record_input_hashes": {record.record_id: record.input_hashes},
        "stage1_stage3_inputs": None,
        "ground_first_refinement": [],
    }
    if mutate_metadata:
        metadata.update(mutate_metadata)
    path = root / "output.npz"
    np.savez_compressed(
        path,
        schema_version=np.array("stage4-model-only-sam3d-output-1.0"),
        record_ids=np.asarray([record.record_id] if record_ids is None else record_ids),
        sequence_ids=np.asarray([record.sequence_id]),
        bboxes_xyxy=np.asarray([record.bbox_xyxy], dtype=np.float32),
        skel_2d_px=native_2d,
        skel_3d_relative_m=native_3d,
        pred_cam_t_m=native_translation,
        focal_length_px=np.asarray([100.0], dtype=np.float32),
        pose23_world_m=xyz[None, :, :],
        pose23_ground_first_world_m=xyz[None, :, :],
        valid_mask=np.asarray([True]),
        joint_names=np.asarray(MHR70_NAMES),
        metadata_json=np.array(json.dumps(metadata)),
    )
    return path


def test_model_only_evaluator_computes_common_cohort_metrics(tmp_path: Path):
    manifest_path = _write_fixture(tmp_path)
    output_path = _write_output(tmp_path, manifest_path=manifest_path)
    report = evaluate_model_only_output(manifest_path=manifest_path, output_path=output_path)
    assert report["schema_version"] == "stage4-model-only-evaluation-report-1.1"
    assert report["metric_computation_completed"] is True
    assert report["accuracy_claim_allowed"] is False
    assert report["coverage"]["valid_model_records"] == 1
    assert report["metrics"]["direct_common_cohort"]["global_mpjpe_m"] == pytest.approx(0.0, abs=1e-7)
    assert report["metrics"]["ground_first_common_cohort"]["global_mpjpe_m"] == pytest.approx(0.0, abs=1e-7)
    assert report["failure_attribution"]["native_reprojection_px"]["count"] == 70


def test_model_only_evaluator_rejects_manifest_hash_mismatch(tmp_path: Path):
    manifest_path = _write_fixture(tmp_path)
    output_path = _write_output(tmp_path, manifest_path=manifest_path, mutate_metadata={"manifest_sha256": "wrong"})
    with pytest.raises(ValueError, match="manifest_sha256"):
        evaluate_model_only_output(manifest_path=manifest_path, output_path=output_path)


def test_model_only_evaluator_rejects_record_order_mismatch(tmp_path: Path):
    manifest_path = _write_fixture(tmp_path)
    output_path = _write_output(tmp_path, manifest_path=manifest_path, record_ids=["wrong-record"])
    with pytest.raises(ValueError, match="record_ids mismatch"):
        evaluate_model_only_output(manifest_path=manifest_path, output_path=output_path)


def test_model_only_evaluator_writes_report(tmp_path: Path):
    manifest_path = _write_fixture(tmp_path)
    output_path = _write_output(tmp_path, manifest_path=manifest_path)
    report_path = tmp_path / "report.json"
    evaluate_model_only_output(manifest_path=manifest_path, output_path=output_path, report_path=report_path)
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["output_sha256"] == sha256_file(output_path)
    assert saved["upstream_dependencies"] is None


def test_model_only_evaluator_rejects_partial_finite_invalid_record(tmp_path: Path):
    manifest_path = _write_fixture(tmp_path)
    output_path = _write_output(tmp_path, manifest_path=manifest_path)
    with np.load(output_path, allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]) for name in data.files}
    arrays["valid_mask"] = np.asarray([False])
    arrays["pose23_world_m"] = np.full((1, 23, 3), np.nan, dtype=np.float32)
    arrays["pose23_ground_first_world_m"] = np.zeros((1, 23, 3), dtype=np.float32)
    np.savez_compressed(output_path, **arrays)
    with pytest.raises(ValueError, match="partial finite output"):
        evaluate_model_only_output(manifest_path=manifest_path, output_path=output_path)


def test_model_only_report_exposes_uncertainty_and_missingness_contract(tmp_path: Path):
    manifest_path = _write_fixture(tmp_path)
    output_path = _write_output(tmp_path, manifest_path=manifest_path)
    report = evaluate_model_only_output(manifest_path=manifest_path, output_path=output_path)
    assert report["uncertainty"]["status"] == "NOT_CALIBRATED"
    assert report["uncertainty"]["scope"] == "SENSITIVITY_ONLY"
    assert report["missingness_summary"]["semantics"]["DEGRADED"]
