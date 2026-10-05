from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from stage4_metric3d.backends.sam3d_pitch_refined.cache import MHR70_NAMES
from stage4_metric3d.model_only import load_model_only_manifest, sha256_file
from stage4_metric3d.model_only_systems import compare_model_only_systems
from stage4_metric3d.wholebody import POSE23_NAMES


def _fixture_manifest(root: Path) -> Path:
    assert cv2.imwrite(str(root / "frame.png"), np.zeros((120, 200, 3), dtype=np.uint8))
    camera = {
        "schema_version": "benchmark-camera-1.0", "coordinate_frame": "BENCHMARK_PITCH_WORLD",
        "frame_index": 0, "status": "VALID", "image": {"width": 200, "height": 120},
        "intrinsics": {"K": [[100.0, 0.0, 100.0], [0.0, 100.0, 60.0], [0.0, 0.0, 1.0]]},
        "extrinsics": {"R_world_to_camera": np.eye(3).tolist(), "camera_center_world_m": [0.0, 0.0, -10.0]},
        "distortion": {"radial": [0, 0, 0, 0, 0, 0], "tangential": [0, 0]},
        "pitch": {"origin": "center", "length_m": 105.0, "width_m": 68.0},
    }
    (root / "camera.json").write_text(json.dumps(camera), encoding="utf-8")
    record_id = "seq_a_frame_0000_person_00"
    (root / "gt.json").write_text(json.dumps({
        "schema_version": "metric-pose23-gt-1.0",
        "poses": [{"record_id": record_id, "joint_names": list(POSE23_NAMES),
                    "xyz23_world_m": [[1.0, 2.0, 0.1] for _ in range(23)], "visible23": [True] * 23}],
    }), encoding="utf-8")
    manifest = {
        "schema_version": "stage4-model-only-benchmark-manifest-1.0",
        "dataset": {"name": "licensed-fixture", "release": "fixture-1.0", "license_id": "TEST-ONLY", "license_url": "https://example.invalid/license"},
        "split": "development",
        "coordinate_frame": {"name": "BENCHMARK_PITCH_WORLD", "origin": "pitch_center", "scope": "PITCH_WORLD_METRIC", "units": "m", "axes": {"x": "goal-to-goal", "y": "touchline-to-touchline", "z": "up"}},
        "records": [{"record_id": record_id, "sequence_id": "seq_a", "split": "development", "image": "frame.png", "bbox_xyxy": [10, 10, 100, 110], "camera": "camera.json", "gt": "gt.json"}],
    }
    path = root / "benchmark.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _fixture_output(root: Path, manifest_path: Path) -> Path:
    manifest = load_model_only_manifest(manifest_path)
    record = manifest.records[0]
    xyz = record.gt_xyz_world_m.astype(np.float32)
    metadata = {
        "schema_version": "stage4-model-only-sam3d-output-1.0", "manifest_sha256": manifest.sha256,
        "checkpoint": "checkpoint", "checkpoint_sha256": "checkpoint-hash", "mhr_path": "mhr",
        "mhr_model_sha256": "mhr-hash", "record_input_hashes": {record.record_id: record.input_hashes},
        "ground_first_refinement": [], "stage1_stage3_inputs": None,
    }
    output = root / "sam3d.npz"
    np.savez_compressed(
        output, schema_version=np.array("stage4-model-only-sam3d-output-1.0"),
        record_ids=np.asarray([record.record_id]), sequence_ids=np.asarray([record.sequence_id]),
        bboxes_xyxy=np.asarray([record.bbox_xyxy], dtype=np.float32), skel_2d_px=np.full((1, 70, 2), [100, 60], dtype=np.float32),
        skel_3d_relative_m=np.zeros((1, 70, 3), dtype=np.float32), pred_cam_t_m=np.asarray([[0, 0, 1]], dtype=np.float32),
        focal_length_px=np.asarray([100], dtype=np.float32), pose23_world_m=xyz[None], pose23_ground_first_world_m=xyz[None],
        valid_mask=np.asarray([True]), joint_names=np.asarray(MHR70_NAMES), metadata_json=np.array(json.dumps(metadata)),
    )
    return output


def _systems_manifest(root: Path, benchmark: Path, output: Path, *, checkpoint_hash: str = "checkpoint-hash") -> Path:
    data = {
        "schema_version": "stage4-model-only-systems-1.0",
        "benchmark_manifest": benchmark.name,
        "benchmark_manifest_sha256": load_model_only_manifest(benchmark).sha256,
        "systems": [
            {"name": "sam3d_direct", "backend": "SAM3D-Body", "variant": "direct", "coordinate_scope": "PITCH_WORLD_METRIC", "output": output.name, "checkpoint_sha256": checkpoint_hash, "repository_revision": "rev", "config_sha256": "cfg"},
            {"name": "sam3d_ground_first", "backend": "SAM3D-Body", "variant": "ground_first", "coordinate_scope": "PITCH_WORLD_METRIC", "output": output.name, "checkpoint_sha256": checkpoint_hash, "repository_revision": "rev", "config_sha256": "cfg"},
        ],
    }
    path = root / "systems.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_model_only_system_comparison_uses_same_cohort(tmp_path: Path):
    benchmark = _fixture_manifest(tmp_path)
    output = _fixture_output(tmp_path, benchmark)
    systems = _systems_manifest(tmp_path, benchmark, output)
    report = compare_model_only_systems(systems_manifest_path=systems)
    assert report["metric_computation_completed"] is True
    assert report["accuracy_claim_allowed"] is False
    assert len(report["systems"]) == 2
    assert report["systems"][0]["coverage"]["evaluated_record_count"] == 1
    assert report["pairwise"][0]["paired_record_count"] == 1


def test_model_only_system_comparison_rejects_checkpoint_hash_mismatch(tmp_path: Path):
    benchmark = _fixture_manifest(tmp_path)
    output = _fixture_output(tmp_path, benchmark)
    systems = _systems_manifest(tmp_path, benchmark, output, checkpoint_hash="wrong")
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        compare_model_only_systems(systems_manifest_path=systems)


def test_model_only_system_comparison_writes_report(tmp_path: Path):
    benchmark = _fixture_manifest(tmp_path)
    output = _fixture_output(tmp_path, benchmark)
    systems = _systems_manifest(tmp_path, benchmark, output)
    report_path = tmp_path / "systems-report.json"
    compare_model_only_systems(systems_manifest_path=systems, report_path=report_path)
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["systems_manifest_sha256"] == sha256_file(systems)
