import json
from pathlib import Path

import numpy as np

from stage4_metric3d.comparison import (
    PITCH_WORLD_METRIC,
    ROOT_RELATIVE_METRIC,
    SCALE_AMBIGUOUS,
    compare_pipeline_outputs,
    convert_external_prediction,
    evaluate_system_manifest,
    load_prediction,
)
from stage4_metric3d.wholebody import POSE23_NAMES


def _write(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    return path


def _pose(offset_x: float = 0.0) -> np.ndarray:
    pose = np.zeros((23, 3), dtype=float)
    pose[:, 0] = np.linspace(1.0, 2.1, 23) + offset_x
    pose[:, 1] = np.linspace(-0.2, 0.2, 23)
    pose[:, 2] = np.linspace(1.8, 0.0, 23)
    return pose


def _native_state(path: Path, pose: np.ndarray, *, status: str, version: str) -> Path:
    joints = [
        {"index": index, "name": name, "xyz_world_m": pose[index].tolist()}
        for index, name in enumerate(POSE23_NAMES)
    ]
    return _write(path, {
        "schema_version": "metric-pose-3d-state-1.0",
        "stage4_version": version,
        "tracks": [{
            "track_id": "track_001",
            "selected_frame_pose_status": status,
            "observations": [{"frame_index": 86, "metric_pose23": joints}],
        }],
    })


def _canonical(path: Path, pose: np.ndarray, scope: str) -> Path:
    return _write(path, {
        "schema_version": "metric-pose23-prediction-1.0",
        "coordinate_scope": scope,
        "poses": [{
            "frame_index": 86,
            "track_id": "track_001",
            "xyz23_m": pose.tolist(),
        }],
    })


def _gt(path: Path, pose: np.ndarray) -> Path:
    return _write(path, {
        "schema_version": "metric-pose23-gt-1.0",
        "poses": [{
            "frame_index": 86,
            "track_id": "track_001",
            "xyz23_world_m": pose.tolist(),
        }],
    })


def test_pipeline_regression_reports_coordinate_drift_and_status_separately(tmp_path):
    baseline = _native_state(tmp_path / "baseline.json", _pose(), status="REJECTED", version="v0.1")
    candidate = _native_state(tmp_path / "candidate.json", _pose(), status="MISSING", version="v0.2")
    result = compare_pipeline_outputs(
        candidate,
        baseline,
        max_abs_delta_m=0.0,
        require_no_finite_loss=True,
    )
    assert result["coordinate_drift"]["max_euclidean_m"] == 0.0
    assert result["selected_frame_status_transitions"]["counts"] == {"REJECTED->MISSING": 1}
    assert result["regression_gate"]["status"] == "PASS"


def test_system_comparison_blocks_global_metrics_for_root_relative_model(tmp_path):
    gt_pose = _pose()
    stage4 = _canonical(tmp_path / "stage4.json", gt_pose + np.asarray([0.1, 0.0, 0.0]), PITCH_WORLD_METRIC)
    relative = gt_pose - np.mean(gt_pose[[11, 12]], axis=0)
    third_party = _canonical(tmp_path / "third_party.json", relative, ROOT_RELATIVE_METRIC)
    gt = _gt(tmp_path / "gt.json", gt_pose)
    manifest = _write(tmp_path / "systems.json", {
        "schema_version": "stage4-benchmark-systems-1.0",
        "gt": gt.name,
        "primary_system": "stage4",
        "bootstrap": {"samples": 32, "seed": 7},
        "systems": {
            "stage4": {"pred": stage4.name, "role": "candidate"},
            "external": {"pred": third_party.name, "role": "third_party"},
        },
    })
    result = evaluate_system_manifest(manifest)
    external_metrics = result["systems_common_cohort"]["external"]["metrics"]
    assert external_metrics["GlobalMPJPE_m"] is None
    assert external_metrics["GALE_m"] is None
    assert np.isclose(external_metrics["RootAlignedMPJPE_m"], 0.0)
    comparison = result["comparisons"]["stage4__vs__external"]
    assert comparison["metrics"]["GlobalMPJPE_m"]["count"] == 0
    assert comparison["metrics"]["RootAlignedMPJPE_m"]["count"] == 1


def test_external_adapter_requires_explicit_scope_and_joint_mapping(tmp_path):
    input_path = _write(tmp_path / "external.json", {
        "poses": [{
            "frame_index": 86,
            "track_id": "track_001",
            "xyz": [[1000.0, 2000.0, 3000.0], [4000.0, 5000.0, 6000.0]],
        }],
    })
    adapter_path = _write(tmp_path / "adapter.json", {
        "schema_version": "stage4-external-pose-adapter-1.0",
        "coordinate_scope": ROOT_RELATIVE_METRIC,
        "units": "mm",
        "source_joint_names": ["head", "pelvis"],
        "joint_map": {"nose": "head", "left_hip": "pelvis"},
        "model": {"name": "example"},
    })
    output_path = tmp_path / "canonical.json"
    converted = convert_external_prediction(input_path, adapter_path, output_path)
    assert converted["poses"][0]["xyz23_m"][0] == [1.0, 2.0, 3.0]
    assert converted["poses"][0]["xyz23_m"][11] == [4.0, 5.0, 6.0]
    assert converted["poses"][0]["xyz23_m"][1] == [None, None, None]
    bundle = load_prediction(output_path)
    assert bundle.coordinate_scope == ROOT_RELATIVE_METRIC


def test_external_adapter_accepts_model_units_only_for_scale_ambiguous(tmp_path):
    input_path = _write(tmp_path / "external.json", {
        "poses": [{"frame_index": 86, "track_id": "track_001", "xyz": [[2.0, 3.0, 4.0]]}],
    })
    adapter_path = _write(tmp_path / "adapter.json", {
        "coordinate_scope": SCALE_AMBIGUOUS,
        "units": "model",
        "source_joint_names": ["head"],
        "joint_map": {"nose": "head"},
    })
    converted = convert_external_prediction(input_path, adapter_path, tmp_path / "canonical.json")
    assert converted["poses"][0]["xyz23_m"][0] == [2.0, 3.0, 4.0]
