from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from stage4_metric3d.model_only import MODEL_ONLY_MANIFEST_SCHEMA, load_model_only_manifest
from stage4_metric3d.wholebody import POSE23_NAMES


def _write_fixture(root: Path, *, image_name: str = "frame.png", bbox=None, camera_overrides=None, gt_overrides=None) -> Path:
    image = np.zeros((120, 200, 3), dtype=np.uint8)
    assert cv2.imwrite(str(root / image_name), image)
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
    camera.update(camera_overrides or {})
    (root / "camera.json").write_text(json.dumps(camera), encoding="utf-8")
    gt = {
        "schema_version": "metric-pose23-gt-1.0",
        "poses": [{
            "record_id": "seq_a_frame_0000_person_00",
            "joint_names": list(POSE23_NAMES),
            "xyz23_world_m": [[1.0, 2.0, 0.1] for _ in range(23)],
            "visible23": [True] * 23,
        }],
    }
    gt.update(gt_overrides or {})
    (root / "gt.json").write_text(json.dumps(gt), encoding="utf-8")
    manifest = {
        "schema_version": MODEL_ONLY_MANIFEST_SCHEMA,
        "dataset": {
            "name": "licensed-fixture",
            "release": "fixture-1.0",
            "license_id": "TEST-ONLY",
            "license_url": "https://example.invalid/license",
        },
        "split": "development",
        "coordinate_frame": {
            "name": "BENCHMARK_PITCH_WORLD",
            "origin": "pitch_center",
            "scope": "PITCH_WORLD_METRIC",
            "units": "m",
            "axes": {"x": "goal-to-goal", "y": "touchline-to-touchline", "z": "up"},
        },
        "records": [{
            "record_id": "seq_a_frame_0000_person_00",
            "sequence_id": "seq_a",
            "split": "development",
            "image": image_name,
            "bbox_xyxy": [10, 10, 100, 110] if bbox is None else bbox,
            "camera": "camera.json",
            "gt": "gt.json",
        }],
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_valid_model_only_manifest_resolves_and_hashes_inputs(tmp_path: Path):
    manifest = load_model_only_manifest(_write_fixture(tmp_path))
    assert manifest.split == "development"
    assert manifest.sha256
    assert len(manifest.records) == 1
    record = manifest.records[0]
    assert record.image_width == 200 and record.image_height == 120
    assert record.gt_xyz_world_m.shape == (23, 3)
    assert set(record.input_hashes) == {"image_sha256", "camera_sha256", "gt_sha256"}


@pytest.mark.parametrize("field", ["image", "camera", "gt"])
def test_missing_required_asset_fails_closed(tmp_path: Path, field: str):
    manifest_path = _write_fixture(tmp_path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["records"][0][field] = "missing.file"
    manifest_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises((FileNotFoundError, ValueError)):
        load_model_only_manifest(manifest_path)


def test_invalid_bbox_fails_closed(tmp_path: Path):
    manifest_path = _write_fixture(tmp_path, bbox=[0, 0, 300, 50])
    with pytest.raises(ValueError, match="bbox_xyxy"):
        load_model_only_manifest(manifest_path)


def test_camera_dimension_mismatch_fails_closed(tmp_path: Path):
    manifest_path = _write_fixture(tmp_path, camera_overrides={"image": {"width": 201, "height": 120}})
    with pytest.raises(ValueError, match="dimension mismatch"):
        load_model_only_manifest(manifest_path)


def test_missing_pitch_dimensions_fails_closed(tmp_path: Path):
    manifest_path = _write_fixture(tmp_path, camera_overrides={"pitch": {"origin": "center"}})
    with pytest.raises(ValueError, match="pitch origin=center"):
        load_model_only_manifest(manifest_path)


def test_missing_visible_gt_joint_fails_closed(tmp_path: Path):
    gt_overrides = {"poses": [{
        "record_id": "seq_a_frame_0000_person_00",
        "joint_names": list(POSE23_NAMES),
        "xyz23_world_m": [[1.0, 2.0, 0.1] for _ in range(23)],
        "visible23": [True] * 22 + [False],
    }]}
    manifest_path = _write_fixture(tmp_path, gt_overrides=gt_overrides)
    load_model_only_manifest(manifest_path)


def test_stage_dependencies_are_rejected(tmp_path: Path):
    manifest_path = _write_fixture(tmp_path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["records"][0]["stage3_state"] = "not_allowed.json"
    manifest_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="Stage 1/3"):
        load_model_only_manifest(manifest_path)


def test_manifest_hash_is_deterministic(tmp_path: Path):
    manifest_path = _write_fixture(tmp_path)
    first = load_model_only_manifest(manifest_path).sha256
    second = load_model_only_manifest(manifest_path).sha256
    assert first == second


def test_unfilled_documentation_template_fails_closed(tmp_path: Path):
    manifest_path = _write_fixture(tmp_path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["dataset"]["release"] = "RECORD_THE_EXACT_RELEASE_VERSION"
    manifest_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="dataset.release"):
        load_model_only_manifest(manifest_path)


def test_model_only_modules_do_not_import_stage_adapters():
    root = Path(__file__).resolve().parents[1]
    for relative in ("stage4_metric3d/model_only.py", "stage4_metric3d/model_only_refinement.py", "run_sam3d_model_only.py"):
        source = (root / relative).read_text(encoding="utf-8")
        assert "stage3_adapter" not in source
        assert "load_stage3_state" not in source
        assert "stage1_camera" not in source.lower()
