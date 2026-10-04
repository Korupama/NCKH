from __future__ import annotations

import json
from pathlib import Path

from tools.stage4_phase0_audit import build_report


def test_phase0_audit_fails_closed_when_real_inputs_are_missing(tmp_path: Path):
    report = build_report(
        paths={
            "stage3_state": tmp_path / "missing_stage3.json",
            "stage1_camera_dir": tmp_path / "missing_cameras",
            "sam3d_cache": tmp_path / "missing_cache.npz",
        },
        tests_passed=48,
        tests_failed=0,
    )

    assert report["acceptance"]["implementation_gate"] == "PASS_IMPLEMENTATION"
    assert report["acceptance"]["integration_frame104"] == "NOT_AVAILABLE_INTEGRATION_INPUTS"
    assert report["acceptance"]["metric_accuracy"] == "NOT_EVALUATED"
    assert report["acceptance"]["accuracy_claim_allowed"] is False
    assert report["upstream_read_only"] is True


def test_phase0_audit_records_existing_output_as_sanity_only(tmp_path: Path):
    stage3 = tmp_path / "stage3.json"
    stage3.write_text("{}", encoding="utf-8")
    cameras = tmp_path / "cameras"
    cameras.mkdir()
    (cameras / "camera.json").write_text("{}", encoding="utf-8")
    cache = tmp_path / "cache.npz"
    cache.write_bytes(b"cache")
    refined = tmp_path / "refined.json"
    refined.write_text(json.dumps({
        "schema_version": "world-grounded-pose-state-1.1",
        "stage4_version": "stage4-sam3d-pitch-refined-0.5.1",
        "selected_frame": 104,
        "tracks": [],
        "quality_gates": {"research_accuracy_frozen": False},
    }), encoding="utf-8")
    direct = tmp_path / "direct.json"
    direct.write_text(refined.read_text(encoding="utf-8"), encoding="utf-8")
    report = build_report(
        paths={
            "stage3_state": stage3,
            "stage1_camera_dir": cameras,
            "sam3d_cache": cache,
            "refined_output": refined,
            "direct_output": direct,
        },
        tests_passed=48,
        tests_failed=0,
    )

    assert report["acceptance"]["integration_frame104"] == "ARTIFACTS_AVAILABLE"
    assert report["acceptance"]["self_consistency"] == "SANITY_ONLY"
    assert report["acceptance"]["metric_accuracy"] == "NOT_EVALUATED"
    assert report["outputs"]["sam3d_pitch_refined"]["summary"]["schema_version"] == "world-grounded-pose-state-1.1"


def test_phase0_manifest_paths_are_resolved_relative_to_manifest(tmp_path: Path):
    manifest = tmp_path / "inputs.json"
    stage3 = tmp_path / "stage3.json"
    stage3.write_text("{}", encoding="utf-8")
    manifest.write_text(json.dumps({
        "schema_version": "stage4-phase0-input-manifest-1.0",
        "paths": {"stage3_state": "stage3.json"},
    }), encoding="utf-8")

    from tools.stage4_phase0_audit import _load_manifest

    loaded = _load_manifest(manifest)
    assert loaded["stage3_state"] == stage3.resolve()


def test_phase0_separates_model_only_from_integration_blockers(tmp_path: Path):
    report = build_report(
        paths={
            "stage3_state": tmp_path / "missing_stage3.json",
            "stage1_camera_dir": tmp_path / "missing_cameras",
            "sam3d_cache": tmp_path / "missing_cache.npz",
        },
        tests_passed=50,
        tests_failed=0,
    )

    assert report["acceptance"]["replay_inputs_available"] is False
    assert report["acceptance"]["integration_cache_generation_inputs_available"] is False
    assert report["acceptance"]["model_only_development"] == "ALLOWED"
    assert report["acceptance"]["model_only_pretrained_inference"] == "BLOCKED_MISSING_STAGE4_MODEL_ASSETS"
    assert "sam3d_checkpoint" in report["acceptance"]["missing_model_only_keys"]
    assert "stage3_state" in report["acceptance"]["missing_integration_keys"]
    assert "stage3_state" in report["recovery"]["replay_cache"]["missing"]
    assert "source_video_or_frames_dir" in report["recovery"]["generate_integration_cache"]["missing"]
