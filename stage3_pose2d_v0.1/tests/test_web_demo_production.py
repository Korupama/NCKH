from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from web_demo import production
from conftest import write_stage2_fixture


def _read_frame(_input_path, frame_index):
    return np.zeros((1080, 1920, 3), dtype=np.uint8), {
        "frame_index": int(frame_index),
        "fps": 25.0,
        "timestamp_seconds": int(frame_index) / 25.0,
    }


def _options(stage2_dir: Path, model_path: Path, **extra):
    options = {
        "stage2_dir": str(stage2_dir),
        "rtmw_model": str(model_path),
        "device": "cpu",
        "confirm_stage2_match": True,
    }
    options.update(extra)
    return options


def test_production_requires_existing_handoff(tmp_path):
    with pytest.raises(FileNotFoundError, match="MISSING_INPUT"):
        production.run_existing_stage2_frame(
            tmp_path / "video.mp4",
            tmp_path / "out",
            10,
            _options(tmp_path / "missing", tmp_path / "model.onnx"),
            _read_frame,
            tmp_path / "model.onnx",
        )


def test_production_requires_confirmation(stage2_fixture, tmp_path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"test-model")
    with pytest.raises(ValueError, match="MISSING_CONFIRMATION"):
        production.run_existing_stage2_frame(
            tmp_path / "video.mp4",
            tmp_path / "out",
            10,
            _options(stage2_fixture, model, confirm_stage2_match=False),
            _read_frame,
            model,
        )


def test_production_rejects_frame_mismatch(stage2_fixture, tmp_path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"test-model")
    with pytest.raises(ValueError, match="FRAME_MISMATCH"):
        production.run_existing_stage2_frame(
            tmp_path / "video.mp4",
            tmp_path / "out",
            11,
            _options(stage2_fixture, model),
            _read_frame,
            model,
        )


def test_production_rejects_video_mismatch(stage2_fixture, tmp_path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"test-model")

    def wrong_size(_input_path, frame_index):
        return np.zeros((720, 1280, 3), dtype=np.uint8), {
            "frame_index": int(frame_index),
            "fps": 30.0,
            "timestamp_seconds": int(frame_index) / 30.0,
        }

    with pytest.raises(ValueError, match="VIDEO_MISMATCH"):
        production.run_existing_stage2_frame(
            tmp_path / "video.mp4",
            tmp_path / "out",
            10,
            _options(stage2_fixture, model),
            wrong_size,
            model,
        )


def test_production_success_is_read_only_and_uses_stage3(monkeypatch, stage2_fixture, tmp_path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"test-model")
    before = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in stage2_fixture.iterdir()
        if path.is_file()
    }
    fake_bundle = production.load_stage2_bundle(stage2_dir=stage2_fixture, strict=True)
    monkeypatch.setattr(production, "load_stage2_bundle", lambda **_: fake_bundle)
    monkeypatch.setattr(
        production,
        "run_stage3",
        lambda **_: {
            "stage3_version": "test-stage3",
            "source_stage2": {"stage2_version": "test-stage2"},
            "metrics": {
                "candidate_tracks": 1,
                "selected_frame_status_counts": {"VALID": 1},
                "PoseCoverageAtT0_given_stage2_candidate": 1.0,
                "ValidPoseCoverageAtT0_given_stage2_candidate": 1.0,
            },
            "selected_frame_poses": [{
                "track_id": "track_001",
                "upstream_role": "player",
                "observation": {
                    "source_bbox_xyxy": [100, 100, 200, 300],
                    "pose_status": "VALID",
                    "qa": {},
                    "keypoints_133": [],
                },
            }],
            "artifacts": {},
        },
    )
    monkeypatch.setattr(production, "draw_pose", lambda frame, *_: frame)
    video = tmp_path / "video.mp4"
    out = tmp_path / "out"
    summary = production.run_existing_stage2_frame(
        video,
        out,
        10,
        _options(stage2_fixture, model),
        _read_frame,
        model,
    )
    assert summary["backend"]["pipeline_mode"] == "production_stage2_stage3"
    assert summary["metrics"]["person_boxes"] == 1
    assert (out / "frame_pose_overlay.jpg").is_file()
    after = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in stage2_fixture.iterdir()
        if path.is_file()
    }
    assert before == after
