from __future__ import annotations

import json
from pathlib import Path

import pytest

from stage2_entities.consolidation import consolidate_human_detections, box_iou
from stage2_entities.contracts import ReplayContext
from stage2_entities.legacy_adapter import build_manifest_from_legacy_jsons
from stage2_entities.tracking import build_entity_track_state

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples" / "legacy_validation"


def _load_frame85():
    p = SAMPLES / "clip_000_source_00000085_sst_pose.json"
    return json.loads(p.read_text(encoding="utf-8"))


def test_cross_class_physical_human_consolidation_real_legacy_frame():
    payload = _load_frame85()
    raw = payload["detections"]
    by_id = {d["detection_id"]: d for d in raw}

    assert box_iou(by_id["player_009"]["bbox_xyxy"], by_id["main_referee_001"]["bbox_xyxy"]) > 0.91
    assert box_iou(by_id["player_004"]["bbox_xyxy"], by_id["goalkeeper_002"]["bbox_xyxy"]) > 0.90

    humans = consolidate_human_detections(raw, cross_class_iou_threshold=0.85, ambiguous_margin=0.05)
    raw_humans = [d for d in raw if int(d["label_id"]) in {2, 3, 4, 5, 6}]

    # The supplied v0.4 frame contains 13 human-class detections. Two high-IoU
    # cross-class pairs collapse, yielding 11 physical humans and 10 offside candidates.
    assert len(raw_humans) == 13
    assert len(humans) == 11
    assert sum(h.resolved_role in {"player", "goalkeeper"} for h in humans) == 10

    groups = {frozenset(h.source_detection_ids): h for h in humans}
    ref_group = groups[frozenset({"player_009", "main_referee_001"})]
    assert ref_group.resolved_role == "referee"
    assert ref_group.pose_required is False

    player_group = groups[frozenset({"player_004", "goalkeeper_002"})]
    assert player_group.resolved_role == "player"
    assert player_group.pose_required is True


def test_legacy_real_three_frame_sequence_migrates_to_entity_track_state(tmp_path: Path):
    paths = sorted(SAMPLES.glob("*.json"))
    manifest = build_manifest_from_legacy_jsons(paths, tmp_path / "perception")
    ctx = ReplayContext(
        schema_version="1.0-test",
        video_path="/nonexistent/football_scene.mp4",
        video_id="legacy-real-three-frame",
        fps=25.0,
        frame_count=630,
        image_width=1280,
        image_height=720,
        selected_frame=86,
        window_start=85,
        window_end=87,
        shot_start=0,
        shot_end=629,
    )
    ctx.validate()

    state = build_entity_track_state(manifest, ctx, tmp_path / "stage2")
    assert state.status == "VALID"
    assert len(state.tracks) == 10
    assert sum(t.candidate_for_stage3 for t in state.tracks) == 10
    assert state.diagnostics["raw_human_detections"] == 33
    assert state.diagnostics["consolidated_human_hypotheses"] == 31
    assert state.diagnostics["cross_class_duplicate_groups_collapsed"] == 2
    assert state.stage3_handoff["stage3_input_ready"] is True
    assert state.stage3_handoff["candidate_tracks_missing_rtmw_at_selected_frame"] == []

    # One extra border player in frame 85 is intentionally orphaned because tracks are
    # anchored to the 10 humans visible at the user-selected frame 86.
    assert len(state.diagnostics["orphan_observations_by_frame"][85]) == 1

    cache_path = Path(state.artifacts["rtmw_track_cache"])
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    assert cache["score_semantics"] == "RTMW_RAW_SIMCC_MAX_NOT_CALIBRATED_PROBABILITY"
    assert set(cache["tracks"]) == {t.track_id for t in state.tracks}
    assert all(key.startswith("track_") for key in cache["tracks"])
    assert all(
        "frame_index" in obs and "pose" in obs
        for track in cache["tracks"].values()
        for obs in track["observations"]
    )


def test_replay_context_rejects_non_raw_coordinate_space():
    ctx = ReplayContext(
        schema_version="bad",
        video_path="x.mp4",
        video_id="x",
        fps=25.0,
        frame_count=100,
        image_width=1280,
        image_height=720,
        selected_frame=50,
        window_start=40,
        window_end=60,
        shot_start=0,
        shot_end=99,
        coordinate_space="RESIZED_640",
    )
    with pytest.raises(ValueError):
        ctx.validate()


def test_production_perception_consolidates_before_rtmw(monkeypatch, tmp_path: Path):
    import cv2
    import numpy as np
    import stage2_entities.perception as perception
    from stage2_entities.backend_core.sst_rtmw_pose_pipeline import Detection, PoseResult

    image_path = tmp_path / "frame_000000085.png"
    cv2.imwrite(str(image_path), np.zeros((360, 640, 3), dtype=np.uint8))

    detections = [
        Detection("player_001", 2, "Player", 0.91, (100, 100, 150, 250)),
        Detection("main_referee_001", 4, "Main referee", 0.99, (101, 101, 151, 251)),
        Detection("player_002", 2, "Player", 0.98, (300, 100, 350, 250)),
        Detection("ball_001", 1, "Ball", 0.95, (400, 250, 410, 260)),
    ]

    monkeypatch.setattr(perception, "predict_frame", lambda *args, **kwargs: {"dummy": True})
    monkeypatch.setattr(perception, "postprocess_sst_prediction", lambda *args, **kwargs: detections)

    class FakePoseEstimator:
        name = "fake_rtmw"
        input_width = 288
        input_height = 384
        bbox_padding = 1.25
        device = "cpu"
        def __init__(self):
            self.seen_boxes = []
        def infer(self, image_bgr, boxes):
            self.seen_boxes = list(boxes)
            result = []
            for box in boxes:
                xy = np.zeros((133, 2), dtype=np.float32)
                scores = np.full((133,), 4.0, dtype=np.float32)
                result.append(PoseResult(xy, scores, self.name, (288, 384), tuple(box)))
            return result

    backend = perception.SSTRTMWStage2Backend.__new__(perception.SSTRTMWStage2Backend)
    backend.sst_device = "cpu"
    backend.sst_model = object()
    backend.sst_metadata = {}
    backend.sst_load_seconds = 0.0
    backend.pose_estimator = FakePoseEstimator()
    backend.class_thresholds = {1: .35, 2: .5, 3: .5, 4: .55, 5: .55, 6: .6}
    backend.person_nms_iou = .65
    backend.ball_nms_iou = .30
    backend.ball_top_k = 1
    backend.cross_class_iou_threshold = .85
    backend.role_ambiguous_margin = .05
    backend.person_box_expansion = 1.0
    backend.keypoint_threshold = 1.0
    backend.sst_checkpoint = "fake.pth"
    backend.rtmw_model = "fake.onnx"

    frame = backend.process_frame(image_path, 85)
    assert len(frame.humans) == 2
    by_role = {h["resolved_role"]: h for h in frame.humans}
    assert set(by_role) == {"referee", "player"}
    assert by_role["referee"]["pose_required"] is False
    assert by_role["player"]["pose_required"] is True
    # The duplicate Player+Referee physical person must be consolidated/excluded
    # before RTMW, so only the true player box reaches the pose backend.
    assert len(backend.pose_estimator.seen_boxes) == 1
    assert len(frame.pose_cache) == 1
    assert len(frame.ball_detections) == 1
