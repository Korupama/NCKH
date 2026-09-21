from __future__ import annotations

import json
from pathlib import Path

import pytest

from stage2_entities.consolidation import consolidate_human_detections, box_iou
from stage2_entities.contracts import ReplayContext
from stage2_entities.tracking import build_entity_track_state
from stage2_entities.benchmark.metrics_detection import (
    gt_duplicate_diagnostics,
    evaluate_selected_frame,
)
from stage2_entities.benchmark.metrics_tracking import evaluate_tracking_window


def _det(did, lid, label, score, box):
    return {"detection_id": did, "label_id": lid, "label": label, "score": score, "bbox_xyxy": box}


def test_geometry_aware_cross_class_consolidation_merges_moderate_iou_same_person():
    # IoU is deliberately below the direct .82 gate, but geometry is highly compatible.
    a = _det("player_001", 2, "Player", .93, [100, 100, 150, 250])
    b = _det("goalkeeper_001", 3, "Goalkeeper", .72, [106, 105, 156, 255])
    assert 0.60 < box_iou(a["bbox_xyxy"], b["bbox_xyxy"]) < 0.82
    humans = consolidate_human_detections([a, b])
    assert len(humans) == 1
    h = humans[0]
    assert set(h.source_detection_ids) == {"player_001", "goalkeeper_001"}
    assert h.superclass == "footballer"
    assert h.candidate_hint is True
    assert h.consolidation_diagnostics["merge_edges"][0]["reason"] in {"geometry_consistent", "high_iou"}


def test_geometry_aware_consolidation_does_not_merge_unrelated_people():
    a = _det("player_001", 2, "Player", .95, [100, 100, 150, 250])
    b = _det("goalkeeper_001", 3, "Goalkeeper", .90, [130, 100, 180, 250])
    # Some overlap, but center displacement is too large for the geometry rescue rule.
    assert box_iou(a["bbox_xyxy"], b["bbox_xyxy"]) < 0.60
    humans = consolidate_human_detections([a, b])
    assert len(humans) == 2


def _human(hid: str, x: float, score: float, *, rescue=False):
    return {
        "physical_human_id": hid,
        "representative_detection_id": hid,
        "representative_label_id": 2,
        "representative_label": "Player",
        "bbox_xyxy": [x, 40, x + 20, 100],
        "detector_score": score,
        "source_detection_ids": [hid],
        "class_evidence": {"Player": score},
        "role_evidence": {"player": score, "goalkeeper": 0, "referee": 0, "other": 0},
        "resolved_role": "player",
        "role_score": 1.0,
        "role_margin": 1.0,
        "role_status": "VALID",
        "superclass": "footballer",
        "candidate_hint": True,
        "pose_required": not rescue,
        "observation_tier": "rescue" if rescue else "primary",
        "rescue_only": rescue,
    }


def _write_rescue_manifest(tmp_path: Path):
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    records = []
    # t0=2 is primary. Frames 1 and 3 intentionally have only low-score rescue boxes.
    for fi in range(5):
        primary = [] if fi in {1, 3} else [_human(f"p{fi}", 100 + fi, .9)]
        rescue = [_human(f"r{fi}", 100 + fi, .28, rescue=True)] if fi in {1, 3} else []
        payload = {
            "frame_index": fi,
            "image_path": "synthetic",
            "image_width": 320,
            "image_height": 180,
            "raw_detections": [],
            "raw_low_score_detections": [],
            "humans": primary,
            "rescue_humans": rescue,
            "pose_cache": {},
            "ball_detections": [],
            "timing_seconds": {},
            "provenance": {},
        }
        path = frame_dir / f"f{fi}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        records.append({"frame_index": fi, "perception_json": str(path)})
    return {"schema_version": "test", "frames": records}


def _ctx():
    ctx = ReplayContext(
        schema_version="test", video_path="x.mp4", video_id="x", fps=5,
        frame_count=5, image_width=320, image_height=180, selected_frame=2,
        window_start=0, window_end=4, shot_start=0, shot_end=4,
    )
    ctx.validate()
    return ctx


def test_temporal_rescue_fills_dropout_but_never_creates_anchor(tmp_path: Path):
    manifest = _write_rescue_manifest(tmp_path)
    with_rescue = build_entity_track_state(manifest, _ctx(), tmp_path / "with", use_temporal_rescue=True)
    without = build_entity_track_state(manifest, _ctx(), tmp_path / "without", use_temporal_rescue=False)
    assert len(with_rescue.tracks) == 1
    assert len(without.tracks) == 1
    assert with_rescue.tracks[0].observed_frames == 5
    assert without.tracks[0].observed_frames == 3
    assert with_rescue.tracks[0].diagnostics["rescue_observation_count"] == 2

    # If t0 contains only a rescue hypothesis, no Stage-2 entity may be created.
    t0_path = Path(manifest["frames"][2]["perception_json"])
    payload = json.loads(t0_path.read_text())
    payload["rescue_humans"] = [_human("rescue_at_t0", 102, .25, rescue=True)]
    payload["humans"] = []
    t0_path.write_text(json.dumps(payload))
    no_anchor = build_entity_track_state(manifest, _ctx(), tmp_path / "no_anchor", use_temporal_rescue=True)
    assert len(no_anchor.tracks) == 0


def test_duplicate_metrics_are_apples_to_apples_after_merge():
    gt = [{"bbox_xyxy": [100,100,150,250], "role":"player"}]
    raw = [
        _det("player_001",2,"Player",.9,[100,100,150,250]),
        _det("goalkeeper_001",3,"Goalkeeper",.8,[103,102,153,252]),
    ]
    merged = [h.to_dict() for h in consolidate_human_detections(raw)]
    before = gt_duplicate_diagnostics(raw, gt, consolidated=False)
    after = gt_duplicate_diagnostics(merged, gt, consolidated=True)
    assert before["AnyDuplicateRate"] == 1.0
    assert before["CrossClassConflictRate"] == 1.0
    assert after["AnyDuplicateRate"] == 0.0
    assert after["CrossClassConflictRate"] == 0.0


def test_candidate_precision_is_candidate_only():
    gt = [
        {"bbox_xyxy":[0,0,10,20], "role":"player"},
        {"bbox_xyxy":[30,0,40,20], "role":"referee"},
    ]
    pred = [
        {"bbox_xyxy":[0,0,10,20], "resolved_role":"player", "candidate_hint":True},
        {"bbox_xyxy":[30,0,40,20], "resolved_role":"player", "candidate_hint":True},
    ]
    m = evaluate_selected_frame(pred, gt)
    assert m["CandidateRecall"] == 1.0
    assert m["CandidatePrecision"] == pytest.approx(0.5)
    assert m["RefereeLeakageRate"] == 1.0


def test_tcr_decomposes_anchor_coverage_and_conditional_continuity():
    frames = [0,1,2]
    gt=[]; pred=[]
    for fi in frames:
        gt += [
            {"frame_index":fi,"track_id":"g1","bbox_xyxy":[10,10,30,50]},
            {"frame_index":fi,"track_id":"g2","bbox_xyxy":[100,10,120,50]},
        ]
        pred.append({"frame_index":fi,"track_id":"p1","bbox_xyxy":[10,10,30,50]})
    m = evaluate_tracking_window(gt,pred,frames,target_frame=1)
    assert m["AnchorCoverage"] == pytest.approx(0.5)
    assert m["ConditionalTCR"] == pytest.approx(1.0)
    assert m["TCR"] == pytest.approx(0.5)
