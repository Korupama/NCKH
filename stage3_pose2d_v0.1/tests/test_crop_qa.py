from __future__ import annotations

import numpy as np

from stage3_pose2d.crop_qa import analyze_crop
from stage3_pose2d.quality import evaluate_pose
from stage3_pose2d.schemas import Stage3Config
from stage3_pose2d.wholebody133 import keypoint_records_to_arrays
from conftest import make_pose_records


def test_crop_geometry_is_ok_for_normal_player_box():
    qa = analyze_crop([100, 100, 200, 300], [1280, 720])
    assert qa["crop_status"] == "OK"
    assert len(qa["affine_matrix_source_to_model"]) == 2
    assert qa["visible_person_area_method"] == "clipped_stage2_bbox_area_proxy"


def test_crop_geometry_marks_border_truncation():
    qa = analyze_crop([0, 0, 500, 500], [500, 500])
    assert qa["crop_status"] == "SEVERE_BORDER_TRUNCATION"
    assert "crop_border_truncation_high" in qa["status_reasons"]


def test_crop_geometry_marks_overlap_and_pose_qa_downgrades_valid():
    records = make_pose_records(bbox=(100, 100, 200, 300), score=4.0)
    xy, scores = keypoint_records_to_arrays(records)
    diagnostics = analyze_crop(
        [100, 100, 200, 300],
        [1280, 720],
        neighbors=[{"track_id": "track_002", "bbox_xyxy": [145, 100, 245, 300]}],
        pose_xy=xy,
        pose_scores=scores,
    )
    assert diagnostics["crop_status"] == "HIGH_OVERLAP"
    qa, _ = evaluate_pose(xy, scores, [100, 100, 200, 300], Stage3Config(), crop_diagnostics=diagnostics)
    assert qa["pose_status"] == "DEGRADED"
    assert "crop_status_high_overlap" in qa["status_reasons"]


def test_cross_person_status_rejects_pose_when_body_is_inside_neighbor():
    records = make_pose_records(bbox=(100, 100, 200, 300), score=4.0)
    xy, scores = keypoint_records_to_arrays(records)
    xy[0:23, 0] += 150.0
    diagnostics = analyze_crop(
        [100, 100, 200, 300],
        [1280, 720],
        neighbors=[{"track_id": "track_002", "bbox_xyxy": [200, 100, 350, 300]}],
        pose_xy=xy,
        pose_scores=scores,
    )
    assert diagnostics["crop_status"] == "LIKELY_CROSS_PERSON"
    qa, _ = evaluate_pose(xy, scores, [100, 100, 200, 300], Stage3Config(), crop_diagnostics=diagnostics)
    assert qa["pose_status"] == "REJECTED"
    assert "likely_cross_person" in qa["status_reasons"]


def test_pose_ownership_support_rejects_pose_displaced_outside_source_bbox():
    records = make_pose_records(bbox=(100, 100, 200, 300), score=4.0)
    xy, scores = keypoint_records_to_arrays(records)
    xy[:23, 0] += 100.0
    diagnostics = analyze_crop(
        [100, 100, 200, 300],
        [1280, 720],
        pose_xy=xy,
        pose_scores=scores,
    )
    assert diagnostics["ownership_status"] == "OUTSIDE_SOURCE"
    qa, _ = evaluate_pose(xy, scores, [100, 100, 200, 300], Stage3Config(), crop_diagnostics=diagnostics)
    assert qa["pose_status"] == "REJECTED"
    assert "body17_support_below_reject_floor" in qa["status_reasons"]


def test_pose_ownership_records_center_evidence_for_supported_pose():
    records = make_pose_records(bbox=(100, 100, 200, 300), score=4.0)
    xy, scores = keypoint_records_to_arrays(records)
    diagnostics = analyze_crop(
        [100, 100, 200, 300],
        [1280, 720],
        pose_xy=xy,
        pose_scores=scores,
    )
    assert diagnostics["ownership_status"] == "SUPPORTED"
    assert diagnostics["body17_center_to_source_center_normalized"] is not None
