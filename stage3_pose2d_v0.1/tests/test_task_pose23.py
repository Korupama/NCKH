import copy

import pytest

from stage3_pose2d.task_pose23 import (
    POSE23_NAMES,
    Pose23ValidationError,
    load_and_validate_manifest,
    pose23_manifest_template,
    validate_pose23_manifest,
    wholebody133_to_pose23,
)
from stage3_pose2d.wholebody133 import WHOLEBODY_KEYPOINT_NAMES


def _sample(sample_id="sample_001", split="train", split_group_id="match_001", frame_index=100):
    keypoints = []
    for index, name in enumerate(POSE23_NAMES):
        keypoints.append({
            "index": index,
            "name": name,
            "x": float(100 + index),
            "y": float(200 + index),
            "visibility": "VISIBLE",
            "annotation_confidence": 1.0,
            "review_status": "DOUBLE_REVIEWED",
        })
    return {
        "sample_id": sample_id,
        "case_id": "case_001",
        "video_id": "video_001",
        "match_id": "match_001",
        "sequence_id": "seq_001",
        "split_group_id": split_group_id,
        "split": split,
        "frame_index": frame_index,
        "track_id": "track_001",
        "image_path": "authorized/frames/frame_000100.jpg",
        "image_width": 1920,
        "image_height": 1080,
        "coordinate_space": "RAW_DISTORTED_PIXEL",
        "bbox_coordinate_space": "RAW_DISTORTED_PIXEL",
        "t0_selection": "USER_SELECTED",
        "bbox_xyxy": [90.0, 180.0, 180.0, 360.0],
        "annotation_review_status": "DOUBLE_REVIEWED",
        "tags": {
            "player_pixel_height": 180.0,
            "scale_bin": "LARGE",
            "occlusion_level": "NONE",
            "motion_blur": "NONE",
            "view": "SIDE",
            "border_truncated": False,
            "crowd_level": "ISOLATED",
        },
        "keypoints_23": keypoints,
    }


def test_empty_template_is_valid_only_when_explicitly_allowed():
    manifest = pose23_manifest_template()
    assert validate_pose23_manifest(manifest) == [
        "manifest.samples: empty manifest is only valid with allow_empty=True"
    ]
    assert validate_pose23_manifest(manifest, allow_empty=True) == []


def test_validator_rejects_cross_split_match_sequence_leakage():
    manifest = pose23_manifest_template()
    manifest["samples"] = [_sample(), _sample("sample_002", split="test", frame_index=101)]
    issues = validate_pose23_manifest(manifest)
    assert any("match/sequence" in issue for issue in issues)


def test_out_of_frame_point_has_no_coordinate():
    manifest = pose23_manifest_template()
    sample = _sample()
    point = sample["keypoints_23"][17]
    point.update({"x": None, "y": None, "visibility": "OUT_OF_FRAME"})
    manifest["samples"] = [sample]
    assert validate_pose23_manifest(manifest) == []

    invalid = copy.deepcopy(sample)
    invalid["keypoints_23"][17].update({"x": 123.0, "y": 456.0})
    manifest["samples"] = [invalid]
    assert any("x/y must be null" in issue for issue in validate_pose23_manifest(manifest))


def test_validator_rejects_duplicate_sample_identity_and_out_of_bounds_point():
    manifest = pose23_manifest_template()
    first = _sample()
    second = _sample("sample_002")
    second["keypoints_23"][0]["x"] = 1920.0
    manifest["samples"] = [first, second]
    issues = validate_pose23_manifest(manifest)
    assert any("duplicate match/sequence/frame/track" in issue for issue in issues)
    assert any("inside image bounds" in issue for issue in issues)


def test_validator_rejects_overlapping_visibility_policy():
    manifest = pose23_manifest_template()
    manifest["evaluation_policy"]["secondary"].append("VISIBLE")
    assert any("disjoint complete partition" in issue for issue in validate_pose23_manifest(manifest, allow_empty=True))


def test_wholebody_prediction_conversion_preserves_first_23_evidence():
    records = [
        {
            "index": index,
            "name": name,
            "x": float(index),
            "y": float(index + 1),
            "raw_model_score": 2.0,
            "state": "VALID",
            "source": "RTMW_CACHE",
            "coordinate_evidence_kind": "RAW_OBSERVED",
            "temporal_estimate_xy": None,
        }
        for index, name in enumerate(WHOLEBODY_KEYPOINT_NAMES)
    ]
    pose23 = wholebody133_to_pose23(records)
    assert len(pose23) == 23
    assert tuple(item["name"] for item in pose23) == POSE23_NAMES
    assert pose23[17]["x"] == 17.0
    assert pose23[17]["coordinate_evidence_kind"] == "RAW_OBSERVED"


def test_loader_reports_invalid_manifest(tmp_path):
    manifest = pose23_manifest_template()
    manifest["samples"] = [_sample()]
    path = tmp_path / "manifest.json"
    path.write_text(__import__("json").dumps(manifest), encoding="utf-8")
    assert load_and_validate_manifest(path)["samples"][0]["sample_id"] == "sample_001"
