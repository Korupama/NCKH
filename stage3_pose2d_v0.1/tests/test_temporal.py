from copy import deepcopy
from stage3_pose2d.temporal import annotate_temporal
from stage3_pose2d.schemas import Stage3Config
from stage3_pose2d.quality import evaluate_pose
from stage3_pose2d.wholebody133 import keypoint_records_to_arrays
from conftest import make_pose_records


def obs(frame, dx=0.0):
    r=make_pose_records()
    for x in r: x["x"]+=dx
    xy,s=keypoint_records_to_arrays(r)
    qa,k=evaluate_pose(xy,s,[100+dx,100,200+dx,300],Stage3Config())
    return {"frame_index":frame,"source_bbox_xyxy":[100+dx,100,200+dx,300],"keypoints_133":k,"qa":qa,"pose_status":qa["pose_status"]}


def test_temporal_no_silent_imputation():
    seq=[obs(1),obs(2,2),obs(3,4)]
    seq[1]["keypoints_133"][0]["x"]=None; seq[1]["keypoints_133"][0]["y"]=None
    out,summary=annotate_temporal(seq,Stage3Config(emit_temporal_estimates=True))
    assert out[1]["keypoints_133"][0]["x"] is None
    assert out[1]["keypoints_133"][0]["temporal_estimate_xy"] is not None
    assert summary["silent_imputation"] is False


def test_temporal_jitter_near_zero_under_constant_bbox_relative_motion():
    out,summary=annotate_temporal([obs(1,0),obs(2,2),obs(3,4)],Stage3Config())
    assert summary["normalized_temporal_jitter"] < 1e-6


def test_temporal_estimate_records_insufficient_context_without_overwriting_raw_coordinates():
    seq=[obs(1),obs(2,2),obs(4,4)]
    seq[1]["keypoints_133"][0]["x"]=None
    seq[1]["keypoints_133"][0]["y"]=None
    out,summary=annotate_temporal(seq,Stage3Config(emit_temporal_estimates=True))

    middle=out[1]["keypoints_133"][0]
    assert middle["temporal_estimate_status"]=="INSUFFICIENT_CONTEXT"
    assert middle["temporal_estimate_reason"]=="nonconsecutive_frame_context"
    assert middle["x"] is None and middle["y"] is None
    assert summary["temporal_estimates"]["created"]==0
    assert summary["temporal_estimates"]["insufficient_context_reasons"]["nonconsecutive_frame_context"] >= 1
    assert summary["raw_coordinates_modified"] is False


def test_temporal_estimate_marks_track_boundaries():
    seq=[obs(1),obs(2),obs(3)]
    for point in (seq[0]["keypoints_133"][0], seq[2]["keypoints_133"][0]):
        point["x"]=None
        point["y"]=None
    out,summary=annotate_temporal(seq,Stage3Config(emit_temporal_estimates=True))

    assert out[0]["keypoints_133"][0]["temporal_estimate_reason"]=="track_boundary"
    assert out[2]["keypoints_133"][0]["temporal_estimate_reason"]=="track_boundary"
    assert summary["temporal_estimates"]["insufficient_context_reasons"]["track_boundary"] >= 2


def test_temporal_ownership_switch_downgrades_valid_pose_without_changing_raw_points():
    seq = [obs(1), obs(2), obs(3)]
    for item in seq:
        item["qa"]["ownership"] = {
            "ownership_status": "SUPPORTED",
            "body17_fraction_inside_source_bbox": 1.0,
            "body17_center_to_source_center_normalized": 0.1,
        }
        item["crop_diagnostics"] = {"ownership_status": "SUPPORTED"}
    seq[1]["qa"]["ownership"]["ownership_status"] = "CENTER_MISMATCH"
    seq[1]["crop_diagnostics"]["ownership_status"] = "CENTER_MISMATCH"
    raw_before = [(kp["x"], kp["y"]) for kp in seq[1]["keypoints_133"]]

    out, summary = annotate_temporal(seq, Stage3Config())

    assert out[1]["pose_status"] == "DEGRADED"
    assert out[1]["qa"]["pose_status"] == "DEGRADED"
    assert "temporal_ownership_switch_suspected" in out[1]["qa"]["status_reasons"]
    assert summary["ownership_switch_suspected_frames"] == [2]
    assert summary["temporal_downgraded_frames"] == [2]
    assert raw_before == [(kp["x"], kp["y"]) for kp in out[1]["keypoints_133"]]
