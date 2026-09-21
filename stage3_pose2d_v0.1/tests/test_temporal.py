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
