import numpy as np
from stage3_pose2d.quality import evaluate_pose
from stage3_pose2d.schemas import Stage3Config
from stage3_pose2d.wholebody133 import keypoint_records_to_arrays
from conftest import make_pose_records


def test_plausible_pose_is_valid():
    xy,s=keypoint_records_to_arrays(make_pose_records(score=4.0))
    qa,k=evaluate_pose(xy,s,[100,100,200,300],Stage3Config())
    assert qa["pose_status"]=="VALID"
    assert qa["feet_completeness"]==1.0
    assert len(k)==133


def test_raw_simcc_not_probability_thresholded():
    # Very small but positive scores remain available; Stage3 uses relative evidence, not score>0.5.
    r=make_pose_records(score=.12)
    xy,s=keypoint_records_to_arrays(r)
    qa,_=evaluate_pose(xy,s,[100,100,200,300],Stage3Config())
    assert qa["body_completeness"]==1.0
    assert qa["note"].endswith("not a calibrated probability")


def test_missing_lower_body_rejected():
    r=make_pose_records()
    for i in range(11,23):
        r[i]["x"]=None; r[i]["y"]=None; r[i]["raw_score"]=0.0
    xy,s=keypoint_records_to_arrays(r)
    qa,_=evaluate_pose(xy,s,[100,100,200,300],Stage3Config())
    assert qa["pose_status"]=="REJECTED"
