import numpy as np
import pytest
from stage3_pose2d.wholebody133 import WHOLEBODY_KEYPOINT_NAMES, keypoint_records_to_arrays, coco133_to_h36m17
from conftest import make_pose_records


def test_wholebody_count_and_order():
    assert len(WHOLEBODY_KEYPOINT_NAMES)==133
    assert WHOLEBODY_KEYPOINT_NAMES[17:23]==("left_big_toe","left_small_toe","left_heel","right_big_toe","right_small_toe","right_heel")


def test_records_to_arrays_and_h36m():
    xy,s=keypoint_records_to_arrays(make_pose_records())
    assert xy.shape==(133,2) and s.shape==(133,)
    h=coco133_to_h36m17(xy)
    assert h.shape==(17,2)
    assert np.isfinite(h).all()


def test_wrong_name_order_rejected():
    r=make_pose_records(); r[0]["name"]="wrong"
    with pytest.raises(ValueError,match="ordering mismatch"):
        keypoint_records_to_arrays(r)
