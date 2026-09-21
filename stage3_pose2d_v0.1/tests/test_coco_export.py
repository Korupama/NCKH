import numpy as np
from benchmark.coco_wholebody import prediction_record


def test_coco_prediction_field_lengths():
    xy=np.zeros((133,2),np.float32); scores=np.ones(133,np.float32)
    r=prediction_record(1,2,xy,scores)
    assert len(r["keypoints"])==17*3
    assert len(r["foot_kpts"])==6*3
    assert len(r["face_kpts"])==68*3
    assert len(r["lefthand_kpts"])==21*3
    assert len(r["righthand_kpts"])==21*3
