import numpy as np
from types import SimpleNamespace
from stage3_pose2d.processor import _fallback_reinfer
from stage3_pose2d.schemas import Stage3Config
from stage3_pose2d.wholebody133 import keypoint_records_to_arrays
from conftest import make_pose_records

class FakeModel:
    def infer_one(self, frame, bbox, crop_scale=1.0):
        xy,s=keypoint_records_to_arrays(make_pose_records(score=3.0))
        return SimpleNamespace(keypoints_xy=xy,scores=s)


def test_fallback_preserves_upstream_pose_when_selected():
    missing=[{"index":i,"name":r["name"],"x":None,"y":None,"raw_model_score":None,"state":"MISSING","source":"NO_RTMW_CACHE","temporal_estimate_xy":None}
             for i,r in enumerate(make_pose_records())]
    obs={
        "frame_index":10,"source_bbox_xyxy":[100,100,200,300],"keypoints_133":missing,
        "qa":{"pose_status":"MISSING"},"pose_status":"MISSING","source_backend":None,"source_score_semantics":None,
        "provenance":{"source":"NO_RTMW_CACHE"}
    }
    out=_fallback_reinfer(obs,np.zeros((400,400,3),np.uint8),FakeModel(),Stage3Config(fallback_crop_scales=[1.0]))
    assert out["pose_status"]=="VALID"
    assert out["provenance"]["raw_cache_preserved"] is True
    assert "upstream_raw_pose" in out
    assert out["upstream_raw_pose"]["pose_status"]=="MISSING"
