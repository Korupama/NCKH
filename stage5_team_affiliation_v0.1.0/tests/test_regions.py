import numpy as np
from stage5_team_affiliation.regions import polygon_from_pose, region_polygon


def obs_with_pose():
    names = ["left_shoulder","right_shoulder","left_hip","right_hip","left_knee","right_knee"]
    pts = {
        "left_shoulder": (10,10), "right_shoulder": (30,10),
        "left_hip": (12,30), "right_hip": (28,30),
        "left_knee": (13,48), "right_knee": (27,48),
    }
    kps=[]
    for i,n in enumerate(names):
        x,y=pts[n]
        kps.append({"index":i,"name":n,"x":x,"y":y,"state":"VALID"})
    return {"keypoints_133":kps,"source_bbox_xyxy":[5,5,35,60]}


def test_pose_torso_polygon():
    p, src = region_polygon(obs_with_pose(), "torso")
    assert src == "POSE_TORSO"
    assert p.shape == (4,2)


def test_bbox_fallback():
    p, src = region_polygon({"keypoints_133":[],"source_bbox_xyxy":[0,0,100,200]}, "torso")
    assert src == "BBOX_TORSO_FALLBACK"
    assert p.shape == (4,2)
