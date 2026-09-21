import numpy as np, cv2
from stage5_team_affiliation.config import Stage5Config
from stage5_team_affiliation.features import extract_color_feature, aggregate_features


def test_color_feature_separates_red_blue():
    cfg=Stage5Config(torso_erode_fraction=0.0)
    red=np.zeros((100,100,3),np.uint8); red[:]=(0,0,220)
    blue=np.zeros((100,100,3),np.uint8); blue[:]=(220,0,0)
    poly=np.array([[10,10],[90,10],[90,90],[10,90]],np.float32)
    fr,_=extract_color_feature(red,poly,cfg)
    fb,_=extract_color_feature(blue,poly,cfg)
    assert fr is not None and fb is not None
    assert np.linalg.norm(fr-fb) > 0.2


def test_aggregate_normalized():
    x=aggregate_features([np.ones(10),np.ones(10)*2])
    assert abs(np.linalg.norm(x)-1)<1e-6
