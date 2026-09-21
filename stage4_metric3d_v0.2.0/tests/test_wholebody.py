import numpy as np
from stage4_metric3d.wholebody import LEGAL_GEOMETRY_CANDIDATE_23, POSE23_NAMES, derive_pose_features


def test_arm_branch_is_excluded_but_shoulders_are_kept():
    idx = {n:i for i,n in enumerate(POSE23_NAMES)}
    assert LEGAL_GEOMETRY_CANDIDATE_23[idx["left_shoulder"]]
    assert LEGAL_GEOMETRY_CANDIDATE_23[idx["right_shoulder"]]
    for name in ("left_elbow","right_elbow","left_wrist","right_wrist"):
        assert not LEGAL_GEOMETRY_CANDIDATE_23[idx[name]]


def test_derived_foot_proxy_does_not_require_exact_toe():
    xyz = np.full((23,3), np.nan)
    xyz[15] = [1,2,.08]
    xyz[19] = [1,2.1,.02]
    out = derive_pose_features(xyz)
    assert out["left_foot_proxy_xyz_m"] is not None
