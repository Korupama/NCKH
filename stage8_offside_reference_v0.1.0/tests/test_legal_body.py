from stage8_offside_reference.legal_body import LEGAL_LANDMARKS, legal_landmark_extent
from helpers import joint, track


def test_arm_landmarks_are_excluded_from_extent():
    tr = track("d1", 104, [joint("left_wrist", 60.0), joint("left_shoulder", 48.0), joint("nose", 47.0)])
    out = legal_landmark_extent(tr, 104, 1)
    assert out["goalward_q_m"] == 48.0
    assert out["anchor"]["name"] == "left_shoulder"


def test_rtl_uses_min_x_via_goalward_transform():
    tr = track("d1", 104, [joint("nose", -45.0), joint("left_big_toe", -48.0)])
    out = legal_landmark_extent(tr, 104, -1)
    assert out["goalward_q_m"] == 48.0
    assert out["goalward_x_m"] == -48.0
    assert out["anchor"]["name"] == "left_big_toe"


def test_invalid_joint_is_ignored():
    tr = track("d1", 104, [joint("nose", 51.0, valid=False), joint("right_heel", 47.0)])
    out = legal_landmark_extent(tr, 104, 1)
    assert out["goalward_q_m"] == 47.0
    assert out["anchor"]["name"] == "right_heel"


def test_missing_observation_is_missing():
    tr = track("d1", 103, [joint("nose", 50.0)])
    out = legal_landmark_extent(tr, 104, 1)
    assert out["status"] == "MISSING"
    assert out["goalward_q_m"] is None


def test_policy_contains_no_elbow_or_wrist():
    assert "left_elbow" not in LEGAL_LANDMARKS
    assert "right_wrist" not in LEGAL_LANDMARKS
    assert "left_shoulder" in LEGAL_LANDMARKS


def test_rejected_upstream_geometry_is_not_rehabilitated():
    tr = track("d1", 104, [joint("nose", 50.0)], status="REJECTED")
    out = legal_landmark_extent(tr, 104, 1)
    assert out["status"] == "UNUSABLE"
    assert out["goalward_q_m"] is None
    assert out["reason"] == "UPSTREAM_SELECTED_FRAME_STATUS_REJECTED"
