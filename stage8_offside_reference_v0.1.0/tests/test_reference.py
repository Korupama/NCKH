import pytest
from stage8_offside_reference.reference import ball_goalward_extent, build_reference


def test_ball_extent_ltr_uses_xmax():
    b = ball_goalward_extent([44.0, 44.22], 1)
    assert b["goalward_q_m"] == 44.22
    assert b["goalward_x_m"] == 44.22


def test_ball_extent_rtl_uses_xmin():
    b = ball_goalward_extent([-44.22, -44.0], -1)
    assert b["goalward_q_m"] == 44.22
    assert b["goalward_x_m"] == -44.22


def test_unordered_ball_extent_rejected():
    with pytest.raises(ValueError):
        ball_goalward_extent([44.22, 44.0], 1)


def test_reference_uses_defender_when_more_goalward():
    second = {"goalward_q_m": 47.0}
    ball = {"goalward_q_m": 44.2}
    r = build_reference(second, ball, 1)
    assert r["source"] == "SECOND_LAST_OPPONENT"
    assert r["X_world_m"] == 47.0
    assert r["vertical_plane"]["equation"] == {"normal": [1.0,0.0,0.0], "offset": -47.0}


def test_reference_uses_ball_when_more_goalward():
    r = build_reference({"goalward_q_m": 47.0}, {"goalward_q_m": 49.2}, 1)
    assert r["source"] == "BALL"
    assert r["X_world_m"] == 49.2


def test_reference_level_case_explicit():
    r = build_reference({"goalward_q_m": 47.0}, {"goalward_q_m": 47.0}, -1)
    assert r["source"] == "BALL_AND_SECOND_LAST_OPPONENT_LEVEL"
    assert r["X_world_m"] == -47.0
