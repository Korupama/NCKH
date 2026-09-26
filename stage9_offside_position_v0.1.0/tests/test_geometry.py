from stage9_offside_position.geometry import goalward_q, legal_extent_from_stage4_track, reference_from_stage8_best_effort
from .helpers import joints


def test_goalward_coordinate_both_directions():
    assert goalward_q(40, 1) == 40
    assert goalward_q(-40, -1) == 40


def test_legal_extent_selects_most_goalward_landmark():
    track={"observations":[{"frame_index":104,"joints_world":joints([48.0,47.5,47.2,47.9])}]}
    r=legal_extent_from_stage4_track(track,104,1)
    assert r["goalward_q_m"] == 48.0
    assert r["anchor"]["name"] == "nose"


def test_arms_do_not_enter_legal_extent():
    track={"observations":[{"frame_index":104,"joints_world":[
        {"name":"nose","xyz_world_m":[46,0,1.7],"valid":True},
        {"name":"right_wrist","xyz_world_m":[60,0,1.2],"valid":True},
    ]}]}
    r=legal_extent_from_stage4_track(track,104,1)
    assert r["goalward_q_m"] == 46


def test_root_fallback():
    track={"observations":[{"frame_index":104,"root_world_m":[42,0,1]}]}
    r=legal_extent_from_stage4_track(track,104,1)
    assert r["status"] == "FALLBACK"
    assert r["goalward_q_m"] == 42


def test_reference_direct():
    r=reference_from_stage8_best_effort({"reference":{"goalward_q_m":47,"X_world_m":47,"source":"BALL"}},1)
    assert r["status"] == "DIRECT" and r["goalward_q_m"] == 47


def test_reference_reconstructs_from_ball_and_second_last():
    r=reference_from_stage8_best_effort({"ball":{"goalward_q_m":50},"second_last_opponent":{"goalward_q_m":47}},1)
    assert r["status"] == "RECONSTRUCTED"
    assert r["source"] == "BALL"
    assert r["goalward_q_m"] == 50
