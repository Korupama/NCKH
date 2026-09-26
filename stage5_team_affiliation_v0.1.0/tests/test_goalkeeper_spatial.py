import numpy as np
from stage5_team_affiliation.config import Stage5Config
from stage5_team_affiliation.goalkeeper import assign_goalkeeper, assign_goalkeeper_spatial


def test_spatial_pitch_goal_affinity():
    cfg = Stage5Config(goalkeeper_assignment_mode="spatial_pitch")
    team_pitch = {
        0: [-25.0, -20.0, -15.0, -10.0],  # Team 0 defends negative X (left goal)
        1: [10.0, 15.0, 20.0, 25.0],      # Team 1 defends positive X (right goal)
    }
    
    # Left goalkeeper
    gk_left = [-48.0, -47.5, -48.2]
    res_left = assign_goalkeeper_spatial(gk_pitch_x=gk_left, team_pitch_x=team_pitch, config=cfg)
    assert res_left["status"] == "VALID"
    assert res_left["team_id"] == 0
    assert res_left["method"] == "SPATIAL_PITCH_GOAL_AFFINITY"

    # Right goalkeeper
    gk_right = [49.0, 48.5, 49.2]
    res_right = assign_goalkeeper_spatial(gk_pitch_x=gk_right, team_pitch_x=team_pitch, config=cfg)
    assert res_right["status"] == "VALID"
    assert res_right["team_id"] == 1


def test_spatial_image_centroid_affinity():
    cfg = Stage5Config(goalkeeper_assignment_mode="spatial_image", goalkeeper_min_spatial_margin_px=50.0)
    team_image = {
        0: [300.0, 400.0, 500.0],
        1: [1400.0, 1500.0, 1600.0],
    }

    # Near team 0
    res_0 = assign_goalkeeper_spatial(gk_image_x=[250.0, 260.0], team_image_x=team_image, config=cfg)
    assert res_0["status"] == "VALID"
    assert res_0["team_id"] == 0

    # Near team 1
    res_1 = assign_goalkeeper_spatial(gk_image_x=[1650.0, 1700.0], team_image_x=team_image, config=cfg)
    assert res_1["status"] == "VALID"
    assert res_1["team_id"] == 1


def test_fallback_to_color_when_spatial_ambiguous():
    cfg = Stage5Config(goalkeeper_assignment_mode="spatial_image", goalkeeper_min_spatial_margin_px=200.0)
    team_image = {
        0: [800.0, 900.0],
        1: [1000.0, 1100.0],
    }
    # GK exactly in the middle between both teams
    gk_x = [950.0]
    
    # Color centroids: team 0 is [1, 0], team 1 is [0, 1]
    team_lower = {0: np.array([1.0, 0.0], np.float32), 1: np.array([0.0, 1.0], np.float32)}
    gk_color = np.array([0.9, 0.1], np.float32)

    res = assign_goalkeeper(
        gk_color, team_lower, cfg,
        gk_image_x=gk_x, team_image_x=team_image,
    )
    # Spatial was ambiguous, so it fell back to color and matched team 0!
    assert res["status"] == "VALID"
    assert res["team_id"] == 0
    assert res["method"] == "LOWER_BODY_APPEARANCE_AFFINITY"
