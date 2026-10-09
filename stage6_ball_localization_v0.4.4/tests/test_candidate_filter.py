import math
import cv2
import numpy as np
import pytest

from ball_localization.camera import CameraStateLite
from ball_localization.candidate_filter import (
    compute_aspect_ratio,
    diameter_validity,
    goal_net_ray_check,
    grid_structure_score,
    ray_intersects_aabb_3d,
    shape_aspect_ratio_score,
    structure_tensor_coherence,
)
from ball_localization.contracts import BallCandidate2D
from ball_localization.pitch_prior import apply_pitch_prior, pitch_soft_prior
from ball_localization.tracking import ViterbiConfig, select_ball_path


def test_aspect_ratio_computation_and_scoring():
    # Perfect square
    assert compute_aspect_ratio([10, 10, 20, 20]) == pytest.approx(1.0)
    assert shape_aspect_ratio_score([10, 10, 20, 20]) == 1.0

    # Slight elongation (e.g. motion blur AR ~ 1.14)
    assert shape_aspect_ratio_score([10, 10, 21.4, 20]) == 1.0

    # Moderate elongation (AR ~ 1.25)
    score_125 = shape_aspect_ratio_score([10, 10, 22.5, 20])
    assert 0.5 < score_125 < 1.0

    # Net mesh cord or line stripe (AR = 2.0)
    assert compute_aspect_ratio([10, 10, 30, 20]) == pytest.approx(2.0)
    assert shape_aspect_ratio_score([10, 10, 30, 20]) == 0.0

    # Very thin vertical goal post segment (AR = 5.0)
    assert shape_aspect_ratio_score([10, 10, 12, 20]) == 0.0


def test_diameter_validity_bounds():
    # Normal soccer ball in broadcast
    assert diameter_validity(12.0) == 1.0
    assert diameter_validity(25.0) == 1.0

    # Single-pixel noise or distant speckle
    assert diameter_validity(1.5) == 0.0

    # Large player body / advertising board
    assert diameter_validity(75.0) == 0.0


def _build_test_camera(pos_xyz: np.ndarray, target_xyz: np.ndarray) -> CameraStateLite:
    C = np.asarray(pos_xyz, dtype=float)
    target = np.asarray(target_xyz, dtype=float)
    forward = target - C
    forward /= np.linalg.norm(forward)
    up = np.asarray([0.0, 0.0, 1.0], dtype=float)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    down /= np.linalg.norm(down)
    R = np.stack([right, down, forward], axis=0)
    K = np.asarray([[2000.0, 0.0, 960.0], [0.0, 2000.0, 540.0], [0.0, 0.0, 1.0]], dtype=float)
    return CameraStateLite(
        0, 1920, 1080, K, R, C,
        np.zeros(12, dtype=float),
        "VALID",
        {"length_m": 105.0, "width_m": 68.0},
        0.0,
    )


def test_goal_net_ray_check_detects_net_and_clears_pitch():
    # Camera at side of pitch viewing the right goal area
    cam = _build_test_camera([20.0, -50.0, 18.0], [45.0, 0.0, 1.0])

    # 1. Point inside Right Goal net (behind goal line at X=+53.5m, Y=0, Z=1.2m)
    right_net_pt = np.array([53.5, 0.0, 1.2])
    uv_right_net = cam.project_world(right_net_pt)[0]
    res_right = goal_net_ray_check(cam, uv_right_net)
    assert res_right["in_goal_net"] is True
    assert res_right["goal_side"] == "right"
    assert res_right["ray_distance_m"] is not None

    # 2. Point inside pitch near penalty spot (X=+41.5m, Y=0, Z=0.11m)
    penalty_spot = np.array([41.5, 0.0, 0.11])
    uv_spot = cam.project_world(penalty_spot)[0]
    res_spot = goal_net_ray_check(cam, uv_spot)
    assert res_spot["in_goal_net"] is False

    # 3. Point at center circle (0, 0, 0.11m)
    center_pt = np.array([0.0, 0.0, 0.11])
    uv_center = cam.project_world(center_pt)[0]
    res_center = goal_net_ray_check(cam, uv_center)
    assert res_center["in_goal_net"] is False


def test_structure_tensor_coherence_distinguishes_ball_from_line():
    # 1. Ball patch: circular blob
    ball_crop = np.full((32, 32, 3), [40, 160, 40], dtype=np.uint8)  # green turf
    cv2.circle(ball_crop, (16, 16), 10, (240, 240, 240), -1)  # white ball
    cv2.circle(ball_crop, (16, 16), 4, (30, 30, 30), -1)  # dark patch
    c_ball = structure_tensor_coherence(ball_crop)
    assert c_ball < 0.35  # Isotropic gradients

    # 2. Line patch: straight white line marking
    line_crop = np.full((32, 32, 3), [40, 160, 40], dtype=np.uint8)  # green turf
    cv2.line(line_crop, (0, 16), (31, 16), (250, 250, 250), 5)  # white line
    c_line = structure_tensor_coherence(line_crop)
    assert c_line > 0.85  # Highly directional gradients


def test_grid_structure_score_distinguishes_frame_from_ball():
    grid_crop = np.full((40, 40, 3), [40, 160, 40], dtype=np.uint8)
    cv2.rectangle(grid_crop, (3, 3), (36, 36), (250, 250, 250), 2)
    cv2.line(grid_crop, (3, 20), (36, 20), (250, 250, 250), 2)
    cv2.line(grid_crop, (20, 3), (20, 36), (250, 250, 250), 2)
    assert grid_structure_score(grid_crop) >= 0.75

    ball_crop = np.full((40, 40, 3), [40, 160, 40], dtype=np.uint8)
    cv2.circle(ball_crop, (20, 20), 12, (240, 240, 240), -1)
    cv2.circle(ball_crop, (15, 15), 3, (30, 30, 30), -1)
    assert grid_structure_score(ball_crop) < 0.75


def test_apply_pitch_prior_suppresses_goal_net_and_elongated_objects():
    cam = _build_test_camera([20.0, -50.0, 18.0], [45.0, 0.0, 1.0])

    # Valid ball in play
    ball_3d = np.array([35.0, 5.0, 0.11])
    uv_ball = cam.project_world(ball_3d)[0]
    c_valid = BallCandidate2D(
        0, "ball_valid",
        [uv_ball[0] - 6, uv_ball[1] - 6, uv_ball[0] + 6, uv_ball[1] + 6],
        uv_ball.tolist(), 0.90, "yolo", 12.0
    )

    # Net knot false positive (in net at X=+53.5m)
    net_3d = np.array([53.5, 0.5, 1.2])
    uv_net = cam.project_world(net_3d)[0]
    c_net = BallCandidate2D(
        0, "net_knot",
        [uv_net[0] - 5, uv_net[1] - 5, uv_net[0] + 5, uv_net[1] + 5],
        uv_net.tolist(), 0.90, "yolo", 10.0
    )

    # Elongated line fragment (AR = 2.5) on field
    uv_line = [800.0, 600.0]
    c_line = BallCandidate2D(
        0, "line_frag",
        [uv_line[0] - 15, uv_line[1] - 5, uv_line[0] + 15, uv_line[1] + 5],
        uv_line, 0.90, "yolo", 10.0
    )

    results = apply_pitch_prior([c_valid, c_net, c_line], cam)
    r_map = {c.candidate_id: c for c in results}

    # True ball keeps high score
    assert r_map["ball_valid"].pitch_prior > 0.90
    assert r_map["ball_valid"].ranking_score > 0.80

    # Net knot is heavily suppressed
    assert r_map["net_knot"].metadata["in_goal_net"] is True
    assert r_map["net_knot"].pitch_prior <= 0.15
    assert r_map["net_knot"].ranking_score <= 0.90 * 0.15

    # Elongated line fragment is completely rejected (shape_score = 0.0)
    assert r_map["line_frag"].metadata["shape_score"] == 0.0
    assert r_map["line_frag"].pitch_prior == 0.0
    assert r_map["line_frag"].ranking_score == 0.0


def test_apply_pitch_prior_rejects_square_grid_frame():
    cam = _build_test_camera([20.0, -50.0, 18.0], [45.0, 0.0, 1.0])
    crop = np.full((40, 40, 3), [40, 160, 40], dtype=np.uint8)
    cv2.rectangle(crop, (3, 3), (36, 36), (250, 250, 250), 2)
    cv2.line(crop, (3, 20), (36, 20), (250, 250, 250), 2)
    cv2.line(crop, (20, 3), (20, 36), (250, 250, 250), 2)
    candidate = BallCandidate2D(
        0, "grid_frame", [0.0, 0.0, 40.0, 40.0], [20.0, 20.0],
        0.90, "yolo", 40.0,
    )
    result = apply_pitch_prior([candidate], cam, image_bgr=crop)[0]
    assert result.metadata["grid_structure_score"] >= 0.75
    assert result.pitch_prior == 0.0
    assert result.ranking_score == 0.0


def test_viterbi_rejects_stationary_goal_net_knot_in_favor_of_moving_ball():
    # Scenario:
    # A true ball is moving smoothly across frames: (100, 100) -> (120, 102) -> (140, 105)
    # with detector score 0.78 (combined ranking 0.78).
    # A false positive knot on the goal net is stationary: (500, 500) -> (500, 500) -> (500, 500)
    # Even if its raw detector score was 0.85, it is in the net (in_goal_net=True).
    frames = [0, 1, 2]
    candidates = {}

    for fi in frames:
        ball_x = 100.0 + fi * 20.0
        ball_y = 100.0 + fi * 2.0
        c_ball = BallCandidate2D(
            fi, f"true_ball_{fi}",
            [ball_x - 6, ball_y - 6, ball_x + 6, ball_y + 6],
            [ball_x, ball_y],
            0.80, "yolo", 12.0,
            pitch_prior=1.0,
            ranking_score=0.80,
            metadata={"in_goal_net": False}
        )

        # Net knot: stationary at (500, 500), marked in_goal_net
        c_net = BallCandidate2D(
            fi, f"net_knot_{fi}",
            [495.0, 495.0, 505.0, 505.0],
            [500.0, 500.0],
            0.85, "yolo", 10.0,
            pitch_prior=0.15,
            ranking_score=0.85 * 0.15,
            metadata={"in_goal_net": True}
        )
        candidates[fi] = [c_ball, c_net]

    cfg = ViterbiConfig()
    path = select_ball_path(candidates, frames, image_width=1920, image_height=1080, config=cfg)

    # Tracker MUST track the true moving ball, NOT the net knot!
    assert [path[fi].candidate_id for fi in frames] == [
        "true_ball_0",
        "true_ball_1",
        "true_ball_2",
    ]
