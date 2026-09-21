import numpy as np

from stage_1_camera.evidence import summarize_keypoint_coverage


def test_keypoint_spatial_coverage_metrics():
    kps = {
        1: {"x": 100.0, "y": 100.0},
        2: {"x": 900.0, "y": 100.0},
        3: {"x": 900.0, "y": 500.0},
        4: {"x": 100.0, "y": 500.0},
    }
    world = {1: (-40.0, -20.0), 2: (40.0, -20.0), 3: (40.0, 20.0), 4: (-40.0, 20.0)}
    out = summarize_keypoint_coverage(kps, 1000, 600, world_xy_lookup=world)
    assert out["num_valid_keypoints"] == 4
    assert np.isclose(out["image_x_span_ratio"], 0.8)
    assert np.isclose(out["image_y_span_ratio"], 400.0 / 600.0)
    assert out["image_convex_hull_area_ratio"] > 0.5
    assert out["image_quadrants_occupied"] == 4
    assert np.isclose(out["world_x_span_m"], 80.0)
    assert np.isclose(out["world_y_span_m"], 40.0)


def test_empty_keypoint_coverage_is_explicit():
    out = summarize_keypoint_coverage({}, 1920, 1080)
    assert out["num_valid_keypoints"] == 0
    assert out["image_convex_hull_area_ratio"] is None
    assert out["world_x_span_m"] is None
