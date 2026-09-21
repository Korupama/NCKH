from stage4_metric3d.proxy_quality import compactness_diagnostics, robust_ground_anchor
from stage4_metric3d.proxy_schemas import Stage4ProjectionConfig


def _point(name, group, x, y):
    return {
        "name": name,
        "anatomical_group": group,
        "xyz_proxy_world_m": [x, y, 0.1],
    }


def test_compactness_rejects_multi_metre_body_spread():
    config = Stage4ProjectionConfig()
    points = [
        _point("left_big_toe", "foot", 0.0, 0.0),
        _point("right_big_toe", "foot", 0.3, 0.1),
        _point("left_shoulder", "shoulder", 0.1, 4.0),
        _point("right_shoulder", "shoulder", 0.3, 4.2),
        _point("nose", "head", 0.2, 5.0),
    ]
    result = compactness_diagnostics(points, config)
    assert result["status"] == "REJECTED"
    assert "TRANSVERSE_SPAN_EXCEEDS_REJECT_LIMIT" in result["reasons"]


def test_ground_anchor_uses_dense_foot_cluster_and_ignores_outlier():
    config = Stage4ProjectionConfig()
    points = [
        _point("left_big_toe", "foot", 10.0, 5.0),
        _point("left_small_toe", "foot", 10.1, 5.1),
        _point("left_heel", "foot", 9.9, 5.0),
        _point("left_ankle", "ankle", 10.0, 5.2),
        _point("right_big_toe", "foot", 16.0, 12.0),
    ]
    result = robust_ground_anchor(points, config)
    assert result["status"] == "VALID"
    assert result["inlier_count"] == 4
    assert abs(result["xyz_ground_m"][0] - 10.0) < 0.11
    assert abs(result["xyz_ground_m"][1] - 5.05) < 0.11
