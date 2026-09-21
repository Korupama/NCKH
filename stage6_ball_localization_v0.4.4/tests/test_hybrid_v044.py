import pytest

from ball_localization.evaluation.issia3d import HybridConfig, _hybrid_rows, calibrate_issia3d_hybrid


def _row(record_id, xyz):
    return {"record_id": record_id, "camera": 3, "segment": 0, "frame": int(record_id[-1]), "gt_xyz": [0.0, 0.0, 0.0], "pred_xyz": xyz, "status": "VALID"}


def test_hybrid_uses_only_observation_proxies_for_ground_and_ballistic_selection():
    collected = {
        "V03_SIZE_PRIOR": [_row("r0", [1, 1, .2]), _row("r1", [2, 1, 2.0])],
        "GROUND_PLANE": [_row("r0", [9, 9, .11]), _row("r1", [9, 9, .11])],
        "V04_TEMPORAL": [_row("r0", [3, 3, .2]), _row("r1", [3, 3, 2.0])],
        "BALLISTIC_G9_81": [_row("r0", [4, 4, .2]), _row("r1", [2.1, 1, 2.0])],
        "BALLISTIC_FIT_G": [_row("r0", [4, 4, .2]), _row("r1", [2.05, 1, 2.0])],
    }
    rows, diagnostics = _hybrid_rows(collected, HybridConfig(ground_proxy_height_m=.5, ballistic_consensus_distance_m=.5, median_window_frames=1))
    assert rows[0]["hybrid"]["selected_method"] == "HYBRID_GROUND_PLANE"
    assert rows[1]["hybrid"]["selected_method"] == "HYBRID_BALLISTIC_FIT_G"
    assert diagnostics["regime_counts"] == {"GROUND": 1, "AIRBORNE": 1}


def test_hybrid_calibration_rejects_test_cameras_before_reading_inputs():
    with pytest.raises(ValueError, match="held-out test"):
        calibrate_issia3d_hybrid(csv_path="not-used.csv", calibration_path="not-used.json", output_dir="not-used", cameras=[1, 2])
