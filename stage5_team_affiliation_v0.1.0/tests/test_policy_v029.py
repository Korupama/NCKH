from __future__ import annotations

from stage5_team_affiliation.residual_calibration import (
    search_player_recovery_policy_grouped,
)


def _row(sequence_id, track_id, gt_role, distance, goal_distance, top2_rate):
    goalkeeper = gt_role == "goalkeeper"
    return {
        "sequence_id": sequence_id,
        "track_id": track_id,
        "gt_role": gt_role,
        "gt_team": "left" if gt_role == "player" else None,
        "mapping": {"0": "left", "1": "right"},
        "mapping_available": True,
        "appearance_group": "RESIDUAL",
        "margin": 0.50,
        "nearest_team": 0,
        "nearest_distance": distance,
        "base_role": "goalkeeper" if goalkeeper else "unknown_residual",
        "base_role_status": "VALID" if goalkeeper else "UNKNOWN",
        "base_team": None,
        "base_team_status": "UNKNOWN",
        "median_goal_distance_m": goal_distance,
        "goal_top2_rate": top2_rate,
    }


def test_geometry_veto_keeps_distant_player_but_rejects_referee() -> None:
    rows = []
    for index in range(10):
        sid = f"train-{index:02d}"
        rows.extend([
            _row(sid, "easy-player", "player", 0.20, 30.0, 0.05),
            _row(sid, "distant-player", "player", 0.35, 30.0, 0.10),
            _row(sid, "referee", "referee", 0.35, 30.0, 0.55),
            _row(sid, "goalkeeper", "goalkeeper", 0.60, 2.0, 0.90),
        ])
    result = search_player_recovery_policy_grouped(
        rows, baseline_outfield_accuracy=0.0, n_folds=5)
    selected = result["selected"]
    assert selected["feasible"] is True
    assert selected["parameters"]["residual_player_referee_veto_enabled"] is True
    assert selected["parameters"]["residual_player_max_distance"] >= 0.40
    assert selected["metrics"]["outfield_coverage"] == 1.0
    assert selected["metrics"]["referee_to_player_rate"] == 0.0
    assert selected["worst_fold_referee_to_player_rate"] == 0.0

