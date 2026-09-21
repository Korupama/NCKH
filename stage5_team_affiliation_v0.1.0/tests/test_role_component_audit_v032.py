from stage5_team_affiliation.role_component_audit_v032 import (
    evaluate_referee_rule,
    search_referee_diagnostic_ceiling,
)


def _row(track, role, margin, distance, goal, top2):
    return {
        "sequence_id": "seq", "track_id": track,
        "gt_role": role, "appearance_group": "RESIDUAL",
        "base_role": "unknown", "margin": margin,
        "nearest_distance": distance,
        "median_goal_distance_m": goal, "goal_top2_rate": top2,
    }


def test_isolated_referee_rule_reports_precision_recall_and_player_contamination():
    rows = [
        _row("ref", "referee", 0.05, 0.8, 30.0, 0.1),
        _row("player", "player", 0.50, 0.2, 25.0, 0.5),
    ]
    result = evaluate_referee_rule(
        rows, max_margin=0.10, min_distance=0.4,
        min_goal_distance_m=26.0, max_top2_rate=0.2)
    assert result["tp"] == 1
    assert result["fp"] == 0
    assert result["fn"] == 0
    assert result["f1"] == 1.0
    assert result["player_to_referee_rate"] == 0.0


def test_referee_ceiling_is_explicitly_non_freezable():
    rows = [
        _row("ref", "referee", 0.05, 0.8, 30.0, 0.1),
        _row("player", "player", 0.50, 0.2, 25.0, 0.5),
    ]
    result = search_referee_diagnostic_ceiling(rows)
    assert result["selected"]["metrics"]["f1"] == 1.0
    assert result["label_use"] == "VALID_GT_DIAGNOSTIC_CEILING_NOT_FREEZABLE"
