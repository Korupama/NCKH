from stage5_team_affiliation.policy_v023 import (
    GoalkeeperRolePolicy,
    ResidualRolePolicy,
    classify_residual_role,
    goalkeeper_candidates,
    research_freeze_gate,
    retain_minimum_core,
    select_goalkeepers_by_geometry,
)
from stage5_team_affiliation.residual_calibration import (
    apply_goalkeeper_role_selection,
    evaluate_goalkeeper_role_policy,
    safe_referee_goal_distance,
)


def _row(track_id, distance, top2, observations=20, rank_frames=20, side="RIGHT"):
    return {
        "sequence_id": "seq",
        "track_id": track_id,
        "goal_context": {
            "goal_side": side,
            "median_goal_distance_m": distance,
            "top2_rate": top2,
            "observations": observations,
            "rank_frames": rank_frames,
        },
    }


def test_goalkeeper_gates_run_before_deepest_selection():
    sparse_deepest = _row("sparse", 2.0, 1.0, observations=1, rank_frames=1)
    supported = _row("gk", 5.0, 0.9)
    policy = GoalkeeperRolePolicy()
    assert [row["track_id"] for row in goalkeeper_candidates([sparse_deepest, supported], policy)] == ["gk"]
    result = select_goalkeepers_by_geometry([sparse_deepest, supported], policy)
    assert result["gk"]["role"] == "GOALKEEPER"
    assert "sparse" not in result


def test_ambiguous_goal_depth_abstains():
    policy = GoalkeeperRolePolicy(ambiguity_margin_m=0.5)
    result = select_goalkeepers_by_geometry([_row("a", 5.0, 0.9), _row("b", 5.2, 0.8)], policy)
    assert result["a"]["role"] == "UNKNOWN"
    assert result["b"]["role"] == "UNKNOWN"


def test_referee_requires_appearance_and_geometry():
    policy = ResidualRolePolicy()
    referee = _row("ref", 30.0, 0.1)
    referee.update(team_margin=0.05, nearest_team_distance=0.8)
    assert classify_residual_role(referee, policy)["role"] == "REFEREE"
    near_goal = _row("not-ref", 8.0, 0.1)
    near_goal.update(team_margin=0.05, nearest_team_distance=0.8)
    assert classify_residual_role(near_goal, policy)["role"] == "UNKNOWN"


def test_freeze_gate_requires_every_recovery_gate():
    result = research_freeze_gate(
        complete_train_split=True,
        goalkeeper_role_feasible=True,
        residual_role_feasible=False,
        goalkeeper_team_feasible=True,
    )
    assert result["eligible"] is False
    assert result["failed_checks"] == ["residual_role_feasible"]


def test_minimum_core_retains_nearest_deterministically():
    kept, diagnostics = retain_minimum_core([0.4, 0.1, 0.2], [4, 1, 3], 2)
    assert kept == [1, 3]
    assert diagnostics["retained"] == 2


def test_train_calibrator_reselects_goalkeeper_without_source_role():
    rows = [
        {
            "sequence_id": "seq",
            "track_id": "gk",
            "gt_role": "goalkeeper",
            "appearance_group": "RESIDUAL",
            "goal_sign": 1,
            "goal_observations": 20,
            "goal_rank_frames": 20,
            "goal_top2_rate": 0.9,
            "median_goal_distance_m": 5.0,
            "median_goalward_depth_m": 47.5,
        },
        {
            "sequence_id": "seq",
            "track_id": "ref",
            "gt_role": "referee",
            "appearance_group": "RESIDUAL",
            "goal_sign": 1,
            "goal_observations": 20,
            "goal_rank_frames": 20,
            "goal_top2_rate": 0.1,
            "median_goal_distance_m": 28.0,
            "median_goalward_depth_m": 24.5,
        },
    ]
    metrics, selected = evaluate_goalkeeper_role_policy(
        rows,
        goal_distance_m=18.0,
        min_top2_rate=0.6,
        min_observations=5,
        ordering_margin_m=0.5,
    )
    assert selected == ["seq::gk"]
    assert metrics["f1"] == 1.0


def test_calibrated_goalkeeper_selection_replaces_source_decision():
    rows = [
        {
            "sequence_id": "seq",
            "track_id": "gk",
            "appearance_group": "RESIDUAL",
            "base_role": "unknown_residual",
            "base_role_status": "UNKNOWN",
            "base_team": None,
            "base_team_status": "UNKNOWN",
        },
        {
            "sequence_id": "seq",
            "track_id": "p",
            "appearance_group": "TEAM_0",
            "base_role": "goalkeeper",
            "base_role_status": "VALID",
            "base_team": 0,
            "base_team_status": "VALID",
        },
    ]
    updated = apply_goalkeeper_role_selection(rows, ["seq::gk"])
    by_id = {row["track_id"]: row for row in updated}
    assert by_id["gk"]["base_role"] == "goalkeeper"
    assert by_id["p"]["base_role"] == "player"


def test_provisional_config_keeps_referee_boundary_outside_new_goalkeeper_zone():
    assert safe_referee_goal_distance(22.0, 25.0, 0.5) == 25.5
    assert safe_referee_goal_distance(30.0, 18.0, 0.5) == 30.0
