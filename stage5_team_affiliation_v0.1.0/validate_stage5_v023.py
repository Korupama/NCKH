from __future__ import annotations

import json

import stage5_team_affiliation
from stage5_team_affiliation.policy_v023 import (
    GoalkeeperRolePolicy,
    ResidualRolePolicy,
    classify_residual_role,
    research_freeze_gate,
    select_goalkeepers_by_geometry,
)
from stage5_team_affiliation.residual_calibration import evaluate_goalkeeper_role_policy


def _row(track_id: str, distance: float, top2: float, observations: int = 20) -> dict:
    return {
        "sequence_id": "synthetic",
        "track_id": track_id,
        "goal_context": {
            "goal_side": "RIGHT",
            "median_goal_distance_m": distance,
            "top2_rate": top2,
            "observations": observations,
            "rank_frames": observations,
        },
    }


def main() -> None:
    checks: dict[str, bool] = {}

    sparse = _row("sparse", 1.0, 1.0, observations=1)
    goalkeeper = _row("goalkeeper", 5.0, 0.9)
    decisions = select_goalkeepers_by_geometry(
        [sparse, goalkeeper], GoalkeeperRolePolicy()
    )
    checks["gates_before_deepest_selection"] = (
        decisions.get("goalkeeper", {}).get("role") == "GOALKEEPER"
        and "sparse" not in decisions
    )

    referee = _row("referee", 30.0, 0.1)
    referee.update(team_margin=0.05, nearest_team_distance=0.8)
    checks["referee_requires_geometry"] = (
        classify_residual_role(referee, ResidualRolePolicy()).get("role")
        == "REFEREE"
    )

    freeze = research_freeze_gate(
        complete_train_split=True,
        goalkeeper_role_feasible=True,
        residual_role_feasible=False,
        goalkeeper_team_feasible=True,
    )
    checks["unsafe_calibration_not_frozen"] = freeze["eligible"] is False

    calibration_rows = [
        {
            "sequence_id": "synthetic",
            "track_id": "goalkeeper",
            "gt_role": "goalkeeper",
            "appearance_group": "RESIDUAL",
            "goal_sign": 1,
            "goal_observations": 20,
            "goal_rank_frames": 20,
            "goal_top2_rate": 0.9,
            "median_goal_distance_m": 5.0,
            "median_goalward_depth_m": 47.5,
        }
    ]
    gk_metrics, selected = evaluate_goalkeeper_role_policy(
        calibration_rows, goal_distance_m=18.0, min_top2_rate=0.6,
        min_observations=5, ordering_margin_m=0.5,
    )
    checks["train_goalkeeper_role_recalibration"] = (
        selected == ["synthetic::goalkeeper"] and gk_metrics["f1"] == 1.0
    )

    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "package_version": stage5_team_affiliation.__version__,
        "schema_version": "stage5-policy-validation-0.2.3",
        "checks": checks,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
