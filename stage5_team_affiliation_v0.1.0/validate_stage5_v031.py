from __future__ import annotations

import json

from stage5_team_affiliation import __version__
from stage5_team_affiliation.residual_calibration import (
    search_goalkeeper_role_policy,
)


def main() -> None:
    rows = [
        {
            "sequence_id": "fixture", "track_id": "gk",
            "gt_role": "goalkeeper", "appearance_group": "RESIDUAL",
            "goal_sign": 1, "goal_observations": 20,
            "goal_rank_frames": 20, "goal_top2_rate": 1.0,
            "median_goal_distance_m": 5.0,
            "median_goalward_depth_m": 47.5,
        },
        {
            "sequence_id": "fixture", "track_id": "player",
            "gt_role": "player", "appearance_group": "TEAM_0",
            "goal_sign": None, "goal_observations": 0,
            "goal_rank_frames": 0, "goal_top2_rate": None,
            "median_goal_distance_m": None,
            "median_goalward_depth_m": None,
        },
    ]
    result = search_goalkeeper_role_policy(
        rows, min_observations=5, ordering_margin_m=0.5,
        min_precision=0.0, min_f1=0.0,
        max_player_goalkeeper_rate=1.0)
    checks = {
        "package_version_is_v031_or_later": tuple(
            map(int, __version__.split('.'))) >= (0, 3, 1),
        "equivalent_plateau_detected": result["equivalent_best_plateau_size"] > 1,
        "interior_point_selected": not any(
            result["selected_at_grid_boundary"].values()),
        "spurious_grid_expansion_avoided": (
            result["requires_grid_expansion"] is False),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    print(json.dumps({
        "status": status,
        "package_version": __version__,
        "schema_version": "stage5-plateau-selection-validation-0.3.1",
        "checks": checks,
    }, indent=2))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
