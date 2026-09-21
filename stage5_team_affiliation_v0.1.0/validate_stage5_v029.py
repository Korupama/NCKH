from __future__ import annotations

import json

from stage5_team_affiliation import __version__
from stage5_team_affiliation.residual_calibration import (
    search_player_recovery_policy_grouped,
)


def _row(sid, tid, role, distance, goal_distance, top2):
    goalkeeper = role == "goalkeeper"
    return {
        "sequence_id": sid,
        "track_id": tid,
        "gt_role": role,
        "gt_team": "left" if role == "player" else None,
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
        "goal_top2_rate": top2,
    }


def main() -> None:
    rows = []
    for index in range(10):
        sid = f"validation-{index:02d}"
        rows.extend([
            _row(sid, "easy-player", "player", 0.20, 30.0, 0.05),
            _row(sid, "distant-player", "player", 0.35, 30.0, 0.10),
            _row(sid, "referee", "referee", 0.35, 30.0, 0.55),
            _row(sid, "goalkeeper", "goalkeeper", 0.60, 2.0, 0.90),
        ])
    result = search_player_recovery_policy_grouped(
        rows, baseline_outfield_accuracy=0.0, n_folds=5)
    selected = result["selected"]
    checks = {
        "package_version_is_v029_or_later": tuple(map(int, __version__.split('.'))) >= (0, 2, 9),
        "geometry_veto_selected": selected["parameters"].get(
            "residual_player_referee_veto_enabled") is True,
        "distant_players_recovered": selected["metrics"]["outfield_coverage"] == 1.0,
        "referee_contamination_zero": (
            selected["metrics"]["referee_to_player_rate"] == 0.0),
        "all_train_folds_referee_safe": (
            selected["worst_fold_referee_to_player_rate"] == 0.0),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    print(json.dumps({
        "status": status,
        "package_version": __version__,
        "schema_version": "stage5-referee-veto-validation-0.2.9",
        "checks": checks,
    }, indent=2))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
