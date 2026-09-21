from __future__ import annotations

import json

from stage5_team_affiliation import __version__
from stage5_team_affiliation.residual_calibration import (
    search_player_recovery_policy_grouped,
)


def _row(sequence_id: str, track_id: str, gt_role: str, distance: float) -> dict:
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
    }


def main() -> None:
    rows = []
    for index in range(10):
        sid = f"validation-{index:02d}"
        rows.extend([
            _row(sid, "player", "player", 0.20),
            _row(sid, "referee", "referee", 0.35),
            _row(sid, "goalkeeper", "goalkeeper", 0.60),
        ])
    result = search_player_recovery_policy_grouped(
        rows, baseline_outfield_accuracy=0.0, n_folds=5)
    selected = result["selected"]
    checks = {
        "package_version_supports_v028": tuple(
            int(part) for part in __version__.split(".")) >= (0, 2, 8),
        "sequence_grouped_folds": result["fold_count"] == 5,
        "zero_referee_contamination_global": (
            selected["metrics"]["referee_to_player_rate"] == 0.0),
        "zero_referee_contamination_each_fold": (
            selected["worst_fold_referee_to_player_rate"] == 0.0),
        "safe_candidate_selected": selected["feasible"] is True,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    print(json.dumps({
        "status": status,
        "package_version": __version__,
        "schema_version": "stage5-grouped-train-validation-0.2.8",
        "checks": checks,
    }, indent=2))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
