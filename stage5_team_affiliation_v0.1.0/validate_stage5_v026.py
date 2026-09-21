from __future__ import annotations

import json

from stage5_team_affiliation import __version__
from stage5_team_affiliation.residual_calibration import (
    evaluate_player_recovery_policy,
    search_player_recovery_policy,
)


def _row(track_id: str, gt_role: str, *, base_role: str = "unknown") -> dict:
    return {
        "sequence_id": "validation-001",
        "track_id": track_id,
        "gt_role": gt_role,
        "gt_team": "left" if gt_role == "player" else None,
        "mapping": {"0": "left", "1": "right"},
        "mapping_available": True,
        "appearance_group": "RESIDUAL",
        "margin": 0.50,
        "nearest_team": 0,
        "nearest_distance": 0.05,
        "base_role": base_role,
        "base_role_status": "VALID" if base_role != "unknown" else "UNKNOWN",
        "base_team": None,
        "base_team_status": "UNKNOWN",
    }


def main() -> None:
    rows = [
        _row("player-1", "player"),
        _row("referee-1", "referee"),
        _row("goalkeeper-1", "goalkeeper", base_role="goalkeeper"),
    ]
    metrics = evaluate_player_recovery_policy(
        rows,
        player_margin=0.08,
        player_max_distance=0.20,
    )
    search = search_player_recovery_policy(
        rows,
        baseline_outfield_accuracy=0.0,
        max_referee_to_player_rate=0.05,
    )

    checks = {
        "referee_contamination_is_reported": (
            metrics["referee_tracks"] == 1
            and metrics["referee_to_player"] == 1
            and metrics["referee_to_player_rate"] == 1.0
        ),
        "unsafe_player_recovery_is_rejected": (
            search["feasible_count"] == 0
            and search["selected"]["feasible"] is False
            and search["selected"]["metrics"]["referee_to_player_rate"] > 0.05
        ),
        "package_version_supports_v026": tuple(
            int(part) for part in __version__.split(".")) >= (0, 2, 6),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    report = {
        "status": status,
        "package_version": __version__,
        "schema_version": "stage5-policy-validation-0.2.6",
        "checks": checks,
    }
    print(json.dumps(report, indent=2))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
