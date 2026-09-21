from __future__ import annotations

from stage5_team_affiliation.residual_calibration import (
    evaluate_player_recovery_policy,
    search_player_recovery_policy,
)


def _row(
    track_id: str,
    gt_role: str,
    *,
    appearance_group: str = "TEAM_0",
    margin: float = 0.50,
    nearest_distance: float = 0.05,
    base_role: str = "unknown",
) -> dict:
    return {
        "sequence_id": "synthetic-001",
        "track_id": track_id,
        "gt_role": gt_role,
        "gt_team": "left" if gt_role == "player" else None,
        "mapping": {"0": "left", "1": "right"},
        "mapping_available": True,
        "appearance_group": "RESIDUAL",
        "margin": margin,
        "nearest_team": 0,
        "nearest_distance": nearest_distance,
        "base_role": base_role,
        "base_role_status": "VALID" if base_role != "unknown" else "UNKNOWN",
        "base_team": None,
        "base_team_status": "UNKNOWN",
    }


def test_player_recovery_reports_referee_contamination() -> None:
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

    assert metrics["referee_tracks"] == 1
    assert metrics["referee_to_player"] == 1
    assert metrics["referee_to_player_rate"] == 1.0


def test_unsafe_player_recovery_is_not_freezable() -> None:
    rows = [
        _row("player-1", "player"),
        _row("referee-1", "referee"),
        _row("goalkeeper-1", "goalkeeper", base_role="goalkeeper"),
    ]

    result = search_player_recovery_policy(
        rows,
        baseline_outfield_accuracy=0.0,
        max_referee_to_player_rate=0.05,
    )

    assert result["feasible_count"] == 0
    assert result["selected"]["feasible"] is False
    assert result["selected"]["metrics"]["referee_to_player_rate"] > 0.05
