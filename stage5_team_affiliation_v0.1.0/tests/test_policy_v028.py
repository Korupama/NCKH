from __future__ import annotations

from stage5_team_affiliation.residual_calibration import (
    search_player_recovery_policy_grouped,
)


def _row(sequence_id: str, track_id: str, gt_role: str, distance: float) -> dict:
    base_role = "goalkeeper" if gt_role == "goalkeeper" else "unknown_residual"
    base_valid = gt_role == "goalkeeper"
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
        "base_role": base_role,
        "base_role_status": "VALID" if base_valid else "UNKNOWN",
        "base_team": None,
        "base_team_status": "UNKNOWN",
    }


def test_grouped_train_selection_prefers_zero_referee_contamination() -> None:
    rows = []
    for index in range(10):
        sid = f"train-{index:02d}"
        rows.extend([
            _row(sid, "player", "player", 0.20),
            _row(sid, "referee", "referee", 0.35),
            _row(sid, "goalkeeper", "goalkeeper", 0.60),
        ])

    result = search_player_recovery_policy_grouped(
        rows,
        baseline_outfield_accuracy=0.0,
        n_folds=5,
    )

    assert result["selected"]["feasible"] is True
    assert result["selected"]["parameters"]["residual_player_max_distance"] == 0.30
    assert result["selected"]["metrics"]["referee_to_player"] == 0
    assert result["selected"]["worst_fold_referee_to_player_rate"] == 0.0
    assert result["fold_count"] == 5


def test_grouped_train_selection_rejects_when_no_safe_player_policy_exists() -> None:
    rows = []
    for index in range(5):
        sid = f"train-{index:02d}"
        rows.extend([
            _row(sid, "player", "player", 0.20),
            _row(sid, "referee", "referee", 0.20),
            _row(sid, "goalkeeper", "goalkeeper", 0.60),
        ])

    result = search_player_recovery_policy_grouped(
        rows,
        baseline_outfield_accuracy=0.0,
        n_folds=5,
    )

    assert result["feasible_count"] == 0
    assert result["selected"]["feasible"] is False

