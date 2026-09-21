from __future__ import annotations

from stage5_team_affiliation.residual_cache_v028 import reset_and_recover_prediction
from stage5_team_affiliation.residual_config import ResidualConfig


def _record(track_id, distance, *, method="RESIDUAL_HIGH_MARGIN_PLAYER_RECOVERY"):
    return {
        "track_id": track_id,
        "appearance_group": "RESIDUAL",
        "stage5_role": "player",
        "stage5_role_status": "VALID",
        "stage5_role_method": method,
        "team_id": 0,
        "team_status": "VALID",
        "assignment_method": "TRAIN_CALIBRATED_NEAREST_TEAM_PROTOTYPE",
        "distances": [distance, distance + 0.50],
        "margin": 0.50,
        "goal_context": {},
    }


def test_cached_policy_resets_only_old_player_recovery() -> None:
    prediction = {
        "tracks": [
            _record("safe-player", 0.20),
            _record("unsafe-residual", 0.35),
            {
                **_record("goalkeeper", 0.10, method="TEMPORAL_GOALKEEPER_GEOMETRY"),
                "stage5_role": "goalkeeper",
                "team_id": None,
                "team_status": "UNKNOWN",
            },
        ]
    }
    cfg = ResidualConfig(
        residual_player_recovery_enabled=True,
        residual_player_min_margin=0.15,
        residual_player_max_distance=0.30,
    )
    result = reset_and_recover_prediction(prediction, cfg)
    records = {record["track_id"]: record for record in result["tracks"]}
    assert records["safe-player"]["stage5_role"] == "player"
    assert records["unsafe-residual"]["stage5_role_status"] == "UNKNOWN"
    assert records["goalkeeper"]["stage5_role"] == "goalkeeper"
    assert result["cache_reevaluation"]["images_read"] is False
