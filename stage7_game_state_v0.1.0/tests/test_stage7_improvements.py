from stage7_game_state.core import build_game_state_context
from preflight_stage7 import build_preflight
import json
import tempfile
from pathlib import Path


def test_attack_direction_fallback_to_explicit():
    # Centre ray hit is exactly 0 (ambiguous), but explicit attack_direction_s is present
    s1 = {"view": {"centre_ray_pitch_hit_m": [0.0, 0.0, 0.0]}, "attack_direction_s": 1}
    s5 = {"players": [
        {"track_id": "1", "team_id": 0, "role": "PLAYER"},
        {"track_id": "2", "team_id": 1, "role": "PLAYER"},
    ]}
    s6 = {"contact_track_id": "1"}
    ctx = build_game_state_context(s1, s5, s6)
    assert ctx.status == "VALID"
    assert ctx.attack_direction["s"] == 1
    assert ctx.attack_direction["source"] == "upstream_explicit_attack_direction"


def test_attack_direction_fallback_to_view_half():
    # Centre ray hit is missing, but view pitch half is "LEFT"
    s1 = {"view": {"view_pitch_half": "LEFT"}}
    s5 = {"players": [
        {"track_id": "1", "team_id": 0, "role": "PLAYER"},
        {"track_id": "2", "team_id": 1, "role": "PLAYER"},
    ]}
    s6 = {"contact_track_id": "1"}
    ctx = build_game_state_context(s1, s5, s6)
    assert ctx.status == "VALID"
    assert ctx.attack_direction["s"] == -1
    assert ctx.attack_direction["source"] == "stage1.view_pitch_half_label"


def test_preflight_tentative_contact_degraded_ready():
    with tempfile.TemporaryDirectory() as tmpdir:
        p1 = Path(tmpdir) / "stage1.json"
        p5 = Path(tmpdir) / "stage5.json"
        p6 = Path(tmpdir) / "stage6.json"

        p1.write_text(json.dumps({"view": {"centre_ray_pitch_hit_m": [15.0, 0.0, 0.0]}}))
        p5.write_text(json.dumps({"players": [
            {"track_id": "1", "team_id": 0, "role": "PLAYER"},
            {"track_id": "2", "team_id": 1, "role": "PLAYER"},
        ]}))
        # Tentative spatial contact with nearest_track_id
        p6.write_text(json.dumps({
            "contact": {
                "track_id": None,
                "status": "INSUFFICIENT_TEMPORAL_SUPPORT",
                "nearest_track_id": "1",
                "image_distance_px": 8.5,
                "threshold_px": 25.0
            }
        }))

        res = build_preflight(str(p1), str(p5), str(p6))
        assert res["status"] == "DEGRADED_READY"
        assert "CONTACT_TENTATIVE_SPATIAL_ONLY" in res["warnings"]
        assert len(res["blockers"]) == 0
