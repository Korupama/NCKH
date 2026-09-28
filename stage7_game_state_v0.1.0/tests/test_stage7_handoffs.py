from stage7_game_state.adapters import extract_stage5_players, extract_stage6_contact
from stage7_game_state.core import build_game_state_context


def test_native_handoffs_preserve_contact_frame_and_team_zero():
    s5 = {"selected_frame": 104, "track_team": {
        "track_006": {"team_id": 1, "role": "player"},
        "track_016": {"team_id": 0, "role": "goalkeeper"},
        "track_ref": {"team_id": None, "role": "referee"},
    }}
    s6 = {"schema_version": "stage6-downstream-handoff-1.0",
          "selected_frame": 104,
          "stage7": {"track_id": "track_006", "status": "SUPPORTED", "region": "FOOT"}}
    assert extract_stage6_contact(s6)["frame_index"] == 104
    assert len(extract_stage5_players(s5)) == 3
    ctx = build_game_state_context({"view": {"centre_ray_pitch_hit_m": [30, 0, 0]}}, s5, s6)
    assert ctx.status == "VALID"
    assert ctx.frame_index == 104
    assert ctx.sets["attackers"] == ["track_006"]
    assert ctx.sets["opponents"] == ["track_016"]
    assert ctx.sets["referees_excluded"] == ["track_ref"]


def test_unassigned_native_contact_stays_unresolved():
    s6 = {"schema_version": "stage6-downstream-handoff-1.0", "selected_frame": 104,
          "stage7": {"track_id": None, "status": "UNASSIGNED"}}
    ctx = build_game_state_context({}, {"track_team": {}}, s6)
    assert ctx.status == "UNRESOLVED"
    assert "MISSING_CONTACT_TRACK_ID" in ctx.reasons
