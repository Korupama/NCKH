from stage7_game_state.core import build_game_state_context


def test_valid_context_right_half():
    s1 = {"view": {"centre_ray_pitch_hit_m": [12.5, 0.0, 0.0]}}
    s5 = {"players": [
        {"track_id": "3", "team_id": 0, "role": "PLAYER"},
        {"track_id": "4", "team_id": 0, "role": "GOALKEEPER"},
        {"track_id": "7", "team_id": 1, "role": "PLAYER"},
        {"track_id": "8", "team_id": 1, "role": "GOALKEEPER"},
        {"track_id": "9", "role": "REFEREE"},
    ]}
    s6 = {"selected_frame_ball": {"contact": {"track_id": "3"}}}
    ctx = build_game_state_context(s1, s5, s6)
    assert ctx.status == "VALID"
    assert ctx.attacking_team_id == 0
    assert ctx.attack_direction["s"] == 1
    assert set(ctx.sets["attackers"]) == {"3", "4"}
    assert set(ctx.sets["opponents"]) == {"7", "8"}
    assert ctx.sets["referees_excluded"] == ["9"]
    assert ctx.diagnostics["invariant_pass"] is True


def test_left_half_direction():
    s1 = {"view": {"centre_ray_pitch_hit_m": [-8.0, 1.0, 0.0]}}
    s5 = {"by_track": {"1": {"team_id": "A"}, "2": {"team_id": "B"}}}
    s6 = {"contact_track_id": "2"}
    ctx = build_game_state_context(s1, s5, s6)
    assert ctx.status == "VALID"
    assert ctx.attack_direction["s"] == -1
    assert ctx.attacking_team_id == "B"


def test_missing_toucher_is_unresolved():
    s1 = {"view": {"centre_ray_pitch_hit_m": [8.0, 0.0, 0.0]}}
    s5 = {"players": [{"track_id": "1", "team_id": 0}, {"track_id": "2", "team_id": 1}]}
    s6 = {}
    ctx = build_game_state_context(s1, s5, s6)
    assert ctx.status == "UNRESOLVED"
    assert "MISSING_CONTACT_TRACK_ID" in ctx.reasons


def test_zero_center_ray_is_unresolved():
    s1 = {"view": {"centre_ray_pitch_hit_m": [0.0, 0.0, 0.0]}}
    s5 = {"players": [{"track_id": "1", "team_id": 0}, {"track_id": "2", "team_id": 1}]}
    s6 = {"contact_track_id": "1"}
    ctx = build_game_state_context(s1, s5, s6)
    assert ctx.status == "UNRESOLVED"
    assert "CENTRE_RAY_X_AMBIGUOUS" in ctx.reasons


def test_goalkeeper_is_normal_opponent_member():
    s1 = {"view": {"centre_ray_pitch_hit_m": [20.0, 0.0, 0.0]}}
    s5 = {"players": [
        {"track_id": "a", "team_id": "A"},
        {"track_id": "gk", "team_id": "B", "role": "GOALKEEPER"},
    ]}
    s6 = {"contact_track_id": "a"}
    ctx = build_game_state_context(s1, s5, s6)
    assert ctx.status == "VALID"
    assert ctx.sets["opponents"] == ["gk"]
