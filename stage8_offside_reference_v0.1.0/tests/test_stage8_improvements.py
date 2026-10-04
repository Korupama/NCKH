from stage8_offside_reference.core import build_offside_reference
from stage8_offside_reference.preflight import build_preflight
from helpers import joint, track, stage4, stage6, stage7


def test_partial_opponents_fallback_degraded_ready():
    # d1 and d2 have usable geometry, d3 is missing from stage4
    s4 = stage4(104, [track("d1", 104, [joint("nose", 50.0)]), track("d2", 104, [joint("nose", 47.0)])])
    s6_data = stage6(104)
    s7_data = stage7(104, 1, ["d1", "d2", "d3"])

    # Strict mode: blocked
    pre_strict = build_preflight(s4, s6_data, s7_data, allow_partial_opponents=False)
    assert pre_strict["status"] == "BLOCKED"
    assert "OPPONENT_STAGE4_TRACK_MISSING" in pre_strict["blockers"]

    # Permissive fallback mode: DEGRADED_READY
    pre_permissive = build_preflight(s4, s6_data, s7_data, allow_partial_opponents=True)
    assert pre_permissive["status"] == "DEGRADED_READY"
    assert "OPPONENT_GEOMETRY_PARTIAL_DEGRADED" in pre_permissive["warnings"]

    # build_offside_reference succeeds with DEGRADED status
    state = build_offside_reference(s4, s6_data, s7_data, allow_partial_opponents=True)
    assert state.status == "DEGRADED"
    assert state.second_last_opponent["track_id"] == "d2"
    assert state.reference["source"] == "SECOND_LAST_OPPONENT"
    assert state.reference["X_world_m"] == 47.0


def test_root_fallback_when_landmarks_missing():
    # Track d1 has root_world_m but empty joints
    tr_d1 = {
        "track_id": "d1",
        "selected_frame_status": "VALID",
        "observations": [{
            "frame_index": 104,
            "root_world_m": [49.5, 0.0, 1.0],
            "joints_world": []
        }]
    }
    s4 = stage4(104, [tr_d1, track("d2", 104, [joint("nose", 47.0)])])
    s6_data = stage6(104)
    s7_data = stage7(104, 1, ["d1", "d2"])

    # Without root fallback: d1 is unusable -> FEWER_THAN_TWO_USABLE_OPPONENTS
    pre_no_root = build_preflight(s4, s6_data, s7_data, allow_root_fallback=False)
    assert pre_no_root["status"] == "BLOCKED"

    # With root fallback: d1 is degraded usable -> READY
    pre_root = build_preflight(s4, s6_data, s7_data, allow_root_fallback=True)
    assert pre_root["status"] == "READY"
    state = build_offside_reference(s4, s6_data, s7_data, allow_root_fallback=True)
    assert state.status == "VALID"
    assert state.second_last_opponent["track_id"] == "d2"


def test_ball_fallback_to_second_last():
    s4 = stage4(104, [track("d1", 104, [joint("nose", 50.0)]), track("d2", 104, [joint("nose", 47.0)])])
    # Ball is unusable
    s6_data = stage6(104, usable=False)
    s7_data = stage7(104, 1, ["d1", "d2"])

    # Strict: blocked
    pre_strict = build_preflight(s4, s6_data, s7_data, allow_ball_fallback=False)
    assert pre_strict["status"] == "BLOCKED"

    # Fallback: DEGRADED_READY
    pre_fb = build_preflight(s4, s6_data, s7_data, allow_ball_fallback=True)
    assert pre_fb["status"] == "DEGRADED_READY"

    state = build_offside_reference(s4, s6_data, s7_data, allow_ball_fallback=True)
    assert state.status == "DEGRADED"
    assert state.reference["source"] == "SECOND_LAST_OPPONENT"
    assert state.reference["X_world_m"] == 47.0
