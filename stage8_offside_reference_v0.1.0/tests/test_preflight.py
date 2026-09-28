from stage8_offside_reference.preflight import build_preflight
from helpers import joint, track, stage4, stage6, stage7


def test_ready_real_schema_shape():
    s4 = stage4(104, [track("d1",104,[joint("nose",50)]), track("d2",104,[joint("nose",47)])])
    p = build_preflight(s4, stage6(104), stage7(104,1,["d1","d2"]))
    assert p["status"] == "READY"
    assert p["stage4"]["opponent_tracks_usable"] == 2
    assert "BALL_METRIC_ACCURACY_NOT_VALIDATED" in p["warnings"]


def test_frame_mismatch_blocks():
    s4 = stage4(104, [track("d1",104,[joint("nose",50)]), track("d2",104,[joint("nose",47)])])
    p = build_preflight(s4, stage6(105), stage7(104,1,["d1","d2"]))
    assert "FRAME_INDEX_MISMATCH" in p["blockers"]


def test_invalid_coordinate_frame_blocks():
    coord = {"name":"WRONG","units":"m","x":"goal-to-goal","y":"touchline-to-touchline","z":"up","pitch_plane":"z=0"}
    s4 = stage4(104, [track("d1",104,[joint("nose",50)]), track("d2",104,[joint("nose",47)])], coord=coord)
    p = build_preflight(s4, stage6(104), stage7(104,1,["d1","d2"]))
    assert "COORDINATE_FRAME_MISMATCH" in p["blockers"]


def test_missing_any_opponent_geometry_fail_closed():
    s4 = stage4(104, [track("d1",104,[joint("nose",50)]), track("d2",104,[joint("nose",47)])])
    p = build_preflight(s4, stage6(104), stage7(104,1,["d1","d2","d3"]))
    assert "OPPONENT_STAGE4_TRACK_MISSING" in p["blockers"]
    assert "OPPONENT_GEOMETRY_INCOMPLETE" in p["blockers"]


def test_unresolved_stage7_blocks():
    s4 = stage4(104, [track("d1",104,[joint("nose",50)]), track("d2",104,[joint("nose",47)])])
    p = build_preflight(s4, stage6(104), stage7(104,1,["d1","d2"], status="UNRESOLVED"))
    assert "STAGE7_CONTEXT_UNRESOLVED" in p["blockers"]


def test_ball_unusable_blocks():
    s4 = stage4(104, [track("d1",104,[joint("nose",50)]), track("d2",104,[joint("nose",47)])])
    p = build_preflight(s4, stage6(104, usable=False), stage7(104,1,["d1","d2"]))
    assert "BALL_LONGITUDINAL_GEOMETRY_UNUSABLE" in p["blockers"]
