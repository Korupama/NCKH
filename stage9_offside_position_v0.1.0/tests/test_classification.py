from stage9_offside_position.classification import classify_attacker


def test_offside_when_goalward_of_reference():
    r = classify_attacker(track_id="a", goalward_q_m=48, reference_q_m=47, toucher_track_id="p")
    assert r["label"] == "OFFSIDE_POSITION"
    assert r["flag"] == "OFF"


def test_onside_when_level():
    r = classify_attacker(track_id="a", goalward_q_m=47, reference_q_m=47, toucher_track_id="p")
    assert r["label"] == "ONSIDE"


def test_onside_in_own_half_even_if_reference_is_behind():
    r = classify_attacker(track_id="a", goalward_q_m=-1, reference_q_m=-5, toucher_track_id="p")
    assert r["label"] == "ONSIDE"
    assert r["own_half_or_halfway"] is True


def test_toucher_excluded():
    r = classify_attacker(track_id="a", goalward_q_m=50, reference_q_m=47, toucher_track_id="a")
    assert r["label"] == "TOUCHER_EXCLUDED"
    assert r["candidate"] is False


def test_missing_geometry_best_effort_defaults_visual_on():
    r = classify_attacker(track_id="a", goalward_q_m=None, reference_q_m=47, toucher_track_id=None)
    assert r["label"] == "UNAVAILABLE"
    assert r["flag"] == "ON"
