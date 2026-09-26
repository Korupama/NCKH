from stage8_offside_reference.core import build_offside_reference
from helpers import joint, track, stage4, stage6, stage7


def test_ltr_valid_reference():
    s4 = stage4(104, [
        track("gk",104,[joint("left_big_toe",50)] , role="goalkeeper"),
        track("d2",104,[joint("right_shoulder",47)]),
        track("d3",104,[joint("nose",42)]),
    ])
    out = build_offside_reference(s4, stage6(104,(44,44.22)), stage7(104,1,["gk","d2","d3"])).to_dict()
    assert out["status"] == "VALID"
    assert out["second_last_opponent"]["track_id"] == "d2"
    assert out["reference"]["source"] == "SECOND_LAST_OPPONENT"
    assert out["reference"]["X_world_m"] == 47
    assert out["quality"]["upstream_metric_accuracy_validated"] is False


def test_rtl_valid_reference():
    s4 = stage4(104, [track("d1",104,[joint("nose",-50)]),track("d2",104,[joint("nose",-47)])])
    out = build_offside_reference(s4, stage6(104,(-44.22,-44)), stage7(104,-1,["d1","d2"])).to_dict()
    assert out["status"] == "VALID"
    assert out["reference"]["X_world_m"] == -47


def test_ball_ahead_is_reference():
    s4 = stage4(104, [track("d1",104,[joint("nose",48)]),track("d2",104,[joint("nose",46)])])
    out = build_offside_reference(s4, stage6(104,(49,49.22)), stage7(104,1,["d1","d2"])).to_dict()
    assert out["reference"]["source"] == "BALL"
    assert out["reference"]["X_world_m"] == 49.22


def test_goalkeeper_has_no_special_case():
    s4 = stage4(104, [track("gk",104,[joint("nose",49)],role="goalkeeper"),track("p",104,[joint("nose",47)])])
    out = build_offside_reference(s4, stage6(104,(40,40.22)), stage7(104,1,["gk","p"])).to_dict()
    assert [r["track_id"] for r in out["opponent_ranking"]] == ["gk","p"]
    assert out["second_last_opponent"]["track_id"] == "p"


def test_partial_geometry_never_emits_reference():
    s4 = stage4(104, [track("d1",104,[joint("nose",49)]),track("d2",104,[joint("nose",47)])])
    out = build_offside_reference(s4, stage6(104), stage7(104,1,["d1","d2","missing"])).to_dict()
    assert out["status"] == "UNRESOLVED"
    assert out["reference"] is None
    assert out["diagnostics"]["ranking_complete"] is False
