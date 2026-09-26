from stage9_offside_position.core import build_offside_position_state
from .helpers import stage4_basic, stage7_basic, stage8_basic


def labels(state):
    return {r["track_id"]:r["label"] for r in state.attackers}


def test_ltr_classification():
    s=build_offside_position_state(stage4_basic(1),stage7_basic(1),stage8_basic(1))
    assert labels(s)=={"a0":"TOUCHER_EXCLUDED","a1":"OFFSIDE_POSITION","a2":"ONSIDE"}
    assert s.status=="DEMO_BEST_EFFORT"


def test_rtl_classification_is_symmetric():
    s=build_offside_position_state(stage4_basic(-1),stage7_basic(-1),stage8_basic(-1))
    assert labels(s)=={"a0":"TOUCHER_EXCLUDED","a1":"OFFSIDE_POSITION","a2":"ONSIDE"}


def test_demo_ignores_upstream_unresolved():
    s=build_offside_position_state(stage4_basic(1),stage7_basic(1,"UNRESOLVED"),stage8_basic(1,"UNRESOLVED"),best_effort=True)
    assert labels(s)["a1"]=="OFFSIDE_POSITION"
    assert s.diagnostics["ignore_upstream_status_for_demo"] is True


def test_strict_honors_upstream_status():
    s=build_offside_position_state(stage4_basic(1),stage7_basic(1,"UNRESOLVED"),stage8_basic(1,"UNRESOLVED"),best_effort=False)
    assert s.status=="UNRESOLVED"


def test_missing_attacker_geometry_is_still_visible():
    s4=stage4_basic(1)
    s4["tracks"]=[t for t in s4["tracks"] if t["track_id"]!="a2"]
    s=build_offside_position_state(s4,stage7_basic(1),stage8_basic(1))
    row=next(r for r in s.attackers if r["track_id"]=="a2")
    assert row["label"]=="UNAVAILABLE"
    assert row["flag"]=="ON"


def test_opponent_ranking_is_carried_for_demo():
    s=build_offside_position_state(stage4_basic(1),stage7_basic(1),stage8_basic(1))
    d2=next(r for r in s.opponents if r["track_id"]=="d2")
    assert d2["rank"]==2 and d2["second_last"] is True


def test_stage9_can_recompute_reference_when_stage8_has_none():
    s8={"frame_index":104,"status":"UNRESOLVED","attack_direction":{"s":1},"opponent_ranking":[],"reference":None}
    s6={"selected_frame":104,"stage8":{"X_world_m":44.0,"ball_center_x_extent_m":[43.9,44.1]}}
    s=build_offside_position_state(stage4_basic(1),stage7_basic(1,"UNRESOLVED"),s8,stage6_input=s6,best_effort=True)
    assert s.reference["status"]=="STAGE9_RECOMPUTED_BEST_EFFORT"
    assert abs(s.reference["goalward_q_m"]-47.0)<1e-9
    assert labels(s)["a1"]=="OFFSIDE_POSITION"


def test_stage8_is_optional_in_best_effort_mode():
    s6={"selected_frame":104,"stage8":{"X_world_m":44.0,"ball_center_x_extent_m":[43.9,44.1]}}
    s=build_offside_position_state(stage4_basic(1),stage7_basic(1,"UNRESOLVED"),None,stage6_input=s6,best_effort=True)
    assert s.reference["status"]=="STAGE9_RECOMPUTED_BEST_EFFORT"
    assert labels(s)["a1"]=="OFFSIDE_POSITION"


def test_stage7_referees_are_carried_as_other_objects():
    s7=stage7_basic(1)
    s7["sets"]["referees_excluded"]=["r1"]
    s=build_offside_position_state(stage4_basic(1),s7,stage8_basic(1))
    assert s.others == [{"track_id":"r1","label":"REFEREE"}]
