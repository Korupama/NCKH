from stage8_offside_reference.adapters import extract_stage4_context, extract_stage6_ball, extract_stage7_context


def test_stage6_downstream_handoff_stage8_block():
    payload = {"selected_frame":104,"stage8":{"X_world_m":12.3,"ball_center_x_extent_m":[12.19,12.41],"usable_for_offside":True,"accuracy_validated":False}}
    out = extract_stage6_ball(payload)
    assert out["frame_index"] == 104
    assert out["X_world_m"] == 12.3
    assert out["ball_center_x_extent_m"] == [12.19,12.41]
    assert out["source"] == "stage6.downstream_handoff.stage8"


def test_full_stage6_state_fallback_supported():
    payload = {"selected_frame_ball":{"frame_index":104,"localization":{"X_world_m":1.2,"ball_center_x_extent_m":[1.09,1.31],"usable_for_offside":True}}}
    out = extract_stage6_ball(payload)
    assert out["frame_index"] == 104
    assert out["source"] == "stage6.selected_frame_ball.localization"


def test_stage7_context_reads_opponents_and_direction():
    out = extract_stage7_context({"frame_index":104,"status":"VALID","attack_direction":{"s":1},"sets":{"opponents":["track_1",2]}})
    assert out["opponents"] == ["track_1","2"]
    assert out["s"] == 1


def test_stage4_context_indexes_tracks():
    out = extract_stage4_context({"selected_frame":104,"coordinate_frame":{},"tracks":[{"track_id":"x"}]})
    assert "x" in out["tracks"]
