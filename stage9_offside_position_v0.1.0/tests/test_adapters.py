from stage9_offside_position.adapters import extract_stage7, extract_stage8, bbox_from_stage3_observation, keypoints_from_stage3_observation


def test_stage7_extracts_sets_even_if_unresolved():
    r=extract_stage7({"status":"UNRESOLVED","frame_index":104,"attack_direction":{"s":1},"sets":{"attackers":[1],"opponents":[2]},"toucher":{"track_id":1}})
    assert r["attackers"]==["1"] and r["opponents"]==["2"] and r["toucher_track_id"]=="1"


def test_stage8_extracts_reference():
    r=extract_stage8({"frame_index":104,"status":"VALID","attack_direction":{"s":-1},"reference":{"goalward_q_m":48,"X_world_m":-48,"source":"BALL"}})
    assert r["reference_q_m"]==48 and r["reference_x_m"]==-48


def test_stage3_bbox_and_keypoints():
    obs={"bbox_xyxy":[1,2,30,40],"keypoints_133":[{"name":"nose","x":4,"y":5,"state":"VALID"}]}
    assert bbox_from_stage3_observation(obs)==[1.0,2.0,30.0,40.0]
    assert keypoints_from_stage3_observation(obs)[0]["name"]=="nose"


def test_stage6_extract_ball_extent():
    from stage9_offside_position.adapters import extract_stage6
    r=extract_stage6({"selected_frame":104,"stage8":{"X_world_m":44.0,"ball_center_x_extent_m":[43.9,44.1]}})
    assert r["frame_index"]==104 and r["ball_center_x_extent_m"]==[43.9,44.1]


def test_stage7_extracts_excluded_groups():
    r=extract_stage7({"sets":{"referees_excluded":["r1"],"unknown_team_excluded":["u1"],"inactive_excluded":["i1"]}})
    assert r["referees"]==["r1"] and r["unknown"]==["u1"] and r["inactive"]==["i1"]
