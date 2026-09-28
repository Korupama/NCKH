def joints(xs):
    names = ["nose", "left_shoulder", "right_hip", "left_big_toe"]
    return [
        {"name": name, "xyz_world_m": [float(x), 0.0, 1.0 if "toe" not in name else 0.02], "valid": True}
        for name, x in zip(names, xs)
    ]


def stage4_basic(s=1):
    sign = 1 if s == 1 else -1
    return {
        "schema_version": "stage4-downstream-handoff-2.1",
        "producer": "stage4-sam3d-pitch-refined-0.5.1",
        "selected_frame": 104,
        "tracks": [
            {"track_id":"a0","role":"player","observations":[{"frame_index":104,"joints_world":joints([sign*40]*4)}]},
            {"track_id":"a1","role":"player","observations":[{"frame_index":104,"joints_world":joints([sign*48,sign*47.5,sign*47.2,sign*47.8])}]},
            {"track_id":"a2","role":"player","observations":[{"frame_index":104,"joints_world":joints([sign*46,sign*45.5,sign*45.2,sign*45.8])}]},
            {"track_id":"d1","role":"player","observations":[{"frame_index":104,"joints_world":joints([sign*49]*4)}]},
            {"track_id":"d2","role":"goalkeeper","observations":[{"frame_index":104,"joints_world":joints([sign*47]*4)}]},
        ]
    }


def stage7_basic(s=1, status="VALID"):
    return {
        "frame_index":104,
        "status":status,
        "attack_direction":{"s":s,"label":"LEFT_TO_RIGHT" if s==1 else "RIGHT_TO_LEFT"},
        "toucher":{"track_id":"a0"},
        "sets":{"attackers":["a0","a1","a2"],"opponents":["d1","d2"]}
    }


def stage8_basic(s=1, status="VALID", q=47.0):
    return {
        "frame_index":104,
        "status":status,
        "attack_direction":{"s":s},
        "reference":{"goalward_q_m":q,"X_world_m":s*q,"source":"SECOND_LAST_OPPONENT"},
        "second_last_opponent":{"track_id":"d2","candidate_track_ids":["d2"],"goalward_q_m":q,"X_world_m":s*q},
        "opponent_ranking":[
            {"track_id":"d1","rank":1,"goalward_q_m":49.0,"goalward_x_m":s*49.0,"anchor":{"name":"nose","xyz_world_m":[s*49,0,1.7]}},
            {"track_id":"d2","rank":2,"goalward_q_m":47.0,"goalward_x_m":s*47.0,"anchor":{"name":"nose","xyz_world_m":[s*47,0,1.7]}},
        ],
        "ball":{"goalward_q_m":44.0,"goalward_x_m":s*44.0}
    }
