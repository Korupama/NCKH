def joint(name, x, y=0.0, z=0.5, valid=True):
    return {"name": name, "xyz_world_m": [x, y, z], "valid": valid}


def track(tid, frame, joints, role="player", status="VALID"):
    return {"track_id": tid, "role": role, "selected_frame_status": status, "observations": [{"frame_index": frame, "joints_world": joints, "quality": status}]}


def stage4(frame, tracks, coord=None):
    return {
        "schema_version": "stage4-downstream-handoff-2.1",
        "producer": "stage4-sam3d-pitch-refined-0.5.1",
        "selected_frame": frame,
        "coordinate_frame": coord or {"name": "STAGE1_PITCH_WORLD", "units": "m", "x": "goal-to-goal", "y": "touchline-to-touchline", "z": "up", "pitch_plane": "z=0"},
        "tracks": tracks,
    }


def stage6(frame, extent=(40.0, 40.22), usable=True, validated=False):
    return {
        "schema_version": "stage6-downstream-handoff-1.0",
        "selected_frame": frame,
        "stage8": {
            "selected_method": "CONTACT_GROUND_PLANE",
            "center_xyz_world_m": [sum(extent)/2, 0.0, 0.11] if extent else None,
            "X_world_m": sum(extent)/2 if extent else None,
            "ball_center_x_extent_m": list(extent) if extent else None,
            "usable_for_offside": usable,
            "accuracy_validated": validated,
        },
        "research_accuracy_frozen": False,
    }


def stage7(frame, s, opponents, status="VALID"):
    return {
        "frame_index": frame,
        "status": status,
        "attack_direction": {"s": s, "label": "LEFT_TO_RIGHT" if s == 1 else "RIGHT_TO_LEFT", "source": "stage1.centre_ray_pitch_hit"},
        "sets": {"attackers": ["a"], "opponents": list(opponents), "referees_excluded": []},
    }
