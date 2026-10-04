from stage7_game_state.adapters import extract_stage1_view
from stage7_game_state.core import build_game_state_context


def _stage5():
    return {"players": [
        {"track_id": "track_006", "team_id": 1, "role": "PLAYER"},
        {"track_id": "track_007", "team_id": 0, "role": "PLAYER"},
    ]}


def _stage6():
    return {"frame_index": 104, "contact": {"track_id": "track_006", "region": "FOOT", "status": "SUPPORTED"}}


def test_reads_explicit_stage1_v12_view_contract():
    s1 = {
        "frame_index": 104,
        "status": "DEGRADED",
        "view": {
            "principal_point_px": [959.5, 539.5],
            "centre_ray_pitch_hit_m": [31.42, 2.17, 0.0],
            "centre_ray_pitch_hit_valid": True,
            "view_pitch_half": "RIGHT",
        },
    }
    ctx = build_game_state_context(s1, _stage5(), _stage6())
    assert ctx.status == "VALID"
    assert ctx.attack_direction["s"] == 1
    assert ctx.attack_direction["source"] == "stage1.centre_ray_pitch_hit"
    assert ctx.diagnostics["centre_ray_derived"] is False
    assert ctx.diagnostics["view_pitch_half"] == "RIGHT"


def test_fallback_understands_actual_stage1_v12_extrinsics_keys():
    s1 = {
        "frame_index": 104,
        "status": "DEGRADED",
        "pitch": {"length_m": 105.0, "width_m": 68.0},
        "extrinsics": {
            "R_world_to_camera": [
                [0.866025403784, 0.0, 0.5],
                [0.0, 1.0, 0.0],
                [0.5, 0.0, -0.866025403784],
            ],
            "camera_center_world_m": [0.0, 0.0, 10.0],
        },
    }
    x, hit, meta = extract_stage1_view(s1)
    assert hit is not None
    assert x is not None
    assert abs(x - 5.7735026919) < 1e-6
    assert meta["source"] == "stage1.camera_geometry_fallback"
    assert meta["rotation_path"] == "extrinsics.R_world_to_camera"
    assert meta["camera_center_path"] == "extrinsics.camera_center_world_m"
    assert meta["valid"] is True


def test_outside_pitch_view_resolves_direction_without_rederiving_geometry():
    s1 = {
        "view": {
            "centre_ray_pitch_hit_m": [80.0, 0.0, 0.0],
            "centre_ray_pitch_hit_valid": False,
            "view_pitch_half": None,
            "reason": "CENTRE_RAY_PITCH_HIT_OUT_OF_BOUNDS",
        },
        "extrinsics": {
            "R_world_to_camera": [[1, 0, 0], [0, 1, 0], [0, 0, -1]],
            "camera_center_world_m": [0, 0, 10],
        },
    }
    ctx = build_game_state_context(s1, _stage5(), _stage6())
    assert ctx.status == "DEGRADED"
    assert ctx.attack_direction["s"] == 1
    assert ctx.attack_direction["centre_ray_x_m"] == 80.0
    assert "ATTACK_DIRECTION_FROM_OUT_OF_BOUNDS_CENTRE_RAY" in ctx.reasons
    assert ctx.diagnostics["centre_ray_valid"] is False
    assert ctx.diagnostics["centre_ray_derived"] is False


def test_negative_outside_pitch_view_resolves_left():
    s1 = {"view": {"centre_ray_pitch_hit_m": [-65.61, 20.1, 0],
                   "centre_ray_pitch_hit_valid": False,
                   "reason": "CENTRE_RAY_PITCH_HIT_OUT_OF_BOUNDS"}}
    ctx = build_game_state_context(s1, _stage5(), _stage6())
    assert ctx.attack_direction["s"] == -1
    assert ctx.attack_direction["view_pitch_half"] == "LEFT"
    assert ctx.status == "DEGRADED"


def test_outside_pitch_midfield_stays_ambiguous():
    s1 = {"view": {"centre_ray_pitch_hit_m": [0, 80, 0],
                   "centre_ray_pitch_hit_valid": False,
                   "reason": "CENTRE_RAY_PITCH_HIT_OUT_OF_BOUNDS"}}
    ctx = build_game_state_context(s1, _stage5(), _stage6())
    assert ctx.attack_direction is None
    assert "CENTRE_RAY_X_AMBIGUOUS" in ctx.reasons


def test_other_invalid_geometry_still_rejected():
    s1 = {"view": {"centre_ray_pitch_hit_m": [80, 0, 0],
                   "centre_ray_pitch_hit_valid": False,
                   "reason": "PITCH_INTERSECTION_BEHIND_CAMERA"}}
    ctx = build_game_state_context(s1, _stage5(), _stage6())
    assert ctx.attack_direction is None
    assert "CENTRE_RAY_PITCH_HIT_INVALID" in ctx.reasons
