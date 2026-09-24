import json
from pathlib import Path
from preflight_stage7 import build_preflight


def _write(path: Path, obj):
    path.write_text(json.dumps(obj), encoding="utf-8")
    return str(path)


def test_preflight_accepts_actual_stage1_v12_fallback(tmp_path):
    s1 = _write(tmp_path / "s1.json", {
        "frame_index": 104,
        "status": "DEGRADED",
        "pitch": {"length_m": 105.0, "width_m": 68.0},
        "extrinsics": {
            "R_world_to_camera": [[0.866025403784, 0, 0.5], [0, 1, 0], [0.5, 0, -0.866025403784]],
            "camera_center_world_m": [0, 0, 10],
        },
    })
    s5 = _write(tmp_path / "s5.json", {
        "frame_index": 104,
        "players": [
            {"track_id": "track_006", "team_id": 1, "role": "PLAYER"},
            {"track_id": "track_007", "team_id": 0, "role": "PLAYER"},
        ],
    })
    s6 = _write(tmp_path / "s6.json", {
        "frame_index": 104,
        "contact": {"track_id": "track_006", "region": "FOOT", "status": "SUPPORTED"},
    })
    report = build_preflight(s1, s5, s6)
    assert report["status"] == "READY"
    assert report["centre_ray"]["derived"] is True
    assert report["centre_ray"]["valid"] is True
    assert report["centre_ray"]["geometry"]["rotation_path"] == "extrinsics.R_world_to_camera"
    assert "CAMERA_DEGRADED_REVIEW_REQUIRED" in report["warnings"]
