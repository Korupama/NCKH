import json
from pathlib import Path

from preflight_stage7 import build_preflight


def _write(path: Path, obj):
    path.write_text(json.dumps(obj), encoding="utf-8")
    return str(path)


def test_preflight_uses_camera_fallback_and_flags_known_referee(tmp_path):
    s1 = _write(tmp_path / "s1.json", {
        "frame_index": 104,
        "status": "DEGRADED",
        "R": [[0.866025403784, 0, 0.5], [0, 1, 0], [0.5, 0, -0.866025403784]],
        "C": [0, 0, 10],
    })
    s5 = _write(tmp_path / "s5.json", {
        "frame_index": 104,
        "players": [
            {"track_id": "track_003", "team_id": 0, "role": "PLAYER"},
            {"track_id": "track_006", "team_id": 1, "role": "PLAYER"},
        ],
    })
    s6 = _write(tmp_path / "s6.json", {
        "frame_index": 104,
        "contact": {"track_id": "track_006", "region": "FOOT", "status": "SUPPORTED"},
    })
    report = build_preflight(s1, s5, s6, known_referee_tracks=["track_003"])
    assert report["centre_ray"]["derived"] is True
    assert report["centre_ray_pitch_hit_m"] is not None
    assert "CAMERA_DEGRADED_REVIEW_REQUIRED" in report["warnings"]
    assert "KNOWN_REFEREE_MISLABELLED_UPSTREAM" in report["blockers"]
    assert report["status"] == "BLOCKED"
