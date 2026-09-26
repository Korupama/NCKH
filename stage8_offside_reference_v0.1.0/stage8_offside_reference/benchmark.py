from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .core import build_offside_reference


def _joint(name: str, x: float, y: float = 0.0, z: float = 0.5):
    return {"name": name, "xyz_world_m": [x, y, z], "valid": True}


def _track(tid: str, frame: int, joints, role="player", status="VALID"):
    return {"track_id": tid, "role": role, "selected_frame_status": status, "observations": [{"frame_index": frame, "joints_world": joints, "quality": status}]}


def _s4(frame: int, tracks):
    return {
        "schema_version": "stage4-downstream-handoff-2.1",
        "producer": "stage4-sam3d-pitch-refined-0.5.1",
        "selected_frame": frame,
        "coordinate_frame": {"name": "STAGE1_PITCH_WORLD", "units": "m", "x": "goal-to-goal", "y": "touchline-to-touchline", "z": "up", "pitch_plane": "z=0"},
        "tracks": tracks,
    }


def _s6(frame: int, extent, usable=True):
    return {"schema_version": "stage6-downstream-handoff-1.0", "selected_frame": frame, "stage8": {"X_world_m": sum(extent)/2 if extent else None, "ball_center_x_extent_m": extent, "usable_for_offside": usable, "accuracy_validated": False}}


def _s7(frame: int, s: int, opponents):
    return {"frame_index": frame, "status": "VALID", "attack_direction": {"s": s, "label": "LEFT_TO_RIGHT" if s == 1 else "RIGHT_TO_LEFT"}, "sets": {"attackers": ["a"], "opponents": opponents}}


def oracle_fixtures() -> List[Dict[str, Any]]:
    f = 104
    return [
        {
            "name": "ltr_defender_reference",
            "inputs": (_s4(f, [_track("d1", f, [_joint("left_big_toe", 49)]), _track("d2", f, [_joint("right_shoulder", 47)]), _track("d3", f, [_joint("nose", 40)])]), _s6(f, [44.0, 44.22]), _s7(f, 1, ["d1","d2","d3"])),
            "check": lambda x: x["status"] == "VALID" and x["second_last_opponent"]["track_id"] == "d2" and x["reference"]["source"] == "SECOND_LAST_OPPONENT" and abs(x["reference"]["X_world_m"] - 47.0) < 1e-9,
        },
        {
            "name": "rtl_defender_reference",
            "inputs": (_s4(f, [_track("d1", f, [_joint("left_big_toe", -49)]), _track("d2", f, [_joint("nose", -47)]), _track("d3", f, [_joint("nose", -40)])]), _s6(f, [-44.22, -44.0]), _s7(f, -1, ["d1","d2","d3"])),
            "check": lambda x: x["status"] == "VALID" and x["second_last_opponent"]["track_id"] == "d2" and abs(x["reference"]["X_world_m"] + 47.0) < 1e-9,
        },
        {
            "name": "ball_is_reference",
            "inputs": (_s4(f, [_track("d1", f, [_joint("nose", 48)]), _track("d2", f, [_joint("nose", 46)])]), _s6(f, [49.0, 49.22]), _s7(f, 1, ["d1","d2"])),
            "check": lambda x: x["status"] == "VALID" and x["reference"]["source"] == "BALL" and abs(x["reference"]["X_world_m"] - 49.22) < 1e-9,
        },
        {
            "name": "arms_excluded",
            "inputs": (_s4(f, [_track("d1", f, [_joint("left_wrist", 60), _joint("left_shoulder", 48)]), _track("d2", f, [_joint("nose", 46)])]), _s6(f, [40.0,40.22]), _s7(f, 1, ["d1","d2"])),
            "check": lambda x: x["status"] == "VALID" and x["opponent_ranking"][0]["anchor"]["name"] == "left_shoulder" and abs(x["opponent_ranking"][0]["goalward_q_m"] - 48.0) < 1e-9,
        },
        {
            "name": "second_last_tie_reference_resolved",
            "inputs": (_s4(f, [_track("d1", f, [_joint("nose", 50)]), _track("d2", f, [_joint("nose", 47)]), _track("d3", f, [_joint("nose", 47)])]), _s6(f,[42.0,42.22]), _s7(f,1,["d1","d2","d3"])),
            "check": lambda x: x["status"] == "VALID" and x["second_last_opponent"]["identity_ambiguous"] is True and set(x["second_last_opponent"]["candidate_track_ids"]) == {"d2","d3"} and x["reference"]["source"] == "SECOND_LAST_OPPONENT",
        },
        {
            "name": "ball_level_with_second_last",
            "inputs": (_s4(f, [_track("d1", f, [_joint("nose",50)]), _track("d2", f, [_joint("nose",47)])]), _s6(f,[46.78,47.0]), _s7(f,1,["d1","d2"])),
            "check": lambda x: x["status"] == "VALID" and x["reference"]["source"] == "BALL_AND_SECOND_LAST_OPPONENT_LEVEL",
        },
        {
            "name": "missing_opponent_pose_abstains",
            "inputs": (_s4(f, [_track("d1", f, [_joint("nose",50)]), _track("d2", f, [_joint("nose",47)])]), _s6(f,[40,40.22]), _s7(f,1,["d1","d2","d3"])),
            "check": lambda x: x["status"] == "UNRESOLVED" and "OPPONENT_GEOMETRY_INCOMPLETE" in x["reasons"],
        },
        {
            "name": "one_opponent_abstains",
            "inputs": (_s4(f, [_track("d1", f, [_joint("nose",50)])]), _s6(f,[40,40.22]), _s7(f,1,["d1"])),
            "check": lambda x: x["status"] == "UNRESOLVED" and "FEWER_THAN_TWO_OPPONENTS" in x["reasons"],
        },
        {
            "name": "ball_unusable_abstains",
            "inputs": (_s4(f, [_track("d1", f, [_joint("nose",50)]), _track("d2", f, [_joint("nose",47)])]), _s6(f,[40,40.22],False), _s7(f,1,["d1","d2"])),
            "check": lambda x: x["status"] == "UNRESOLVED" and "BALL_LONGITUDINAL_GEOMETRY_UNUSABLE" in x["reasons"],
        },
        {
            "name": "frame_mismatch_abstains",
            "inputs": (_s4(104, [_track("d1",104,[_joint("nose",50)]),_track("d2",104,[_joint("nose",47)])]), _s6(105,[40,40.22]), _s7(104,1,["d1","d2"])),
            "check": lambda x: x["status"] == "UNRESOLVED" and "FRAME_INDEX_MISMATCH" in x["reasons"],
        },
        {
            "name": "degraded_stage4_geometry_is_usable_but_propagated",
            "inputs": (_s4(f, [_track("d1", f, [_joint("nose",50)],status="DEGRADED"), _track("d2", f, [_joint("nose",47)])]), _s6(f,[40,40.22]), _s7(f,1,["d1","d2"])),
            "check": lambda x: x["status"] == "VALID" and any(r["status"] == "DEGRADED" for r in x["opponent_ranking"]),
        },
    ]


def run_oracle_benchmark(output_dir: str) -> Dict[str, Any]:
    rows = []
    for fixture in oracle_fixtures():
        s4, s6, s7 = fixture["inputs"]
        pred = build_offside_reference(s4, s6, s7).to_dict()
        ok = bool(fixture["check"](pred))
        rows.append({"fixture": fixture["name"], "pass": ok, "prediction": pred})
    accuracy = sum(int(r["pass"]) for r in rows) / max(len(rows), 1)
    report = {"status": "PASS" if accuracy == 1.0 else "FAIL", "oracle_logic_accuracy": accuracy, "target": 1.0, "rows": rows}
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "stage8_oracle_benchmark.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report
