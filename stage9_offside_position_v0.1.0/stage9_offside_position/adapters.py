from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


def load_json(path_or_obj: Any) -> Dict[str, Any]:
    if path_or_obj is None:
        return {}
    if isinstance(path_or_obj, dict):
        return path_or_obj
    p = Path(path_or_obj)
    return json.loads(p.read_text(encoding="utf-8"))


def normalize_track_id(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def finite_xyz(value: Any) -> Optional[List[float]]:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    try:
        out = [float(value[0]), float(value[1]), float(value[2])]
    except (TypeError, ValueError):
        return None
    return out if all(math.isfinite(v) for v in out) else None


def source_path(value: Any) -> Optional[str]:
    if value is None or isinstance(value, dict):
        return None
    try:
        return str(Path(value).resolve())
    except Exception:
        return str(value)


def extract_stage7(stage7: Mapping[str, Any]) -> Dict[str, Any]:
    attack = stage7.get("attack_direction") if isinstance(stage7.get("attack_direction"), dict) else {}
    sets = stage7.get("sets") if isinstance(stage7.get("sets"), dict) else {}
    toucher = stage7.get("toucher") if isinstance(stage7.get("toucher"), dict) else {}
    try:
        frame = int(stage7.get("frame_index")) if stage7.get("frame_index") is not None else None
    except (TypeError, ValueError):
        frame = None
    try:
        s = int(attack.get("s")) if attack.get("s") is not None else None
    except (TypeError, ValueError):
        s = None
    attackers = [normalize_track_id(x) for x in (sets.get("attackers") or [])]
    opponents = [normalize_track_id(x) for x in (sets.get("opponents") or [])]
    referees = [normalize_track_id(x) for x in (sets.get("referees_excluded") or [])]
    unknown = [normalize_track_id(x) for x in (sets.get("unknown_team_excluded") or [])]
    inactive = [normalize_track_id(x) for x in (sets.get("inactive_excluded") or [])]
    return {
        "frame_index": frame,
        "status": str(stage7.get("status") or "UNKNOWN").upper(),
        "s": s if s in (-1, 1) else None,
        "direction_label": attack.get("label"),
        "attackers": [x for x in attackers if x is not None],
        "opponents": [x for x in opponents if x is not None],
        "referees": [x for x in referees if x is not None],
        "unknown": [x for x in unknown if x is not None],
        "inactive": [x for x in inactive if x is not None],
        "toucher_track_id": normalize_track_id(toucher.get("track_id")) if toucher else None,
        "raw": dict(stage7),
    }


def extract_stage8(stage8: Mapping[str, Any]) -> Dict[str, Any]:
    attack = stage8.get("attack_direction") if isinstance(stage8.get("attack_direction"), dict) else {}
    reference = stage8.get("reference") if isinstance(stage8.get("reference"), dict) else {}
    ball = stage8.get("ball") if isinstance(stage8.get("ball"), dict) else {}
    second = stage8.get("second_last_opponent") if isinstance(stage8.get("second_last_opponent"), dict) else {}
    try:
        frame = int(stage8.get("frame_index")) if stage8.get("frame_index") is not None else None
    except (TypeError, ValueError):
        frame = None
    try:
        s = int(attack.get("s")) if attack.get("s") is not None else None
    except (TypeError, ValueError):
        s = None
    q = reference.get("goalward_q_m")
    x = reference.get("X_world_m")
    q_out = float(q) if finite_number(q) else None
    x_out = float(x) if finite_number(x) else None
    return {
        "frame_index": frame,
        "status": str(stage8.get("status") or "UNKNOWN").upper(),
        "s": s if s in (-1, 1) else None,
        "reference_q_m": q_out,
        "reference_x_m": x_out,
        "reference_source": reference.get("source"),
        "reference": dict(reference),
        "ball": dict(ball),
        "second_last": dict(second),
        "opponent_ranking": [dict(x) for x in (stage8.get("opponent_ranking") or []) if isinstance(x, dict)],
        "raw": dict(stage8),
    }


def _track_map(stage4: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in stage4.get("tracks") or []:
        if not isinstance(row, dict):
            continue
        tid = normalize_track_id(row.get("track_id"))
        if tid is not None:
            out[tid] = row
    return out


def extract_stage4(stage4: Mapping[str, Any]) -> Dict[str, Any]:
    try:
        frame = int(stage4.get("selected_frame")) if stage4.get("selected_frame") is not None else None
    except (TypeError, ValueError):
        frame = None
    return {
        "frame_index": frame,
        "tracks": _track_map(stage4),
        "coordinate_frame": dict(stage4.get("coordinate_frame") or {}),
        "schema_version": stage4.get("schema_version"),
        "producer": stage4.get("producer"),
        "raw": dict(stage4),
    }


def selected_stage4_observation(track: Mapping[str, Any], frame_index: Optional[int]) -> Optional[Dict[str, Any]]:
    if frame_index is None:
        return None
    for obs in track.get("observations") or []:
        if not isinstance(obs, dict):
            continue
        try:
            fi = int(obs.get("frame_index"))
        except (TypeError, ValueError):
            continue
        if fi == int(frame_index):
            return dict(obs)
    return None


def stage3_track_map(stage3: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in stage3.get("tracks") or []:
        if not isinstance(row, dict):
            continue
        tid = normalize_track_id(row.get("track_id"))
        if tid is not None:
            out[tid] = row
    return out


def selected_stage3_observation(track: Mapping[str, Any], frame_index: Optional[int]) -> Optional[Dict[str, Any]]:
    if frame_index is None:
        return None
    for obs in track.get("observations") or []:
        if not isinstance(obs, dict):
            continue
        try:
            fi = int(obs.get("frame_index"))
        except (TypeError, ValueError):
            continue
        if fi == int(frame_index):
            return dict(obs)
    return None


def bbox_from_stage3_observation(obs: Mapping[str, Any] | None) -> Optional[List[float]]:
    if not isinstance(obs, Mapping):
        return None
    candidates = [obs.get("bbox_xyxy"), obs.get("bbox"), obs.get("box_xyxy")]
    for value in candidates:
        if isinstance(value, (list, tuple)) and len(value) >= 4 and all(finite_number(v) for v in value[:4]):
            return [float(v) for v in value[:4]]
    return None


def keypoints_from_stage3_observation(obs: Mapping[str, Any] | None) -> List[Dict[str, Any]]:
    if not isinstance(obs, Mapping):
        return []
    raw = obs.get("keypoints_133") or obs.get("keypoints") or []
    out: List[Dict[str, Any]] = []
    if isinstance(raw, list):
        for i, kp in enumerate(raw):
            if isinstance(kp, dict):
                x, y = kp.get("x"), kp.get("y")
                if finite_number(x) and finite_number(y):
                    out.append({"name": kp.get("name", str(i)), "x": float(x), "y": float(y), "state": kp.get("state")})
            elif isinstance(kp, (list, tuple)) and len(kp) >= 2 and finite_number(kp[0]) and finite_number(kp[1]):
                out.append({"name": str(i), "x": float(kp[0]), "y": float(kp[1]), "state": None})
    return out


def replay_context(stage3: Mapping[str, Any]) -> Dict[str, Any]:
    return dict(stage3.get("replay_context") or {})


def extract_stage6(stage6: Mapping[str, Any]) -> Dict[str, Any]:
    if not stage6:
        return {"frame_index": None, "X_world_m": None, "ball_center_x_extent_m": None, "center_xyz_world_m": None, "raw": {}}
    block = stage6.get("stage8") if isinstance(stage6.get("stage8"), Mapping) else None
    frame_value = stage6.get("selected_frame")
    if block is None:
        selected = stage6.get("selected_frame_ball") if isinstance(stage6.get("selected_frame_ball"), Mapping) else {}
        block = selected.get("localization") if isinstance(selected.get("localization"), Mapping) else selected
        frame_value = selected.get("frame_index", frame_value)
    block = dict(block or {})
    try:
        frame = int(frame_value) if frame_value is not None else None
    except (TypeError, ValueError):
        frame = None
    extent = block.get("ball_center_x_extent_m")
    if isinstance(extent, (list, tuple)) and len(extent) >= 2 and all(finite_number(v) for v in extent[:2]):
        extent_out = [float(extent[0]), float(extent[1])]
    else:
        extent_out = None
    center = finite_xyz(block.get("center_xyz_world_m"))
    x = float(block.get("X_world_m")) if finite_number(block.get("X_world_m")) else (center[0] if center else None)
    return {"frame_index": frame, "X_world_m": x, "ball_center_x_extent_m": extent_out, "center_xyz_world_m": center, "raw": block}
