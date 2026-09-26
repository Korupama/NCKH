from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


def load_json(path_or_obj: Any) -> Dict[str, Any]:
    if isinstance(path_or_obj, dict):
        return path_or_obj
    path = Path(path_or_obj)
    return json.loads(path.read_text(encoding="utf-8"))


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
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        out = [float(value[0]), float(value[1]), float(value[2])]
    except (TypeError, ValueError):
        return None
    return out if all(math.isfinite(v) for v in out) else None


def extract_stage7_context(stage7: Mapping[str, Any]) -> Dict[str, Any]:
    attack = stage7.get("attack_direction") if isinstance(stage7.get("attack_direction"), dict) else {}
    sets = stage7.get("sets") if isinstance(stage7.get("sets"), dict) else {}
    try:
        frame = int(stage7.get("frame_index")) if stage7.get("frame_index") is not None else None
    except (TypeError, ValueError):
        frame = None
    try:
        s = int(attack.get("s")) if attack.get("s") is not None else None
    except (TypeError, ValueError):
        s = None
    opponents = [normalize_track_id(v) for v in (sets.get("opponents") or [])]
    opponents = [v for v in opponents if v is not None]
    attackers = [normalize_track_id(v) for v in (sets.get("attackers") or [])]
    attackers = [v for v in attackers if v is not None]
    toucher = stage7.get("toucher") if isinstance(stage7.get("toucher"), dict) else None
    return {
        "frame_index": frame,
        "status": str(stage7.get("status") or "UNRESOLVED").upper(),
        "s": s,
        "label": attack.get("label"),
        "source": attack.get("source"),
        "opponents": opponents,
        "attackers": attackers,
        "toucher_track_id": normalize_track_id(toucher.get("track_id")) if toucher else None,
    }


def extract_stage4_context(stage4: Mapping[str, Any]) -> Dict[str, Any]:
    try:
        selected_frame = int(stage4.get("selected_frame")) if stage4.get("selected_frame") is not None else None
    except (TypeError, ValueError):
        selected_frame = None
    coordinate = stage4.get("coordinate_frame") if isinstance(stage4.get("coordinate_frame"), dict) else {}
    tracks: Dict[str, Dict[str, Any]] = {}
    for row in stage4.get("tracks") or []:
        if not isinstance(row, dict):
            continue
        tid = normalize_track_id(row.get("track_id"))
        if tid is None:
            continue
        tracks[tid] = row
    return {
        "selected_frame": selected_frame,
        "coordinate_frame": dict(coordinate),
        "tracks": tracks,
        "schema_version": stage4.get("schema_version"),
        "producer": stage4.get("producer"),
        "source_world_grounded_pose_state": stage4.get("source_world_grounded_pose_state"),
    }


def selected_stage4_observation(track: Mapping[str, Any], frame_index: int) -> Optional[Dict[str, Any]]:
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


def extract_stage6_ball(stage6: Mapping[str, Any]) -> Dict[str, Any]:
    # Primary production handoff shape: top-level selected_frame + stage8 localization block.
    block = stage6.get("stage8") if isinstance(stage6.get("stage8"), dict) else None
    source = "stage6.downstream_handoff.stage8"
    frame_value = stage6.get("selected_frame")

    # Full contact-aware state remains supported for integration/debugging.
    if block is None:
        selected = stage6.get("selected_frame_ball") if isinstance(stage6.get("selected_frame_ball"), dict) else {}
        block = selected.get("localization") if isinstance(selected.get("localization"), dict) else selected
        frame_value = selected.get("frame_index", frame_value)
        source = "stage6.selected_frame_ball.localization"

    try:
        frame = int(frame_value) if frame_value is not None else None
    except (TypeError, ValueError):
        frame = None

    block = block or {}
    extent = block.get("ball_center_x_extent_m")
    extent_out = None
    if isinstance(extent, (list, tuple)) and len(extent) == 2 and all(finite_number(v) for v in extent):
        extent_out = [float(extent[0]), float(extent[1])]

    center = finite_xyz(block.get("center_xyz_world_m"))
    x = block.get("X_world_m")
    x_out = float(x) if finite_number(x) else (center[0] if center else None)
    usable = bool(block.get("usable_for_offside", block.get("usable_for_offside_longitudinal_coordinate", False)))
    accuracy_validated = bool(block.get("accuracy_validated", False))
    return {
        "frame_index": frame,
        "source": source,
        "usable_for_offside": usable,
        "accuracy_validated": accuracy_validated,
        "selected_method": block.get("selected_method"),
        "center_xyz_world_m": center,
        "X_world_m": x_out,
        "ball_center_x_extent_m": extent_out,
        "raw": dict(block),
    }
