from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def load_json(path_or_obj: Any) -> Dict[str, Any]:
    if isinstance(path_or_obj, dict):
        return path_or_obj
    path = Path(path_or_obj)
    return json.loads(path.read_text(encoding="utf-8"))


def _first_non_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _dig(obj: Any, path: Iterable[str]) -> Any:
    cur = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def normalize_track_id(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def team_key(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        v = value.strip()
        if not v or v.upper() in {"UNKNOWN", "UNK", "NONE", "NULL", "NA", "N/A"}:
            return None
        return v
    return str(value)


def extract_stage1_view(stage1: Dict[str, Any]) -> Tuple[Optional[float], Optional[List[float]], Dict[str, Any]]:
    candidates = [
        _dig(stage1, ["view", "centre_ray_pitch_hit_m"]),
        _dig(stage1, ["view", "center_ray_pitch_hit_m"]),
        stage1.get("centre_ray_pitch_hit_m"),
        stage1.get("center_ray_pitch_hit_m"),
        _dig(stage1, ["camera", "view", "centre_ray_pitch_hit_m"]),
        _dig(stage1, ["camera", "view", "center_ray_pitch_hit_m"]),
    ]
    point = next((v for v in candidates if isinstance(v, (list, tuple)) and len(v) >= 2), None)
    if point is None:
        return None, None, {"source": None}
    try:
        p = [float(point[0]), float(point[1]), float(point[2]) if len(point) > 2 else 0.0]
    except (TypeError, ValueError):
        return None, None, {"source": None}
    return p[0], p, {"source": "stage1.centre_ray_pitch_hit"}


def extract_stage6_contact(stage6: Dict[str, Any]) -> Dict[str, Any]:
    contacts = [
        stage6.get("stage7") if str(stage6.get("schema_version", "")).startswith("stage6-downstream-handoff-") else None,
        _dig(stage6, ["selected_frame_ball", "contact"]),
        stage6.get("contact"),
        _dig(stage6, ["selected_frame", "contact"]),
    ]
    contact = next((v for v in contacts if isinstance(v, dict)), {})
    track_id = _first_non_none(
        contact.get("track_id"),
        contact.get("contact_track_id"),
        stage6.get("contact_track_id"),
        _dig(stage6, ["toucher", "track_id"]),
    )
    region = _first_non_none(contact.get("region"), contact.get("body_region"), stage6.get("contact_body_region"))
    status = _first_non_none(contact.get("status"), stage6.get("contact_status"))
    confidence = _first_non_none(contact.get("confidence"), contact.get("score"), stage6.get("contact_confidence"))
    frame_index = _first_non_none(
        _dig(stage6, ["selected_frame_ball", "frame_index"]),
        _dig(stage6, ["selected_frame", "frame_index"]),
        stage6.get("frame_index"),
        stage6.get("selected_frame") if not isinstance(stage6.get("selected_frame"), dict) else None,
    )
    try:
        if frame_index is not None:
            frame_index = int(frame_index)
    except (TypeError, ValueError):
        frame_index = None
    return {
        "track_id": normalize_track_id(track_id),
        "region": region,
        "status": status,
        "confidence": confidence,
        "frame_index": frame_index,
        "source": "stage6.contact",
    }


def _looks_like_player_record(obj: Any) -> bool:
    return isinstance(obj, dict) and any(k in obj for k in ("track_id", "id", "player_track_id"))


def _candidate_player_containers(stage5: Dict[str, Any]) -> List[Any]:
    return [
        stage5.get("track_team"),
        stage5.get("players"),
        stage5.get("tracks"),
        stage5.get("player_states"),
        stage5.get("entities"),
        _dig(stage5, ["state", "players"]),
        _dig(stage5, ["state", "tracks"]),
        stage5.get("by_track"),
    ]


def extract_stage5_players(stage5: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = None
    for cand in _candidate_player_containers(stage5):
        if isinstance(cand, list):
            raw = cand
            break
        if isinstance(cand, dict):
            converted = []
            for key, value in cand.items():
                if isinstance(value, dict):
                    row = dict(value)
                    row.setdefault("track_id", key)
                    converted.append(row)
            if converted:
                raw = converted
                break
    if raw is None and _looks_like_player_record(stage5):
        raw = [stage5]
    if raw is None:
        return []

    players: List[Dict[str, Any]] = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        tid = normalize_track_id(_first_non_none(row.get("track_id"), row.get("player_track_id"), row.get("id")))
        if tid is None:
            continue
        team = _first_non_none(row.get("team_id"), row.get("team"), row.get("team_label"), row.get("affiliation"))
        role = _first_non_none(row.get("role"), row.get("role_name"), row.get("entity_type"), row.get("class_name"), row.get("class"))
        is_ref = bool(row.get("is_referee", False))
        role_text = str(role).lower() if role is not None else ""
        if any(token in role_text for token in ("referee", "official", "linesman", "assistant_ref")):
            is_ref = True
        active = _first_non_none(row.get("active_at_t0"), row.get("active"), row.get("visible"), row.get("eligible"), True)
        active = bool(active)
        players.append({
            "track_id": tid,
            "team_id": team,
            "team_key": team_key(team),
            "role": role,
            "is_referee": is_ref,
            "active": active,
            "raw": row,
        })
    return players
