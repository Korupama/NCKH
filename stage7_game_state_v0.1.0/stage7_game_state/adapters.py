from __future__ import annotations

import json
import math
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


def _as_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "1", "yes", "y", "valid"}:
            return True
        if v in {"false", "0", "no", "n", "invalid"}:
            return False
    return None


def _as_vec3(value: Any) -> Optional[List[float]]:
    if isinstance(value, dict):
        xyz = [value.get(k) for k in ("x", "y", "z")]
        if all(v is not None for v in xyz):
            value = xyz
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    try:
        out = [float(value[0]), float(value[1]), float(value[2])]
    except (TypeError, ValueError):
        return None
    return out if all(math.isfinite(v) for v in out) else None


def _as_mat3(value: Any) -> Optional[List[List[float]]]:
    if isinstance(value, (list, tuple)) and len(value) == 9 and not isinstance(value[0], (list, tuple)):
        value = [value[0:3], value[3:6], value[6:9]]
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    rows: List[List[float]] = []
    try:
        for row in value[:3]:
            if not isinstance(row, (list, tuple)) or len(row) < 3:
                return None
            rows.append([float(row[0]), float(row[1]), float(row[2])])
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for row in rows for v in row):
        return None
    return rows


def _mat_t_vec(R: List[List[float]], v: List[float]) -> List[float]:
    return [
        R[0][0] * v[0] + R[1][0] * v[1] + R[2][0] * v[2],
        R[0][1] * v[0] + R[1][1] * v[1] + R[2][1] * v[2],
        R[0][2] * v[0] + R[1][2] * v[1] + R[2][2] * v[2],
    ]


def _mat_vec(R: List[List[float]], v: List[float]) -> List[float]:
    return [sum(R[i][j] * v[j] for j in range(3)) for i in range(3)]


def _camera_geometry_candidates(stage1: Dict[str, Any]) -> Tuple[Optional[List[List[float]]], Optional[List[float]], Optional[str], Dict[str, Any]]:
    # The first entries are the *actual* Stage-1 CameraState v1.2 serialization.
    rotation_candidates = [
        ("extrinsics.R_world_to_camera", _dig(stage1, ["extrinsics", "R_world_to_camera"]), "WORLD_TO_CAMERA"),
        ("R_world_to_camera", stage1.get("R_world_to_camera"), "WORLD_TO_CAMERA"),
        ("R", stage1.get("R"), None),
        ("rotation_matrix", stage1.get("rotation_matrix"), None),
        ("camera.R", _dig(stage1, ["camera", "R"]), None),
        ("camera.rotation_matrix", _dig(stage1, ["camera", "rotation_matrix"]), None),
        ("extrinsics.R", _dig(stage1, ["extrinsics", "R"]), None),
        ("pose.R", _dig(stage1, ["pose", "R"]), None),
        ("calibration.rotation_matrix", _dig(stage1, ["calibration", "rotation_matrix"]), None),
        ("camera_state.R", _dig(stage1, ["camera_state", "R"]), None),
    ]
    center_candidates = [
        ("extrinsics.camera_center_world_m", _dig(stage1, ["extrinsics", "camera_center_world_m"])),
        ("camera_center_world_m", stage1.get("camera_center_world_m")),
        ("C", stage1.get("C")),
        ("camera_center", stage1.get("camera_center")),
        ("camera_center_m", stage1.get("camera_center_m")),
        ("position_meters", stage1.get("position_meters")),
        ("camera.C", _dig(stage1, ["camera", "C"])),
        ("camera.position_meters", _dig(stage1, ["camera", "position_meters"])),
        ("extrinsics.C", _dig(stage1, ["extrinsics", "C"])),
        ("calibration.position_meters", _dig(stage1, ["calibration", "position_meters"])),
        ("camera_state.C", _dig(stage1, ["camera_state", "C"])),
    ]
    translation_candidates = [
        ("extrinsics.t_world_to_camera", _dig(stage1, ["extrinsics", "t_world_to_camera"])),
        ("t_world_to_camera", stage1.get("t_world_to_camera")),
        ("t", stage1.get("t")),
        ("translation", stage1.get("translation")),
        ("translation_vector", stage1.get("translation_vector")),
        ("camera.t", _dig(stage1, ["camera", "t"])),
        ("extrinsics.t", _dig(stage1, ["extrinsics", "t"])),
        ("camera_state.t", _dig(stage1, ["camera_state", "t"])),
    ]

    R = None
    r_path = None
    implied_convention = None
    for path, value, conv in rotation_candidates:
        R = _as_mat3(value)
        if R is not None:
            r_path = path
            implied_convention = conv
            break
    if R is None:
        return None, None, None, {"geometry_reason": "ROTATION_MATRIX_MISSING"}

    C = None
    c_path = None
    for path, value in center_candidates:
        C = _as_vec3(value)
        if C is not None:
            c_path = path
            break

    if C is None:
        for path, value in translation_candidates:
            t = _as_vec3(value)
            if t is None:
                continue
            # Stage 1 contract: x_cam = R (X_world - C) = R X + t.
            C = [-x for x in _mat_t_vec(R, t)]
            c_path = f"derived_from_{path}"
            break

    if C is None:
        return R, None, None, {"rotation_path": r_path, "geometry_reason": "CAMERA_CENTER_MISSING"}

    convention = _first_non_none(
        stage1.get("rotation_convention"),
        _dig(stage1, ["camera", "rotation_convention"]),
        _dig(stage1, ["extrinsics", "rotation_convention"]),
        _dig(stage1, ["camera_state", "rotation_convention"]),
    )
    if convention is None and implied_convention is not None:
        resolved = implied_convention
        explicit = True  # explicit in the serialized key name
    else:
        convention_text = str(convention).strip().upper() if convention is not None else "WORLD_TO_CAMERA"
        if any(token in convention_text for token in ("CAMERA_TO_WORLD", "C2W", "CAM2WORLD")):
            resolved = "CAMERA_TO_WORLD"
        else:
            resolved = "WORLD_TO_CAMERA"
        explicit = convention is not None

    return R, C, resolved, {
        "rotation_path": r_path,
        "camera_center_path": c_path,
        "rotation_convention": resolved,
        "rotation_convention_explicit": explicit,
    }


def _pitch_bounds(stage1: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    pitch = stage1.get("pitch")
    if not isinstance(pitch, dict):
        return None
    try:
        length = float(pitch.get("length_m", 105.0))
        width = float(pitch.get("width_m", 68.0))
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(length) and math.isfinite(width) and length > 0 and width > 0):
        return None
    return length, width


def _derive_centre_ray_pitch_hit(stage1: Dict[str, Any], *, epsilon: float = 1e-9, pitch_tolerance_m: float = 1.0) -> Tuple[Optional[List[float]], Dict[str, Any]]:
    R, C, convention, meta = _camera_geometry_candidates(stage1)
    if R is None or C is None or convention is None:
        return None, meta

    optical_axis_cam = [0.0, 0.0, 1.0]
    if convention == "WORLD_TO_CAMERA":
        d_world = _mat_t_vec(R, optical_axis_cam)
    else:
        d_world = _mat_vec(R, optical_axis_cam)

    dz = float(d_world[2])
    if not math.isfinite(dz) or abs(dz) <= epsilon:
        meta["geometry_reason"] = "CENTRE_RAY_PARALLEL_TO_PITCH"
        return None, meta

    lam = -float(C[2]) / dz
    if not math.isfinite(lam) or lam <= 0.0:
        meta["geometry_reason"] = "PITCH_INTERSECTION_BEHIND_CAMERA"
        meta["lambda"] = lam
        return None, meta

    p = [float(C[i] + lam * d_world[i]) for i in range(3)]
    if not all(math.isfinite(v) for v in p):
        meta["geometry_reason"] = "NONFINITE_PITCH_INTERSECTION"
        return None, meta
    p[2] = 0.0

    bounds = _pitch_bounds(stage1)
    in_bounds = None
    if bounds is not None:
        length, width = bounds
        tol = max(0.0, float(pitch_tolerance_m))
        in_bounds = bool(abs(p[0]) <= length / 2.0 + tol and abs(p[1]) <= width / 2.0 + tol)

    meta.update({
        "geometry_reason": None if in_bounds is not False else "CENTRE_RAY_PITCH_HIT_OUT_OF_BOUNDS",
        "lambda": lam,
        "camera_center_m": [float(v) for v in C],
        "centre_ray_world_direction": [float(v) for v in d_world],
        "pitch_bounds_checked": bounds is not None,
        "centre_ray_pitch_hit_in_bounds": in_bounds,
    })
    return p, meta


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
    view = stage1.get("view") if isinstance(stage1.get("view"), dict) else {}
    candidates = [
        view.get("centre_ray_pitch_hit_m"),
        view.get("center_ray_pitch_hit_m"),
        stage1.get("centre_ray_pitch_hit_m"),
        stage1.get("center_ray_pitch_hit_m"),
        _dig(stage1, ["camera", "view", "centre_ray_pitch_hit_m"]),
        _dig(stage1, ["camera", "view", "center_ray_pitch_hit_m"]),
    ]
    point = next((v for v in candidates if isinstance(v, (list, tuple)) and len(v) >= 2), None)
    if point is not None:
        try:
            p = [float(point[0]), float(point[1]), float(point[2]) if len(point) > 2 else 0.0]
        except (TypeError, ValueError):
            p = None
        if p is not None and all(math.isfinite(v) for v in p):
            valid_flag = _as_bool(_first_non_none(
                view.get("centre_ray_pitch_hit_valid"),
                stage1.get("centre_ray_pitch_hit_valid"),
            ))
            # Backward-compatible artifacts did not carry an explicit validity bit.
            valid = True if valid_flag is None else valid_flag
            meta = {
                "source": "stage1.centre_ray_pitch_hit",
                "derived": False,
                "valid": valid,
                "view_pitch_half": view.get("view_pitch_half"),
                "centre_ray_pitch_hit_in_bounds": _as_bool(view.get("centre_ray_pitch_hit_in_bounds")),
                "geometry_reason": view.get("reason"),
            }
            return (p[0] if valid else None), p, meta

    derived, geometry_meta = _derive_centre_ray_pitch_hit(stage1)
    if derived is None:
        return None, None, {"source": None, "derived": False, "valid": False, **geometry_meta}

    in_bounds = geometry_meta.get("centre_ray_pitch_hit_in_bounds")
    valid = in_bounds is not False
    half = "LEFT" if derived[0] < 0 else ("RIGHT" if derived[0] > 0 else "MIDFIELD")
    return (derived[0] if valid else None), derived, {
        "source": "stage1.camera_geometry_fallback",
        "derived": True,
        "valid": valid,
        "view_pitch_half": half if valid else None,
        **geometry_meta,
    }


def extract_stage6_contact(stage6: Dict[str, Any]) -> Dict[str, Any]:
    # Production Stage-6 contact-aware handoff stores the Stage-7 payload under
    # top-level ``stage7`` and the selected frame under top-level
    # ``selected_frame``.  Full ball_trajectory_state.json remains supported.
    handoff_contact = _dig(stage6, ["stage7"])
    contacts = [
        _dig(stage6, ["selected_frame_ball", "contact"]),
        handoff_contact,
        stage6.get("contact"),
        _dig(stage6, ["selected_frame", "contact"]),
    ]
    contact = next((v for v in contacts if isinstance(v, dict)), {})
    source = (
        "stage6.downstream_handoff.stage7"
        if isinstance(handoff_contact, dict) and contact is handoff_contact
        else "stage6.contact"
    )
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
        contact.get("frame_index"),
        _dig(stage6, ["selected_frame_ball", "frame_index"]),
        stage6.get("selected_frame") if not isinstance(stage6.get("selected_frame"), dict) else None,
        _dig(stage6, ["selected_frame", "frame_index"]),
        stage6.get("frame_index"),
    )
    try:
        if frame_index is not None:
            frame_index = int(frame_index)
    except (TypeError, ValueError):
        frame_index = None
    return {
        "track_id": normalize_track_id(track_id),
        "nearest_track_id": normalize_track_id(contact.get("nearest_track_id")),
        "region": region,
        "status": status,
        "confidence": confidence,
        "image_distance_px": contact.get("image_distance_px"),
        "threshold_px": contact.get("threshold_px"),
        "temporal_contact_evidence": contact.get("temporal_contact_evidence"),
        "frame_index": frame_index,
        "source": source,
    }


def _looks_like_player_record(obj: Any) -> bool:
    return isinstance(obj, dict) and any(k in obj for k in ("track_id", "id", "player_track_id"))


def _candidate_player_containers(stage5: Dict[str, Any]) -> List[Any]:
    return [
        # stage5-downstream-handoff-{1.0,3.0}
        stage5.get("track_team"),
        # full Stage-5 state variants
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
        # Residual Stage-5 v3 handoff exposes effective_role/stage5_role rather
        # than the legacy ``role`` key.  effective_role is the downstream
        # semantic authority when present.
        role = _first_non_none(
            row.get("effective_role"),
            row.get("stage5_role"),
            row.get("role"),
            row.get("upstream_role"),
            row.get("role_name"),
            row.get("entity_type"),
            row.get("class_name"),
            row.get("class"),
        )
        is_ref = bool(row.get("is_referee", False))
        role_text = str(role).lower() if role is not None else ""
        if any(token in role_text for token in ("referee", "official", "linesman", "assistant_ref")):
            is_ref = True

        if "selected_frame_bbox_xyxy" in row:
            # Full Stage-5 state: selected-frame visibility is explicit.
            active = row.get("selected_frame_bbox_xyxy") is not None
        elif isinstance(stage5.get("track_team"), dict):
            # Residual Stage-5 handoff v3 contains selected-frame tracks only.
            # Legacy v1 handoff did not make visibility explicit; the preflight
            # emits a warning for that older schema.
            active = True
        else:
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
