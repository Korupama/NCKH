from __future__ import annotations

from typing import Any, Dict, List, Mapping

from .adapters import extract_stage4_context, extract_stage6_ball, extract_stage7_context, load_json
from .legal_body import legal_landmark_extent


EXPECTED_COORDINATE_FRAME = {
    "name": "STAGE1_PITCH_WORLD",
    "units": "m",
    "x": "goal-to-goal",
    "y": "touchline-to-touchline",
    "z": "up",
    "pitch_plane": "z=0",
}


def _coord_ok(coord: Mapping[str, Any]) -> bool:
    return all(coord.get(k) == v for k, v in EXPECTED_COORDINATE_FRAME.items())


def build_preflight(
    stage4_input: Any,
    stage6_input: Any,
    stage7_input: Any,
    *,
    allow_partial_opponents: bool = False,
    allow_ball_fallback: bool = False,
    allow_root_fallback: bool = False,
) -> Dict[str, Any]:
    stage4 = load_json(stage4_input)
    stage6 = load_json(stage6_input)
    stage7 = load_json(stage7_input)
    s4 = extract_stage4_context(stage4)
    s6 = extract_stage6_ball(stage6)
    s7 = extract_stage7_context(stage7)

    blockers: List[str] = []
    warnings: List[str] = []

    frames = {"stage4": s4["selected_frame"], "stage6": s6["frame_index"], "stage7": s7["frame_index"]}
    known_frames = [v for v in frames.values() if v is not None]
    if len(known_frames) != 3:
        blockers.append("FRAME_INDEX_MISSING")
    elif len(set(known_frames)) != 1:
        blockers.append("FRAME_INDEX_MISMATCH")

    stage7_tentative_only = bool(
        s7["status"] == "DEGRADED"
        and set(stage7.get("reasons") or []) == {"CONTACT_TENTATIVE_SPATIAL_ONLY"}
        and ((stage7.get("toucher") or {}).get("evidence_level") == "TENTATIVE_SPATIAL_ONLY")
    )
    goalkeeper_reference_only = bool(
        allow_ball_fallback and s7["status"] == "DEGRADED"
        and "ATTACKING_TEAM_INFERRED_FROM_GOALKEEPER" in (stage7.get("reasons") or [])
        and (stage7.get("diagnostics", {}).get("team_resolution") or {}).get("reference_only")
    )
    if goalkeeper_reference_only:
        warnings.append("GOALKEEPER_TEAM_ASSUMPTION_DEFENDER_REFERENCE_ONLY")
    elif s7["status"] == "DEGRADED" and stage7_tentative_only:
        warnings.append("STAGE7_CONTACT_TENTATIVE_SPATIAL_ONLY")
    elif s7["status"] != "VALID":
        blockers.append("STAGE7_CONTEXT_UNRESOLVED")
    if s7["s"] not in (-1, 1):
        blockers.append("ATTACK_DIRECTION_INVALID")
    if not _coord_ok(s4["coordinate_frame"]):
        blockers.append("COORDINATE_FRAME_MISMATCH")

    opponents = list(dict.fromkeys(s7["opponents"]))
    if len(opponents) < 2:
        blockers.append("FEWER_THAN_TWO_OPPONENTS")

    extent_rows = []
    missing_tracks = []
    unusable_tracks = []
    if s7["s"] in (-1, 1) and s4["selected_frame"] is not None:
        for tid in opponents:
            track = s4["tracks"].get(tid)
            if track is None:
                missing_tracks.append(tid)
                continue
            row = legal_landmark_extent(track, s4["selected_frame"], s7["s"], allow_root_fallback=allow_root_fallback)
            extent_rows.append(row)
            if row["status"] not in {"VALID", "DEGRADED"} or row.get("goalward_q_m") is None:
                unusable_tracks.append(tid)

    usable_count = sum(1 for row in extent_rows if row.get("goalward_q_m") is not None and row.get("status") in {"VALID", "DEGRADED"})

    if allow_partial_opponents and usable_count >= 2:
        if missing_tracks:
            warnings.append("OPPONENT_STAGE4_TRACK_MISSING_FALLBACK_ACTIVE")
        if unusable_tracks:
            warnings.append("OPPONENT_LEGAL_GEOMETRY_UNUSABLE_FALLBACK_ACTIVE")
        if missing_tracks or unusable_tracks:
            warnings.append("OPPONENT_GEOMETRY_PARTIAL_DEGRADED")
    else:
        if missing_tracks:
            blockers.append("OPPONENT_STAGE4_TRACK_MISSING")
        if unusable_tracks:
            blockers.append("OPPONENT_LEGAL_GEOMETRY_UNUSABLE")
        if missing_tracks or unusable_tracks:
            blockers.append("OPPONENT_GEOMETRY_INCOMPLETE")

    if usable_count < 2:
        blockers.append("FEWER_THAN_TWO_USABLE_OPPONENTS")

    extent = s6["ball_center_x_extent_m"]
    if goalkeeper_reference_only:
        warnings.append("BALL_EXCLUDED_FROM_DEFENDER_REFERENCE")
    elif allow_ball_fallback:
        if not s6["usable_for_offside"]:
            warnings.append("BALL_LONGITUDINAL_GEOMETRY_UNUSABLE_FALLBACK_ACTIVE")
        if extent is None:
            warnings.append("BALL_X_EXTENT_MISSING_FALLBACK_ACTIVE")
        elif extent[0] > extent[1]:
            blockers.append("BALL_X_EXTENT_ORDER_INVALID")
    else:
        if not s6["usable_for_offside"]:
            blockers.append("BALL_LONGITUDINAL_GEOMETRY_UNUSABLE")
        if extent is None:
            blockers.append("BALL_X_EXTENT_MISSING")
        elif extent[0] > extent[1]:
            blockers.append("BALL_X_EXTENT_ORDER_INVALID")

    if not s6["accuracy_validated"]:
        warnings.append("BALL_METRIC_ACCURACY_NOT_VALIDATED")
    if not bool(stage6.get("research_accuracy_frozen", False)):
        warnings.append("STAGE6_RESEARCH_ACCURACY_NOT_FROZEN")
    warnings.append("STAGE4_LEGAL_EXTENT_IS_LANDMARK_PROXY_NOT_EXACT_BODY_SURFACE")

    blockers = list(dict.fromkeys(blockers))
    warnings = list(dict.fromkeys(warnings))
    is_degraded = bool(
        stage7_tentative_only or goalkeeper_reference_only
        or (allow_partial_opponents and (missing_tracks or unusable_tracks) and usable_count >= 2)
        or (allow_ball_fallback and (not s6["usable_for_offside"] or extent is None))
    )
    status = "BLOCKED" if blockers else ("DEGRADED_READY" if is_degraded else "READY")
    return {
        "schema_version": "stage8-preflight-1.0",
        "status": status,
        "frames": frames,
        "stage7": {
            "status": s7["status"],
            "s": s7["s"],
            "opponents": opponents,
            "tentative_spatial_contact_only": stage7_tentative_only,
            "goalkeeper_reference_only": goalkeeper_reference_only,
        },
        "stage4": {
            "schema_version": s4["schema_version"],
            "producer": s4["producer"],
            "coordinate_frame": s4["coordinate_frame"],
            "opponent_tracks_found": len(opponents) - len(missing_tracks),
            "opponent_tracks_usable": usable_count,
            "missing_opponent_track_ids": missing_tracks,
            "unusable_opponent_track_ids": unusable_tracks,
        },
        "stage6": {
            "source": s6["source"],
            "usable_for_offside": s6["usable_for_offside"],
            "accuracy_validated": s6["accuracy_validated"],
            "X_world_m": s6["X_world_m"],
            "ball_center_x_extent_m": extent,
        },
        "blockers": blockers,
        "warnings": warnings,
    }
