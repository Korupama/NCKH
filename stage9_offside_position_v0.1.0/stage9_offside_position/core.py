from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .adapters import extract_stage4, extract_stage6, extract_stage7, extract_stage8, load_json, source_path
from .classification import classify_attacker
from .constants import DEFAULT_EPSILON_M
from .geometry import legal_extent_from_stage4_track, reference_from_stage8_best_effort
from .models import OffsidePositionState


def _frame_choice(*values: Optional[int]) -> Optional[int]:
    for v in values:
        if v is not None:
            return int(v)
    return None


def _direction_choice(s7: Optional[int], s8: Optional[int]) -> Optional[int]:
    return s8 if s8 in (-1, 1) else s7 if s7 in (-1, 1) else None


def build_offside_position_state(
    stage4_input: Any,
    stage7_input: Any,
    stage8_input: Any = None,
    *,
    stage6_input: Any = None,
    epsilon_m: float = DEFAULT_EPSILON_M,
    best_effort: bool = True,
    allow_defender_only: bool = False,
    allow_tentative_context: bool = False,
) -> OffsidePositionState:
    stage4_raw = load_json(stage4_input)
    stage7_raw = load_json(stage7_input)
    stage8_raw = load_json(stage8_input)
    stage6_raw = load_json(stage6_input) if stage6_input is not None else {}
    s4 = extract_stage4(stage4_raw)
    s6 = extract_stage6(stage6_raw)
    s7 = extract_stage7(stage7_raw)
    s8 = extract_stage8(stage8_raw)

    state = OffsidePositionState()
    state.mode = "BEST_EFFORT_DEMO" if best_effort else "STRICT"
    state.frame_index = _frame_choice(s8["frame_index"], s7["frame_index"], s4["frame_index"])
    s = _direction_choice(s7["s"], s8["s"])
    state.attack_direction = {
        "s": s,
        "label": s7.get("direction_label") or (("LEFT_TO_RIGHT" if s == 1 else "RIGHT_TO_LEFT") if s in (-1, 1) else None),
        "source": "stage8" if s8["s"] in (-1, 1) else "stage7" if s7["s"] in (-1, 1) else "missing",
    }
    state.toucher_track_id = s7["toucher_track_id"]

    ref_input = stage8_raw if best_effort else {"reference": stage8_raw.get("reference")}
    ref = reference_from_stage8_best_effort(ref_input, s)
    # Explicit demo policy: missing ball/contact does not block a measured
    # second-last-defender reference. Other unresolved geometry still blocks it.
    defender_only = bool(
        allow_defender_only
        and s8['reference'].get('kind') == 'DEFENDER_ONLY'
        and s8['reference'].get('reference_only')
        and s8['reference_source'] == 'SECOND_LAST_OPPONENT'
        and s8['reference_q_m'] is not None
        and s8['second_last'].get('goalward_q_m') == s8['reference_q_m']
        and len(s8['opponent_ranking']) >= 2
        and s7['attackers'] and len(s7['opponents']) >= 2
        and s8['status'] == 'DEGRADED'
        and (s7['status'] == 'VALID' or (
            s7['status'] == 'DEGRADED'
            and ('ATTACKING_TEAM_INFERRED_FROM_GOALKEEPER' in (stage7_raw.get('reasons') or [])
                 or (allow_tentative_context
                     and set(stage7_raw.get('reasons') or []) == {'CONTACT_TENTATIVE_SPATIAL_ONLY'}
                     and (stage7_raw.get('toucher') or {}).get('evidence_level') == 'TENTATIVE_SPATIAL_ONLY'))
        ))
        and s7['s'] == s8['s'] and s in (-1, 1)
        and s4['frame_index'] == s7['frame_index'] == s8['frame_index']
        and s4['frame_index'] is not None
    )
    if defender_only:
        state.mode = 'DEFENDER_REFERENCE'
    tentative_context = bool(
        allow_tentative_context
        and s7['status'] == s8['status'] == 'DEGRADED'
        and set(stage7_raw.get('reasons') or []) == {'CONTACT_TENTATIVE_SPATIAL_ONLY'}
        and set(stage8_raw.get('reasons') or []) == {'STAGE7_CONTACT_TENTATIVE_SPATIAL_ONLY'}
        and (stage7_raw.get('toucher') or {}).get('evidence_level') == 'TENTATIVE_SPATIAL_ONLY'
        and s7['toucher_track_id'] in s7['attackers']
        and s8['reference_q_m'] is not None
        and s8['second_last'].get('goalward_q_m') is not None
        and len(s8['opponent_ranking']) >= 2 and len(s7['opponents']) >= 2
        and s7['s'] == s8['s'] and s in (-1, 1)
        and s4['frame_index'] == s7['frame_index'] == s8['frame_index']
        and s4['frame_index'] is not None
    )
    if tentative_context:
        state.mode = 'TENTATIVE_CONTACT_REFERENCE'
    relaxed_reference = defender_only or tentative_context
    basis = 'DEFENDER_ONLY' if defender_only else 'STAGE8_REFERENCE_TENTATIVE_CONTACT' if tentative_context else 'STANDARD'
    reference_fallback_rows: List[Dict[str, Any]] = []
    if best_effort and ref.get("goalward_q_m") is None and s in (-1, 1) and state.frame_index is not None:
        for tid in list(dict.fromkeys(s7["opponents"])):
            track = s4["tracks"].get(tid)
            if track is None:
                continue
            row = legal_extent_from_stage4_track(track, state.frame_index, s)
            if row.get("goalward_q_m") is not None:
                reference_fallback_rows.append({"track_id": tid, **row})
        reference_fallback_rows.sort(key=lambda r: float(r["goalward_q_m"]), reverse=True)
        q_second = float(reference_fallback_rows[1]["goalward_q_m"]) if len(reference_fallback_rows) >= 2 else None
        ball_extent = s6.get("ball_center_x_extent_m")
        q_ball = None
        if isinstance(ball_extent, list) and len(ball_extent) >= 2:
            q_ball = max(float(s) * float(ball_extent[0]), float(s) * float(ball_extent[1]))
        elif s6.get("X_world_m") is not None:
            q_ball = float(s) * float(s6["X_world_m"])
        available = [(q_second, "SECOND_LAST_OPPONENT_STAGE9_FALLBACK"), (q_ball, "BALL_STAGE9_FALLBACK")]
        available = [(qv, src) for qv, src in available if qv is not None]
        if available:
            q_ref, src = max(available, key=lambda r: r[0])
            ref = {
                "status": "STAGE9_RECOMPUTED_BEST_EFFORT",
                "goalward_q_m": float(q_ref),
                "X_world_m": float(s) * float(q_ref),
                "source": src,
            }
    state.reference = {
        **ref,
        "epsilon_m": float(epsilon_m),
        "upstream_stage8_status": s8["status"],
        "classification_basis": basis,
    }
    state.ball = dict(s8.get("ball") or {}) or (
        {"X_world_m": s6.get("X_world_m"), "ball_center_x_extent_m": s6.get("ball_center_x_extent_m"), "center_xyz_world_m": s6.get("center_xyz_world_m"), "source": "stage6"}
        if s6.get("X_world_m") is not None or s6.get("ball_center_x_extent_m") is not None else None
    )

    attackers: List[Dict[str, Any]] = []
    fallback_count = 0
    missing_count = 0
    offside_count = 0
    onside_count = 0
    toucher_count = 0

    for tid in list(dict.fromkeys(s7["attackers"])):
        track = s4["tracks"].get(tid)
        if track is None or state.frame_index is None or s not in (-1, 1):
            extent = {"status": "MISSING", "source": None, "goalward_q_m": None, "goalward_x_m": None, "anchor": None, "legal_landmark_count": 0}
        else:
            extent = legal_extent_from_stage4_track(track, state.frame_index, s)
        classification = classify_attacker(
            track_id=tid,
            goalward_q_m=extent.get("goalward_q_m"),
            reference_q_m=ref.get("goalward_q_m"),
            toucher_track_id=state.toucher_track_id,
            epsilon_m=epsilon_m,
            geometry_source=extent.get("source"),
        )
        if not best_effort and not relaxed_reference and (s7["status"] != "VALID" or s8["status"] != "VALID"
                                or (stage8_raw.get("reference") or {}).get("reference_only")):
            classification.update(label="UNAVAILABLE", flag="?", is_offside_position=None,
                                  candidate=False, reason="UPSTREAM_CONTEXT_OR_REFERENCE_NOT_CONFIRMED")
        if relaxed_reference:
            classification['classification_basis'] = basis
            if extent.get('goalward_q_m') is None:
                classification.update(label='UNAVAILABLE', flag='?', is_offside_position=None,
                                      candidate=False, reason='PLAYER_GEOMETRY_MISSING')
        if extent.get("status") == "FALLBACK":
            fallback_count += 1
        if extent.get("goalward_q_m") is None:
            missing_count += 1
        if classification["label"] == "OFFSIDE_POSITION":
            offside_count += 1
        elif classification["label"] == "ONSIDE":
            onside_count += 1
        elif classification["label"] == "TOUCHER_EXCLUDED":
            toucher_count += 1
        attackers.append({
            "track_id": tid,
            "role": track.get("role") if isinstance(track, dict) else None,
            "geometry": extent,
            **classification,
        })

    opponents: List[Dict[str, Any]] = []
    ranking_by_id = {str(r.get("track_id")): r for r in s8["opponent_ranking"] if r.get("track_id") is not None}
    for tid in list(dict.fromkeys(s7["opponents"])):
        row = ranking_by_id.get(str(tid), {})
        opponents.append({
            "track_id": tid,
            "rank": row.get("rank"),
            "goalward_q_m": row.get("goalward_q_m"),
            "goalward_x_m": row.get("goalward_x_m"),
            "anchor": row.get("anchor"),
            "second_last": str(tid) in {str(x) for x in (s8.get("second_last") or {}).get("candidate_track_ids", [])},
        })

    state.attackers = attackers
    state.opponents = opponents
    state.others = (
        [{"track_id": tid, "label": "REFEREE"} for tid in s7.get("referees", [])]
        + [{"track_id": tid, "label": "UNKNOWN_TEAM"} for tid in s7.get("unknown", [])]
        + [{"track_id": tid, "label": "INACTIVE"} for tid in s7.get("inactive", [])]
    )

    critical_missing = s not in (-1, 1) or ref.get("goalward_q_m") is None
    if relaxed_reference:
        state.status = 'DEGRADED' if missing_count < len(attackers) else 'UNRESOLVED'
    elif best_effort:
        state.status = "DEMO_BEST_EFFORT" if not critical_missing else "DEMO_PARTIAL"
    else:
        state.status = "VALID" if not critical_missing and missing_count == 0 and s7["status"] == "VALID" and s8["status"] == "VALID" else "UNRESOLVED"

    state.diagnostics = {
        "ignore_upstream_status_for_demo": bool(best_effort),
        "classification_basis": basis,
        "allow_defender_only": bool(allow_defender_only),
        "allow_tentative_context": bool(allow_tentative_context),
        "ball_used_for_reference": not defender_only,
        "upstream_status": {"stage7": s7["status"], "stage8": s8["status"]},
        "frame_alignment": {"stage4": s4["frame_index"], "stage7": s7["frame_index"], "stage8": s8["frame_index"]},
        "attacker_count": len(attackers),
        "opponent_count": len(opponents),
        "offside_position_count": offside_count,
        "onside_count": onside_count,
        "toucher_excluded_count": toucher_count,
        "fallback_geometry_count": fallback_count,
        "missing_geometry_defaulted_onside_count": missing_count,
        "stage9_reference_fallback_opponent_rows": reference_fallback_rows,
        "warning": ("Position classified against the Stage 8 reference; toucher/team context is based on tentative spatial contact."
                    if tentative_context else "Position classified against the second-last defender only; ball/contact unavailable or unconfirmed."
                    if defender_only else "DEMO MODE intentionally ignores DEGRADED/UNRESOLVED/uncertainty gates; labels are visualization outputs, not validated referee decisions."
                    if best_effort else "STRICT: unconfirmed context or reference does not produce offside/onside labels."),
    }
    state.provenance = {
        "stage4": source_path(stage4_input),
        "stage7": source_path(stage7_input),
        "stage8": source_path(stage8_input),
        "stage6": source_path(stage6_input),
        "stage4_schema": s4.get("schema_version"),
        "stage8_schema": stage8_raw.get("schema_version"),
    }
    return state
