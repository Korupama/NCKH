from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .adapters import extract_stage4_context, extract_stage6_ball, extract_stage7_context, load_json
from .legal_body import EXCLUDED_ARM_LANDMARKS, LEGAL_LANDMARKS, POLICY_ID, legal_landmark_extent
from .models import OffsideReferenceState
from .preflight import build_preflight
from .ranking import DEFAULT_TIE_EPSILON_M, rank_opponents, select_second_last
from .reference import DEFAULT_COMPARISON_EPSILON_M, ball_goalward_extent, build_reference


def _source_path(value: Any) -> str | None:
    if isinstance(value, dict):
        return None
    try:
        return str(Path(value).resolve())
    except Exception:
        return str(value)


def build_offside_reference(
    stage4_input: Any,
    stage6_input: Any,
    stage7_input: Any,
    *,
    tie_epsilon_m: float = DEFAULT_TIE_EPSILON_M,
    comparison_epsilon_m: float = DEFAULT_COMPARISON_EPSILON_M,
) -> OffsideReferenceState:
    stage4 = load_json(stage4_input)
    stage6 = load_json(stage6_input)
    stage7 = load_json(stage7_input)
    s4 = extract_stage4_context(stage4)
    s6 = extract_stage6_ball(stage6)
    s7 = extract_stage7_context(stage7)
    pre = build_preflight(stage4, stage6, stage7)

    state = OffsideReferenceState()
    state.frame_index = s7["frame_index"] if s7["frame_index"] is not None else s4["selected_frame"]
    state.attack_direction = None if s7["s"] not in (-1, 1) else {
        "s": s7["s"],
        "label": s7["label"],
        "source": "stage7",
        "upstream_source": s7["source"],
        "goalward_coordinate": "q = s * X_world_m",
    }
    state.legal_body_policy = {
        "policy_id": POLICY_ID,
        "semantics": "landmark-based proxy for legal offside body geometry; not an exact body surface",
        "included_landmarks": list(LEGAL_LANDMARKS),
        "excluded_arm_landmarks": list(EXCLUDED_ARM_LANDMARKS),
    }

    extent_rows: List[Dict[str, Any]] = []
    if s7["s"] in (-1, 1) and s4["selected_frame"] is not None:
        for tid in list(dict.fromkeys(s7["opponents"])):
            track = s4["tracks"].get(tid)
            if track is None:
                extent_rows.append({
                    "track_id": tid,
                    "status": "MISSING",
                    "reason": "STAGE4_TRACK_MISSING",
                    "goalward_q_m": None,
                    "goalward_x_m": None,
                    "anchor": None,
                    "legal_landmark_count": 0,
                })
            else:
                extent_rows.append(legal_landmark_extent(track, s4["selected_frame"], s7["s"]))

    usable = [row for row in extent_rows if row.get("goalward_q_m") is not None and row.get("status") in {"VALID", "DEGRADED"}]
    state.opponent_ranking = rank_opponents(usable, tie_epsilon_m=tie_epsilon_m)

    ball = None
    if s7["s"] in (-1, 1) and s6["ball_center_x_extent_m"] is not None:
        try:
            ball = ball_goalward_extent(s6["ball_center_x_extent_m"], s7["s"])
            ball.update({
                "X_world_m": s6["X_world_m"],
                "center_xyz_world_m": s6["center_xyz_world_m"],
                "selected_method": s6["selected_method"],
                "usable_for_offside": s6["usable_for_offside"],
                "accuracy_validated": s6["accuracy_validated"],
                "source": s6["source"],
            })
        except ValueError:
            ball = None
    state.ball = ball

    if pre["status"] in {"READY", "DEGRADED_READY"}:
        second = select_second_last(state.opponent_ranking, tie_epsilon_m=tie_epsilon_m)
        if second is not None and ball is not None:
            state.second_last_opponent = second
            state.reference = build_reference(second, ball, s7["s"], epsilon_m=comparison_epsilon_m)
            state.status = "DEGRADED" if pre["status"] == "DEGRADED_READY" else "VALID"
            if state.status == "DEGRADED":
                state.reasons = ["STAGE7_CONTACT_TENTATIVE_SPATIAL_ONLY"]
        else:
            state.status = "UNRESOLVED"
            state.reasons = ["REFERENCE_GEOMETRY_NOT_RESOLVED"]
    else:
        state.status = "UNRESOLVED"
        state.reasons = list(pre["blockers"])

    state.diagnostics = {
        "preflight": pre,
        "opponent_extent_rows": extent_rows,
        "opponent_count_stage7": len(s7["opponents"]),
        "opponent_count_with_usable_geometry": len(usable),
        "ranking_complete": pre["status"] in {"READY", "DEGRADED_READY"},
        "tie_epsilon_m": float(tie_epsilon_m),
        "reference_comparison_epsilon_m": float(comparison_epsilon_m),
        "invariants": {
            "ranking_descending_q": all(
                state.opponent_ranking[i]["goalward_q_m"] >= state.opponent_ranking[i + 1]["goalward_q_m"]
                for i in range(max(0, len(state.opponent_ranking) - 1))
            ),
            "reference_not_behind_second_last": (
                True if state.reference is None or state.second_last_opponent is None
                else state.reference["goalward_q_m"] + comparison_epsilon_m >= state.second_last_opponent["goalward_q_m"]
            ),
            "reference_not_behind_ball": (
                True if state.reference is None or state.ball is None
                else state.reference["goalward_q_m"] + comparison_epsilon_m >= state.ball["goalward_q_m"]
            ),
        },
    }
    state.quality = {
        "implementation_valid": state.status == "VALID",
        "upstream_metric_accuracy_validated": False,
        "research_accuracy_frozen": False,
        "stage4_geometry_semantics": "landmark proxy; independent offside metric accuracy not established by this stage",
        "stage6_ball_accuracy_validated": bool(s6["accuracy_validated"]),
    }
    state.provenance = {
        "stage4": _source_path(stage4_input),
        "stage6": _source_path(stage6_input),
        "stage7": _source_path(stage7_input),
        "stage4_schema_version": s4["schema_version"],
        "stage4_producer": s4["producer"],
    }
    return state
