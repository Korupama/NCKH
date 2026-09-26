from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from .constants import DEFAULT_EPSILON_M, LABEL_OFFSIDE, LABEL_ONSIDE, LABEL_TOUCHER, LABEL_UNAVAILABLE


def classify_attacker(
    *,
    track_id: str,
    goalward_q_m: Optional[float],
    reference_q_m: Optional[float],
    toucher_track_id: Optional[str],
    epsilon_m: float = DEFAULT_EPSILON_M,
    geometry_source: Optional[str] = None,
) -> Dict[str, Any]:
    if toucher_track_id is not None and str(track_id) == str(toucher_track_id):
        return {
            "label": LABEL_TOUCHER,
            "flag": "PASSER",
            "is_offside_position": False,
            "candidate": False,
            "own_half_or_halfway": None,
            "delta_q_m": None if goalward_q_m is None or reference_q_m is None else float(goalward_q_m - reference_q_m),
            "reason": "TOUCHER_EXCLUDED_FROM_OFFSIDE_POSITION_CANDIDATES",
        }
    if goalward_q_m is None or reference_q_m is None:
        return {
            "label": LABEL_UNAVAILABLE,
            "flag": "ON",
            "is_offside_position": False,
            "candidate": True,
            "own_half_or_halfway": None,
            "delta_q_m": None,
            "reason": "GEOMETRY_MISSING_BEST_EFFORT_DEFAULT_ONSIDE",
        }

    q = float(goalward_q_m)
    q_ref = float(reference_q_m)
    delta = q - q_ref
    own_half = q <= float(epsilon_m)
    is_offside = (not own_half) and delta > float(epsilon_m)
    return {
        "label": LABEL_OFFSIDE if is_offside else LABEL_ONSIDE,
        "flag": "OFF" if is_offside else "ON",
        "is_offside_position": bool(is_offside),
        "candidate": True,
        "own_half_or_halfway": bool(own_half),
        "delta_q_m": float(delta),
        "reason": (
            "GOALWARD_OF_REFERENCE_IN_OPPONENT_HALF"
            if is_offside
            else "OWN_HALF_OR_HALFWAY" if own_half else "LEVEL_OR_BEHIND_REFERENCE"
        ),
        "geometry_source": geometry_source,
    }
