from __future__ import annotations

import math
from typing import Any, Dict, List

from .legal_body import goalward_coordinate


DEFAULT_COMPARISON_EPSILON_M = 1e-9


def ball_goalward_extent(x_extent_m: List[float], s: int) -> Dict[str, Any]:
    if not isinstance(x_extent_m, (list, tuple)) or len(x_extent_m) != 2:
        raise ValueError("ball x extent must contain [xmin, xmax]")
    xmin, xmax = float(x_extent_m[0]), float(x_extent_m[1])
    if not (math.isfinite(xmin) and math.isfinite(xmax)):
        raise ValueError("ball x extent must be finite")
    if xmin > xmax:
        raise ValueError("ball x extent must be ordered xmin <= xmax")
    q_values = [goalward_coordinate(xmin, s), goalward_coordinate(xmax, s)]
    q = max(q_values)
    x = xmin if q_values[0] >= q_values[1] else xmax
    return {
        "x_extent_m": [xmin, xmax],
        "goalward_q_m": float(q),
        "goalward_x_m": float(x),
    }


def build_reference(
    second_last: Dict[str, Any],
    ball: Optional[Dict[str, Any]],
    s: int,
    *,
    epsilon_m: float = DEFAULT_COMPARISON_EPSILON_M
) -> Dict[str, Any]:
    q_second = float(second_last["goalward_q_m"])
    if ball is None or ball.get("goalward_q_m") is None:
        source = "SECOND_LAST_OPPONENT"
        q_ref = q_second
    else:
        q_ball = float(ball["goalward_q_m"])
        if q_ball > q_second + epsilon_m:
            source = "BALL"
            q_ref = q_ball
        elif q_second > q_ball + epsilon_m:
            source = "SECOND_LAST_OPPONENT"
            q_ref = q_second
        else:
            source = "BALL_AND_SECOND_LAST_OPPONENT_LEVEL"
            q_ref = max(q_ball, q_second)
    x_ref = float(s) * q_ref
    return {
        "source": source,
        "goalward_q_m": float(q_ref),
        "X_world_m": x_ref,
        "vertical_plane": {
            "axis": "X",
            "X_world_m": x_ref,
            "equation": {"normal": [1.0, 0.0, 0.0], "offset": -x_ref},
            "meaning": "X = X_world_m in STAGE1_PITCH_WORLD",
        },
    }
