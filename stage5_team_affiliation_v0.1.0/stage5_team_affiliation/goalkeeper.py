from __future__ import annotations

from typing import Any, Dict, Mapping, Optional
import numpy as np

from .config import Stage5Config


def defensive_tail_deltas(gk_id, side, points, player_teams, config):
    """Signed TEAM_0-minus-TEAM_1 goalward tail deltas on GK-visible frames."""
    deltas = []
    for frame in sorted(points.get(gk_id, {})):
        tails = []
        for team in (0, 1):
            q = sorted((side * points[t][frame][0] for t, c in player_teams.items()
                        if c == team and frame in points.get(t, {})), reverse=True)
            if len(q) < config.defensive_tail_k:
                break
            tails.append(float(np.median(q[:config.defensive_tail_k])))
        if len(tails) == 2:
            deltas.append(tails[0] - tails[1])
    return deltas


def assign_by_defensive_tail(gk_id, side, points, player_teams, config):
    return forced_defending_assignment()


def lower_body_centroids_by_team(
    team_labels: Mapping[str, int],
    team_status: Mapping[str, str],
    lower_features: Mapping[str, np.ndarray],
) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    for team in (0, 1):
        feats = [lower_features[tid] for tid, lab in team_labels.items() if lab == team and team_status.get(tid) == "VALID" and tid in lower_features]
        if feats:
            x = np.median(np.stack(feats, axis=0), axis=0)
            n = float(np.linalg.norm(x))
            out[team] = (x / n if n > 0 else x).astype(np.float32)
    return out


def assign_goalkeeper_spatial(
    *,
    gk_pitch_x: Optional[Any] = None,
    gk_image_x: Optional[Any] = None,
    team_pitch_x: Optional[Mapping[int, Any]] = None,
    team_image_x: Optional[Mapping[int, Any]] = None,
    config: Stage5Config,
) -> Dict[str, Any]:
    return forced_defending_assignment()


def assign_goalkeeper(
    feature: Optional[np.ndarray],
    team_lower_centroids: Mapping[int, np.ndarray],
    config: Stage5Config,
    *,
    gk_pitch_x: Optional[Any] = None,
    gk_image_x: Optional[Any] = None,
    team_pitch_x: Optional[Mapping[int, Any]] = None,
    team_image_x: Optional[Mapping[int, Any]] = None,
) -> Dict[str, Any]:
    return forced_defending_assignment()


def forced_defending_assignment():
    """GK membership is fixed; the defending cluster ID is resolved in Stage 7."""
    return {"team_id": None, "status": "PENDING_CONTEXT", "team_status": "PENDING_CONTEXT",
            "team_side": "DEFENDING", "method": "GOALKEEPER_ALWAYS_DEFENDING",
            "assignment_method": "GOALKEEPER_ALWAYS_DEFENDING"}
