from __future__ import annotations

from typing import Any, Dict, Mapping, Optional
import numpy as np

from .clustering import assign_to_centroids
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
    """Top-K team tails computed on the SAME frames as the GK, then voted over time."""
    deltas = defensive_tail_deltas(gk_id, side, points, player_teams, config)
    result = {'team_id': None, 'team_status': 'UNKNOWN', 'assignment_method': 'DEFENSIVE_TAIL_ASSOCIATION',
              'support_frames': len(deltas), 'tail_deltas_m': deltas,
              'reason': 'INSUFFICIENT_SIMULTANEOUS_TEAM_SUPPORT'}
    if len(deltas) < config.min_observations:
        return result
    delta = float(np.median(deltas)); winner = 0 if delta > 0 else 1
    sign = 1 if winner == 0 else -1
    votes = float(np.mean(np.array(deltas) * sign >= config.min_tail_separation_m))
    result.update(median_tail_delta_m=delta, agreement_rate=votes, reason='AMBIGUOUS_DEFENSIVE_TAIL')
    if abs(delta) >= config.min_tail_separation_m and votes >= config.min_tail_vote_rate:
        result.update(team_id=winner, team_status='VALID', reason=None)
    return result


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


def assign_goalkeeper(feature: Optional[np.ndarray], team_lower_centroids: Mapping[int, np.ndarray], config: Stage5Config) -> Dict[str, Any]:
    if feature is None:
        return {"team_id": None, "status": "UNKNOWN", "method": "NO_LOWER_BODY_FEATURE"}
    if set(team_lower_centroids) != {0, 1}:
        return {"team_id": None, "status": "UNKNOWN", "method": "INSUFFICIENT_TEAM_LOWER_BODY_REFERENCE"}
    centroids = np.stack([team_lower_centroids[0], team_lower_centroids[1]], axis=0)
    result = assign_to_centroids(
        feature, centroids,
        min_margin=config.goalkeeper_min_margin,
        max_distance_ratio=config.goalkeeper_max_distance_ratio,
    )
    result["method"] = "LOWER_BODY_APPEARANCE_AFFINITY"
    return result
