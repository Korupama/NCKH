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


def assign_goalkeeper_spatial(
    *,
    gk_pitch_x: Optional[Any] = None,
    gk_image_x: Optional[Any] = None,
    team_pitch_x: Optional[Mapping[int, Any]] = None,
    team_image_x: Optional[Mapping[int, Any]] = None,
    config: Stage5Config,
) -> Dict[str, Any]:
    mode = getattr(config, "goalkeeper_assignment_mode", "spatial_hybrid")

    # 1. Pitch spatial reasoning (if enabled and coordinates available)
    if mode in ("spatial_hybrid", "spatial_pitch"):
        if (
            gk_pitch_x is not None and len(gk_pitch_x) > 0
            and team_pitch_x is not None and 0 in team_pitch_x and 1 in team_pitch_x
            and len(team_pitch_x[0]) > 0 and len(team_pitch_x[1]) > 0
        ):
            med0 = float(np.median(team_pitch_x[0]))
            med1 = float(np.median(team_pitch_x[1]))
            sep = abs(med0 - med1)
            min_sep = getattr(config, "goalkeeper_min_pitch_separation_m", 1.0)
            if sep >= min_sep:
                left_team = 0 if med0 < med1 else 1
                right_team = 1 - left_team
                gk_med = float(np.median(gk_pitch_x))
                team_id = left_team if gk_med < 0 else right_team
                margin = abs(gk_med)
                return {
                    "team_id": team_id,
                    "status": "VALID",
                    "method": "SPATIAL_PITCH_GOAL_AFFINITY",
                    "margin": margin,
                    "pitch_separation_m": sep,
                    "gk_pitch_median_m": gk_med,
                }
        if mode == "spatial_pitch":
            return {"team_id": None, "status": "UNKNOWN", "method": "INSUFFICIENT_PITCH_COORDINATES"}

    # 2. Image spatial reasoning (if enabled and coordinates available)
    if mode in ("spatial_hybrid", "spatial_image"):
        if (
            gk_image_x is not None and len(gk_image_x) > 0
            and team_image_x is not None and 0 in team_image_x and 1 in team_image_x
            and len(team_image_x[0]) > 0 and len(team_image_x[1]) > 0
        ):
            med0 = float(np.median(team_image_x[0]))
            med1 = float(np.median(team_image_x[1]))
            team_sep = abs(med0 - med1)
            min_team_sep = getattr(config, "goalkeeper_min_team_image_separation_px", 80.0)
            if team_sep < min_team_sep:
                if not getattr(config, "goalkeeper_fallback_to_color", True):
                    return {
                        "team_id": None,
                        "status": "UNKNOWN",
                        "method": "AMBIGUOUS_TEAM_IMAGE_SEPARATION",
                        "team_separation_px": team_sep,
                    }
            else:
                gk_med = float(np.median(gk_image_x))
                # In soccer, a goalkeeper is at the defensive extremity.
                # If GK is located between team centroids (interior), image X distance is unreliable (e.g. set-piece scramble).
                is_interior = min(med0, med1) < gk_med < max(med0, med1)
                if is_interior:
                    if not getattr(config, "goalkeeper_fallback_to_color", True):
                        return {
                            "team_id": None,
                            "status": "UNKNOWN",
                            "method": "AMBIGUOUS_GK_INTERIOR_POSITION",
                            "gk_image_median_px": gk_med,
                        }
                else:
                    d0 = abs(gk_med - med0)
                    d1 = abs(gk_med - med1)
                    margin = abs(d0 - d1)
                    min_margin = getattr(config, "goalkeeper_min_spatial_margin_px", 30.0)
                    if margin >= min_margin:
                        team_id = 0 if d0 < d1 else 1
                        return {
                            "team_id": team_id,
                            "status": "VALID",
                            "method": "SPATIAL_IMAGE_CENTROID_AFFINITY",
                            "margin": margin,
                            "d0_px": d0,
                            "d1_px": d1,
                            "gk_image_median_px": gk_med,
                        }
                    elif not getattr(config, "goalkeeper_fallback_to_color", True):
                        return {
                            "team_id": None,
                            "status": "UNKNOWN",
                            "method": "AMBIGUOUS_IMAGE_SPATIAL_SEPARATION",
                            "margin": margin,
                        }

    return {"team_id": None, "status": "UNKNOWN", "method": "SPATIAL_UNAVAILABLE"}


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
    mode = getattr(config, "goalkeeper_assignment_mode", "spatial_hybrid")
    if mode != "color_lower_body":
        spatial_res = assign_goalkeeper_spatial(
            gk_pitch_x=gk_pitch_x,
            gk_image_x=gk_image_x,
            team_pitch_x=team_pitch_x,
            team_image_x=team_image_x,
            config=config,
        )
        if spatial_res.get("status") == "VALID":
            return spatial_res
        if not getattr(config, "goalkeeper_fallback_to_color", True):
            return spatial_res

    # Color fallback
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
