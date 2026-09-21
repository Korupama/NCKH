"""Goalkeeper-to-team association from whole-team defended-half geometry.

No semantic role/team GT enters this module.  ``player_teams`` contains only the
two dominant appearance clusters inferred upstream.
"""
from __future__ import annotations

import numpy as np


def defended_half_deltas(goalkeeper_id, side, points, player_teams):
    """Return per-frame TEAM_0 minus TEAM_1 median goalward depth.

    Positive values mean TEAM_0 is collectively deeper toward the selected
    goal; negative values mean TEAM_1.  Medians over all available outfield
    players are intentionally used instead of only the deepest one or two
    players, making the signal less sensitive to a single overlapping attacker.
    """
    frames = sorted(points.get(goalkeeper_id, {}))
    deltas = []
    for frame in frames:
        depths = {0: [], 1: []}
        for track_id, team_id in player_teams.items():
            if team_id not in (0, 1) or frame not in points.get(track_id, {}):
                continue
            depths[int(team_id)].append(float(side) * float(points[track_id][frame][0]))
        if not depths[0] or not depths[1]:
            continue
        deltas.append(float(np.median(depths[0]) - np.median(depths[1])))
    return deltas


def assign_by_defended_half(goalkeeper_id, side, points, player_teams, config):
    deltas = np.asarray(
        defended_half_deltas(goalkeeper_id, side, points, player_teams), dtype=float)
    diagnostics = {
        'assignment_method': 'DEFENDED_HALF_TEAM_DISTRIBUTION',
        'half_deltas_m': deltas.tolist(),
        'observations': int(len(deltas)),
        'team_id': None,
        'team_status': 'UNKNOWN',
    }
    if len(deltas) < config.min_observations:
        diagnostics['reason'] = 'INSUFFICIENT_DEFENDED_HALF_SUPPORT'
        return diagnostics
    median = float(np.median(deltas))
    winner = 0 if median > 0 else 1
    sign = 1.0 if winner == 0 else -1.0
    vote_rate = float(np.mean(deltas * sign >= config.min_defended_half_separation_m))
    diagnostics.update(
        median_delta_m=median,
        absolute_median_delta_m=abs(median),
        vote_rate=vote_rate,
        suggested_team_id=winner,
    )
    if (abs(median) < config.min_defended_half_separation_m
            or vote_rate < config.min_defended_half_vote_rate):
        diagnostics['reason'] = 'AMBIGUOUS_DEFENDED_HALF_ORDERING'
        return diagnostics
    diagnostics.update(
        team_id=winner,
        team_status='VALID',
        reason='DEFENDED_HALF_GATES_PASSED',
    )
    return diagnostics

