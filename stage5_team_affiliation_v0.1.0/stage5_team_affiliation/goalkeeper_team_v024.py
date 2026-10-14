"""Historical geometry diagnostics; assignment now uses fixed defending policy."""
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
    from .goalkeeper import forced_defending_assignment
    return forced_defending_assignment()

