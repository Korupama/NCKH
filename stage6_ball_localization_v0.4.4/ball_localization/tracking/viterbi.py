from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import math
import numpy as np
from ..contracts import BallCandidate2D


@dataclass
class ViterbiConfig:
    missing_cost: float = 2.6
    enter_exit_cost: float = 0.8
    motion_weight: float = 7.0
    size_weight: float = 0.35
    max_normalized_jump: float = 0.30
    epsilon: float = 1e-6
    net_stationary_penalty: float = 2.5
    static_line_penalty: float = 1.5


def _emission(c: Optional[BallCandidate2D], cfg: ViterbiConfig) -> float:
    if c is None:
        return cfg.missing_cost
    return -math.log(max(cfg.epsilon, min(1.0, float(c.ranking_score))))


def _transition(a: Optional[BallCandidate2D], b: Optional[BallCandidate2D], diag: float, cfg: ViterbiConfig) -> float:
    if a is None and b is None:
        return 0.0
    if a is None or b is None:
        return cfg.enter_exit_cost
    dist_px = float(np.linalg.norm(np.asarray(b.center_uv) - np.asarray(a.center_uv)))
    jump = dist_px / max(diag, 1.0)
    cost = cfg.motion_weight * jump
    if jump > cfg.max_normalized_jump:
        cost += 8.0 * (jump - cfg.max_normalized_jump)
    cost += cfg.size_weight * abs(math.log(max(b.diameter_px, 1e-3) / max(a.diameter_px, 1e-3)))

    # Anti-stagnation penalties for goal net and static line artifacts:
    # A candidate staying nearly motionless on the goal net is a net knot artifact.
    b_meta = b.metadata or {}
    if (b_meta.get("in_goal_net") or b_meta.get("is_goal_net")) and dist_px < 2.0:
        cost += cfg.net_stationary_penalty
    if float(b_meta.get("line_coherence", 0.0)) > 0.60 and dist_px < 1.0:
        cost += cfg.static_line_penalty

    return cost


def select_ball_path(
    candidates_by_frame: Dict[int, List[BallCandidate2D]],
    frame_indices: List[int],
    *,
    image_width: int,
    image_height: int,
    config: ViterbiConfig | None = None,
) -> Dict[int, Optional[BallCandidate2D]]:
    cfg = config or ViterbiConfig()
    if not frame_indices:
        return {}
    diag = float(math.hypot(image_width, image_height))
    states = [list(candidates_by_frame.get(fi, [])) + [None] for fi in frame_indices]
    costs = [np.asarray([_emission(s, cfg) for s in states[0]], float)]
    back = [np.full(len(states[0]), -1, int)]
    for t in range(1, len(frame_indices)):
        cur = np.full(len(states[t]), np.inf)
        b = np.full(len(states[t]), -1, int)
        for j, sj in enumerate(states[t]):
            vals = [costs[t - 1][i] + _transition(si, sj, diag, cfg) for i, si in enumerate(states[t - 1])]
            i = int(np.argmin(vals))
            cur[j] = vals[i] + _emission(sj, cfg)
            b[j] = i
        costs.append(cur)
        back.append(b)
    j = int(np.argmin(costs[-1]))
    chosen = {}
    for t in range(len(frame_indices) - 1, -1, -1):
        chosen[frame_indices[t]] = states[t][j]
        j = int(back[t][j]) if t > 0 else -1
    return chosen
