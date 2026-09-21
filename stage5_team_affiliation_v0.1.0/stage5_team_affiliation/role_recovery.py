"""Metric pitch reasoning only. No GT or upstream semantic roles enter this module."""
import numpy as np


def clean_pitch_tracks(points, config):
    result = {}
    for tid, observations in points.items():
        result[tid] = {}
        for frame, xy in observations.items():
            p = np.asarray(xy, dtype=float)
            if (p.shape == (2,) and np.isfinite(p).all()
                    and abs(p[0]) <= config.pitch_length_m / 2 + config.pitch_margin_m
                    and abs(p[1]) <= config.pitch_width_m / 2 + config.pitch_margin_m):
                result[tid][int(frame)] = p
    return result


def project_track_to_pitch(observations, intersect_pitch):
    """Stage1 adapter seam: caller supplies intersect_pitch(frame, raw_pixel).

    Prefer distal feet; otherwise bbox bottom-center. No camera implementation or
    Stage4 proxy is invented here. Exceptions from the camera adapter propagate.
    """
    result = {}
    for obs in observations:
        feet = [(k.get('x'), k.get('y')) for k in obs.get('keypoints_133', [])
                if k.get('name') in {'left_big_toe', 'right_big_toe', 'left_heel', 'right_heel'}
                and k.get('state') == 'VALID' and k.get('x') is not None and k.get('y') is not None]
        feet = [p for p in feet if np.isfinite(p).all()]
        if feet:
            pixel = np.median(feet, axis=0)
        else:
            box = obs.get('source_bbox_xyxy')
            if box is None or len(box) != 4 or not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
                continue
            pixel = [(box[0] + box[2]) / 2, box[3]]
        fi = int(obs['frame_index'])
        xyz = intersect_pitch(fi, pixel)
        if xyz is not None:
            xyz = np.asarray(xyz, dtype=float)
            if xyz.shape in ((2,), (3,)) and np.isfinite(xyz).all():
                result[fi] = xyz[:2].tolist()
    return result


def compute_depth_rank(tid, points, side, config):
    ranks = []
    for frame, xy in points.get(tid, {}).items():
        others = [side * p[frame][0] for key, p in points.items() if key != tid and frame in p]
        if len(others) + 1 < config.min_rank_humans:
            continue
        # Ties are conservative: all nearly tied people count as ahead.
        ranks.append(1 + sum(q >= side * xy[0] - config.ordering_margin_m for q in others))
    return {'rank_frames': len(ranks), 'top1_rate': float(np.mean(np.array(ranks) == 1)) if ranks else None,
            'top2_rate': float(np.mean(np.array(ranks) <= 2)) if ranks else None}


def compute_goal_depth(tid, points, config):
    obs = points.get(tid, {})
    if not obs:
        return {'observations': 0, 'side': None}
    x = np.array([p[0] for p in obs.values()]); median_x = float(np.median(x))
    side = 1 if median_x > 0 else -1 if median_x < 0 else 0
    return {'observations': len(obs), 'side': 'RIGHT' if side == 1 else 'LEFT' if side == -1 else None,
            'sign': side, 'median_goalward_depth_m': abs(median_x),
            'median_goal_distance_m': config.pitch_length_m / 2 - abs(median_x),
            **compute_depth_rank(tid, points, side, config)}


def pairwise_order(a, b, points, side, config):
    overlap = sorted(set(points.get(a, {})) & set(points.get(b, {})))
    wins = [side * (points[a][f][0] - points[b][f][0]) > config.ordering_margin_m for f in overlap]
    return {'overlap_frames': len(overlap), 'rate': float(np.mean(wins)) if wins else None}


def classify_residual_roles(residual_ids, points, config, *, temporal=True):
    contexts = {t: compute_goal_depth(t, points, config) for t in residual_ids}
    result = {t: {'stage5_role': 'unknown_residual', 'stage5_role_status': 'UNKNOWN',
                  'stage5_role_method': 'RESIDUAL_GOAL_DEPTH_V021',
                  'role_decision_reason': 'NOT_A_GOALKEEPER_CANDIDATE',
                  'goal_context': c} for t, c in contexts.items()}
    winners = []
    for side in (-1, 1):
        geometric = [t for t, c in contexts.items() if c.get('sign') == side
                     and c['observations'] >= config.min_observations
                     and c['median_goal_distance_m'] <= config.goal_distance_m]
        candidates = []
        for tid in geometric:
            c = contexts[tid]
            if temporal and c['rank_frames'] < config.min_observations:
                result[tid]['role_decision_reason'] = 'INSUFFICIENT_ALL_HUMAN_RANK_SUPPORT'
                continue
            if temporal and c['top2_rate'] < config.min_top2_rate:
                result[tid]['role_decision_reason'] = 'LOW_ALL_HUMAN_TOP2_RATE'
                continue
            candidates.append(tid)
        candidates.sort(key=lambda t: (-contexts[t]['median_goalward_depth_m'], t))
        for tid in candidates:
            contexts[tid]['goalkeeper_candidate'] = True
            contexts[tid]['goalkeeper_candidate_side'] = 'RIGHT' if side == 1 else 'LEFT'
            result[tid]['role_decision_reason'] = 'NOT_DEEPEST_GOAL_CANDIDATE'
        if not candidates:
            continue
        best = candidates[0]
        decision = {
            'candidate_ids': list(candidates),
            'pairwise_compared_ids': [],
            'pairwise_policy': ('REQUIRED' if config.pairwise_ordering_required else 'DIAGNOSTIC_ONLY'),
            'pairwise_ignored_non_candidate_ids': sorted(
                t for t in residual_ids
                if t != best and contexts[t].get('sign') == side and t not in geometric
            ),
        }
        result[best]['role_gate_diagnostics'] = decision
        if len(candidates) > 1 and (contexts[best]['median_goalward_depth_m'] -
                                   contexts[candidates[1]]['median_goalward_depth_m'] <= config.ordering_margin_m):
            result[best]['role_decision_reason'] = 'AMBIGUOUS_GOAL_DEPTH_ORDER'
            continue
        c = contexts[best]
        if temporal:
            # Selection eligibility and pairwise diagnostics intentionally use
            # different pools.  The winner must pass the temporal rank/top-2
            # gates, but every residual that independently passes the spatial
            # goal-candidate gate remains a pairwise rival.  Otherwise a rival
            # with insufficient/no temporal overlap silently disappears and a
            # frozen config with pairwise_ordering_required=True can incorrectly
            # accept the winner instead of abstaining.
            rivals = [t for t in geometric if t != best]
            evidence = {t: pairwise_order(best, t, points, side, config) for t in rivals}
            c['pairwise_order'] = evidence
            decision['pairwise_compared_ids'] = list(rivals)
            insufficient = [t for t, e in evidence.items() if e['overlap_frames'] < config.min_observations]
            failed = [t for t, e in evidence.items()
                      if e['overlap_frames'] >= config.min_observations and e['rate'] < config.min_pairwise_rate]
            decision['pairwise_insufficient_overlap_ids'] = insufficient
            decision['pairwise_failed_ids'] = failed
            if config.pairwise_ordering_required and insufficient:
                result[best]['role_decision_reason'] = 'INSUFFICIENT_GOAL_CANDIDATE_OVERLAP'
                continue
            if config.pairwise_ordering_required and failed:
                result[best]['role_decision_reason'] = 'FAILED_GOAL_CANDIDATE_ORDERING'
                continue
        result[best].update(stage5_role='goalkeeper', stage5_role_status='VALID',
                            role_decision_reason='GOALKEEPER_GATES_PASSED')
        winners.append(best)
    # Referee recovery is deliberately disabled by default in v0.2.1.  The v0.2.0
    # rule confused ordinary appearance-outlier players with referees in the smoke
    # benchmark.  It may only be opted into by a frozen TRAIN-derived config.
    if not config.referee_recovery_enabled:
        for tid in result:
            if tid not in winners:
                result[tid]['referee_recovery'] = {
                    'enabled': False,
                    'reason': 'NOT_CALIBRATED_ON_TRAIN',
                }
        return result

    for tid, c in contexts.items():
        if tid in winners or c['observations'] < config.min_observations:
            continue
        if c.get('median_goal_distance_m', 0) < config.referee_min_goal_distance_m:
            continue
        if not temporal:
            continue  # V2 identifies goal candidates, not roaming officials.
        rank_sides = [compute_depth_rank(tid, points, s, config) for s in (-1, 1)]
        if any(r['rank_frames'] < config.min_observations or r['top2_rate'] > config.referee_max_top2_rate for r in rank_sides):
            continue
        beaten = [pairwise_order(g, tid, points, contexts[g]['sign'], config) for g in winners]
        if any(e['overlap_frames'] >= config.min_observations and e['rate'] >= config.min_pairwise_rate for e in beaten):
            result[tid].update(stage5_role='referee', stage5_role_status='VALID',
                               stage5_role_method='RESIDUAL_REFEREE_HEURISTIC_OPT_IN',
                               role_decision_reason='REFEREE_OPT_IN_GATES_PASSED',
                               referee_recovery={'enabled': True})
    return result
