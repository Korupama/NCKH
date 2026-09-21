"""TRAIN-only calibration from a completed residual-v3 prediction directory."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import hashlib
import itertools
import json
import math
import numpy as np

from .gsr_benchmark import SoccerNetGSRDataset
from .policy_v023 import research_freeze_gate
from .residual_config import ResidualConfig
from . import __version__


PLAYER_MARGINS = (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
PLAYER_MAX_DISTANCES = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 1.00)
PLAYER_VETO_GOAL_DISTANCES = (15.0, 18.0, 20.0, 22.0, 25.0, 30.0, 35.0, 40.0)
PLAYER_VETO_TOP2_RATES = (0.20, 0.30, 0.40, 0.45, 0.50, 0.60, 0.70)
REFEREE_MARGINS = (0.05, 0.10, 0.15, 0.20, 0.25)
REFEREE_MIN_DISTANCES = (0.40, 0.50, 0.60, 0.70, 0.80)
REFEREE_GOAL_DISTANCES = (18.0, 22.0, 26.0, 30.0)
REFEREE_TOP2_RATES = (0.10, 0.20, 0.30)
GOALKEEPER_GOAL_DISTANCES = (8.0, 10.0, 12.0, 15.0, 18.0, 20.0, 22.0, 25.0,
                             28.0, 30.0, 35.0)
GOALKEEPER_TOP2_RATES = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60,
                         0.70, 0.80, 0.90)
TAIL_SEPARATIONS = (0.50, 1.00, 1.50, 2.00, 2.50, 3.00, 4.00, 5.00)
TAIL_VOTE_RATES = (0.50, 0.60, 0.70, 0.80, 0.90)
HALF_SEPARATIONS = (0.50, 1.00, 1.50, 2.00, 3.00, 4.00, 5.00, 7.50, 10.00)
HALF_VOTE_RATES = (0.50, 0.60, 0.70, 0.80, 0.90)


def _prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return {
        'tp': tp, 'fp': fp, 'fn': fn, 'precision': precision, 'recall': recall,
        'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
    }


def _row_key(row):
    return f"{row['sequence_id']}::{row['track_id']}"


def safe_referee_goal_distance(current_distance, goalkeeper_distance, ordering_margin):
    """Return a referee boundary that is strictly beyond the GK zone."""
    return max(float(current_distance), float(goalkeeper_distance) + float(ordering_margin))


def evaluate_goalkeeper_role_policy(rows, *, goal_distance_m, min_top2_rate,
                                    min_observations, ordering_margin_m):
    """Re-evaluate goalkeeper roles from cached prediction-time geometry only.

    GT is consulted only after selections have been made.  The selector mirrors
    runtime semantics: appearance residuals are grouped by sequence/goal side,
    independent observation/rank/top-2 gates run first, then the deepest
    surviving candidate is selected.  Near-tied candidates cause abstention.
    """
    groups = {}
    for row in rows:
        if row.get('appearance_group') != 'RESIDUAL':
            continue
        sign = row.get('goal_sign')
        if sign not in (-1, 1):
            continue
        if row.get('goal_observations', 0) < min_observations:
            continue
        if row.get('goal_rank_frames', 0) < min_observations:
            continue
        top2 = row.get('goal_top2_rate')
        distance = row.get('median_goal_distance_m')
        if top2 is None or top2 < min_top2_rate:
            continue
        if distance is None or distance > goal_distance_m:
            continue
        groups.setdefault((row['sequence_id'], sign), []).append(row)

    selected = set()
    ambiguous_groups = 0
    for candidates in groups.values():
        candidates.sort(key=lambda r: (-r.get('median_goalward_depth_m', -math.inf), r['track_id']))
        if len(candidates) > 1:
            gap = (candidates[0].get('median_goalward_depth_m', -math.inf)
                   - candidates[1].get('median_goalward_depth_m', -math.inf))
            if gap <= ordering_margin_m:
                ambiguous_groups += 1
                continue
        selected.add(_row_key(candidates[0]))

    tp = sum(row['gt_role'] == 'goalkeeper' and _row_key(row) in selected for row in rows)
    fp = sum(row['gt_role'] != 'goalkeeper' and _row_key(row) in selected for row in rows)
    fn = sum(row['gt_role'] == 'goalkeeper' and _row_key(row) not in selected for row in rows)
    outfield = [row for row in rows if row['gt_role'] == 'player']
    player_fp = sum(_row_key(row) in selected for row in outfield)
    metrics = _prf(tp, fp, fn)
    metrics.update({
        'selected_tracks': len(selected),
        'ambiguous_groups': ambiguous_groups,
        'player_to_goalkeeper': player_fp,
        'player_to_goalkeeper_rate': player_fp / len(outfield) if outfield else None,
    })
    return metrics, sorted(selected)


def search_goalkeeper_role_policy(rows, *, min_observations, ordering_margin_m,
                                  min_precision=0.95, min_f1=0.90,
                                  max_player_goalkeeper_rate=0.005):
    candidates = []
    for distance, top2 in itertools.product(
            GOALKEEPER_GOAL_DISTANCES, GOALKEEPER_TOP2_RATES):
        metrics, selected = evaluate_goalkeeper_role_policy(
            rows, goal_distance_m=distance, min_top2_rate=top2,
            min_observations=min_observations, ordering_margin_m=ordering_margin_m,
        )
        feasible = (
            metrics['precision'] is not None and metrics['precision'] >= min_precision
            and (metrics['f1'] or 0) >= min_f1
            and metrics['player_to_goalkeeper_rate'] <= max_player_goalkeeper_rate
        )
        candidates.append({
            'parameters': {'goal_distance_m': distance, 'min_top2_rate': top2},
            'metrics': metrics,
            'selected_track_keys': selected,
            'feasible': feasible,
        })
    candidates.sort(key=lambda c: (
        c['feasible'], c['metrics']['f1'] or 0, c['metrics']['recall'] or 0,
        c['metrics']['precision'] or 0, -c['metrics']['player_to_goalkeeper_rate'],
    ), reverse=True)
    objective_best = candidates[0]
    # Threshold grids often contain a flat optimum: several adjacent settings
    # yield the exact same selected track set and therefore identical metrics.
    # Choosing the first tuple in iteration order can spuriously land on a grid
    # boundary and demand another expansion.  Within an equivalent best plateau,
    # prefer the point with the largest discrete distance from every grid edge.
    plateau = [candidate for candidate in candidates
               if candidate['feasible'] == objective_best['feasible']
               and candidate['selected_track_keys'] == objective_best['selected_track_keys']]

    def interior_score(candidate):
        distance_index = GOALKEEPER_GOAL_DISTANCES.index(
            candidate['parameters']['goal_distance_m'])
        top2_index = GOALKEEPER_TOP2_RATES.index(
            candidate['parameters']['min_top2_rate'])
        distance_margin = min(
            distance_index, len(GOALKEEPER_GOAL_DISTANCES) - 1 - distance_index)
        top2_margin = min(
            top2_index, len(GOALKEEPER_TOP2_RATES) - 1 - top2_index)
        return (
            min(distance_margin, top2_margin),
            distance_margin + top2_margin,
            -abs(distance_index - (len(GOALKEEPER_GOAL_DISTANCES) - 1) / 2),
            -abs(top2_index - (len(GOALKEEPER_TOP2_RATES) - 1) / 2),
        )

    selected = max(plateau, key=interior_score)
    selected_distance = selected['parameters']['goal_distance_m']
    selected_top2 = selected['parameters']['min_top2_rate']
    boundary = {
        'goal_distance_m': selected_distance in {
            min(GOALKEEPER_GOAL_DISTANCES), max(GOALKEEPER_GOAL_DISTANCES)},
        'min_top2_rate': selected_top2 in {
            min(GOALKEEPER_TOP2_RATES), max(GOALKEEPER_TOP2_RATES)},
    }
    return {
        'selected': selected,
        'feasible_count': sum(c['feasible'] for c in candidates),
        'evaluated_count': len(candidates),
        'grid': {
            'goal_distance_m': list(GOALKEEPER_GOAL_DISTANCES),
            'min_top2_rate': list(GOALKEEPER_TOP2_RATES),
        },
        'selected_at_grid_boundary': boundary,
        'requires_grid_expansion': any(boundary.values()),
        'selection_policy': 'BEST_OBJECTIVE_THEN_EQUIVALENT_PLATEAU_INTERIOR',
        'equivalent_best_plateau_size': len(plateau),
        'equivalent_best_plateau_parameters': [
            candidate['parameters'] for candidate in plateau],
        'top_candidates': candidates[:20],
    }


def apply_goalkeeper_role_selection(rows, selected_track_keys):
    """Build calibration rows without inheriting the source GK thresholds."""
    selected = set(selected_track_keys)
    updated = []
    for source in rows:
        row = dict(source)
        if str(row.get('appearance_group') or '').startswith('TEAM_'):
            row['base_role'] = 'player'
            row['base_role_status'] = 'VALID'
        elif _row_key(row) in selected:
            row['base_role'] = 'goalkeeper'
            row['base_role_status'] = 'VALID'
            row['base_team'] = None
            row['base_team_status'] = 'UNKNOWN'
        else:
            row['base_role'] = 'unknown_residual'
            row['base_role_status'] = 'UNKNOWN'
            row['base_team'] = None
            row['base_team_status'] = 'UNKNOWN'
        updated.append(row)
    return updated


def evaluate_appearance_policy(rows, *, player_margin, player_max_distance,
                               referee_margin, referee_min_distance,
                               referee_min_goal_distance_m=None,
                               referee_max_top2_rate=None):
    predicted = []
    for row in rows:
        role = row['base_role'] if row['base_role_status'] == 'VALID' else 'unknown'
        team = row['base_team'] if row['base_team_status'] == 'VALID' else None
        if role == 'unknown' and row['appearance_group'] == 'RESIDUAL':
            if row['margin'] >= player_margin and row['nearest_distance'] <= player_max_distance:
                role, team = 'player', row['nearest_team']
            elif (row['margin'] <= referee_margin
                  and row['nearest_distance'] >= referee_min_distance
                  and (referee_min_goal_distance_m is None
                       or (row.get('median_goal_distance_m') is not None
                           and row['median_goal_distance_m'] >= referee_min_goal_distance_m))
                  and (referee_max_top2_rate is None
                       or (row.get('goal_top2_rate') is not None
                           and row['goal_top2_rate'] <= referee_max_top2_rate))):
                role, team = 'referee', None
        predicted.append((row, role, team))

    roles = {}
    for target in ('player', 'goalkeeper', 'referee'):
        tp = sum(r['gt_role'] == target and p == target for r, p, _ in predicted)
        fp = sum(r['gt_role'] != target and p == target for r, p, _ in predicted)
        fn = sum(r['gt_role'] == target and p != target for r, p, _ in predicted)
        roles[target] = _prf(tp, fp, fn)

    outfield = [x for x in predicted if x[0]['gt_role'] == 'player']
    correct = 0; assigned = 0
    for row, role, team in outfield:
        if role == 'player' and team in (0, 1) and row['mapping_available']:
            assigned += 1
            correct += int(row['mapping'][str(team)] == row['gt_team'])
    player_to_ref = sum(r['gt_role'] == 'player' and p == 'referee' for r, p, _ in predicted)
    referee_to_player = sum(r['gt_role'] == 'referee' and p == 'player' for r, p, _ in predicted)
    return {
        'outfield_tracks': len(outfield),
        'outfield_assigned': assigned,
        'outfield_correct': correct,
        'outfield_overall_accuracy': correct / len(outfield) if outfield else None,
        'outfield_coverage': assigned / len(outfield) if outfield else None,
        'roles': roles,
        'player_to_referee': player_to_ref,
        'player_to_referee_rate': player_to_ref / len(outfield) if outfield else None,
        'referee_to_player': referee_to_player,
    }


def evaluate_player_recovery_policy(
        rows, *, player_margin, player_max_distance,
        player_referee_veto_enabled=False,
        player_veto_min_goal_distance_m=22.0,
        player_veto_min_top2_rate=0.40):
    predicted = []
    vetoed_tracks = 0
    missing_veto_context = 0
    for row in rows:
        role = row['base_role'] if row['base_role_status'] == 'VALID' else 'unknown'
        team = row['base_team'] if row['base_team_status'] == 'VALID' else None
        goal_distance = row.get('median_goal_distance_m')
        top2_rate = row.get('goal_top2_rate')
        veto_context_valid = (
            isinstance(goal_distance, (int, float)) and math.isfinite(goal_distance)
            and isinstance(top2_rate, (int, float)) and math.isfinite(top2_rate)
        )
        eligible_residual = (
            role == 'unknown' and row['appearance_group'] == 'RESIDUAL')
        referee_veto = player_referee_veto_enabled and eligible_residual and (
            not veto_context_valid
            or (goal_distance >= player_veto_min_goal_distance_m
                and top2_rate >= player_veto_min_top2_rate)
        )
        if (player_referee_veto_enabled and eligible_residual
                and not veto_context_valid):
            missing_veto_context += 1
        if referee_veto:
            vetoed_tracks += 1
        if (role == 'unknown' and row['appearance_group'] == 'RESIDUAL'
                and row['margin'] >= player_margin
                and row['nearest_distance'] <= player_max_distance
                and not referee_veto):
            role, team = 'player', row['nearest_team']
        predicted.append((row, role, team))

    roles = {}
    for target in ('player', 'goalkeeper', 'referee'):
        tp = sum(r['gt_role'] == target and p == target for r, p, _ in predicted)
        fp = sum(r['gt_role'] != target and p == target for r, p, _ in predicted)
        fn = sum(r['gt_role'] == target and p != target for r, p, _ in predicted)
        roles[target] = _prf(tp, fp, fn)
    outfield = [x for x in predicted if x[0]['gt_role'] == 'player']
    assigned = 0; correct = 0
    for row, role, team in outfield:
        if role == 'player' and team in (0, 1) and row['mapping_available']:
            assigned += 1
            correct += int(row['mapping'][str(team)] == row['gt_team'])
    referees = [x for x in predicted if x[0]['gt_role'] == 'referee']
    referee_to_player = sum(role == 'player' for _, role, _ in referees)
    return {
        'outfield_tracks': len(outfield),
        'outfield_assigned': assigned,
        'outfield_correct': correct,
        'outfield_overall_accuracy': correct / len(outfield) if outfield else None,
        'outfield_coverage': assigned / len(outfield) if outfield else None,
        'roles': roles,
        'referee_tracks': len(referees),
        'referee_to_player': referee_to_player,
        'referee_to_player_rate': (
            referee_to_player / len(referees) if referees else 0.0),
        'player_referee_veto_enabled': bool(player_referee_veto_enabled),
        'vetoed_tracks': vetoed_tracks,
        'missing_veto_context': missing_veto_context,
    }


def search_player_recovery_policy(rows, *, baseline_outfield_accuracy,
                                  min_player_precision=0.97,
                                  min_goalkeeper_f1=0.90,
                                  max_referee_to_player_rate=0.05,
    max_outfield_drop=0.005):
    candidates = []
    for margin, distance in itertools.product(PLAYER_MARGINS, PLAYER_MAX_DISTANCES):
        metrics = evaluate_player_recovery_policy(
            rows, player_margin=margin, player_max_distance=distance)
        player = metrics['roles']['player']; goalkeeper = metrics['roles']['goalkeeper']
        feasible = (
            player['precision'] is not None and player['precision'] >= min_player_precision
            and (goalkeeper['f1'] or 0) >= min_goalkeeper_f1
            and metrics['referee_to_player_rate'] <= max_referee_to_player_rate
            and metrics['outfield_overall_accuracy'] >= baseline_outfield_accuracy - max_outfield_drop
        )
        candidates.append({
            'parameters': {
                'residual_player_min_margin': margin,
                'residual_player_max_distance': distance,
            },
            'metrics': metrics,
            'feasible': feasible,
        })
    candidates.sort(key=lambda c: (
        c['feasible'], c['metrics']['outfield_overall_accuracy'],
        c['metrics']['roles']['player']['f1'] or 0,
        c['metrics']['roles']['player']['precision'] or 0,
        -c['metrics']['referee_to_player_rate'],
    ), reverse=True)
    return {
        'selected': candidates[0],
        'feasible_count': sum(c['feasible'] for c in candidates),
        'evaluated_count': len(candidates),
        'top_candidates': candidates[:20],
    }


def search_player_recovery_policy_grouped(
        rows, *, baseline_outfield_accuracy, n_folds=5,
        min_player_precision=0.97, min_goalkeeper_f1=0.90,
        max_referee_to_player_rate=0.0, max_outfield_drop=0.005):
    """Select player recovery only when it is referee-safe in every TRAIN fold.

    Folds are grouped by sequence, so tracks from one clip never appear in two
    folds.  No VALID labels or metrics enter this selection.  The zero-referee
    constraint is deliberately stricter than the final held-out 5% gate.
    """
    sequence_ids = sorted({str(row['sequence_id']) for row in rows})
    if not sequence_ids:
        raise ValueError('Grouped player calibration requires TRAIN rows')
    grouped_evidence_sufficient = len(sequence_ids) >= 2
    fold_count = min(int(n_folds), len(sequence_ids))
    folds = [set(sequence_ids[index::fold_count]) for index in range(fold_count)]
    candidates = []
    veto_policies = [(False, None, None)] + [
        (True, goal_distance, top2_rate)
        for goal_distance, top2_rate in itertools.product(
            PLAYER_VETO_GOAL_DISTANCES, PLAYER_VETO_TOP2_RATES)
    ]
    for margin, distance, veto_policy in itertools.product(
            PLAYER_MARGINS, PLAYER_MAX_DISTANCES, veto_policies):
        veto_enabled, veto_goal_distance, veto_top2_rate = veto_policy
        evaluation_kwargs = {
            'player_margin': margin,
            'player_max_distance': distance,
            'player_referee_veto_enabled': veto_enabled,
            'player_veto_min_goal_distance_m': (
                veto_goal_distance if veto_enabled else 22.0),
            'player_veto_min_top2_rate': (
                veto_top2_rate if veto_enabled else 0.40),
        }
        metrics = evaluate_player_recovery_policy(
            rows, **evaluation_kwargs)
        fold_metrics = []
        for index, sequence_fold in enumerate(folds):
            fold_rows = [row for row in rows if str(row['sequence_id']) in sequence_fold]
            current = evaluate_player_recovery_policy(
                fold_rows, **evaluation_kwargs)
            fold_metrics.append({
                'fold': index,
                'sequence_ids': sorted(sequence_fold),
                'metrics': current,
            })
        player = metrics['roles']['player']
        goalkeeper = metrics['roles']['goalkeeper']
        fold_referee_safe = all(
            fold['metrics']['referee_to_player_rate'] <= max_referee_to_player_rate
            for fold in fold_metrics)
        feasible = (
            player['precision'] is not None and player['precision'] >= min_player_precision
            and (goalkeeper['f1'] or 0) >= min_goalkeeper_f1
            and metrics['referee_to_player_rate'] <= max_referee_to_player_rate
            and metrics['outfield_overall_accuracy'] >= baseline_outfield_accuracy - max_outfield_drop
            and fold_referee_safe
            and grouped_evidence_sufficient
        )
        candidates.append({
            'parameters': {
                'residual_player_min_margin': margin,
                'residual_player_max_distance': distance,
                'residual_player_referee_veto_enabled': veto_enabled,
                **({
                    'residual_player_veto_min_goal_distance_m': veto_goal_distance,
                    'residual_player_veto_min_top2_rate': veto_top2_rate,
                } if veto_enabled else {}),
            },
            'metrics': metrics,
            'grouped_train_folds': fold_metrics,
            'worst_fold_referee_to_player_rate': max(
                fold['metrics']['referee_to_player_rate'] for fold in fold_metrics),
            'feasible': feasible,
        })
    candidates.sort(key=lambda candidate: (
        candidate['feasible'],
        candidate['metrics']['outfield_overall_accuracy'],
        candidate['metrics']['roles']['player']['f1'] or 0,
        candidate['metrics']['roles']['player']['precision'] or 0,
        -candidate['worst_fold_referee_to_player_rate'],
    ), reverse=True)
    return {
        'selection_protocol': 'TRAIN_SEQUENCE_GROUPED_REFEREE_VETO_V029',
        'fold_count': fold_count,
        'sequence_count': len(sequence_ids),
        'grouped_evidence_sufficient': grouped_evidence_sufficient,
        'constraints': {
            'min_player_precision': min_player_precision,
            'min_goalkeeper_f1': min_goalkeeper_f1,
            'max_referee_to_player_rate_global': max_referee_to_player_rate,
            'max_referee_to_player_rate_each_fold': max_referee_to_player_rate,
            'max_outfield_drop': max_outfield_drop,
        },
        'selected': candidates[0],
        'feasible_count': sum(candidate['feasible'] for candidate in candidates),
        'evaluated_count': len(candidates),
        'top_candidates': candidates[:20],
    }


def search_appearance_policy(rows, *, baseline_outfield_accuracy,
                             min_referee_precision=0.95, min_referee_f1=0.80,
                             min_goalkeeper_f1=0.90, max_player_referee_rate=0.005,
                             max_outfield_drop=0.005,
                             min_referee_goal_distance_exclusive=None):
    candidates = []
    has_geometry = any(
        'median_goal_distance_m' in row or 'goal_top2_rate' in row
        for row in rows
    )
    geometry_grid = (
        itertools.product(REFEREE_GOAL_DISTANCES, REFEREE_TOP2_RATES)
        if has_geometry else [(None, None)]
    )
    appearance_grid = list(itertools.product(
        PLAYER_MARGINS, PLAYER_MAX_DISTANCES, REFEREE_MARGINS,
        REFEREE_MIN_DISTANCES))
    for (pm, pd, rm, rd), (rg, rt) in itertools.product(
            appearance_grid, geometry_grid):
        if rm >= pm:
            continue
        # ResidualConfig requires the referee's minimum goal distance to be
        # strictly outside the goalkeeper candidate zone.  Enforce that joint
        # invariant during the search instead of selecting two independently
        # optimal but mutually invalid thresholds.
        if (rg is not None and min_referee_goal_distance_exclusive is not None
                and rg <= min_referee_goal_distance_exclusive):
            continue
        metrics = evaluate_appearance_policy(
            rows, player_margin=pm, player_max_distance=pd,
            referee_margin=rm, referee_min_distance=rd,
            referee_min_goal_distance_m=rg, referee_max_top2_rate=rt,
        )
        ref = metrics['roles']['referee']; gk = metrics['roles']['goalkeeper']
        feasible = (
            ref['precision'] is not None and ref['precision'] >= min_referee_precision
            and (ref['f1'] or 0) >= min_referee_f1
            and (gk['f1'] or 0) >= min_goalkeeper_f1
            and metrics['player_to_referee_rate'] <= max_player_referee_rate
            and metrics['outfield_overall_accuracy'] >= baseline_outfield_accuracy - max_outfield_drop
        )
        candidates.append({
            'parameters': {
                'residual_player_min_margin': pm,
                'residual_player_max_distance': pd,
                'residual_referee_max_margin': rm,
                'residual_referee_min_distance': rd,
                **({'referee_min_goal_distance_m': rg,
                    'referee_max_top2_rate': rt} if has_geometry else {}),
            },
            'metrics': metrics,
            'feasible': feasible,
        })
    candidates.sort(key=lambda c: (
        c['feasible'], c['metrics']['outfield_overall_accuracy'],
        c['metrics']['roles']['referee']['f1'] or 0,
        c['metrics']['roles']['referee']['precision'] or 0,
        -c['metrics']['player_to_referee_rate'],
    ), reverse=True)
    return {'selected': candidates[0], 'feasible_count': sum(c['feasible'] for c in candidates),
            'evaluated_count': len(candidates), 'top_candidates': candidates[:20]}


def evaluate_tail_policy(rows, *, separation, vote_rate, min_observations):
    goalkeepers = [r for r in rows if r['gt_role'] == 'goalkeeper']
    correct = 0; assigned = 0
    for row in goalkeepers:
        deltas = np.asarray(row.get('tail_deltas_m') or [], dtype=float)
        if (row['base_role'] != 'goalkeeper' or row['base_role_status'] != 'VALID'
                or len(deltas) < min_observations or not row['mapping_available']):
            continue
        median = float(np.median(deltas))
        winner = 0 if median > 0 else 1
        sign = 1 if winner == 0 else -1
        agreement = float(np.mean(deltas * sign >= separation))
        if abs(median) >= separation and agreement >= vote_rate:
            assigned += 1
            correct += int(row['mapping'][str(winner)] == row['gt_team'])
    return {
        'tracks': len(goalkeepers), 'assigned': assigned, 'correct': correct,
        'coverage': assigned / len(goalkeepers) if goalkeepers else None,
        'selective_accuracy': correct / assigned if assigned else None,
        'overall_accuracy': correct / len(goalkeepers) if goalkeepers else None,
    }


def search_tail_policy(rows, *, min_observations, minimum_overall_accuracy=0.80):
    candidates = []
    for separation, vote_rate in itertools.product(TAIL_SEPARATIONS, TAIL_VOTE_RATES):
        metrics = evaluate_tail_policy(rows, separation=separation, vote_rate=vote_rate,
                                       min_observations=min_observations)
        candidates.append({
            'parameters': {'min_tail_separation_m': separation, 'min_tail_vote_rate': vote_rate},
            'metrics': metrics,
            'feasible': metrics['overall_accuracy'] is not None
                        and metrics['overall_accuracy'] >= minimum_overall_accuracy,
        })
    candidates.sort(key=lambda c: (
        c['feasible'], c['metrics']['overall_accuracy'] or 0,
        c['metrics']['selective_accuracy'] or 0, c['metrics']['coverage'] or 0,
    ), reverse=True)
    return {'selected': candidates[0], 'feasible_count': sum(c['feasible'] for c in candidates),
            'evaluated_count': len(candidates), 'top_candidates': candidates[:20]}


def evaluate_defended_half_policy(rows, *, separation, vote_rate, min_observations):
    goalkeepers = [r for r in rows if r['gt_role'] == 'goalkeeper']
    correct = 0; assigned = 0
    for row in goalkeepers:
        deltas = np.asarray(row.get('half_deltas_m') or [], dtype=float)
        if (row['base_role'] != 'goalkeeper' or row['base_role_status'] != 'VALID'
                or len(deltas) < min_observations or not row['mapping_available']):
            continue
        median = float(np.median(deltas))
        winner = 0 if median > 0 else 1
        sign = 1 if winner == 0 else -1
        agreement = float(np.mean(deltas * sign >= separation))
        if abs(median) >= separation and agreement >= vote_rate:
            assigned += 1
            correct += int(row['mapping'][str(winner)] == row['gt_team'])
    return {
        'tracks': len(goalkeepers), 'assigned': assigned, 'correct': correct,
        'coverage': assigned / len(goalkeepers) if goalkeepers else None,
        'selective_accuracy': correct / assigned if assigned else None,
        'overall_accuracy': correct / len(goalkeepers) if goalkeepers else None,
    }


def search_defended_half_policy(rows, *, min_observations,
                                minimum_overall_accuracy=0.80):
    candidates = []
    for separation, vote_rate in itertools.product(HALF_SEPARATIONS, HALF_VOTE_RATES):
        metrics = evaluate_defended_half_policy(
            rows, separation=separation, vote_rate=vote_rate,
            min_observations=min_observations)
        candidates.append({
            'parameters': {
                'min_defended_half_separation_m': separation,
                'min_defended_half_vote_rate': vote_rate,
            },
            'metrics': metrics,
            'feasible': metrics['overall_accuracy'] is not None
                        and metrics['overall_accuracy'] >= minimum_overall_accuracy,
        })
    candidates.sort(key=lambda c: (
        c['feasible'], c['metrics']['overall_accuracy'] or 0,
        c['metrics']['selective_accuracy'] or 0, c['metrics']['coverage'] or 0,
    ), reverse=True)
    return {
        'selected': candidates[0],
        'feasible_count': sum(c['feasible'] for c in candidates),
        'evaluated_count': len(candidates),
        'top_candidates': candidates[:20],
    }


def _load_rows(dataset_root, split, prediction_dir):
    root = Path(prediction_dir).resolve()
    manifest = json.loads((root / 'run_manifest.json').read_text(encoding='utf-8'))
    summary = json.loads((root / 'benchmark_summary.json').read_text(encoding='utf-8'))
    if manifest.get('status') != 'COMPLETE' or manifest.get('variant') != 'V3':
        raise ValueError('Calibration requires a COMPLETE residual-v3 source run')
    if summary.get('split') != split or summary.get('method') != 'residual-v3':
        raise ValueError('Source summary split/method mismatch')
    if Path(summary.get('dataset_root', '')).resolve() != Path(dataset_root).resolve():
        raise ValueError('Source prediction dataset root mismatch')
    source_cfg = ResidualConfig(**summary['configuration']); source_cfg.validate()
    if (source_cfg.residual_appearance_recovery_enabled
            or source_cfg.residual_player_recovery_enabled
            or source_cfg.residual_player_referee_veto_enabled
            or source_cfg.residual_referee_appearance_recovery_enabled
            or source_cfg.defensive_tail_assignment_enabled
            or source_cfg.defended_half_assignment_enabled
            or source_cfg.referee_recovery_enabled):
        raise ValueError('Calibration source must use abstaining/default recovery switches')

    ds = SoccerNetGSRDataset(dataset_root, split)
    rows = []
    for sid in manifest['sequence_ids']:
        seq = ds.load(sid)
        prediction = json.loads((root / 'sequences' / sid / 'prediction.json').read_text(encoding='utf-8'))
        metrics = json.loads((root / 'sequences' / sid / 'metrics.json').read_text(encoding='utf-8'))
        mapping = {str(k): int(v) for k, v in (metrics.get('mapping_pred_to_gt') or {}).items()}
        gt_roles = seq.role_by_track(); gt_teams = seq.team_gt_by_track()
        records = {str(r['track_id']): r for r in prediction['tracks']}
        if set(gt_roles) - set(records):
            raise ValueError(f'{sid}: prediction missing GT human tracks')
        for tid, gt_role in gt_roles.items():
            record = records[tid]
            goal = record.get('goal_context') or {}
            distances = np.asarray(record.get('distances', []), dtype=float)
            if distances.shape == (2,) and np.isfinite(distances).all():
                nearest_team = int(np.argmin(distances)); nearest_distance = float(distances[nearest_team])
                margin = float(record.get('margin', 0.0))
            else:
                nearest_team = None; nearest_distance = math.inf; margin = 0.0
            rows.append({
                'sequence_id': sid, 'track_id': tid, 'gt_role': gt_role,
                'gt_team': gt_teams.get(tid), 'mapping': mapping,
                'mapping_available': set(mapping) == {'0', '1'},
                'appearance_group': record.get('appearance_group'),
                'margin': margin, 'nearest_team': nearest_team,
                'nearest_distance': nearest_distance,
                'base_role': record.get('stage5_role'),
                'base_role_status': record.get('stage5_role_status'),
                'base_team': record.get('team_id'),
                'base_team_status': record.get('team_status'),
                'tail_deltas_m': (record.get('goalkeeper_assignment') or {}).get('tail_deltas_m', []),
                'half_deltas_m': (record.get('goalkeeper_assignment') or {}).get('half_deltas_m', []),
                'goal_sign': goal.get('sign'),
                'goal_side': goal.get('side'),
                'goal_observations': int(goal.get('observations') or 0),
                'goal_rank_frames': int(goal.get('rank_frames') or 0),
                'goal_top2_rate': goal.get('top2_rate'),
                'median_goal_distance_m': goal.get('median_goal_distance_m'),
                'median_goalward_depth_m': goal.get('median_goalward_depth_m'),
            })
    discovered = ds.discover()
    complete_train = set(manifest['sequence_ids']) == set(discovered) and len(manifest['sequence_ids']) == len(discovered)
    return rows, source_cfg, summary, manifest, complete_train


def calibrate_residual_train(*, dataset_root, split, prediction_dir, output_dir):
    if str(split).lower() != 'train':
        raise ValueError('Calibration is TRAIN-only; VALID/TEST are evaluation splits')
    out = Path(output_dir).resolve()
    if out.exists():
        raise FileExistsError(f'Use a NEW calibration output directory: {out}')
    rows, base_cfg, source_summary, source_manifest, complete_train = _load_rows(
        dataset_root, split, prediction_dir)
    out.mkdir(parents=True)
    baseline_outfield = source_summary['groups']['outfield']['micro_overall_accuracy']
    goalkeeper_role = search_goalkeeper_role_policy(
        rows, min_observations=base_cfg.min_observations,
        ordering_margin_m=base_cfg.ordering_margin_m,
    )
    calibrated_rows = apply_goalkeeper_role_selection(
        rows, goalkeeper_role['selected']['selected_track_keys'])
    calibrated_goal_distance = (
        goalkeeper_role['selected']['parameters']['goal_distance_m']
        if goalkeeper_role['selected']['feasible'] else base_cfg.goal_distance_m
    )
    player = search_player_recovery_policy_grouped(
        calibrated_rows, baseline_outfield_accuracy=baseline_outfield)
    appearance = search_appearance_policy(
        calibrated_rows, baseline_outfield_accuracy=baseline_outfield,
        min_referee_goal_distance_exclusive=calibrated_goal_distance)
    tail = search_tail_policy(
        calibrated_rows, min_observations=base_cfg.min_observations)
    half = search_defended_half_policy(
        calibrated_rows, min_observations=base_cfg.min_observations)

    selected = base_cfg
    if goalkeeper_role['selected']['feasible']:
        goalkeeper_parameters = goalkeeper_role['selected']['parameters']
        # Even when residual/referee recovery is infeasible and remains
        # disabled, ResidualConfig still requires the stored referee boundary
        # to sit strictly outside the calibrated goalkeeper candidate zone.
        # Keep a conservative valid boundary so a successful GK calibration
        # can always be emitted as a provisional configuration.
        safe_referee_distance = safe_referee_goal_distance(
            selected.referee_min_goal_distance_m,
            goalkeeper_parameters['goal_distance_m'],
            selected.ordering_margin_m,
        )
        selected = replace(
            selected,
            **goalkeeper_parameters,
            referee_min_goal_distance_m=safe_referee_distance,
        )
    if player['selected']['feasible']:
        selected = replace(
            selected,
            residual_player_recovery_enabled=True,
            **player['selected']['parameters'],
        )
    if appearance['selected']['feasible']:
        selected = replace(
            selected,
            residual_player_recovery_enabled=True,
            residual_referee_appearance_recovery_enabled=True,
            **appearance['selected']['parameters'],
        )
    if half['selected']['feasible']:
        selected = replace(
            selected,
            defended_half_assignment_enabled=True,
            defensive_tail_assignment_enabled=False,
            **half['selected']['parameters'],
        )
    elif tail['selected']['feasible']:
        selected = replace(selected, defensive_tail_assignment_enabled=True,
                           **tail['selected']['parameters'])
    selected.validate()
    goalkeeper_role_grid_stable = bool(
        goalkeeper_role['selected']['feasible']
        and not goalkeeper_role['requires_grid_expansion']
    )
    freeze = research_freeze_gate(
        complete_train_split=complete_train,
        goalkeeper_role_feasible=goalkeeper_role_grid_stable,
        residual_role_feasible=(player['selected']['feasible']
                                and appearance['selected']['feasible']),
        goalkeeper_team_feasible=(half['selected']['feasible']
                                  or tail['selected']['feasible']),
    )
    eligible = freeze['eligible']
    component_valid_eligible = bool(
        complete_train and goalkeeper_role_grid_stable
        and player['selected']['feasible'])
    config_name = 'frozen_residual_config.json' if eligible else 'provisional_residual_config.json'
    config_path = out / config_name
    config_path.write_text(json.dumps(selected.to_dict(), indent=2), encoding='utf-8')
    config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    report = {
        'schema_version': 'stage5-residual-train-calibration-1.8',
        'package_version': __version__, 'status': 'COMPLETE', 'split': split,
        'dataset_root': str(Path(dataset_root).resolve()),
        'source_prediction_dir': str(Path(prediction_dir).resolve()),
        'source_sequence_count': len(source_manifest['sequence_ids']),
        'complete_train_split': complete_train,
        'eligible_for_valid_evaluation': component_valid_eligible,
        'eligible_for_component_valid_evaluation': component_valid_eligible,
        'eligible_for_full_research_freeze': eligible,
        'baseline_outfield_accuracy': baseline_outfield,
        'goalkeeper_role_recovery': goalkeeper_role,
        'goalkeeper_role_grid_stable': goalkeeper_role_grid_stable,
        'player_recovery': player,
        'appearance_recovery': appearance,
        'goalkeeper_team_recovery': tail,
        'goalkeeper_defended_half_recovery': half,
        'component_freezes': {
            'goalkeeper_role': goalkeeper_role_grid_stable,
            'player_recovery': player['selected']['feasible'],
            'referee_recovery': appearance['selected']['feasible'],
            'goalkeeper_team': (half['selected']['feasible']
                                or tail['selected']['feasible']),
        },
        'freeze_gate': freeze,
        'selected_config_file': config_name,
        'heldout_evaluation_contract': {
            'schema_version': 'stage5-heldout-contract-0.3.0',
            'allowed_split': 'valid',
            'config_sha256': config_sha256,
            'requires_complete_valid_split_for_gate': True,
            'requires_paired_bbox_color_baseline': True,
            'tuning_on_valid_forbidden': True,
        },
        'leakage_guard': 'GT labels used only inside TRAIN calibration; command rejects VALID/TEST',
        'research_accuracy_frozen': False,
    }
    (out / 'calibration_report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    return report
