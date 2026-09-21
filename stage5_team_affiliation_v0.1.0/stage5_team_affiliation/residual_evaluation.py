"""Evaluation-only code: GT semantics must not be passed into inference."""
from .gsr_benchmark import evaluate_sequence, aggregate_benchmark, _bootstrap_ci


def _prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return {'tp': tp, 'fp': fp, 'fn': fn, 'precision': precision, 'recall': recall,
            'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None}


def evaluate_roles(gt_roles, records):
    by_id = {r['track_id']: r for r in records}
    predicted = {t: r['stage5_role'] if r['stage5_role_status'] == 'VALID' else 'unknown'
                 for t, r in by_id.items()}
    metrics = {}
    for role in ('player', 'goalkeeper', 'referee'):
        tp = sum(g == role and predicted.get(t) == role for t, g in gt_roles.items())
        fp = sum(g != role and predicted.get(t) == role for t, g in gt_roles.items())
        fn = sum(g == role and predicted.get(t) != role for t, g in gt_roles.items())
        metrics[role] = _prf(tp, fp, fn)
    players = [t for t, g in gt_roles.items() if g == 'player']
    metrics['player_contamination'] = {'tracks': len(players)}
    for role in ('goalkeeper', 'referee'):
        count = sum(predicted.get(t) == role for t in players)
        metrics['player_contamination'][role] = count
        metrics['player_contamination'][role + '_rate'] = count / len(players) if players else None
    return metrics


def evaluate_residual_sequence(seq, prediction):
    pred = {r['track_id']: r['team_id'] if r['team_status'] == 'VALID' else None for r in prediction['tracks']}
    mapping_available = any(pred.get(t) in (0, 1) for t in seq.team_gt_by_track(role='player'))
    if not mapping_available:
        # Without an outfield correspondence, a default identity map would grant
        # arbitrary goalkeeper credit. Treat unalignable affiliations as UNKNOWN.
        pred = {t: None for t in pred}
    metrics = evaluate_sequence(seq, pred)
    metrics['outfield_mapping_available'] = mapping_available
    metrics['roles'] = evaluate_roles(seq.role_by_track(), prediction['tracks'])
    # ARI/NMI on tiny goalkeeper-only groups are permutation-invariant and misleading.
    metrics['goalkeeper']['ARI'] = metrics['goalkeeper']['NMI'] = None
    metrics['metric_scope'] = {'team_f1_ari_nmi': 'assigned-only', 'role_prf': 'all GT humans; UNKNOWN counts as FN'}
    return metrics


def aggregate_residual(rows, method, split):
    summary = aggregate_benchmark(rows, method, split)
    summary['roles'] = {}
    for role in ('player', 'goalkeeper', 'referee'):
        counts = [sum(r['roles'][role][key] for r in rows) for key in ('tp', 'fp', 'fn')]
        summary['roles'][role] = _prf(*counts)
        summary['roles'][role]['sequence_f1_95ci'] = _bootstrap_ci([r['roles'][role]['f1'] for r in rows if r['roles'][role]['f1'] is not None])
    total = sum(r['roles']['player_contamination']['tracks'] for r in rows)
    summary['player_contamination'] = {'tracks': total}
    for role in ('goalkeeper', 'referee'):
        count = sum(r['roles']['player_contamination'][role] for r in rows)
        summary['player_contamination'][role] = count
        summary['player_contamination'][role + '_rate'] = count / total if total else None
    return summary
