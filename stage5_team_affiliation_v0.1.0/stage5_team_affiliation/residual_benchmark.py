"""Role-blind GSR oracle-geometry experiment; distinct from v0.1 GT-role B0."""
from pathlib import Path
import hashlib
import json
import numpy as np
from .config import Stage5Config
from .residual_config import ResidualConfig
from .gsr_benchmark import SoccerNetGSRDataset, _bbox_xyxy, _write_json
from .residual_pipeline import (extract_human_features,
                                extract_human_multiregion_features,
                                infer_residual)
from .residual_evaluation import evaluate_residual_sequence, aggregate_residual
from .heldout_v027 import heldout_component_gate
from . import __version__


def build_blind_inputs(raw, ordered_images):
    """Collapse category IDs to HUMAN population, then discard all role/team fields.

    Human detection/identity is oracle; role within HUMAN is not. Ball/pitch/camera
    and category=other are excluded. This is not an end-to-end detector evaluation.
    """
    human_categories = {str(c['id']) for c in raw.get('categories', [])
                        if c.get('name') in {'player', 'goalkeeper', 'referee'} and c.get('supercategory') == 'object'}
    if not human_categories:
        raise ValueError('Explicit GSR human category metadata required; no fallback to GT role')
    frame_map = {str(im.get('image_id', im.get('id'))): i for i, im in enumerate(ordered_images)}
    if len(frame_map) != len(ordered_images):
        raise ValueError('Duplicate image IDs')
    observations = {}; pitch = {}; seen = set(); missing_pitch = 0
    for ann in raw.get('annotations', []):
        if str(ann.get('category_id')) not in human_categories:
            continue
        fi = frame_map.get(str(ann.get('image_id'))); tid = ann.get('track_id')
        box = _bbox_xyxy(ann)
        if fi is None or tid is None or box is None or not np.isfinite(box).all():
            raise ValueError('Malformed human annotation')
        tid = str(tid)
        if (tid, fi) in seen:
            raise ValueError(f'Duplicate annotation for track/frame: {tid}/{fi}')
        seen.add((tid, fi))
        observations.setdefault(tid, []).append({'frame_index': fi, 'source_bbox_xyxy': box, 'keypoints_133': []})
        b = ann.get('bbox_pitch') or {}
        xy = [b.get('x_bottom_middle'), b.get('y_bottom_middle')]
        if all(isinstance(v, (int, float)) for v in xy) and np.isfinite(xy).all():
            pitch.setdefault(tid, {})[fi] = xy
        else:
            missing_pitch += 1
    return observations, pitch, {'human_observations': len(seen), 'missing_pitch_observations': missing_pitch}


def _baseline_gate(baseline_path, rows, summary, dataset_root, split):
    gates = {'outfield_drop_less_than_0_5pp': 'NOT_EVALUATED',
             'goalkeeper_team_accuracy_ge_80pct': summary['groups']['goalkeeper']['micro_overall_accuracy'] >= .8,
             'goalkeeper_role_f1_ge_90pct': (summary['roles']['goalkeeper']['f1'] or 0) >= .9,
             'research_accuracy_frozen': False}
    if baseline_path:
        path = Path(baseline_path)
        baseline = json.loads(path.read_text(encoding='utf-8'))
        old_rows = [json.loads(line) for line in (path.parent / 'sequence_metrics.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        old = {r['sequence_id']: r for r in old_rows}
        if (baseline.get('method') != 'bbox-color' or baseline.get('split') != split
                or Path(baseline['dataset_root']).resolve() != Path(dataset_root).resolve()
                or set(old) != {r['sequence_id'] for r in rows} or len(old_rows) != len(rows)
                or any(old[r['sequence_id']]['outfield']['tracks'] != r['outfield']['tracks'] for r in rows)):
            raise ValueError('V0 baseline must cover the identical split/sequences/GT track population')
        old_total = sum(r['outfield']['tracks'] for r in old_rows)
        old_acc = sum(r['outfield']['correct_valid'] for r in old_rows) / max(1, old_total)
        delta = summary['groups']['outfield']['micro_overall_accuracy'] - old_acc
        gates.update(outfield_accuracy_delta=delta, outfield_drop_less_than_0_5pp=delta > -.005)
    gate_values = [gates[k] for k in ('outfield_drop_less_than_0_5pp', 'goalkeeper_team_accuracy_ge_80pct', 'goalkeeper_role_f1_ge_90pct')]
    gates['oracle_research_gate'] = 'PASS' if all(v is True for v in gate_values) else 'NOT_EVALUATED' if any(isinstance(v, str) for v in gate_values) else 'FAIL'
    return gates


def run_residual_benchmark(*, dataset_root, split, output_dir, variant='V3', limit=None,
                           progress_every=1, residual_config=None, baseline_summary=None,
                           heldout_evaluation=False, calibration_provenance=None):
    cfg = residual_config or ResidualConfig(); cfg.validate()
    if variant not in {'V1', 'V2', 'V3'} or (limit is not None and limit <= 0) or progress_every < 0:
        raise ValueError('Invalid variant/limit/progress interval')
    if heldout_evaluation:
        if str(split).lower() != 'valid':
            raise ValueError('Held-out component evaluation is VALID-only')
        if limit is not None:
            raise ValueError('Held-out component evaluation must cover the complete VALID split')
        if not baseline_summary:
            raise ValueError('Held-out component evaluation requires a paired bbox-color baseline')
        if not calibration_provenance:
            raise ValueError('Held-out component evaluation requires verified TRAIN provenance')
    ds = SoccerNetGSRDataset(dataset_root, split)
    ids = ds.discover()
    if limit is not None:
        ids = ids[:limit]
    if not ids:
        raise ValueError('No GSR sequences found; cannot score an empty benchmark')
    out = Path(output_dir).resolve()
    if out.exists():
        raise FileExistsError(f'Use a NEW output directory: {out}')
    out.mkdir(parents=True)
    manifest = {'status': 'RUNNING', 'variant': variant, 'split': split,
                'dataset_root': str(Path(dataset_root).resolve()),
                'sequence_ids': ids, 'completed_sequences': [],
                'package_version': __version__, 'residual_configuration': cfg.to_dict(),
                'evaluation_mode': 'HELDOUT_COMPONENT' if heldout_evaluation else 'EXPERIMENTAL',
                'calibration_provenance': calibration_provenance,
                'real_accuracy_claim': False}
    _write_json(out / 'run_manifest.json', manifest)
    rows = []
    try:
        for index, sid in enumerate(ids, 1):
            seq = ds.load(sid)
            raw = json.loads(seq.labels_path.read_text(encoding='utf-8'))
            observations, points, adapter_diag = build_blind_inputs(raw, seq.images)
            if not observations or (variant != 'V1' and not any(points.values())):
                raise ValueError(f'{sid}: missing humans or oracle geometry')
            if cfg.multiregion_appearance_enabled:
                features, appearance_diag, feature_cfg = extract_human_multiregion_features(
                    observations, seq.read_frame, Stage5Config(),
                    torso_weight=cfg.multiregion_torso_weight,
                    lower_weight=cfg.multiregion_lower_weight,
                    require_lower=cfg.multiregion_require_lower,
                )
            else:
                features, appearance_diag, feature_cfg = extract_human_features(
                    observations, seq.read_frame, Stage5Config())
            prediction = infer_residual(observations, features, points if variant != 'V1' else {}, cfg, variant=variant)
            prediction.update(sequence_id=sid, geometry_source='GSR_BBOX_PITCH_ORACLE' if variant != 'V1' else 'NONE',
                              adapter_diagnostics=adapter_diag, appearance_diagnostics=appearance_diag)
            # Prediction is complete before evaluating any semantic labels.
            metrics = evaluate_residual_sequence(seq, prediction)
            metrics['method'] = 'residual-' + variant.lower()
            metrics['labels_sha256'] = hashlib.sha256(seq.labels_path.read_bytes()).hexdigest()
            rows.append(metrics)
            _write_json(out / 'sequences' / sid / 'prediction.json', prediction)
            _write_json(out / 'sequences' / sid / 'metrics.json', metrics)
            manifest['completed_sequences'].append(sid)
            _write_json(out / 'run_manifest.json', manifest)
            if progress_every and (index % progress_every == 0 or index == len(ids)):
                print(f'[{index}/{len(ids)}] {sid}: overall={metrics["all_team_tracks"]["overall_accuracy"]:.4f}', flush=True)
        summary = aggregate_residual(rows, 'residual-' + variant.lower(), split)
        gates = _baseline_gate(baseline_summary, rows, summary, dataset_root, split)
        if heldout_evaluation:
            gates['heldout_component_gate'] = heldout_component_gate(
                summary, gates, complete_valid_split=(len(rows) == len(ids)))
        summary.update(package_version=__version__, dataset_root=str(Path(dataset_root).resolve()), configuration=cfg.to_dict(), appearance_configuration=feature_cfg,
                       appearance_descriptor=('TORSO_LOWER_WEIGHTED_CONCAT'
                                              if cfg.multiregion_appearance_enabled
                                              else 'TORSO_COLOR'),
                       evaluation_mode='HELDOUT_COMPONENT' if heldout_evaluation else 'EXPERIMENTAL',
                       calibration_provenance=calibration_provenance,
                       protocol={'schema_version': 'stage5-role-blind-gsr-3.0', 'gt_role_hidden': True, 'gt_team_hidden': True,
                                 'population': 'GT human boxes/track IDs; categories collapsed before inference',
                                 'geometry': 'GSR bbox_pitch oracle; not Stage1 predicted geometry' if variant != 'V1' else 'NONE',
                                 'pixel_filter': 'preserve dark/green/achromatic kits; V0 unchanged',
                                 'unknown_policy': 'wrong in overall; excluded from selective',
                                 'team_f1_ari_nmi_scope': 'assigned-only; goalkeeper ARI/NMI suppressed',
                                 'v021_safety': 'pairwise only among goal candidates; referee and GK-team recovery default to abstain',
                                 'tuning': 'defaults uncalibrated; tune TRAIN only, never VALID'},
                       gates=gates)
        _write_json(out / 'benchmark_summary.json', summary)
        (out / 'sequence_metrics.jsonl').write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in rows), encoding='utf-8')
        manifest.update(status='COMPLETE', real_accuracy_claim=True)
        _write_json(out / 'run_manifest.json', manifest)
        return summary
    except Exception as exc:
        manifest.update(status='FAILED', error=str(exc))
        _write_json(out / 'run_manifest.json', manifest)
        raise
