import copy
import json
import numpy as np
import pytest
from stage5_team_affiliation.residual_config import ResidualConfig
from stage5_team_affiliation.clustering import fit_dominant_team_prototypes
from stage5_team_affiliation.role_recovery import (clean_pitch_tracks, classify_residual_roles,
                                                   project_track_to_pitch, compute_depth_rank)
from stage5_team_affiliation.goalkeeper import assign_by_defensive_tail
from stage5_team_affiliation.residual_pipeline import infer_residual, recover_residual_appearance
from stage5_team_affiliation.residual_calibration import (evaluate_appearance_policy,
                                                          search_appearance_policy,
                                                          search_tail_policy,
                                                          calibrate_residual_train)
from stage5_team_affiliation.residual_benchmark import build_blind_inputs, _baseline_gate
from stage5_team_affiliation.residual_evaluation import evaluate_roles
from stage5_team_affiliation.residual_replay import load_pitch_state


@pytest.mark.parametrize('colour', [(30, 30, 30), (30, 180, 30), (240, 240, 240)])
def test_residual_pixel_filter_preserves_referee_gk_and_white_kits(colour):
    from stage5_team_affiliation.residual_pipeline import extract_human_features
    from stage5_team_affiliation.config import Stage5Config
    image = np.full((60, 40, 3), colour, dtype=np.uint8)
    obs = {'human': [{'frame_index': i, 'source_bbox_xyxy': [0, 0, 40, 60], 'keypoints_133': []} for i in range(6)]}
    features, diag, cfg = extract_human_features(obs, lambda _: image, Stage5Config())
    assert 'human' in features
    assert diag['human']['valid_torso_frames'] == 3
    assert cfg['green_min_saturation'] == 256


def test_single_residual_can_be_gk_without_pairwise_rival():
    _, points = fixture()
    roles = classify_residual_roles(['gk'], points, ResidualConfig())
    assert roles['gk']['stage5_role'] == 'goalkeeper'


def mock_benchmark_dataset(tmp_path, split='valid'):
    f, p = fixture()
    directory = tmp_path / 'GSR' / split / 'SNGS-999'
    directory.mkdir(parents=True)
    images = [{'image_id': str(i), 'file_name': f'{i}.jpg', 'width': 100, 'height': 100} for i in range(8)]
    categories = [{'id': i, 'name': name, 'supercategory': 'object'}
                  for i, name in enumerate(['player', 'goalkeeper', 'referee'], 1)]
    annotations = []
    for tid in f:
        role = 'goalkeeper' if tid == 'gk' else 'referee' if tid == 'ref' else 'player'
        team = None if tid == 'ref' else 'right' if tid.startswith('b') else 'left'
        for fi in range(8):
            annotations.append({'track_id': tid, 'image_id': str(fi), 'supercategory': 'object',
                                'category_id': {'player': 1, 'goalkeeper': 2, 'referee': 3}[role],
                                'attributes': {'role': role, 'team': team},
                                'bbox_image': {'x': 1, 'y': 1, 'w': 10, 'h': 20},
                                'bbox_pitch': {'x_bottom_middle': p[tid][fi][0], 'y_bottom_middle': 0}})
    (directory / 'Labels-GameState.json').write_text(json.dumps({
        'info': {'version': '1.3'}, 'images': images, 'categories': categories, 'annotations': annotations}))
    return tmp_path / 'GSR', f


def test_benchmark_reports_and_gates_without_image_pipeline(tmp_path, monkeypatch):
    from stage5_team_affiliation.residual_benchmark import run_residual_benchmark
    root, features = mock_benchmark_dataset(tmp_path)
    def mocked_extract(observations, reader, cfg):
        assert all(set(o) == {'frame_index', 'source_bbox_xyxy', 'keypoints_133'} for obs in observations.values() for o in obs)
        return features, {}, {}
    monkeypatch.setattr('stage5_team_affiliation.residual_benchmark.extract_human_features', mocked_extract)
    summary = run_residual_benchmark(dataset_root=root, split='valid', output_dir=tmp_path/'out', progress_every=0)
    # v0.2.1 keeps uncalibrated goalkeeper-team association diagnostic-only.
    assert summary['groups']['goalkeeper']['micro_overall_accuracy'] == 0
    assert summary['roles']['goalkeeper']['f1'] == 1
    assert summary['gates']['oracle_research_gate'] == 'NOT_EVALUATED'  # no paired V0
    assert summary['gates']['research_accuracy_frozen'] is False
    assert json.loads((tmp_path/'out/run_manifest.json').read_text())['status'] == 'COMPLETE'
    assert not (tmp_path/'out/benchmark_summary.md').exists()
    assert summary['groups']['goalkeeper']['macro_ARI'] is None
    with pytest.raises(FileExistsError):
        run_residual_benchmark(dataset_root=root, split='valid', output_dir=tmp_path/'out')


def test_failed_run_never_reports_complete(tmp_path, monkeypatch):
    from stage5_team_affiliation.residual_benchmark import run_residual_benchmark
    root, _ = mock_benchmark_dataset(tmp_path)
    def fail(*args):
        raise ValueError('image decode failure')
    monkeypatch.setattr('stage5_team_affiliation.residual_benchmark.extract_human_features', fail)
    with pytest.raises(ValueError, match='decode failure'):
        run_residual_benchmark(dataset_root=root, split='valid', output_dir=tmp_path/'out')
    assert json.loads((tmp_path/'out/run_manifest.json').read_text())['status'] == 'FAILED'
    assert not (tmp_path/'out/benchmark_summary.json').exists()


def test_paired_v0_gate_requires_same_sequences(tmp_path):
    baseline = tmp_path / 'baseline.json'
    baseline.write_text(json.dumps({'method': 'bbox-color', 'split': 'valid', 'dataset_root': str(tmp_path)}))
    (tmp_path/'sequence_metrics.jsonl').write_text(json.dumps({'sequence_id': 'different', 'outfield': {'tracks': 1, 'correct_valid': 1}}))
    summary = {'groups': {'goalkeeper': {'micro_overall_accuracy': 1}, 'outfield': {'micro_overall_accuracy': 1}},
               'roles': {'goalkeeper': {'f1': 1}}}
    with pytest.raises(ValueError, match='identical'):
        _baseline_gate(baseline, [{'sequence_id': 'A', 'outfield': {'tracks': 1}}], summary, tmp_path, 'valid')


def fixture():
    features = {**{f'a{i}': np.array([1., 0., 0.]) for i in range(4)},
                **{f'b{i}': np.array([0., 1., 0.]) for i in range(4)},
                'gk': np.array([0., 0., 1.]), 'ref': np.array([-1., 0., 0.])}
    xs = dict(zip(features, [-10, 30, 35, 38, -20, 0, 10, 15, 47, 20]))
    points = {t: {f: [x, 0.] for f in range(8)} for t, x in xs.items()}
    return features, points


def test_trimmed_cores_and_role_blind_recovery():
    f, p = fixture()
    result = infer_residual(f, f, p, ResidualConfig())
    r = {x['track_id']: x for x in result['tracks']}
    assert r['gk']['appearance_group'] == r['ref']['appearance_group'] == 'RESIDUAL'
    assert r['gk']['stage5_role'] == 'goalkeeper'
    assert r['ref']['stage5_role'] == 'unknown_residual'
    assert r['ref']['team_id'] is None and r['ref']['team_status'] == 'UNKNOWN'
    assert r['gk']['team_id'] is None and r['gk']['team_status'] == 'UNKNOWN'
    assert r['gk']['assignment_method'] == 'DEFENSIVE_TAIL_DIAGNOSTIC_ONLY'
    assert r['gk']['goalkeeper_assignment']['suggested_team_id'] == r['a0']['team_id'] != r['b0']['team_id']
    assert 'gk' not in sum(result['clustering']['cores'], [])


def test_uncalibrated_role_and_team_heuristics_require_explicit_opt_in():
    f, p = fixture()
    cfg = ResidualConfig(referee_recovery_enabled=True,
                         defensive_tail_assignment_enabled=True)
    r = {x['track_id']: x for x in infer_residual(f, f, p, cfg)['tracks']}
    assert r['ref']['stage5_role'] == 'referee'
    assert r['ref']['team_status'] == 'NOT_APPLICABLE'
    assert r['gk']['team_id'] == r['a0']['team_id'] != r['b0']['team_id']


def test_train_calibrated_residual_appearance_recovery():
    records = {
        'player': {'appearance_group': 'RESIDUAL', 'stage5_role': 'unknown_residual',
                   'stage5_role_status': 'UNKNOWN', 'distances': [0.2, 0.7], 'margin': 0.71,
                   'team_id': None, 'team_status': 'UNKNOWN'},
        'ref': {'appearance_group': 'RESIDUAL', 'stage5_role': 'unknown_residual',
                'stage5_role_status': 'UNKNOWN', 'distances': [0.8, 0.84], 'margin': 0.05,
                'team_id': None, 'team_status': 'UNKNOWN'},
        'gk': {'appearance_group': 'RESIDUAL', 'stage5_role': 'goalkeeper',
               'stage5_role_status': 'VALID', 'distances': [0.7, 0.8], 'margin': 0.12,
               'team_id': None, 'team_status': 'UNKNOWN'},
    }
    cfg = ResidualConfig(residual_appearance_recovery_enabled=True,
                         residual_player_min_margin=.3, residual_player_max_distance=.5,
                         residual_referee_max_margin=.1, residual_referee_min_distance=.6)
    recover_residual_appearance(records, cfg)
    assert records['player']['stage5_role'] == 'player' and records['player']['team_id'] == 0
    assert records['ref']['stage5_role'] == 'referee' and records['ref']['team_status'] == 'NOT_APPLICABLE'
    assert records['gk']['stage5_role'] == 'goalkeeper'


def calibration_rows():
    mapping = {'0': 0, '1': 1}
    common = {'sequence_id': 'train-1', 'mapping': mapping, 'mapping_available': True,
              'tail_deltas_m': []}
    return [
        common | {'track_id': 'p0', 'gt_role': 'player', 'gt_team': 0,
                  'appearance_group': 'TEAM_0', 'margin': .8, 'nearest_team': 0,
                  'nearest_distance': .1, 'base_role': 'player', 'base_role_status': 'VALID',
                  'base_team': 0, 'base_team_status': 'VALID'},
        common | {'track_id': 'p1', 'gt_role': 'player', 'gt_team': 1,
                  'appearance_group': 'RESIDUAL', 'margin': .65, 'nearest_team': 1,
                  'nearest_distance': .2, 'base_role': 'unknown_residual', 'base_role_status': 'UNKNOWN',
                  'base_team': None, 'base_team_status': 'UNKNOWN'},
        common | {'track_id': 'r', 'gt_role': 'referee', 'gt_team': None,
                  'appearance_group': 'RESIDUAL', 'margin': .03, 'nearest_team': 0,
                  'nearest_distance': .85, 'base_role': 'unknown_residual', 'base_role_status': 'UNKNOWN',
                  'base_team': None, 'base_team_status': 'UNKNOWN'},
        common | {'track_id': 'g', 'gt_role': 'goalkeeper', 'gt_team': 0,
                  'appearance_group': 'RESIDUAL', 'margin': .12, 'nearest_team': 1,
                  'nearest_distance': .7, 'base_role': 'goalkeeper', 'base_role_status': 'VALID',
                  'base_team': None, 'base_team_status': 'UNKNOWN',
                  'tail_deltas_m': [3., 3.2, 2.8, 3.1, 3.0]},
    ]


def test_train_search_selects_feasible_appearance_and_tail_policies():
    rows = calibration_rows()
    search = search_appearance_policy(rows, baseline_outfield_accuracy=.5)
    assert search['selected']['feasible'] is True
    assert search['selected']['metrics']['outfield_overall_accuracy'] == 1
    assert search['selected']['metrics']['roles']['referee']['f1'] == 1
    tail = search_tail_policy(rows, min_observations=5)
    assert tail['selected']['feasible'] is True
    assert tail['selected']['metrics']['overall_accuracy'] == 1


def test_calibrator_rejects_non_train_before_io(tmp_path):
    with pytest.raises(ValueError, match='TRAIN-only'):
        calibrate_residual_train(dataset_root=tmp_path, split='valid',
                                 prediction_dir=tmp_path/'pred', output_dir=tmp_path/'out')


def test_train_calibrator_consumes_completed_predictions_without_images(tmp_path, monkeypatch):
    from stage5_team_affiliation.residual_benchmark import run_residual_benchmark
    root, features = mock_benchmark_dataset(tmp_path, split='train')
    monkeypatch.setattr(
        'stage5_team_affiliation.residual_benchmark.extract_human_features',
        lambda observations, reader, cfg: (features, {}, {}),
    )
    source = tmp_path / 'source'
    run_residual_benchmark(dataset_root=root, split='train', output_dir=source,
                           variant='V3', progress_every=0)
    output = tmp_path / 'calibration'
    report = calibrate_residual_train(dataset_root=root, split='train',
                                      prediction_dir=source, output_dir=output)
    assert report['status'] == 'COMPLETE'
    assert report['complete_train_split'] is True
    assert (output / report['selected_config_file']).is_file()
    assert (output / 'calibration_report.json').is_file()
    assert not list(output.glob('*.md'))


@pytest.mark.parametrize('variant', ['V1', 'V2'])
def test_ablation_no_defensive_tail(variant):
    f, p = fixture()
    r = {x['track_id']: x for x in infer_residual(f, f, p, ResidualConfig(), variant=variant)['tracks']}
    assert r['gk']['team_id'] is None
    assert r['gk']['stage5_role'] == ('unknown_residual' if variant == 'V1' else 'goalkeeper')


def test_missing_geometry_is_not_referee():
    f, _ = fixture()
    r = {x['track_id']: x for x in infer_residual(f, f, {}, ResidualConfig())['tracks']}
    assert r['gk']['stage5_role_status'] == r['ref']['stage5_role_status'] == 'UNKNOWN'


def test_missing_appearance_is_not_residual():
    f, p = fixture(); del f['gk']
    result = infer_residual(p, f, p, ResidualConfig())
    gk = next(r for r in result['tracks'] if r['track_id'] == 'gk')
    assert gk['appearance_group'] == 'UNKNOWN'
    assert gk['stage5_role_status'] == 'UNKNOWN'


def test_single_colour_fails_closed():
    f = {str(i): np.array([1., 0.]) for i in range(8)}
    r = infer_residual(f, f, {}, ResidualConfig())
    assert r['clustering']['status'] == 'UNAVAILABLE'
    assert all(t['team_id'] is None for t in r['tracks'])


def test_geometry_is_mirror_symmetric():
    _, p = fixture()
    mirrored = {t: {f: [-xy[0], xy[1]] for f, xy in obs.items()} for t, obs in p.items()}
    roles = classify_residual_roles(['gk', 'ref'], mirrored, ResidualConfig())
    assert roles['gk']['stage5_role'] == 'goalkeeper'
    assert roles['gk']['goal_context']['side'] == 'LEFT'
    assert roles['ref']['stage5_role'] == 'unknown_residual'


def test_rank_uses_all_humans_not_only_residuals():
    _, p = fixture()
    for i in range(3):
        p[f'a{i}'] = {f: [50, 0] for f in range(8)}
    r = classify_residual_roles(['gk', 'ref'], p, ResidualConfig())
    assert r['gk']['stage5_role_status'] == 'UNKNOWN'


def test_tied_residuals_do_not_choose_by_track_id():
    _, p = fixture(); p['ref'] = copy.deepcopy(p['gk'])
    r = classify_residual_roles(['gk', 'ref'], p, ResidualConfig())
    assert all(x['stage5_role_status'] == 'UNKNOWN' for x in r.values())


def test_non_goal_residual_without_overlap_does_not_block_gk():
    _, p = fixture(); p['ref'] = {f + 100: xy for f, xy in p['ref'].items()}
    r = classify_residual_roles(['gk', 'ref'], p, ResidualConfig())
    assert r['gk']['stage5_role'] == 'goalkeeper'
    assert r['gk']['role_gate_diagnostics']['pairwise_compared_ids'] == []
    assert r['gk']['role_gate_diagnostics']['pairwise_ignored_non_candidate_ids'] == ['ref']


def test_goal_candidate_without_overlap_is_diagnostic_by_default():
    _, p = fixture()
    p['rival'] = {f + 100: [46, 0] for f in range(8)}
    r = classify_residual_roles(['gk', 'rival'], p, ResidualConfig())
    assert r['gk']['stage5_role'] == 'goalkeeper'
    assert r['gk']['role_gate_diagnostics']['pairwise_compared_ids'] == ['rival']
    assert r['gk']['role_gate_diagnostics']['pairwise_insufficient_overlap_ids'] == ['rival']
    assert r['gk']['role_gate_diagnostics']['pairwise_policy'] == 'DIAGNOSTIC_ONLY'


def test_frozen_config_can_require_pairwise_overlap():
    _, p = fixture()
    p['rival'] = {f + 100: [46, 0] for f in range(8)}
    cfg = ResidualConfig(pairwise_ordering_required=True)
    r = classify_residual_roles(['gk', 'rival'], p, cfg)
    assert r['gk']['stage5_role_status'] == 'UNKNOWN'
    assert r['gk']['role_decision_reason'] == 'INSUFFICIENT_GOAL_CANDIDATE_OVERLAP'


def test_unrelated_same_side_residual_does_not_enter_pairwise_gate():
    _, p = fixture()
    p['appearance_outlier'] = {f: [25, 0] for f in range(8)}
    r = classify_residual_roles(['gk', 'appearance_outlier'], p, ResidualConfig())
    assert r['gk']['stage5_role'] == 'goalkeeper'
    assert 'appearance_outlier' in r['gk']['role_gate_diagnostics']['pairwise_ignored_non_candidate_ids']


def test_short_support_is_unknown():
    _, p = fixture(); p['gk'] = {0: [47, 0], 1: [47, 0]}
    assert classify_residual_roles(['gk'], p, ResidualConfig())['gk']['stage5_role_status'] == 'UNKNOWN'


def test_no_rank_without_context():
    p = {'gk': {f: [47, 0] for f in range(8)}}
    assert compute_depth_rank('gk', p, 1, ResidualConfig())['rank_frames'] == 0
    assert classify_residual_roles(['gk'], p, ResidualConfig())['gk']['stage5_role_status'] == 'UNKNOWN'


def test_invalid_pitch_observations_removed():
    p = {'x': {0: [float('nan'), 0], 1: [1, 500], 2: [1, 2]}}
    assert list(clean_pitch_tracks(p, ResidualConfig())['x']) == [2]


def test_tail_requires_two_players_per_team():
    _, p = fixture()
    r = assign_by_defensive_tail('gk', 1, p, {'a1': 0, 'a2': 0, 'b2': 1}, ResidualConfig())
    assert r['team_status'] == 'UNKNOWN'


def test_tail_ignores_single_extreme_attacker():
    _, p = fixture(); p['b3'] = {f: [50, 0] for f in range(8)}
    teams = {f'a{i}': 0 for i in range(4)} | {f'b{i}': 1 for i in range(4)}
    assert assign_by_defensive_tail('gk', 1, p, teams, ResidualConfig())['team_id'] == 0


def test_tail_ambiguous_is_unknown():
    _, p = fixture(); p['b2'] = p['a2']; p['b3'] = p['a3']
    teams = {f'a{i}': 0 for i in range(4)} | {f'b{i}': 1 for i in range(4)}
    assert assign_by_defensive_tail('gk', 1, p, teams, ResidualConfig())['team_id'] is None


def test_projection_prefers_valid_feet_and_falls_back_to_bbox():
    calls = []
    def intersect(frame, uv):
        calls.append((frame, list(uv))); return [uv[0], uv[1], 0]
    obs = [{'frame_index': 0, 'source_bbox_xyxy': [0, 0, 10, 20], 'keypoints_133': [
        {'name': 'left_heel', 'x': 3, 'y': 17, 'state': 'VALID'}]},
        {'frame_index': 1, 'source_bbox_xyxy': [0, 0, 10, 20]}]
    project_track_to_pitch(obs, intersect)
    assert calls == [(0, [3, 17]), (1, [5, 20])]


def test_gt_semantics_are_stripped_before_inference():
    raw = {'categories': [{'id': 1, 'name': 'player', 'supercategory': 'object'}], 'annotations': [
        {'track_id': 9, 'image_id': 'a', 'category_id': 1, 'attributes': {'role': 'player', 'team': 'left'},
         'bbox_image': {'x': 0, 'y': 0, 'w': 10, 'h': 20}, 'bbox_pitch': {'x_bottom_middle': 10, 'y_bottom_middle': 4}}]}
    images = [{'image_id': 'a'}]
    before = build_blind_inputs(raw, images)
    raw['annotations'][0]['attributes'] = {'role': 'goalkeeper', 'team': 'right', 'jersey': 99}
    assert before == build_blind_inputs(raw, images)
    assert set(before[0]['9'][0]) == {'frame_index', 'source_bbox_xyxy', 'keypoints_133'}


def test_no_gt_role_fallback_for_human_population():
    with pytest.raises(ValueError, match='category metadata'):
        build_blind_inputs({'annotations': []}, [])


def test_role_metrics_count_unknown_and_contamination():
    gt = {'a': 'player', 'g': 'goalkeeper', 'r': 'referee'}
    rows = [{'track_id': 'a', 'stage5_role': 'goalkeeper', 'stage5_role_status': 'VALID'},
            {'track_id': 'g', 'stage5_role': 'unknown_residual', 'stage5_role_status': 'UNKNOWN'}]
    m = evaluate_roles(gt, rows)
    assert m['goalkeeper']['fp'] == 1 and m['goalkeeper']['fn'] == 1
    assert m['referee']['fn'] == 1
    assert m['player_contamination']['goalkeeper_rate'] == 1


def test_pitch_cache_must_match_replay(tmp_path):
    path = tmp_path / 'pitch.json'
    path.write_text(json.dumps({'schema_version': 'stage5-pitch-tracks-1.0',
                               'coordinate_space': 'PITCH_METERS_X_LONGITUDINAL_CENTER_ORIGIN',
                               'source': 'STAGE1_GROUND_PROJECTION', 'replay_context': {'selected_frame': 86}}))
    with pytest.raises(ValueError, match='replay mismatch'):
        load_pitch_state(path, {'video_id': 'replay', 'selected_frame': 104}, [])


@pytest.mark.parametrize('kwargs', [{'min_observations': 0}, {'core_keep_fraction': 2},
                                    {'ordering_margin_m': float('nan')},
                                    {'referee_recovery_enabled': 'yes'},
                                    {'pairwise_ordering_required': 1},
                                    {'residual_appearance_recovery_enabled': 1},
                                    {'residual_player_min_margin': .1, 'residual_referee_max_margin': .1}])
def test_config_invalid(kwargs):
    with pytest.raises(ValueError):
        ResidualConfig(**kwargs).validate()
