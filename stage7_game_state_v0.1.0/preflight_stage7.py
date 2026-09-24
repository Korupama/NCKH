"""Read-only upstream checks; writes a readiness report, never fabricates GT."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from stage7_game_state.adapters import load_json, extract_stage1_view, extract_stage5_players, extract_stage6_contact


def main():
    root = Path(__file__).resolve().parent
    workspace = root.parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage1', default=str(workspace / 'stage_1_camera_v12/outputs/temporal_v13/batch_video/shot_ptz/optimized_camera_states/camera_state_00000104.json'))
    parser.add_argument('--stage5', default=str(workspace / 'stage5_team_affiliation_v0.1.0/outputs/project_replay/stage5_downstream_handoff.json'))
    parser.add_argument('--stage6', default=str(workspace / 'stage6_ball_localization_v0.4.4/outputs/replay_t0_104_contact_v051/stage6_downstream_handoff.json'))
    parser.add_argument('--output', default=str(root / 'outputs/preflight/stage7_preflight.json'))
    args = parser.parse_args()
    sources, states, blockers = {}, {}, []
    for key in ('stage1', 'stage5', 'stage6'):
        path = Path(getattr(args, key)).resolve()
        sources[key] = {'path': str(path), 'exists': path.is_file()}
        if not path.is_file():
            blockers.append(key.upper() + '_FILE_MISSING')
            states[key] = {}
            continue
        sources[key]['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        states[key] = load_json(path)
    s1, s5, s6 = (states[k] for k in ('stage1', 'stage5', 'stage6'))
    contact = extract_stage6_contact(s6)
    players = extract_stage5_players(s5)
    x, hit, _ = extract_stage1_view(s1)
    frames = {'stage1': s1.get('frame_index'), 'stage5': s5.get('selected_frame'), 'stage6': contact['frame_index']}
    if any(v is None for v in frames.values()) or len(set(frames.values())) != 1:
        blockers.append('UPSTREAM_FRAME_MISMATCH_OR_MISSING')
    if x is None:
        blockers.append('CENTRE_RAY_PITCH_HIT_MISSING')
    elif not all(math.isfinite(v) for v in hit) or abs(x) <= 1e-6:
        blockers.append('CENTRE_RAY_INVALID_OR_AMBIGUOUS')
    if s1.get('status') not in ('VALID', 'DEGRADED'):
        blockers.append('CAMERA_STATUS_NOT_USABLE')
    if contact.get('status') != 'SUPPORTED' or not contact.get('track_id'):
        blockers.append('CONTACT_NOT_SUPPORTED')
    toucher = next((p for p in players if p['track_id'] == contact.get('track_id')), None)
    if toucher is None or toucher['team_key'] is None or toucher['is_referee'] or not toucher['active']:
        blockers.append('TOUCHER_TEAM_OR_ROLE_UNRESOLVED')
    warnings = []
    # This is an explicit, previously confirmed annotation of this replay only.
    if 'project_replay' in str(args.stage5) and frames['stage5'] == 104:
        ref = next((p for p in players if p['track_id'] == 'track_003'), None)
        if ref is not None and not ref['is_referee']:
            blockers.append('KNOWN_REPLAY_REFEREE_TRACK_003_MISLABELLED_UPSTREAM')
    if s1.get('status') == 'DEGRADED':
        warnings.append('CAMERA_DEGRADED_REVIEW_REQUIRED')
    manifest = root / 'data/stage7_eval_manifest.json'
    manifest_ready = False
    if manifest.is_file():
        cases = load_json(manifest).get('cases', [])
        manifest_ready = bool(cases) and all(
            isinstance(c, dict) and bool(c.get('gt')) and all(
                c.get(k) and (manifest.parent / c[k]).is_file() for k in ('stage1', 'stage5', 'stage6')
            ) for c in cases
        )
    report = {
        'status': 'BLOCKED' if blockers else 'READY_FOR_QA',
        'sources': sources, 'frames': frames, 'camera_status': s1.get('status'),
        'centre_ray_pitch_hit_m': hit, 'contact': contact,
        'num_tracks': len(players), 'toucher_team_id': toucher['team_id'] if toucher else None,
        'blockers': blockers, 'warnings': warnings,
        'benchmark_manifest_paths_ready': manifest_ready,
        'independent_ground_truth_verified': False,
        'production_accuracy_evaluated': False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 2 if blockers else 0


if __name__ == '__main__':
    raise SystemExit(main())
