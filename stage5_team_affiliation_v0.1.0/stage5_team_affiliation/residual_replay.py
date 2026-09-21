"""Opt-in replay adapter. Live Stage1 execution is deliberately not a dependency."""
from pathlib import Path
import json
import hashlib
import cv2
from .adapters import load_stage3_state, load_stage2_state, validate_stage2_stage3_alignment
from .config import Stage5Config
from .residual_config import ResidualConfig
from .residual_pipeline import (extract_human_features,
                                extract_human_multiregion_features,
                                infer_residual)
from .visualization import render_selected_frame


def load_pitch_state(path, context, track_ids):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if (data.get('schema_version') != 'stage5-pitch-tracks-1.0'
            or data.get('coordinate_space') != 'PITCH_METERS_X_LONGITUDINAL_CENTER_ORIGIN'
            or data.get('source') != 'STAGE1_GROUND_PROJECTION'):
        raise ValueError('Explicit Stage1 metric pitch-track contract required')
    for key in ('video_id', 'selected_frame', 'image_width', 'image_height', 'fps'):
        if key not in context or data.get('replay_context', {}).get(key) != context[key]:
            raise ValueError(f'Pitch cache replay mismatch: {key}')
    points = {}
    for track in data.get('tracks', []):
        tid = str(track['track_id'])
        if tid not in track_ids or tid in points:
            raise ValueError(f'Unknown/duplicate pitch track ID: {tid}')
        points[tid] = {}
        for obs in track.get('observations', []):
            frame = int(obs['frame_index'])
            if frame in points[tid]:
                raise ValueError('Duplicate pitch frame')
            if context['window_start'] <= frame <= context['window_end'] and obs.get('status') == 'VALID':
                points[tid][frame] = obs['xy_pitch_m']
    return points


def run_residual_replay(*, stage3_state, video_path, output_dir, stage2_state=None,
                        pitch_state=None, variant='V3', config=None, residual_config=None):
    cfg = config or Stage5Config(); cfg.validate()
    rcfg = residual_config or ResidualConfig(); rcfg.validate()
    out = Path(output_dir).resolve()
    if out.exists():
        raise FileExistsError('Use a new output directory')
    s3 = load_stage3_state(stage3_state); context = s3['replay_context']; t0 = int(context['selected_frame'])
    alignment = validate_stage2_stage3_alignment(load_stage2_state(stage2_state), s3)
    if stage2_state and not alignment.get('ready'):
        raise ValueError(f'Stage2/3 alignment failed: {alignment}')
    tracks = {str(t['track_id']): t for t in s3['tracks']}
    if len(tracks) != len(s3['tracks']):
        raise ValueError('Duplicate Stage3 track IDs')
    if not context['window_start'] <= t0 <= context['window_end']:
        raise ValueError('Selected frame outside window')
    observations = {}
    for tid, track in tracks.items():
        # Strip upstream role/confidence and any metric proxies from inference input.
        obs = [{k: o[k] for k in ('frame_index', 'source_bbox_xyxy', 'keypoints_133') if k in o}
               for o in track['observations'] if context['window_start'] <= o['frame_index'] <= context['window_end']]
        if len({o['frame_index'] for o in obs}) != len(obs):
            raise ValueError('Duplicate Stage3 track/frame')
        observations[tid] = obs
    if variant != 'V1' and not pitch_state:
        raise ValueError('V2/V3 replay requires --pitch-state. Validate oracle GSR first; no Stage4 or guessed geometry fallback.')
    points = load_pitch_state(pitch_state, context, tracks) if pitch_state else {}
    video = Path(video_path or context['video_path']).resolve()
    if video != Path(context['video_path']).resolve():
        raise ValueError('Video identity differs from Stage3')
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise ValueError('Cannot open replay')
        if (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) != (context['image_width'], context['image_height']):
            raise ValueError('Video dimensions mismatch')
        if abs(cap.get(cv2.CAP_PROP_FPS) - context['fps']) > .1:
            raise ValueError('Video fps mismatch')
        def read_frame(frame):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
            ok, image = cap.read()
            return image if ok else None
        if rcfg.multiregion_appearance_enabled:
            features, diagnostics, feature_cfg = extract_human_multiregion_features(
                observations, read_frame, cfg,
                torso_weight=rcfg.multiregion_torso_weight,
                lower_weight=rcfg.multiregion_lower_weight,
                require_lower=rcfg.multiregion_require_lower,
            )
        else:
            features, diagnostics, feature_cfg = extract_human_features(
                observations, read_frame, cfg)
        result = infer_residual(tracks, features, points, rcfg, variant=variant,
                                random_state=cfg.random_state, n_init=cfg.kmeans_n_init)
        overlay = read_frame(t0)
        if overlay is None:
            raise ValueError('Selected frame decode failed')
        for row in result['tracks']:
            original = tracks[row['track_id']]
            row['upstream_role'] = original.get('upstream_role')
            row['upstream_role_confidence'] = original.get('upstream_role_score')
            row['effective_role'] = row['stage5_role'] if row['stage5_role_status'] == 'VALID' else row['upstream_role']
            row['role'] = row['effective_role']  # backwards-compatible renderer field
            row['appearance'] = diagnostics[row['track_id']]
            selected = next((o for o in observations[row['track_id']] if o['frame_index'] == t0), {})
            row['selected_frame_bbox_xyxy'] = selected.get('source_bbox_xyxy')
        source = lambda p: {'path': str(Path(p).resolve()), 'sha256': hashlib.sha256(Path(p).read_bytes()).hexdigest()} if p else None
        result.update(schema_version='team-affiliation-state-3.0', stage5_version='stage5-team-affiliation-0.3.0',
                      replay_context=context, configuration=rcfg.to_dict(), appearance_configuration=feature_cfg,
                      source_stage3=source(stage3_state), source_stage2=source(stage2_state), source_pitch=source(pitch_state),
                      alignment=alignment, gates={'real_accuracy': 'NOT_EVALUATED', 'research_accuracy_frozen': False},
                      warnings=['Provisional role/geometry heuristic. VALID means rule passed, not validated accuracy.',
                                'Referee recovery and goalkeeper-team assignment abstain by default until TRAIN calibration.',
                                'Effective-role fallback preserves upstream only; UNKNOWN team stays UNKNOWN.'], artifacts={})
        out.mkdir(parents=True, exist_ok=False)
        image_path = render_selected_frame(overlay, result['tracks'], out / 'selected_frame_team_affiliation.png')
        state_path = out / 'team_affiliation_state.json'; handoff_path = out / 'stage5_downstream_handoff.json'
        result['artifacts'] = {'team_affiliation_state': str(state_path), 'downstream_handoff': str(handoff_path), 'selected_frame_overlay': str(image_path)}
        handoff = {'schema_version': 'stage5-downstream-handoff-3.0', 'selected_frame': t0, 'track_team': {
            r['track_id']: ({k: r[k] for k in ('upstream_role', 'stage5_role', 'stage5_role_status', 'stage5_role_method',
                                               'effective_role', 'team_id', 'team_status', 'assignment_method')}
                            | {'role_decision_reason': r.get('role_decision_reason')})
            for r in result['tracks'] if r['selected_frame_bbox_xyxy'] is not None},
            'cluster_ids_are_arbitrary': True, 'attacking_team_not_resolved': True, 'research_accuracy_frozen': False}
        state_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
        handoff_path.write_text(json.dumps(handoff, indent=2, allow_nan=False), encoding='utf-8')
        return result
    finally:
        cap.release()
