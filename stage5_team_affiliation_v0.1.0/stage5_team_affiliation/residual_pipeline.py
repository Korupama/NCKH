"""V1/V2/V3 label-blind inference. V0 remains in pipeline.py/gsr_benchmark.py."""
from collections import defaultdict
from dataclasses import replace
import numpy as np
from .clustering import fit_dominant_team_prototypes
from .features import extract_color_feature, aggregate_features
from .regions import region_polygon
from .role_recovery import clean_pitch_tracks, classify_residual_roles
from .goalkeeper import assign_by_defensive_tail
from .goalkeeper_team_v024 import assign_by_defended_half


def recover_residual_appearance(records, config):
    """Recover high-margin players / low-margin referees after GK geometry.

    This policy is disabled in the research defaults.  Thresholds must come from
    the TRAIN-only calibrator; no GT semantics enter this function.
    """
    player_recovery_enabled = (
        config.residual_appearance_recovery_enabled
        or config.residual_player_recovery_enabled
    )
    referee_recovery_enabled = (
        config.residual_appearance_recovery_enabled
        or config.residual_referee_appearance_recovery_enabled
    )
    if not player_recovery_enabled and not referee_recovery_enabled:
        return records
    for record in records.values():
        if (record.get('appearance_group') != 'RESIDUAL'
                or record.get('stage5_role_status') == 'VALID'):
            continue
        distances = np.asarray(record.get('distances', []), dtype=float)
        try:
            margin = float(record.get('margin'))
        except (TypeError, ValueError):
            continue
        if distances.shape != (2,) or not np.isfinite(distances).all() or not np.isfinite(margin):
            continue
        nearest = int(np.argmin(distances)); nearest_distance = float(distances[nearest])
        goal = record.get('goal_context') or {}
        try:
            player_goal_distance = float(goal.get('median_goal_distance_m'))
            player_top2_rate = float(goal.get('top2_rate'))
            player_veto_context_valid = (
                np.isfinite(player_goal_distance) and np.isfinite(player_top2_rate))
        except (TypeError, ValueError):
            player_goal_distance = None
            player_top2_rate = None
            player_veto_context_valid = False
        player_referee_veto = config.residual_player_referee_veto_enabled and (
            not player_veto_context_valid
            or (
                player_goal_distance >= config.residual_player_veto_min_goal_distance_m
                and player_top2_rate >= config.residual_player_veto_min_top2_rate
            )
        )
        if (player_recovery_enabled
                and margin >= config.residual_player_min_margin
                and nearest_distance <= config.residual_player_max_distance
                and not player_referee_veto):
            record.update(stage5_role='player', stage5_role_status='VALID',
                          stage5_role_method='RESIDUAL_HIGH_MARGIN_PLAYER_RECOVERY',
                          role_decision_reason='TRAIN_CALIBRATED_PLAYER_APPEARANCE',
                          team_id=nearest, team_status='VALID',
                          assignment_method='TRAIN_CALIBRATED_NEAREST_TEAM_PROTOTYPE')
            continue
        if (player_recovery_enabled and player_referee_veto):
            record['player_recovery_veto'] = {
                'applied': True,
                'reason': ('MISSING_GOAL_CONTEXT' if not player_veto_context_valid
                           else 'REFEREE_LIKE_GOAL_RANK_GEOMETRY'),
                'median_goal_distance_m': player_goal_distance,
                'top2_rate': player_top2_rate,
            }
        # Backward compatibility for stored v0.2.2 records/tests that predate
        # goal_context.  New V2/V3 predictions always carry this field and must
        # pass the geometry gate; an explicitly old record may retain the
        # appearance-only behavior of its original frozen configuration.
        has_goal_context = 'goal_context' in record
        try:
            goal_distance = float(goal.get('median_goal_distance_m'))
            top2_rate = float(goal.get('top2_rate'))
        except (TypeError, ValueError):
            goal_distance = None
            top2_rate = None
        referee_geometry_ok = (not has_goal_context) or (
            goal_distance is not None
            and np.isfinite(goal_distance)
            and goal_distance >= config.referee_min_goal_distance_m
            and top2_rate is not None
            and np.isfinite(top2_rate)
            and top2_rate <= config.referee_max_top2_rate
        )
        if (referee_recovery_enabled
              and margin <= config.residual_referee_max_margin
              and nearest_distance >= config.residual_referee_min_distance
              and referee_geometry_ok):
            record.update(stage5_role='referee', stage5_role_status='VALID',
                          stage5_role_method=('RESIDUAL_APPEARANCE_GEOMETRY_REFEREE_RECOVERY'
                                              if has_goal_context
                                              else 'RESIDUAL_LOW_MARGIN_REFEREE_RECOVERY_LEGACY'),
                          role_decision_reason=('TRAIN_CALIBRATED_REFEREE_APPEARANCE_GEOMETRY'
                                                if has_goal_context
                                                else 'LEGACY_CONFIG_WITHOUT_GOAL_CONTEXT'),
                          team_id=None, team_status='NOT_APPLICABLE',
                          assignment_method='REFEREE_EXCLUDED')
    return records


def extract_human_features(observations, read_frame, appearance_config):
    """Only boxes/pose/frame indices allowed; reader has no semantic labels.

    Preserve descriptor family, but do not remove green/dark/achromatic KIT pixels
    in the role-blind arm. Otherwise green GKs/black refs disappear before recovery.
    This is explicit and has no effect on V0's feature filtering.
    """
    cfg = replace(appearance_config, green_min_saturation=256, min_pixel_saturation=0,
                  min_pixel_value=8, max_pixel_value=255)
    cfg.validate()
    jobs = defaultdict(list); vectors = {t: [] for t in observations}
    diagnostics = {}
    for tid, obs in observations.items():
        ordered = sorted(obs, key=lambda o: o['frame_index'])[::cfg.sample_every_n_frames]
        if len(ordered) > cfg.max_samples_per_track:
            indices = np.linspace(0, len(ordered) - 1, cfg.max_samples_per_track).round().astype(int)
            ordered = [ordered[i] for i in indices]
        diagnostics[tid] = {'sampled_frames': len(ordered), 'valid_torso_frames': 0, 'pose_frames': 0, 'bbox_frames': 0}
        for obs in ordered:
            jobs[int(obs['frame_index'])].append((tid, obs))
    for fi in sorted(jobs):
        image = read_frame(fi)
        if image is None:
            raise ValueError(f'Cannot decode required image/frame {fi}; benchmark is incomplete')
        for tid, obs in jobs[fi]:
            polygon, source = region_polygon(obs, 'torso', cfg.allow_bbox_torso_fallback)
            if polygon is None:
                continue
            vector, _ = extract_color_feature(image, polygon, cfg)
            if vector is not None:
                vectors[tid].append(vector)
                diagnostics[tid]['valid_torso_frames'] += 1
                diagnostics[tid]['pose_frames' if source == 'POSE_TORSO' else 'bbox_frames'] += 1
    features = {t: aggregate_features(v) for t, v in vectors.items() if len(v) >= cfg.min_valid_torso_frames}
    return features, diagnostics, cfg.to_dict()


def combine_region_features(torso, lower, torso_weight, lower_weight):
    """Build a distance-preserving composite descriptor.

    Each region is independently normalized before concatenation.  Scaling by
    sqrt(weight) makes squared Euclidean distance equal the weighted sum of the
    two regional squared distances.
    """
    torso = np.asarray(torso, dtype=np.float32)
    lower = np.asarray(lower, dtype=np.float32)
    if torso.ndim != 1 or lower.ndim != 1 or torso.size == 0 or lower.size == 0:
        raise ValueError('Expected non-empty one-dimensional region features')
    if not np.isfinite(torso).all() or not np.isfinite(lower).all():
        raise ValueError('Region features must be finite')
    torso_norm = float(np.linalg.norm(torso))
    lower_norm = float(np.linalg.norm(lower))
    if torso_norm <= 0 or lower_norm <= 0:
        raise ValueError('Region features must have non-zero norm')
    combined = np.concatenate([
        np.sqrt(float(torso_weight)) * torso / torso_norm,
        np.sqrt(float(lower_weight)) * lower / lower_norm,
    ]).astype(np.float32)
    norm = float(np.linalg.norm(combined))
    if norm > 0:
        combined /= norm
    return combined


def extract_human_multiregion_features(
        observations, read_frame, appearance_config, *,
        torso_weight=0.70, lower_weight=0.30, require_lower=True):
    """Extract independent torso/lower descriptors and combine per track.

    The two regions are aggregated separately across time so a missing lower
    crop cannot silently be replaced by a torso crop.  With ``require_lower``
    enabled (the research default), a track abstains when the lower-body
    evidence is insufficient.
    """
    cfg = replace(appearance_config, green_min_saturation=256,
                  min_pixel_saturation=0, min_pixel_value=8,
                  max_pixel_value=255)
    cfg.validate()
    if not 0 < torso_weight <= 1 or not 0 < lower_weight <= 1:
        raise ValueError('Multiregion weights must be in (0, 1]')
    if not np.isclose(torso_weight + lower_weight, 1.0, atol=1e-9, rtol=0):
        raise ValueError('Multiregion weights must sum to 1')

    jobs = defaultdict(list)
    region_vectors = {
        tid: {'torso': [], 'lower': []} for tid in observations
    }
    diagnostics = {}
    for tid, obs in observations.items():
        ordered = sorted(obs, key=lambda o: o['frame_index'])[::cfg.sample_every_n_frames]
        if len(ordered) > cfg.max_samples_per_track:
            indices = np.linspace(
                0, len(ordered) - 1, cfg.max_samples_per_track
            ).round().astype(int)
            ordered = [ordered[i] for i in indices]
        diagnostics[tid] = {
            'sampled_frames': len(ordered),
            'valid_torso_frames': 0,
            'valid_lower_frames': 0,
            'torso_pose_frames': 0,
            'torso_bbox_frames': 0,
            'lower_pose_frames': 0,
            'lower_bbox_frames': 0,
            'descriptor_status': 'INSUFFICIENT_EVIDENCE',
        }
        for observation in ordered:
            jobs[int(observation['frame_index'])].append((tid, observation))

    for frame_index in sorted(jobs):
        image = read_frame(frame_index)
        if image is None:
            raise ValueError(
                f'Cannot decode required image/frame {frame_index}; benchmark is incomplete')
        for tid, observation in jobs[frame_index]:
            for region, fallback in (
                    ('torso', cfg.allow_bbox_torso_fallback),
                    ('lower', cfg.allow_bbox_lower_body_fallback)):
                polygon, source = region_polygon(observation, region, fallback)
                if polygon is None:
                    continue
                vector, _ = extract_color_feature(image, polygon, cfg)
                if vector is None:
                    continue
                region_vectors[tid][region].append(vector)
                diagnostics[tid][f'valid_{region}_frames'] += 1
                source_key = (
                    f'{region}_pose_frames' if source.startswith('POSE_')
                    else f'{region}_bbox_frames')
                diagnostics[tid][source_key] += 1

    features = {}
    for tid, vectors in region_vectors.items():
        torso = (aggregate_features(vectors['torso'])
                 if len(vectors['torso']) >= cfg.min_valid_torso_frames else None)
        lower = (aggregate_features(vectors['lower'])
                 if len(vectors['lower']) >= cfg.min_valid_lower_body_frames else None)
        if torso is None or (require_lower and lower is None):
            diagnostics[tid]['descriptor_status'] = (
                'INSUFFICIENT_TORSO' if torso is None else 'INSUFFICIENT_LOWER')
            continue
        if lower is None:
            features[tid] = torso
            diagnostics[tid]['descriptor_status'] = 'TORSO_ONLY_FALLBACK'
            diagnostics[tid]['descriptor_dimension'] = int(torso.size)
        else:
            features[tid] = combine_region_features(
                torso, lower, torso_weight, lower_weight)
            diagnostics[tid]['descriptor_status'] = 'MULTIREGION_VALID'
            diagnostics[tid]['descriptor_dimension'] = int(features[tid].size)

    feature_config = cfg.to_dict()
    feature_config.update({
        'descriptor': 'TORSO_LOWER_WEIGHTED_CONCAT',
        'multiregion_torso_weight': float(torso_weight),
        'multiregion_lower_weight': float(lower_weight),
        'multiregion_require_lower': bool(require_lower),
    })
    return features, diagnostics, feature_config


def infer_residual(track_ids, features, pitch_tracks, config, *, variant='V3', random_state=23, n_init=20):
    """No upstream/GT role or team parameter exists, even for geometry fallback."""
    config.validate()
    if variant not in {'V1', 'V2', 'V3'}:
        raise ValueError('Expected V1, V2 or V3')
    track_ids = sorted(track_ids)
    if len(track_ids) != len(set(track_ids)) or set(features) - set(track_ids) or set(pitch_tracks) - set(track_ids):
        raise ValueError('Duplicate or unknown inference track IDs')
    records = {t: {'track_id': t, 'appearance_group': 'UNKNOWN', 'stage5_role': 'unknown_residual',
                   'stage5_role_status': 'UNKNOWN', 'stage5_role_method': 'INSUFFICIENT_APPEARANCE',
                   'team_id': None, 'team_status': 'UNKNOWN', 'assignment_method': 'UNAVAILABLE'} for t in track_ids}
    try:
        prototypes = fit_dominant_team_prototypes(features, config, random_state=random_state, n_init=n_init)
    except ValueError as exc:
        return {'tracks': list(records.values()), 'clustering': {'status': 'UNAVAILABLE', 'reason': str(exc)}, 'variant': variant}
    for tid, a in prototypes['tracks'].items():
        records[tid].update(a)
        if a['team_id'] is not None:
            records[tid].update(stage5_role='player', stage5_role_status='VALID',
                                stage5_role_method='DOMINANT_APPEARANCE_CORE', team_status='VALID',
                                assignment_method=(
                                    'TRIMMED_KMEANS_TORSO_LOWER_COLOR'
                                    if config.multiregion_appearance_enabled
                                    else 'TRIMMED_KMEANS_TORSO_COLOR'))
        else:
            records[tid]['stage5_role_method'] = 'APPEARANCE_RESIDUAL' if a['appearance_group'] == 'RESIDUAL' else 'AMBIGUOUS_APPEARANCE'
    if variant != 'V1':
        points = clean_pitch_tracks(pitch_tracks, config)
        residuals = [t for t, r in records.items() if r['appearance_group'] == 'RESIDUAL']
        roles = classify_residual_roles(residuals, points, config, temporal=variant == 'V3')
        player_teams = {t: r['team_id'] for t, r in records.items() if r['stage5_role'] == 'player'}
        for tid, role in roles.items():
            records[tid].update(role)
            if role['stage5_role'] == 'referee':
                records[tid].update(team_status='NOT_APPLICABLE', assignment_method='REFEREE_EXCLUDED')
            elif role['stage5_role'] == 'goalkeeper' and variant == 'V3':
                tail = assign_by_defensive_tail(
                    tid, role['goal_context']['sign'], points, player_teams, config)
                half = assign_by_defended_half(
                    tid, role['goal_context']['sign'], points, player_teams, config)
                diagnostics = dict(half)
                diagnostics['tail_deltas_m'] = tail.get('tail_deltas_m', [])
                diagnostics['defensive_tail'] = tail
                if config.defended_half_assignment_enabled:
                    diagnostics['selected_policy'] = 'DEFENDED_HALF_TEAM_DISTRIBUTION'
                    records[tid].update(team_id=half['team_id'], team_status=half['team_status'],
                                        assignment_method=half['assignment_method'],
                                        goalkeeper_assignment=diagnostics)
                elif config.defensive_tail_assignment_enabled:
                    diagnostics.update({
                        'selected_policy': 'DEFENSIVE_TAIL',
                        'team_id': tail.get('team_id'),
                        'team_status': tail.get('team_status'),
                        'assignment_method': tail.get('assignment_method'),
                        'reason': tail.get('reason'),
                        'suggested_team_id': tail.get('suggested_team_id', tail.get('team_id')),
                    })
                    records[tid].update(team_id=tail['team_id'], team_status=tail['team_status'],
                                        assignment_method=tail['assignment_method'],
                                        goalkeeper_assignment=diagnostics)
                else:
                    diagnostic = dict(diagnostics)
                    diagnostic.update(enabled=False, selected_policy=None,
                                      calibration_status='NOT_CALIBRATED_ON_TRAIN')
                    records[tid].update(team_id=None, team_status='UNKNOWN',
                                        assignment_method='DEFENSIVE_TAIL_DIAGNOSTIC_ONLY',
                                        goalkeeper_assignment=diagnostic)
    recover_residual_appearance(records, config)
    return {'tracks': list(records.values()), 'clustering': {'status': 'FITTED', **prototypes}, 'variant': variant}
