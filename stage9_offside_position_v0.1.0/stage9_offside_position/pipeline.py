"""Read-only visualization of measured Stage 1–9 artifacts.

Stage 9 is consumed as an artifact produced by its existing pipeline.
Frame-specific observations are joined by both frame index and track ID.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import sys

import cv2
import numpy as np

from .adapters import load_json, selected_stage3_observation, finite_xyz
from .core import build_offside_position_state
from .projection import project_world_points, parse_camera
from .visualization import load_frame_with_source, encode_jpeg, _draw_label


EDGES = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9),
         (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13),
         (13, 15), (12, 14), (14, 16), (15, 17), (15, 19), (16, 20), (16, 22)]
NAMES = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
         'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
         'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee',
         'right_knee', 'left_ankle', 'right_ankle', 'left_big_toe',
         'left_small_toe', 'left_heel', 'right_big_toe', 'right_small_toe', 'right_heel']


@dataclass
class PipelineContext:
    state: dict
    frame: object
    frame_source: dict


def project_paths(root: Path) -> dict:
    cache = root / 'stage6_ball_localization_v0.4.4/outputs/replay_t0_104_cached_v050'
    return {
        'stage1': root / 'stage_1_camera_v12/outputs/temporal_v13/batch_video/shot_ptz/optimized_camera_states/camera_state_00000104_view.json',
        'stage2': cache / 'stage2_reselected/stage2_entity_tracks.json',
        'stage3': cache / 'stage3/tracked_pose_2d_state.json',
        'stage4': root / 'stage4_metric3d_v0.2.0/runs/stage4_v051_frame104/sam3d-pitch-refined/stage4_downstream_handoff.json',
        'stage5': root / 'stage5_team_affiliation_v0.1.0/outputs/project_replay/stage5_downstream_handoff.json',
        'stage6': root / 'stage6_ball_localization_v0.4.4/outputs/replay_t0_104_contact_v051/ball_trajectory_state.json',
        'stage8': root / 'stage8_offside_reference_v0.1.0/outputs/project_replay_104/offside_reference_state.json',
        'stage9': root / 'stage9_offside_position_v0.1.0/outputs/project_replay_104/offside_position_state.json',
    }


def rebuild_project_stage7(root: Path, paths: dict) -> Path:
    """Run the actual Stage 7 implementation, preserving unresolved statuses."""
    package = str(root / 'stage7_game_state_v0.1.0')
    if package not in sys.path:
        sys.path.insert(0, package)
    from stage7_game_state.core import build_game_state_context
    state = build_game_state_context(str(paths['stage1']), str(paths['stage5']), str(paths['stage6'])).to_dict()
    output = root / 'stage9_offside_position_v0.1.0/outputs/upstream_frame104/game_state_context.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding='utf-8')
    return output


def rebuild_project_stage8(root: Path, paths: dict) -> Path:
    """Run the actual Stage 8 implementation for the initial project view."""
    package = str(root / 'stage8_offside_reference_v0.1.0')
    if package not in sys.path:
        sys.path.insert(0, package)
    from stage8_offside_reference.core import build_offside_reference
    state = build_offside_reference(str(paths['stage4']), str(paths['stage6']), str(paths['stage7'])).to_dict()
    output = root / 'stage8_offside_reference_v0.1.0/outputs/project_replay_104/offside_reference_state.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding='utf-8')
    return output


def rebuild_project_stage9(root: Path, paths: dict) -> Path:
    """Run the actual Stage 9 implementation without bypassing upstream gates."""
    state = build_offside_position_state(
        str(paths['stage4']), str(paths['stage7']), str(paths['stage8']),
        stage6_input=str(paths['stage6']), best_effort=False,
    ).to_dict()
    output = root / 'stage9_offside_position_v0.1.0/outputs/project_replay_104/offside_position_state.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding='utf-8')
    return output


def _frame(data):
    return data.get('frame_index', data.get('selected_frame', data.get('replay_context', {}).get('selected_frame')))


def _map(data):
    return {str(t['track_id']): t for t in data.get('tracks', [])}


def _observation(track, frame):
    return selected_stage3_observation(track, frame) or {}


def _points2d(obs):
    if obs.get('pose_status') in ('REJECTED', 'MISSING'):
        return []
    result = []
    for p in obs.get('keypoints_133', [])[:23]:
        valid = p.get('state') not in ('INVALID', 'MISSING', 'OUT_OF_FRAME', 'LOW_CONFIDENCE',
                                      'GEOMETRIC_OUTLIER', 'TEMPORAL_OUTLIER', 'LEFT_RIGHT_SUSPECT',
                                      'TEMPORAL_IMPUTED', 'LOW_MODEL_EVIDENCE')
        xy = [p.get('x'), p.get('y')]
        result.append(xy if valid and all(isinstance(v, (int, float)) and np.isfinite(v) for v in xy) else None)
    return result if any(p is not None for p in result) else []


def _project(camera, xyz):
    """Do not draw points behind the camera or clamp them onto image borders."""
    cam = parse_camera(camera)
    pts = np.asarray(xyz, dtype=float)
    depth = ((cam['R'] @ pts.T).T + cam['t'])[:, 2]
    uv = project_world_points(camera, xyz)
    return [p.tolist() if z > 0 and np.isfinite(p).all() else None for p, z in zip(uv, depth)]


def _pitch_lines(pitch):
    x, y = pitch.get('length_m', 105) / 2, pitch.get('width_m', 68) / 2
    lines = [[[-x, -y, 0], [x, -y, 0], [x, y, 0], [-x, y, 0], [-x, -y, 0]],
             [[0, -y, 0], [0, y, 0]]]
    for sign in [-1, 1]:
        for depth, halfwidth in [(16.5, 20.16), (5.5, 9.16)]:
            lines.append([[sign*x, -halfwidth, 0], [sign*(x-depth), -halfwidth, 0],
                          [sign*(x-depth), halfwidth, 0], [sign*x, halfwidth, 0]])
    lines.append([[9.15*np.cos(a), 9.15*np.sin(a), 0] for a in np.linspace(0, 2*np.pi, 90)])
    return lines


def build_pipeline_context(*, paths: dict, video_path=None, image_path=None):
    data = {name: load_json(path) for name, path in paths.items() if name in {f'stage{i}' for i in range(1, 10)}}
    missing = [f'stage{i}' for i in range(1, 10) if not data.get(f'stage{i}')]
    if missing:
        raise ValueError('Missing artifacts: ' + ', '.join(missing))
    s1, s2, s3, s4, s5, s6, s7, s8, s9 = [data[f'stage{i}'] for i in range(1, 10)]
    frame_index = int(_frame(s4))
    for name, d in data.items():
        if _frame(d) != frame_index:
            raise ValueError(f'{name}: frame {_frame(d)} does not match frame {frame_index}; select aligned artifacts')
    if not image_path and not video_path:
        video_path = s3.get('replay_context', {}).get('video_path')
    frame, source = load_frame_with_source(image_path=image_path, video_path=video_path, frame_index=frame_index)
    if source['kind'] == 'FALLBACK_CANVAS':
        raise ValueError('Cannot read source replay. Supply a clean --video or --image.')
    h, w = frame.shape[:2]
    for name, meta in [('stage1', s1.get('image', {})), ('stage2', s2.get('replay_context', {})), ('stage3', s3.get('replay_context', {}))]:
        width, height = meta.get('width', meta.get('image_width')), meta.get('height', meta.get('image_height'))
        if (width, height) != (w, h):
            raise ValueError(f'{name}: image size {(width, height)} does not match replay {(w, h)}')
    source['origin'] = 'explicit image' if image_path else 'replay video'
    maps = [_map(s) for s in [s2, s3, s4]]
    teams = s5.get('track_team', {})
    sets = s7.get('sets', {})
    toucher_payload = s7.get('toucher') or {}
    toucher_id = toucher_payload.get('track_id')
    toucher_tentative = toucher_payload.get('evidence_level') == 'TENTATIVE_SPATIAL_ONLY'
    offside_by_id = {str(row.get('track_id')): row for row in s9.get('attackers', []) if row.get('track_id') is not None}
    rows = []
    for tid in sorted(set().union(*maps, teams)):
        t2, t3, t4 = [m.get(tid, {}) for m in maps]
        o2, o3, o4 = [_observation(t, frame_index) for t in [t2, t3, t4]]
        joints = {j['name']: finite_xyz(j.get('xyz_world_m')) for j in o4.get('joints_world', []) if j.get('valid', True)}
        world = [joints.get(name) for name in NAMES]
        projected = [_project(s1, [p])[0] if p else None for p in world]
        group = next((key for key, ids in sets.items() if tid in ids), 'unresolved')
        offside = offside_by_id.get(tid, {})
        rows.append({'track_id': tid, 'role': teams.get(tid, {}).get('role', t2.get('role', 'unknown')),
                     'active': bool(o2 or o3 or o4), 'bbox': o2.get('bbox_xyxy'),
                     'pose_bbox': o3.get('source_bbox_xyxy', o3.get('bbox_xyxy')),
                     'pose2d': _points2d(o3), 'pose2d_status': o3.get('pose_status', 'MISSING'),
                     'quality2d': o3.get('qa', {}), 'crop2d': o3.get('crop_diagnostics', {}),
                     'model2d': o3.get('model_diagnostics', {}),
                     'root_world_m': finite_xyz(o4.get('root_world_m')), 'joints_world': world,
                     'projected3d': projected, 'pose3d_status': t4.get('selected_frame_status', 'MISSING'),
                     'quality3d': o4.get('quality', {}), 'team_id': teams.get(tid, {}).get('team_id'),
                     'team_status': teams.get(tid, {}).get('team_status', 'MISSING'), 'group': group,
                     'toucher': tid == toucher_id,
                     'toucher_tentative': bool(tid == toucher_id and toucher_tentative),
                     'opponent_rank': next((r.get('rank') for r in s8.get('opponent_ranking', []) if str(r.get('track_id')) == tid), None),
                     'second_last': tid in {str(x) for x in ((s8.get('second_last_opponent') or {}).get('candidate_track_ids') or [])},
                     'offside_label': offside.get('label'), 'offside_flag': offside.get('flag'),
                     'offside_delta_q_m': offside.get('delta_q_m'), 'offside_reason': offside.get('reason')})
    ball = s6.get('selected_frame_ball', {})
    if ball.get('frame_index') != frame_index:
        raise ValueError('stage6: selected ball frame does not match replay')
    pitch = s1.get('pitch', {})
    lines = _pitch_lines(pitch)
    # S8 owns the line. Never substitute a Stage-9 best-effort ball estimate.
    display_reference = s8.get('reference') if s8.get('status') in {'VALID', 'DEGRADED'} else None
    reference_x = (display_reference or {}).get('X_world_m')
    projected_reference = None
    if reference_x is not None:
        half_width = float(pitch.get('width_m', 68)) / 2.0
        projected_reference = _project(s1, [[float(reference_x), -half_width, 0.0], [float(reference_x), half_width, 0.0]])
    n2 = sum(bool(r['bbox']) for r in rows)
    n3 = sum(bool(r['pose2d']) for r in rows)
    qa3_counts = {label: sum(r['pose2d_status'] == label for r in rows)
                  for label in ('VALID', 'DEGRADED', 'REJECTED')}
    qa3_status = ('DEGRADED' if qa3_counts['DEGRADED'] or qa3_counts['REJECTED'] else 'VALID') if any(qa3_counts.values()) else ('AVAILABLE' if n3 else 'MISSING')
    if qa3_counts['REJECTED'] and not n3:
        qa3_status = 'REJECTED'
    n4 = sum(r['root_world_m'] is not None for r in rows)
    n5 = sum(r['active'] and r['team_id'] is not None for r in rows)
    n9_off = sum(r.get('offside_label') == 'OFFSIDE_POSITION' for r in rows)
    n9_on = sum(r.get('offside_label') == 'ONSIDE' for r in rows)
    caps = s1.get('diagnostics', {}).get('capabilities', {})
    titles = ['Camera & tọa độ sân', 'Phát hiện & tracking', 'Tư thế 2D', 'Tư thế 3D trên sân',
              'Nhận diện đội', 'Bóng & tiếp xúc', 'Ngữ cảnh trận đấu', 'Mốc tham chiếu việt vị',
              'Phân loại vị trí việt vị']
    summaries = [f"Mặt sân: {caps.get('ground_geometry', {}).get('status', 'UNKNOWN')}",
                 f'{n2} người tại frame {frame_index}',
                 (f"{qa3_counts['VALID']} đạt · {qa3_counts['DEGRADED']} cần xem lại · {qa3_counts['REJECTED']} bị loại" if any(qa3_counts.values()) else f'{n3} bộ keypoint tại frame này'),
                 f'{n4}/{len(s4.get("tracks", []))} track có pose 3D', f'{n5} người được gán đội',
                 f"Chạm bóng: {(ball.get('contact') or {}).get('track_id', 'chưa rõ')}",
                 f"{len(sets.get('attackers', []))} tấn công · {len(sets.get('opponents', []))} đối phương",
                 (("Vạch hậu vệ tham khảo · " if (display_reference or {}).get('reference_only') else "") + f"X = {reference_x:.3f} m" if reference_x is not None else 'Chưa xác định được mốc tham chiếu'),
                 f'{n9_off} vị trí việt vị · {n9_on} onside']
    statuses = [s1.get('status', 'UNKNOWN'), s2.get('status', 'UNKNOWN'),
                qa3_status, 'PARTIAL' if n4 < len(s4.get('tracks', [])) else 'AVAILABLE',
                'AVAILABLE' if n5 else 'MISSING', ball.get('status', 'UNKNOWN'), s7.get('status', 'UNKNOWN'), s8.get('status', 'UNKNOWN'),
                s9.get('status', 'UNKNOWN')]
    details = [dict(capabilities=caps, camera=s1.get('extrinsics'), view=s1.get('view')),
               dict(status=s2.get('status'), metrics=s2.get('metrics'), scope='Metrics may cover the full tracking window'),
               dict(acceptance_gate=s3.get('acceptance_gate'), metrics=s3.get('metrics'),
                    execution=s3.get('live_execution'), note='Raw SimCC scores are not calibrated probabilities'),
               dict(producer=s4.get('producer'), coverage=f'{n4}/{len(s4.get("tracks", []))}', note='World joints are estimates; per-track residuals are shown below'),
               dict(research_accuracy_frozen=s5.get('research_accuracy_frozen'), cluster_ids_are_arbitrary=s5.get('cluster_ids_are_arbitrary')),
               dict(status=ball.get('status'), method=ball.get('method'), contact=ball.get('contact'), localization=ball.get('localization')),
               s7,
               dict(reference=s8.get('reference'), second_last_opponent=s8.get('second_last_opponent'), opponent_ranking=s8.get('opponent_ranking'), reasons=s8.get('reasons'), quality=s8.get('quality')),
               dict(mode=s9.get('mode'), reference=s9.get('reference'), attackers=s9.get('attackers'), diagnostics=s9.get('diagnostics'), note='Chỉ phân loại vị trí; không kết luận hành vi phạm luật việt vị.')]
    raw_data = [s1, s2, s3, s4, s5, s6, s7, s8, s9]
    stages = [dict(id=i+1, title=titles[i], summary=summaries[i], status=statuses[i],
                   is_mocked=raw_data[i].get('is_mocked', False),
                   source=str(Path(paths[f'stage{i+1}']).resolve()), detail=details[i]) for i in range(9)]
    state = dict(mode='UPSTREAM_1_7', frame_index=frame_index, image_size=[w, h],
                 timestamp_sec=s1.get('timestamp_sec'), frame_source=source,
                 stages=stages, tracks=rows, game_state=s7,
                 ball=dict(center_uv=(ball.get('candidate') or ball.get('observation') or {}).get('center_uv'),
                           center_xyz_world_m=finite_xyz(ball.get('center_xyz_world_m')), contact=ball.get('contact', {}), status=ball.get('status')),
                 reference=display_reference, second_last_opponent=s8.get('second_last_opponent'),
                 offside_position=s9,
                 projected_reference=projected_reference,
                 pitch=dict(length_m=pitch.get('length_m', 105), width_m=pitch.get('width_m', 68)),
                 pitch_lines=lines, projected_pitch=[_project(s1, line) for line in lines], skeleton_edges=EDGES,
                 counts=dict(detected=n2, pose2d=n3, pose3d=n4, teams=n5, offside_position=n9_off, onside=n9_on),
                 scope=f'Quan sát Stage 1–9 tại frame {frame_index}; Stage 9 chỉ phân loại vị trí, không kết luận lỗi việt vị.')
    return PipelineContext(state, frame, source)


def render_pipeline_frame(ctx, query=None):
    query = query or {}
    def flag(name, default=False):
        on = str(query.get(name, ['1' if default else '0'])[0]).lower() not in ('0', 'false', 'off', 'no')
        if not on:
            return False
        if name.startswith('s'):
            try:
                idx = int(name[1:]) - 1
                if ctx.state['stages'][idx].get('is_mocked'):
                    return False
            except:
                pass
        return True
    if not flag('overlay', True):
        return encode_jpeg(ctx.frame)
    img = ctx.frame.copy()
    h, w = img.shape[:2]
    def point(p):
        return tuple(int(np.clip(round(v), -1000000, 1000000)) for v in p)
    def line(a, b, color, width=2):
        if a is not None and b is not None:
            cv2.line(img, point(a), point(b), color, width, cv2.LINE_AA)
    if flag('s1'):
        for poly in ctx.state['projected_pitch']:
            for a, b in zip(poly, poly[1:]):
                line(a, b, (212, 228, 100))
    occupied = []
    focus = set(query.get('track', [])) - {''}
    enabled_layers = {i: flag(f's{i}') for i in range(1, 10)}
    raw_detection_view = enabled_layers[2] and not any(enabled_layers[i] for i in range(3, 10))
    for row in ctx.state['tracks']:
        if focus and row['track_id'] not in focus:
            continue
        # Stage 2 remains an honest raw detector view. In every downstream
        # composition, detections without pose/context evidence are omitted so
        # spectators and tiny false tracks cannot obscure the analysis.
        useful_downstream_track = bool(
            row['pose2d'] or row['root_world_m'] or row['toucher']
            or row['group'] in ('attackers', 'opponents') or row.get('offside_label')
        )
        if not raw_detection_view and not useful_downstream_track:
            continue
        color = (235, 201, 98)
        if enabled_layers[5]:
            color = {0: (225, 173, 71), 1: (121, 160, 250)}.get(row['team_id'], (175, 175, 175))
        if enabled_layers[7]:
            color = {'attackers': (121, 160, 250), 'opponents': (225, 173, 71)}.get(row['group'], (175, 175, 175))
            if row['toucher']:
                color = (80, 235, 248)
        if enabled_layers[9] and row.get('offside_label'):
            color = {
                'OFFSIDE_POSITION': (70, 70, 255),
                'ONSIDE': (95, 220, 110),
                'TOUCHER_EXCLUDED': (80, 235, 248),
                'UNAVAILABLE': (175, 175, 175),
            }.get(row.get('offside_label'), color)
        bbox = row['bbox'] or row['pose_bbox']
        if bbox and any(enabled_layers[i] for i in (2, 5, 7, 9)):
            box = [int(max(0, min(w-1 if i%2==0 else h-1, v))) for i, v in enumerate(bbox)]
            if flag('labels', True):
                suffix = ''
                if enabled_layers[9] and row.get('offside_label'):
                    suffix = {
                        'OFFSIDE_POSITION': ' OFFSIDE POSITION',
                        'ONSIDE': ' ONSIDE',
                        'TOUCHER_EXCLUDED': ' PASSER',
                        'UNAVAILABLE': ' UNKNOWN',
                    }.get(row.get('offside_label'), ' ?')
                    if row.get('offside_label') == 'TOUCHER_EXCLUDED' and row.get('toucher_tentative'):
                        suffix = ' PASSER?'
                    if (row.get('offside_label') == 'UNAVAILABLE'
                            and (ctx.state.get('reference') or {}).get('reference_only')
                            and row.get('offside_delta_q_m') is not None):
                        suffix = ' BEYOND DEF LINE*' if row['offside_delta_q_m'] > 1e-9 else ' BEHIND/LEVEL DEF LINE*'
                elif enabled_layers[7]:
                    suffix = (' TOUCH?' if row.get('toucher_tentative') else ' TOUCH') if row['toucher'] else {'attackers':' ATT', 'opponents':' OPP'}.get(row['group'], ' ?')
                elif enabled_layers[5]:
                    suffix = f" TEAM {row['team_id']}" if row['team_id'] is not None else ' TEAM ?'
                _draw_label(img, box, row['track_id'].replace('track_', '#') + suffix, color, occupied)
            else:
                cv2.rectangle(img, tuple(box[:2]), tuple(box[2:]), color, 2)
        for key, enabled, col in [('pose2d', enabled_layers[3], (120, 255, 173)), ('projected3d', enabled_layers[4], (224, 136, 225))]:
            pts = row[key]
            if not enabled:
                continue
            for a, b in EDGES:
                if max(a,b) < len(pts):
                    line(pts[a], pts[b], col)
            for p in pts:
                if p is not None and 0 <= p[0] < w and 0 <= p[1] < h:
                    cv2.circle(img, point(p), 3, col, -1, cv2.LINE_AA)
    if flag('s6') and ctx.state['ball']['center_uv']:
        p = point(ctx.state['ball']['center_uv'])
        cv2.circle(img, p, 13, (80, 235, 248), 3, cv2.LINE_AA)
        cv2.circle(img, p, 3, (255, 255, 255), -1)
        if flag('labels', True):
            cv2.putText(img, 'BALL', (p[0]+16, p[1]-10), cv2.FONT_HERSHEY_SIMPLEX, .55, (80,235,248), 2, cv2.LINE_AA)
    if (enabled_layers[8] or enabled_layers[9]) and (ctx.state.get('reference') or {}).get('X_world_m') is not None:
        reference_line = ctx.state.get('projected_reference') or []
        if len(reference_line) < 2:
            return encode_jpeg(img)
        reference_only = (ctx.state.get('reference') or {}).get('reference_only', False)
        line_color = (80, 200, 255) if reference_only else (80, 80, 255)
        line(reference_line[0], reference_line[1], line_color, 4)
        if flag('labels', True):
            anchor = next((p for p in reference_line if p is not None), None)
            if anchor is not None:
                caption = 'DEFENDER REFERENCE ONLY' if reference_only else 'OFFSIDE POSITION REFERENCE'
                cv2.putText(img, caption, point(anchor), cv2.FONT_HERSHEY_SIMPLEX, .55, line_color, 2, cv2.LINE_AA)
    return encode_jpeg(img)
