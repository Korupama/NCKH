"""Outside-pitch detections must never enter the shared Stage 3-9 handoff."""
import ast
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'stage2_sst_rtmw_v1.3'))
sys.path.insert(0, str(ROOT / 'stage6_ball_localization_v0.4.4'))
from stage2_entities.pitch_filter import filter_tracks_to_pitch
from ball_localization.camera import CameraStateLite


def camera():
    # Downward camera: u=X+100, v=100-Y, z=0, with real ray projection.
    return CameraStateLite(7, 200, 200,
                           np.array([[10, 0, 100], [0, 10, 100], [0, 0, 1.]]),
                           np.diag([1., -1., -1.]), np.array([0., 0., 10.]),
                           np.zeros(5), 'DEGRADED', {'length_m': 105., 'width_m': 68.})


def track(tid, x, y, role='player'):
    u, v = 100 + x, 100 - y
    return {'track_id': tid, 'role': role, 'observations': [
        {'frame_index': 7, 'bbox_xyxy': [u-1, v-10, u+1, v], 'detector_score': .9}]}


@pytest.mark.parametrize('role', ['player', 'goalkeeper', 'referee'])
def test_off_pitch_exclusion_has_no_role_exemption(role):
    source = [track('track_002', -50.73, -1.38, 'goalkeeper'),
              track('track_023', -54.39, 35.43, role),
              track('track_025', 0, 34.01, role)]
    original = json.dumps(source)
    kept, report = filter_tracks_to_pitch(source, camera())
    assert [t['track_id'] for t in kept] == ['track_002']
    assert report['excluded_track_count'] == 2
    assert all(r['reason'] == 'OUTSIDE_PITCH' for r in report['excluded'])
    assert json.dumps(source) == original  # audit must not mutate input artifacts


def test_boundary_is_kept_but_no_extra_margin_or_clamping():
    source = [track(str(i), x, y) for i, (x, y) in enumerate(
        [(52.5, 34), (-52.5, -34), (52.501, 0), (0, -34.001)])]
    kept, _ = filter_tracks_to_pitch(source, camera())
    assert [t['track_id'] for t in kept] == ['0', '1']


def test_people_use_feet_ball_uses_centre():
    # Body crosses the projected touchline, but feet remain inside.
    person = track('p', 0, 33)
    ball = track('b', 0, 33, 'ball')
    kept, _ = filter_tracks_to_pitch([person, ball], camera())
    assert [t['track_id'] for t in kept] == ['p']


def test_invalid_camera_and_wrong_frame_are_not_silently_accepted():
    cam = camera()
    cam.status = 'INVALID'
    with pytest.raises(ValueError, match='usable Stage-1'):
        filter_tracks_to_pitch([], cam)
    cam = camera()
    cam.frame_index = 8
    with pytest.raises(ValueError, match='mismatch'):
        filter_tracks_to_pitch([track('p', 0, 0)], cam)


def test_unprojectable_observation_is_excluded():
    cam = camera()
    cam.R_world_to_camera = np.eye(3)  # points away from pitch
    kept, report = filter_tracks_to_pitch([track('p', 0, 0)], cam)
    assert kept == []
    assert report['excluded'][0]['reason'] == 'NO_FORWARD_PITCH_INTERSECTION'


def test_live_stage2_writes_only_filtered_handoff_and_ball_pool(tmp_path, monkeypatch):
    # Execute the actual live function without importing GPU/model initialization.
    path = ROOT / 'stage9_offside_position_v0.1.0/live_single_frame_inference.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'process_stage2')
    inputs = [track('a', 0, 0), track('b', -54.39, 35.43),
              track('c', 0, 0, 'goalkeeper'), track('d', 0, 0, 'ball'),
              track('e', 60, 0, 'ball')]
    detections = [dict(label=t['role'], bbox_xyxy=t['observations'][0]['bbox_xyxy'], score=.9)
                  for t in inputs]
    def detect(**kwargs):
        return None, detections, None
    import ball_localization.camera as camera_module
    monkeypatch.setattr(camera_module, 'load_camera_state', lambda path: camera())
    ns = dict(__file__=str(path), Path=Path, cv2=cv2, json=json, sys=sys, process_frame=detect)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), ns)
    image = tmp_path / 'frame.png'
    assert cv2.imwrite(str(image), np.zeros((200, 200, 3), dtype=np.uint8))
    ns['process_stage2'](image, tmp_path, device='cpu', frame_index=7)
    state = json.loads((tmp_path / 'stage2.json').read_text())
    assert [t['track_id'] for t in state['tracks']] == ['track_001', 'track_003', 'track_004']
    assert len(state['auxiliary_ball_detections_by_frame']['7']) == 1
    assert state['pitch_filter']['excluded_track_count'] == 2
    audit = json.loads((tmp_path / 'stage2_pitch_filter.json').read_text())
    assert {r['track_id'] for r in audit['excluded']} == {'track_002', 'track_005'}
    assert (tmp_path / 'stage2_debug.jpg').is_file()
