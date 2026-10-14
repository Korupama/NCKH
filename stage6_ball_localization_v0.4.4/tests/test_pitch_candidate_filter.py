import numpy as np
import pytest

from ball_localization.camera import CameraStateLite
from ball_localization.contracts import BallCandidate2D
from ball_localization.pitch_prior import apply_pitch_prior, filter_pitch_candidates
from ball_localization import pipeline


def camera():
    return CameraStateLite(0, 1920, 1080,
        np.array([[1000., 0, 960], [0, 1000, 540], [0, 0, 1]]),
        np.diag([1., -1., -1.]), np.array([0., 0., 50.]),
        np.zeros(5), 'VALID', {'length_m': 105., 'width_m': 68.})


def candidate(cam, x, y, name='ball', score=.9):
    u, v = cam.project_world([x, y, 0])[0]
    return BallCandidate2D(0, name, [u-3, v-3, u+3, v+3], [u, v], score, 'test', 6.)


@pytest.mark.parametrize('x,y,keep', [
    (0, 0, True), (52.5, 34, True), (-52.5, -34, True),
    (52.51, 0, False), (-52.51, 0, False), (0, 34.01, False), (0, -34.01, False)])
def test_pitch_boundary(x, y, keep):
    cam = camera()
    rows, rejected = filter_pitch_candidates([candidate(cam, x, y)], cam)
    assert bool(rows) == keep
    assert bool(rejected) != keep
    if rejected:
        assert rejected[0]['reason'] == 'OUTSIDE_PITCH'


def test_strict_removes_high_score_outside_while_soft_preserves_it():
    cam = camera()
    rows = [candidate(cam, 53, 0, 'outside', .99), candidate(cam, 0, 0, 'inside', .2)]
    assert len(apply_pitch_prior(rows, cam)) == 2
    kept, rejected = filter_pitch_candidates(rows, cam)
    assert [c.candidate_id for c in kept] == ['inside']
    assert rejected[0]['candidate']['candidate_id'] == 'outside'


@pytest.mark.parametrize('mode', ['missing', 'invalid', 'behind', 'parallel'])
def test_unverifiable_candidates_rejected(mode):
    cam = camera()
    ball = candidate(cam, 0, 0)
    if mode == 'invalid':
        cam.status = 'INVALID'
    elif mode == 'behind':
        cam.R_world_to_camera = np.eye(3)
    elif mode == 'parallel':
        cam.R_world_to_camera = np.array([[1., 0, 0], [0, 0, -1], [0, 1, 0]])
    kept, rejected = filter_pitch_candidates([ball], None if mode == 'missing' else cam)
    assert kept == [] and len(rejected) == 1


def test_margin_and_degraded_camera_are_explicit():
    cam = camera()
    cam.status = 'DEGRADED'
    rows, _ = filter_pitch_candidates([candidate(cam, 52.6, 0)], cam, margin_m=.2)
    assert len(rows) == 1
    assert rows[0].metadata['pitch_candidate_filter']['camera_status'] == 'DEGRADED'
    for margin in [-1, float('nan'), float('inf')]:
        with pytest.raises(ValueError):
            filter_pitch_candidates([], cam, margin_m=margin)


@pytest.mark.parametrize('provider', ['yolo-first', 'fusion', 'stage2-sst', 'yolo'])
@pytest.mark.parametrize('inside_auxiliary', [True, False])
def test_filter_before_provider_selection_and_tracker(monkeypatch, provider, inside_auxiliary):
    cam = camera()
    primary = candidate(cam, 60, 0, 'outside-yolo', .99)
    auxiliary = candidate(cam, 0 if inside_auxiliary else -60, 0, 'sst', .9)
    replay = dict(window_start=0, window_end=0, selected_frame=0, fps=30,
                  image_width=1920, image_height=1080, video_path='frame.png',
                  coordinate_space='RAW_DISTORTED_PIXEL')
    monkeypatch.setattr(pipeline, 'camera_state_for_frame', lambda *a: cam)
    monkeypatch.setattr(pipeline, 'load_replay_context_from_stage1', lambda *a: replay)
    monkeypatch.setattr(pipeline, 'load_stage2_candidates', lambda *a: (replay, {0: [auxiliary]}))
    monkeypatch.setattr(pipeline, 'SoccerNetV3DYOLOProvider', lambda *a, **kw: type('Provider', (), {'info': lambda self: {'name': 'test'}})())
    monkeypatch.setattr(pipeline, '_infer_yolo_candidates', lambda *a, **kw: {0: [primary]})
    # Keep the real tracker and geometry, omit only artifact I/O and inference.
    monkeypatch.setattr(pipeline, '_finalize_state', lambda **kw: kw)
    result = pipeline.run_stage6(stage1_root='unused', output_dir='unused',
        provider_kind=provider, weights='unused', stage2_entity_tracks='unused',
        pitch_candidate_policy='strict-ground', localization_mode='ground-only')
    expected = inside_auxiliary and provider != 'yolo'
    assert bool(result['candidates'][0]) == expected
    assert (result['frames'][0].candidate is not None) == expected
    report = result['detector_info']['pitch_candidate_filter']
    assert report['policy'] == 'strict-ground'
    if provider != 'stage2-sst':
        assert report['pools']['yolo']['rejected_count'] == 1
    if provider == 'yolo-first':
        assert result['detector_info']['fallback_frames'] == ([0] if expected else [])
