"""Regression checks for same-frame camera, kit and ball integration recovery."""
import sys
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for package in ('stage_1_camera_v12', 'stage5_team_affiliation_v0.1.0',
                'stage6_ball_localization_v0.4.4'):
    sys.path.insert(0, str(ROOT / package))

from stage_1_camera.candidates import build_camera_with_candidate_recovery
from stage5_team_affiliation.config import Stage5Config
from stage5_team_affiliation.features import extract_color_feature
from stage5_team_affiliation.clustering import recover_region_consensus
from ball_localization.contracts import BallCandidate2D
from ball_localization.providers.fusion import prefer_primary_candidates
from ball_localization.pipeline import _infer_yolo_candidates


def camera_result(valid=True, error=2.):
    return {'mode': 'full', 'use_ransac': 0, 'rep_err': error, 'calib_plane': 0,
            'cam_params': {'x_focal_length': 1500 if valid else 70000,
                           'y_focal_length': 1500 if valid else 70000,
                           'principal_point': [960, 540],
                           'position_meters': [0, 60, -15],
                           'rotation_matrix': np.eye(3).tolist()}}


class FakeCalibration:
    def __init__(self, iwidth=1920, iheight=1080, denormalize=False):
        self.image_width, self.image_height = iwidth, iheight
        self.keypoints_dict, self.lines_dict = {}, {}
        self.ord_pts = [0]

    def update(self, keypoints, lines):
        self.keypoints_dict, self.lines_dict = keypoints, lines

    def get_cam_params(self, mode, use_ransac, **kwargs):
        valid = mode == 'full' and use_ransac == 0
        r = camera_result(valid, 30 if valid else 1)
        return r['cam_params'], r['rep_err']


class RecoveryTests(unittest.TestCase):
    def camera(self, result, factory=FakeCalibration):
        return build_camera_with_candidate_recovery(
            FakeCalibration(), factory, result, frame_index=100,
            image_width=1920, image_height=1080)

    def test_camera_prefers_physical_solution_over_smaller_invalid_error(self):
        state = self.camera(camera_result(False))
        self.assertEqual(state.status, 'DEGRADED')
        self.assertEqual(state.evidence['rep_err_px'], 30)
        self.assertTrue(state.diagnostics['candidate_recovery']['used'])
        self.assertEqual(state.diagnostics['quality_gate']['invalid_reasons'], [])

    def test_usable_camera_is_not_reselected(self):
        def forbidden_factory(**kwargs):
            self.fail('Should not reselect an already usable camera')
        self.camera(camera_result(), forbidden_factory)

    def test_all_invalid_candidates_remain_invalid(self):
        with patch.object(FakeCalibration, 'get_cam_params',
                          return_value=(camera_result(False)['cam_params'], 1)):
            state = self.camera(camera_result(False))
        self.assertEqual(state.status, 'INVALID')
        self.assertFalse(state.diagnostics['candidate_recovery']['used'])

    def test_dark_and_white_kits_retained_but_bright_turf_rejected(self):
        polygon = np.array([[3, 3], [35, 3], [35, 35], [3, 35]], np.float32)
        cfg = Stage5Config(include_neutral_kit_pixels=True)
        for color in ([18, 26, 18], [240, 240, 240]):
            frame = np.full((40, 40, 3), color, np.uint8)
            self.assertIsNone(extract_color_feature(frame, polygon, Stage5Config())[0])
            self.assertIsNotNone(extract_color_feature(frame, polygon, cfg)[0])
        grass = np.full((40, 40, 3), [40, 160, 40], np.uint8)
        self.assertIsNone(extract_color_feature(grass, polygon, cfg)[0])

    def region_fixture(self):
        labels = {str(i): i // 3 for i in range(6)}
        status = {tid: 'VALID' for tid in labels}
        torso = {tid: np.array([1., 0.] if team == 0 else [0., 1.])
                 for tid, team in labels.items()}
        lower = deepcopy(torso)
        labels['outlier'], status['outlier'] = 0, 'UNKNOWN'
        torso['outlier'], lower['outlier'] = np.array([.7, .3]), np.array([.98, .02])
        cluster = SimpleNamespace(labels=labels, status=status, margins={'outlier': .5})
        return cluster, torso, lower

    def test_lower_support_requires_independent_torso_agreement(self):
        c, torso, lower = self.region_fixture()
        self.assertIn('outlier', recover_region_consensus(c, torso, lower, Stage5Config()))
        torso['outlier'] = np.array([0., 1.])
        self.assertEqual(recover_region_consensus(c, torso, lower, Stage5Config()), {})

    def test_missing_or_distant_lower_evidence_cannot_recover(self):
        c, torso, lower = self.region_fixture()
        lower['outlier'] = np.array([2., 1.])
        self.assertEqual(recover_region_consensus(c, torso, lower, Stage5Config()), {})
        del lower['outlier']
        self.assertEqual(recover_region_consensus(c, torso, lower, Stage5Config()), {})

    def test_primary_ball_scores_are_not_compared_with_sst_scores(self):
        def ball(fi, source, score):
            return BallCandidate2D(fi, source, [1, 1, 3, 3], [2, 2], score, source, 2)
        primary = {100: [ball(100, 'yolo', .2)], 101: []}
        auxiliary = {100: [ball(100, 'sst', .99)], 101: [ball(101, 'sst', .4)]}
        selected, audit = prefer_primary_candidates(primary, auxiliary)
        self.assertEqual(selected[100][0].source, 'yolo')
        self.assertEqual(selected[101][0].source, 'sst')
        self.assertEqual(audit['fallback_frames'], [101])
        with self.assertRaises(ValueError):
            prefer_primary_candidates({101: primary[100]}, {})

    def test_still_image_keeps_original_frame_index(self):
        replay = dict(video_path='frame.png', window_start=90281, window_end=90281,
                      selected_frame=90281, image_width=20, image_height=10)
        provider = SimpleNamespace(detect=lambda image, fi: [fi])
        with patch('ball_localization.pipeline.cv2.imread', return_value=np.zeros((10, 20, 3))), \
             patch('ball_localization.pipeline.camera_state_for_frame', return_value=None), \
             patch('ball_localization.pipeline.apply_pitch_prior', side_effect=lambda rows, *a, **k: rows):
            result = _infer_yolo_candidates(replay, '.', provider, pitch_margin_m=12, pitch_far_prior=.2)
            self.assertEqual(result, {90281: [90281]})
            replay['window_start'] -= 1
            with self.assertRaises(ValueError):
                _infer_yolo_candidates(replay, '.', provider, pitch_margin_m=12, pitch_far_prior=.2)


if __name__ == '__main__':
    unittest.main()
