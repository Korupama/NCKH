"""Regression for SAM3D pose-only handoffs (no distal-foot anchor contract)."""
import unittest

from ball_localization.contact import ground_anchor


class PoseOnlyHandoffTests(unittest.TestCase):
    def handoff(self):
        return {
            'selected_frame': 64671,
            'coordinate_frame': {'name': 'STAGE1_PITCH_WORLD', 'units': 'm',
                                 'x': 'goal-to-goal', 'y': 'touchline-to-touchline', 'z': 'up'},
            'tracks': [{'track_id': 'p1', 'observations': []}],
        }

    def test_pose_without_anchor_is_missing(self):
        result = ground_anchor(self.handoff(), 'p1', 64671)
        self.assertEqual(result['status'], 'MISSING')
        self.assertIsNone(result['xy_world_m'])

    def test_unconfirmed_or_absent_track_is_missing(self):
        for track in (None, 'absent'):
            self.assertEqual(ground_anchor(self.handoff(), track, 64671)['status'], 'MISSING')

    def test_actual_anchor_still_requires_valid_convention(self):
        hand = self.handoff()
        hand['tracks'][0]['selected_frame_ground_anchor'] = {
            'status': 'VALID', 'method': 'DENSEST_ANKLE_FOOT_CLUSTER_MEDIAN', 'xyz_ground_m': [1, 2, 0]}
        with self.assertRaisesRegex(ValueError, 'convention mismatch'):
            ground_anchor(hand, 'p1', 64671)

    def test_frame_mismatch_still_rejected(self):
        with self.assertRaisesRegex(ValueError, 'selected_frame mismatch'):
            ground_anchor(self.handoff(), 'p1', 0)


if __name__ == '__main__':
    unittest.main()
