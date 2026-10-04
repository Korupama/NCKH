"""Cross-stage regression checks for the optional goalkeeper reference path."""
import copy
import runpy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for package in ('stage7_game_state_v0.1.0', 'stage8_offside_reference_v0.1.0', 'stage9_offside_position_v0.1.0'):
    sys.path.insert(0, str(ROOT / package))
from stage7_game_state.core import build_game_state_context
from stage8_offside_reference.core import build_offside_reference
from stage9_offside_position.core import build_offside_position_state

h = runpy.run_path(str(ROOT / 'stage8_offside_reference_v0.1.0/tests/helpers.py'))


class DefenderReferenceTests(unittest.TestCase):
    def setUp(self):
        self.camera = {'view': {'centre_ray_pitch_hit_m': [-30, 0, 0]}}
        self.teams = {'players': [
            {'track_id': 'g', 'team_id': 1, 'role': 'goalkeeper'},
            {'track_id': 'd', 'team_id': 1, 'role': 'player'},
            {'track_id': 'a', 'team_id': 0, 'role': 'player'},
        ]}
        self.ball = h['stage6'](10, extent=(-33, -32.8))
        self.pose = h['stage4'](10, [h['track'](tid, 10, [h['joint']('left_heel', x)])
                                   for tid, x in [('g', -50), ('d', -42), ('a', -43)]])

    def context(self, enabled=True):
        return build_game_state_context(self.camera, self.teams, self.ball,
                                        allow_goalkeeper_fallback=enabled).to_dict()

    def test_fallback_is_explicit_and_does_not_invent_toucher(self):
        self.assertEqual(self.context(False)['status'], 'UNRESOLVED')
        s7 = self.context()
        self.assertEqual(s7['status'], 'DEGRADED')
        self.assertIsNone(s7['toucher'])
        self.assertEqual(s7['attacking_team_id'], 0)
        self.assertEqual(set(s7['sets']['opponents']), {'g', 'd'})
        self.assertTrue(s7['diagnostics']['invariant_pass'])

    def test_ambiguous_or_unaffiliated_keeper_does_not_resolve(self):
        self.teams['players'].append({'track_id': 'g2', 'team_id': None, 'role': 'goalkeeper'})
        self.assertEqual(self.context()['status'], 'UNRESOLVED')
        self.teams['players'].pop()
        self.teams['players'][0]['team_id'] = None
        self.assertEqual(self.context()['status'], 'UNRESOLVED')

    def test_confirmed_contact_is_not_overridden(self):
        self.ball['stage7'] = {'contact_track_id': 'g'}
        # Use the adapter's direct contact contract.
        self.ball['contact_track_id'] = 'g'
        self.assertEqual(self.context()['attacking_team_id'], 1)

    def test_reference_uses_defender_not_ball_and_never_classifies(self):
        s7 = self.context()
        s8 = build_offside_reference(self.pose, self.ball, s7, allow_ball_fallback=True).to_dict()
        self.assertEqual(s8['status'], 'DEGRADED')
        self.assertEqual(s8['reference']['X_world_m'], -42)
        self.assertTrue(s8['reference']['reference_only'])
        self.assertIsNone(s8['ball'])
        s9 = build_offside_position_state(self.pose, s7, s8, best_effort=False).to_dict()
        self.assertEqual(s9['status'], 'UNRESOLVED')
        self.assertEqual(s9['attackers'][0]['label'], 'UNAVAILABLE')

    def test_missing_and_unusable_ball_keep_defender_line(self):
        s7 = h['stage7'](10, -1, ['g', 'd'])
        for ball in [h['stage6'](10, extent=None, usable=False), h['stage6'](10, extent=(-52, -51), usable=False)]:
            s8 = build_offside_reference(self.pose, ball, s7, allow_ball_fallback=True).to_dict()
            self.assertEqual(s8['status'], 'DEGRADED')
            self.assertEqual(s8['reference']['X_world_m'], -42)

    def test_incomplete_defenders_and_frame_mismatch_do_not_draw(self):
        s7 = self.context()
        pose = copy.deepcopy(self.pose)
        pose['tracks'] = pose['tracks'][:1]
        self.assertIsNone(build_offside_reference(pose, self.ball, s7, allow_ball_fallback=True).reference)
        pose = copy.deepcopy(self.pose)
        pose['selected_frame'] = 11
        self.assertIsNone(build_offside_reference(pose, self.ball, s7, allow_ball_fallback=True).reference)

    def test_strict_stage9_never_reconstructs_ball_only_reference(self):
        s7 = h['stage7'](10, -1, [], status='UNRESOLVED')
        s8 = {'status': 'UNRESOLVED', 'frame_index': 10, 'ball': {'goalward_q_m': 32}}
        s9 = build_offside_position_state(self.pose, s7, s8, stage6_input=self.ball, best_effort=False)
        self.assertIsNone(s9.reference['X_world_m'])

    def test_explicit_defender_policy_classifies_both_sides_without_ball(self):
        self.ball = h['stage6'](10, extent=None, usable=False)
        s7 = self.context()
        s8 = build_offside_reference(self.pose, self.ball, s7, allow_ball_fallback=True).to_dict()
        for x, label in [(-43, 'OFFSIDE_POSITION'), (-42, 'ONSIDE'), (-41, 'ONSIDE'), (5, 'ONSIDE')]:
            pose = copy.deepcopy(self.pose)
            pose['tracks'][-1]['observations'][0]['joints_world'][0]['xyz_world_m'][0] = x
            s9 = build_offside_position_state(pose, s7, s8, best_effort=False, allow_defender_only=True)
            self.assertEqual(s9.mode, 'DEFENDER_REFERENCE')
            self.assertEqual(s9.status, 'DEGRADED')
            self.assertEqual(s9.attackers[0]['label'], label)
            self.assertEqual(s9.attackers[0]['classification_basis'], 'DEFENDER_ONLY')

    def test_defender_policy_still_requires_aligned_frames_and_player_geometry(self):
        s7 = self.context()
        s8 = build_offside_reference(self.pose, self.ball, s7, allow_ball_fallback=True).to_dict()
        pose = copy.deepcopy(self.pose)
        pose['selected_frame'] = 11
        s9 = build_offside_position_state(pose, s7, s8, best_effort=False, allow_defender_only=True)
        self.assertEqual(s9.status, 'UNRESOLVED')
        pose = copy.deepcopy(self.pose)
        pose['tracks'].pop()
        s9 = build_offside_position_state(pose, s7, s8, best_effort=False, allow_defender_only=True)
        self.assertEqual(s9.attackers[0]['label'], 'UNAVAILABLE')
        self.assertEqual(s9.attackers[0]['flag'], '?')

    def tentative_inputs(self, ball_missing=False):
        pose = copy.deepcopy(self.pose)
        pose['tracks'].append(h['track']('passer', 10, [h['joint']('nose', -30)]))
        s7 = h['stage7'](10, -1, ['g', 'd'], status='DEGRADED')
        s7['sets']['attackers'] = ['a', 'passer']
        s7['reasons'] = ['CONTACT_TENTATIVE_SPATIAL_ONLY']
        s7['toucher'] = {'track_id': 'passer', 'evidence_level': 'TENTATIVE_SPATIAL_ONLY'}
        ball = h['stage6'](10, extent=None, usable=False) if ball_missing else self.ball
        s8 = build_offside_reference(pose, ball, s7, allow_ball_fallback=True).to_dict()
        return pose, s7, s8

    def test_tentative_contact_classifies_instead_of_hiding_every_attacker(self):
        pose, s7, s8 = self.tentative_inputs()
        strict = build_offside_position_state(pose, s7, s8, best_effort=False)
        self.assertEqual(strict.attackers[0]['label'], 'UNAVAILABLE')
        result = build_offside_position_state(pose, s7, s8, best_effort=False, allow_tentative_context=True)
        self.assertEqual(result.mode, 'TENTATIVE_CONTACT_REFERENCE')
        self.assertEqual(result.status, 'DEGRADED')
        self.assertEqual([r['label'] for r in result.attackers], ['OFFSIDE_POSITION', 'TOUCHER_EXCLUDED'])

    def test_tentative_contact_with_missing_ball_uses_defender_policy(self):
        pose, s7, s8 = self.tentative_inputs(ball_missing=True)
        result = build_offside_position_state(pose, s7, s8, best_effort=False,
                                             allow_defender_only=True, allow_tentative_context=True)
        self.assertEqual(result.mode, 'DEFENDER_REFERENCE')
        self.assertEqual(result.attackers[0]['label'], 'OFFSIDE_POSITION')

    def test_tentative_policy_does_not_bypass_unrelated_blockers(self):
        pose, s7, s8 = self.tentative_inputs()
        s8['reasons'].append('COORDINATE_FRAME_MISMATCH')
        result = build_offside_position_state(pose, s7, s8, best_effort=False, allow_tentative_context=True)
        self.assertEqual(result.status, 'UNRESOLVED')
        self.assertEqual(result.attackers[0]['label'], 'UNAVAILABLE')
        pose, s7, s8 = self.tentative_inputs()
        s8['frame_index'] = 11
        result = build_offside_position_state(pose, s7, s8, best_effort=False, allow_tentative_context=True)
        self.assertEqual(result.attackers[0]['label'], 'UNAVAILABLE')


if __name__ == '__main__':
    unittest.main()
