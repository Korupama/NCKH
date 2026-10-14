"""Cross-stage policy: a GK is ranked, never removed or classified as attacker."""
import sys
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for folder in ('stage5_team_affiliation_v0.1.0', 'stage7_game_state_v0.1.0',
               'stage8_offside_reference_v0.1.0', 'stage9_offside_position_v0.1.0'):
    sys.path.insert(0, str(ROOT / folder))

from stage7_game_state.core import build_game_state_context
from stage8_offside_reference.core import build_offside_reference
from stage9_offside_position.core import build_offside_position_state
from stage5_team_affiliation.goalkeeper import assign_goalkeeper

h = runpy.run_path(str(ROOT / 'stage8_offside_reference_v0.1.0/tests/helpers.py'))


@pytest.mark.parametrize('direction', [-1, 1])
@pytest.mark.parametrize('gk_x,expected_rank,reference', [(50, 1, 47), (45, 2, 45)])
@pytest.mark.parametrize('old_team', [None, 0, 1])
def test_keeper_is_in_defender_ranking(direction, gk_x, expected_rank, reference, old_team):
    ball = h['stage6'](10, extent=(direction*30-.11, direction*30+.11))
    ball['contact_track_id'] = 'a'
    teams = {'players': [dict(track_id='a', role='player', team_id=1),
                         dict(track_id='d', role='player', team_id=0),
                         dict(track_id='g', role='goalkeeper', team_id=old_team)]}
    s7 = build_game_state_context({'view': {'centre_ray_pitch_hit_m': [direction*20, 0, 0]}}, teams, ball).to_dict()
    assert s7['status'] == 'VALID'
    assert set(s7['sets']['opponents']) == {'d', 'g'}
    assert s7['sets']['attackers'] == ['a']
    assert s7['diagnostics']['goalkeeper_assignments']['g']['team_id'] == 0
    pose = h['stage4'](10, [h['track'](tid, 10, [h['joint']('left_heel', direction*x)], role=role)
                           for tid, x, role in [('a', 48, 'player'), ('d', 47, 'player'), ('g', gk_x, 'goalkeeper')]])
    s8 = build_offside_reference(pose, ball, s7).to_dict()
    assert s8['status'] == 'VALID'
    assert next(r['rank'] for r in s8['opponent_ranking'] if r['track_id'] == 'g') == expected_rank
    assert s8['reference']['goalward_q_m'] == reference
    s9 = build_offside_position_state(pose, s7, s8, best_effort=False).to_dict()
    assert s9['status'] == 'VALID'
    assert [r['track_id'] for r in s9['attackers']] == ['a']
    gk = next(r for r in s9['opponents'] if r['track_id'] == 'g')
    assert gk['rank'] == expected_rank
    assert gk['second_last'] == (expected_rank == 2)


def test_stage5_never_guesses_keeper_cluster():
    # Deliberately unusable appearance/config: fixed policy needs neither.
    result = assign_goalkeeper(None, {}, None)
    assert result['team_side'] == 'DEFENDING'
    assert result['team_id'] is None
    assert result['method'] == 'GOALKEEPER_ALWAYS_DEFENDING'


def test_unresolved_contact_and_multiple_unknown_keepers_still_defenders():
    teams = {'players': [dict(track_id=tid, role='GK', team_id=None) for tid in ['g1', 'g2']]}
    s7 = build_game_state_context({}, teams, {}).to_dict()
    assert set(s7['sets']['opponents']) == {'g1', 'g2'}
    assert s7['sets']['attackers'] == []
    assert s7['status'] == 'UNRESOLVED'


def test_stale_context_cannot_classify_goalkeeper():
    pose = h['stage4'](10, [h['track']('g', 10, [h['joint']('left_heel', 50)], role='goalkeeper')])
    old_context = h['stage7'](10, 1, ['d1', 'd2'])
    old_context['sets']['attackers'] = ['g']
    ball = h['stage6'](10)
    s8 = build_offside_reference(pose, ball, old_context).to_dict()
    assert 'GOALKEEPER_IN_ATTACKERS_RERUN_STAGE7' in s8['reasons']
    with pytest.raises(ValueError, match='GOALKEEPER_IN_ATTACKERS'):
        build_offside_position_state(pose, old_context, s8)
