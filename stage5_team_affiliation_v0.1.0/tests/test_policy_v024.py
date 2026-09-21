from stage5_team_affiliation.goalkeeper_team_v024 import assign_by_defended_half
from stage5_team_affiliation.residual_config import ResidualConfig
from stage5_team_affiliation.residual_pipeline import recover_residual_appearance
from stage5_team_affiliation.residual_calibration import (
    GOALKEEPER_GOAL_DISTANCES,
    GOALKEEPER_TOP2_RATES,
)


def test_expanded_goalkeeper_grid_moves_past_previous_boundaries():
    assert max(GOALKEEPER_GOAL_DISTANCES) > 25.0
    assert min(GOALKEEPER_TOP2_RATES) < 0.4


def test_player_recovery_can_be_enabled_without_referee_recovery():
    records = {
        'player': {
            'appearance_group': 'RESIDUAL', 'stage5_role': 'unknown_residual',
            'stage5_role_status': 'UNKNOWN', 'distances': [0.2, 0.7],
            'margin': 0.71, 'team_id': None, 'team_status': 'UNKNOWN',
        },
        'referee': {
            'appearance_group': 'RESIDUAL', 'stage5_role': 'unknown_residual',
            'stage5_role_status': 'UNKNOWN', 'distances': [0.8, 0.84],
            'margin': 0.05, 'team_id': None, 'team_status': 'UNKNOWN',
            'goal_context': {'median_goal_distance_m': 30.0, 'top2_rate': 0.1},
        },
    }
    config = ResidualConfig(
        residual_player_recovery_enabled=True,
        residual_player_min_margin=0.3,
        residual_player_max_distance=0.5,
    )
    recover_residual_appearance(records, config)
    assert records['player']['stage5_role'] == 'player'
    assert records['referee']['stage5_role_status'] == 'UNKNOWN'


def test_defended_half_distribution_assigns_goalkeeper_team():
    points = {'gk': {}, 'p0a': {}, 'p0b': {}, 'p1a': {}, 'p1b': {}}
    for frame in range(8):
        points['gk'][frame] = [48.0, 0.0]
        points['p0a'][frame] = [30.0, 5.0]
        points['p0b'][frame] = [32.0, -5.0]
        points['p1a'][frame] = [-10.0, 4.0]
        points['p1b'][frame] = [-12.0, -4.0]
    result = assign_by_defended_half(
        'gk', 1, points,
        {'p0a': 0, 'p0b': 0, 'p1a': 1, 'p1b': 1},
        ResidualConfig(),
    )
    assert result['team_status'] == 'VALID'
    assert result['team_id'] == 0
    assert len(result['half_deltas_m']) == 8

