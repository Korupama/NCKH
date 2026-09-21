from stage5_team_affiliation.residual_calibration import (
    GOALKEEPER_GOAL_DISTANCES,
    GOALKEEPER_TOP2_RATES,
    search_goalkeeper_role_policy,
)


def test_final_top2_grid_extends_below_previous_boundary():
    assert GOALKEEPER_TOP2_RATES[:4] == (0.05, 0.10, 0.15, 0.20)
    assert 25.0 < max(GOALKEEPER_GOAL_DISTANCES)


def test_goalkeeper_search_reports_boundary_state():
    rows = [
        {
            'sequence_id': 'seq',
            'track_id': 'gk',
            'gt_role': 'goalkeeper',
            'appearance_group': 'RESIDUAL',
            'goal_sign': 1,
            'goal_observations': 20,
            'goal_rank_frames': 20,
            'goal_top2_rate': 1.0,
            'median_goal_distance_m': 5.0,
            'median_goalward_depth_m': 47.5,
        },
        {
            'sequence_id': 'seq',
            'track_id': 'player',
            'gt_role': 'player',
            'appearance_group': 'TEAM_0',
            'goal_sign': None,
            'goal_observations': 0,
            'goal_rank_frames': 0,
            'goal_top2_rate': None,
            'median_goal_distance_m': None,
            'median_goalward_depth_m': None,
        },
    ]
    result = search_goalkeeper_role_policy(
        rows, min_observations=5, ordering_margin_m=0.5,
        min_precision=0.0, min_f1=0.0,
        max_player_goalkeeper_rate=1.0,
    )
    assert 'selected_at_grid_boundary' in result
    assert result['requires_grid_expansion'] == any(
        result['selected_at_grid_boundary'].values())
    assert result['grid']['min_top2_rate'][0] == 0.05


def test_goalkeeper_search_prefers_interior_of_equivalent_best_plateau():
    rows = [
        {
            'sequence_id': 'seq', 'track_id': 'gk',
            'gt_role': 'goalkeeper', 'appearance_group': 'RESIDUAL',
            'goal_sign': 1, 'goal_observations': 20, 'goal_rank_frames': 20,
            'goal_top2_rate': 1.0, 'median_goal_distance_m': 5.0,
            'median_goalward_depth_m': 47.5,
        },
        {
            'sequence_id': 'seq', 'track_id': 'player',
            'gt_role': 'player', 'appearance_group': 'TEAM_0',
            'goal_sign': None, 'goal_observations': 0, 'goal_rank_frames': 0,
            'goal_top2_rate': None, 'median_goal_distance_m': None,
            'median_goalward_depth_m': None,
        },
    ]
    result = search_goalkeeper_role_policy(
        rows, min_observations=5, ordering_margin_m=0.5,
        min_precision=0.0, min_f1=0.0,
        max_player_goalkeeper_rate=1.0)
    assert result['equivalent_best_plateau_size'] > 1
    assert result['selected_at_grid_boundary'] == {
        'goal_distance_m': False, 'min_top2_rate': False}
    assert result['requires_grid_expansion'] is False
