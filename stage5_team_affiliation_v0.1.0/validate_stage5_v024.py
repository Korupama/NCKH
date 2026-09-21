from __future__ import annotations

import json

import stage5_team_affiliation
from stage5_team_affiliation.goalkeeper_team_v024 import assign_by_defended_half
from stage5_team_affiliation.residual_calibration import (
    GOALKEEPER_GOAL_DISTANCES,
    GOALKEEPER_TOP2_RATES,
)
from stage5_team_affiliation.residual_config import ResidualConfig


def main():
    points = {'gk': {}, 'team0': {}, 'team1': {}}
    for frame in range(8):
        points['gk'][frame] = [48.0, 0.0]
        points['team0'][frame] = [30.0, 0.0]
        points['team1'][frame] = [-10.0, 0.0]
    assignment = assign_by_defended_half(
        'gk', 1, points, {'team0': 0, 'team1': 1}, ResidualConfig())
    checks = {
        'goalkeeper_grid_expanded': (
            max(GOALKEEPER_GOAL_DISTANCES) > 25.0
            and min(GOALKEEPER_TOP2_RATES) < 0.4
        ),
        'split_recovery_switches': (
            hasattr(ResidualConfig(), 'residual_player_recovery_enabled')
            and hasattr(ResidualConfig(), 'residual_referee_appearance_recovery_enabled')
        ),
        'defended_half_assignment': (
            assignment['team_status'] == 'VALID' and assignment['team_id'] == 0
        ),
        'goalkeeper_team_policies_mutually_exclusive': True,
    }
    try:
        ResidualConfig(
            defensive_tail_assignment_enabled=True,
            defended_half_assignment_enabled=True,
        ).validate()
        checks['goalkeeper_team_policies_mutually_exclusive'] = False
    except ValueError:
        pass
    report = {
        'status': 'PASS' if all(checks.values()) else 'FAIL',
        'package_version': stage5_team_affiliation.__version__,
        'schema_version': 'stage5-policy-validation-0.2.4',
        'checks': checks,
    }
    print(json.dumps(report, indent=2))
    if report['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()

