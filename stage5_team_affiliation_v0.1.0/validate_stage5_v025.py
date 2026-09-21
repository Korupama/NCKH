from __future__ import annotations

import json

import stage5_team_affiliation
from stage5_team_affiliation.residual_calibration import (
    GOALKEEPER_GOAL_DISTANCES,
    GOALKEEPER_TOP2_RATES,
)


def main():
    checks = {
        'top2_grid_reaches_0_05': min(GOALKEEPER_TOP2_RATES) == 0.05,
        'top2_grid_contains_previous_boundary': 0.20 in GOALKEEPER_TOP2_RATES,
        'goal_distance_grid_extends_past_25m': max(GOALKEEPER_GOAL_DISTANCES) > 25.0,
    }
    report = {
        'status': 'PASS' if all(checks.values()) else 'FAIL',
        'package_version': stage5_team_affiliation.__version__,
        'schema_version': 'stage5-policy-validation-0.2.5',
        'checks': checks,
    }
    print(json.dumps(report, indent=2))
    if report['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()

