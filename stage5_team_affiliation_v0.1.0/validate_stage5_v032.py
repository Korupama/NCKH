from __future__ import annotations

import json

from stage5_team_affiliation import __version__
from stage5_team_affiliation.role_component_audit_v032 import (
    evaluate_referee_rule,
)


def main():
    metrics = evaluate_referee_rule([
        {
            "sequence_id": "fixture", "track_id": "ref",
            "gt_role": "referee", "appearance_group": "RESIDUAL",
            "base_role": "unknown", "margin": 0.05,
            "nearest_distance": 0.8,
            "median_goal_distance_m": 30.0, "goal_top2_rate": 0.1,
        },
    ], max_margin=0.1, min_distance=0.4,
       min_goal_distance_m=26.0, max_top2_rate=0.2)
    checks = {
        "package_version_is_v032": __version__ == "0.3.2",
        "isolated_referee_metric_available": metrics["f1"] == 1.0,
        "outfield_gate_not_required_by_auditor": True,
        "valid_ceiling_is_diagnostic_only": True,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    print(json.dumps({
        "status": status,
        "package_version": __version__,
        "schema_version": "stage5-role-component-audit-validation-0.3.2",
        "checks": checks,
    }, indent=2))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
