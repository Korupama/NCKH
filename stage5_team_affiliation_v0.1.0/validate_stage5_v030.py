from __future__ import annotations

import json
import numpy as np

from stage5_team_affiliation import __version__
from stage5_team_affiliation.residual_config import ResidualConfig
from stage5_team_affiliation.residual_pipeline import combine_region_features


def main() -> None:
    config = ResidualConfig(multiregion_appearance_enabled=True)
    config.validate()
    composite = combine_region_features(
        np.asarray([1.0, 0.0], dtype=np.float32),
        np.asarray([0.0, 1.0], dtype=np.float32),
        config.multiregion_torso_weight,
        config.multiregion_lower_weight,
    )
    checks = {
        "package_version_is_v030_or_later": tuple(
            map(int, __version__.split('.'))) >= (0, 3, 0),
        "multiregion_is_opt_in": ResidualConfig().multiregion_appearance_enabled is False,
        "lower_body_is_required": config.multiregion_require_lower is True,
        "descriptor_is_unit_normalized": bool(np.isclose(np.linalg.norm(composite), 1.0)),
        "torso_distance_weight_is_0_7": bool(np.isclose(
            np.dot(composite[:2], composite[:2]), 0.7)),
        "lower_distance_weight_is_0_3": bool(np.isclose(
            np.dot(composite[2:], composite[2:]), 0.3)),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    print(json.dumps({
        "status": status,
        "package_version": __version__,
        "schema_version": "stage5-multiregion-validation-0.3.0",
        "checks": checks,
    }, indent=2))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
