from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

from stage5_team_affiliation import __version__
from stage5_team_affiliation.heldout_v027 import (
    heldout_component_gate,
    load_heldout_candidate,
)
from stage5_team_affiliation.residual_config import ResidualConfig


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="stage5_v027_") as raw:
        root = Path(raw)
        dataset = root / "GSR"
        dataset.mkdir()
        cfg = ResidualConfig(residual_player_recovery_enabled=True)
        config_path = root / "provisional_residual_config.json"
        config_path.write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
        digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
        report_path = root / "calibration_report.json"
        report_path.write_text(json.dumps({
            "status": "COMPLETE",
            "split": "train",
            "dataset_root": str(dataset),
            "complete_train_split": True,
            "selected_config_file": config_path.name,
            "component_freezes": {
                "goalkeeper_role": True,
                "player_recovery": True,
                "referee_recovery": False,
                "goalkeeper_team": False,
            },
            "heldout_evaluation_contract": {
                "allowed_split": "valid",
                "config_sha256": digest,
            },
            "research_accuracy_frozen": False,
        }, indent=2), encoding="utf-8")
        loaded, provenance = load_heldout_candidate(
            config_path=config_path,
            calibration_report_path=report_path,
            dataset_root=dataset,
            evaluation_split="valid",
        )

    gate = heldout_component_gate({
        "roles": {
            "player": {"precision": 0.98},
            "goalkeeper": {"f1": 0.91},
        },
        "referee": {"team_contamination_rate": 0.04},
    }, {"outfield_drop_less_than_0_5pp": True}, complete_valid_split=True)
    checks = {
        "package_version_supports_v027": tuple(
            int(part) for part in __version__.split(".")) >= (0, 2, 7),
        "train_to_valid_provenance_verified": (
            provenance["calibration_split"] == "train"
            and provenance["evaluation_split"] == "valid"
        ),
        "player_component_config_loaded": loaded.residual_player_recovery_enabled,
        "heldout_component_gate_passes_safe_fixture": gate["status"] == "PASS",
        "full_stage_freeze_remains_false": (
            gate["full_stage5_research_accuracy_frozen"] is False),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    print(json.dumps({
        "status": status,
        "package_version": __version__,
        "schema_version": "stage5-heldout-validation-0.2.7",
        "checks": checks,
    }, indent=2))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
