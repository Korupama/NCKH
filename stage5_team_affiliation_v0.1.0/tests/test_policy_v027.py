from __future__ import annotations

import hashlib
import json

import pytest

from stage5_team_affiliation.heldout_v027 import (
    heldout_component_gate,
    load_heldout_candidate,
)
from stage5_team_affiliation.residual_config import ResidualConfig


def _candidate_files(tmp_path):
    dataset = tmp_path / "GSR"
    dataset.mkdir()
    cfg = ResidualConfig(residual_player_recovery_enabled=True)
    config_path = tmp_path / "provisional_residual_config.json"
    config_path.write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
    digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
    report = {
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
    }
    report_path = tmp_path / "calibration_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return dataset, config_path, report_path


def test_train_calibrated_component_candidate_is_accepted_for_valid(tmp_path) -> None:
    dataset, config_path, report_path = _candidate_files(tmp_path)
    cfg, provenance = load_heldout_candidate(
        config_path=config_path,
        calibration_report_path=report_path,
        dataset_root=dataset,
        evaluation_split="valid",
    )
    assert cfg.residual_player_recovery_enabled is True
    assert provenance["calibration_split"] == "train"
    assert provenance["evaluation_split"] == "valid"


def test_config_tampering_is_rejected(tmp_path) -> None:
    dataset, config_path, report_path = _candidate_files(tmp_path)
    data = json.loads(config_path.read_text(encoding="utf-8"))
    data["residual_player_min_margin"] = 0.99
    config_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        load_heldout_candidate(
            config_path=config_path,
            calibration_report_path=report_path,
            dataset_root=dataset,
            evaluation_split="valid",
        )


def test_train_cannot_be_used_as_heldout_split(tmp_path) -> None:
    dataset, config_path, report_path = _candidate_files(tmp_path)
    with pytest.raises(ValueError, match="VALID-only"):
        load_heldout_candidate(
            config_path=config_path,
            calibration_report_path=report_path,
            dataset_root=dataset,
            evaluation_split="train",
        )


def test_component_gate_excludes_unfrozen_branches() -> None:
    summary = {
        "roles": {
            "player": {"precision": 0.98},
            "goalkeeper": {"f1": 0.91},
        },
        "referee": {"team_contamination_rate": 0.04},
    }
    result = heldout_component_gate(
        summary,
        {"outfield_drop_less_than_0_5pp": True},
        complete_valid_split=True,
    )
    assert result["status"] == "PASS"
    assert result["component_accuracy_validated"] is True
    assert result["full_stage5_research_accuracy_frozen"] is False
    assert "goalkeeper_team_assignment" in result["excluded_from_this_gate"]


def test_component_gate_rejects_referee_contamination_above_limit() -> None:
    summary = {
        "roles": {
            "player": {"precision": 0.99},
            "goalkeeper": {"f1": 0.95},
        },
        "referee": {"team_contamination_rate": 0.051},
    }
    result = heldout_component_gate(
        summary,
        {"outfield_drop_less_than_0_5pp": True},
        complete_valid_split=True,
    )
    assert result["status"] == "FAIL"
    assert result["checks"]["referee_to_player_rate_le_5pct"] is False

