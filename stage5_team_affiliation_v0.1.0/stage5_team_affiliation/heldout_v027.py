"""Leakage-safe loading of a TRAIN-calibrated Stage 5 held-out candidate."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from .residual_config import ResidualConfig


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _same_path(left: str | Path, right: str | Path) -> bool:
    return os.path.normcase(str(Path(left).expanduser().resolve())) == os.path.normcase(
        str(Path(right).expanduser().resolve())
    )


def load_heldout_candidate(
    *,
    config_path: str | Path,
    calibration_report_path: str | Path,
    dataset_root: str | Path,
    evaluation_split: str,
) -> tuple[ResidualConfig, dict]:
    """Load a config only when its TRAIN provenance permits held-out VALID use."""
    if str(evaluation_split).lower() != "valid":
        raise ValueError("Frozen held-out evaluation is VALID-only")

    config_path = Path(config_path).expanduser().resolve()
    report_path = Path(calibration_report_path).expanduser().resolve()
    config_data = json.loads(config_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    cfg = ResidualConfig(**config_data)
    cfg.validate()

    if report.get("status") != "COMPLETE" or str(report.get("split", "")).lower() != "train":
        raise ValueError("Held-out config must come from a COMPLETE TRAIN calibration")
    if not report.get("complete_train_split"):
        raise ValueError("Partial TRAIN calibration cannot authorize held-out evaluation")
    if not _same_path(report.get("dataset_root", ""), dataset_root):
        raise ValueError("Calibration and held-out evaluation must use the same GSR dataset root")
    if report.get("selected_config_file") != config_path.name:
        raise ValueError("Config filename does not match calibration_report.selected_config_file")

    contract = report.get("heldout_evaluation_contract") or {}
    expected_hash = contract.get("config_sha256")
    if not expected_hash:
        raise ValueError("Calibration report lacks the config hash contract")
    actual_hash = sha256_file(config_path)
    if actual_hash != expected_hash:
        raise ValueError("Residual config hash does not match its TRAIN calibration report")
    if contract.get("allowed_split") != "valid":
        raise ValueError("Calibration report does not authorize VALID evaluation")

    components = report.get("component_freezes") or {}
    if not components.get("goalkeeper_role"):
        raise ValueError("Goalkeeper-role thresholds were not safely calibrated on TRAIN")
    if cfg.residual_player_recovery_enabled and not components.get("player_recovery"):
        raise ValueError("Player recovery is enabled without a passing TRAIN component gate")
    if cfg.residual_referee_appearance_recovery_enabled and not components.get("referee_recovery"):
        raise ValueError("Referee recovery is enabled without a passing TRAIN component gate")
    goalkeeper_team_enabled = (
        cfg.defensive_tail_assignment_enabled or cfg.defended_half_assignment_enabled
    )
    if goalkeeper_team_enabled and not components.get("goalkeeper_team"):
        raise ValueError("Goalkeeper-team assignment is enabled without a passing TRAIN component gate")

    provenance = {
        "schema_version": "stage5-heldout-provenance-0.2.8",
        "calibration_report": str(report_path),
        "calibration_report_sha256": sha256_file(report_path),
        "residual_config": str(config_path),
        "residual_config_sha256": actual_hash,
        "calibration_split": "train",
        "evaluation_split": "valid",
        "complete_train_split": True,
        "component_freezes": components,
        "full_research_freeze_before_valid": bool(report.get("research_accuracy_frozen")),
    }
    return cfg, provenance


def heldout_component_gate(summary: dict, baseline_gates: dict, *, complete_valid_split: bool) -> dict:
    """Gate only the components that TRAIN calibration actually enabled."""
    player = (summary.get("roles") or {}).get("player") or {}
    goalkeeper = (summary.get("roles") or {}).get("goalkeeper") or {}
    referee = summary.get("referee") or {}
    checks = {
        "complete_valid_split": bool(complete_valid_split),
        "outfield_drop_less_than_0_5pp": baseline_gates.get(
            "outfield_drop_less_than_0_5pp") is True,
        "player_precision_ge_97pct": (
            player.get("precision") is not None and player["precision"] >= 0.97),
        "referee_to_player_rate_le_5pct": (
            referee.get("team_contamination_rate") is not None
            and referee["team_contamination_rate"] <= 0.05),
        "goalkeeper_role_f1_ge_90pct": (
            goalkeeper.get("f1") is not None and goalkeeper["f1"] >= 0.90),
    }
    passed = all(checks.values())
    return {
        "status": "PASS" if passed else "FAIL",
        "checks": checks,
        "component_accuracy_validated": passed,
        "full_stage5_research_accuracy_frozen": False,
        "excluded_from_this_gate": ["referee_recovery", "goalkeeper_team_assignment"],
    }
