"""Role/GK-team audit from stored predictions; never reads image pixels.

The diagnostic-ceiling searches intentionally use evaluation labels and must
never be exported as a frozen inference configuration.  Their purpose is to
test whether the currently stored signals are sufficient for a component.
"""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np

from . import __version__
from .gsr_benchmark import SoccerNetGSRDataset, _write_json
from .residual_calibration import (
    REFEREE_GOAL_DISTANCES,
    REFEREE_MIN_DISTANCES,
    REFEREE_MARGINS,
    REFEREE_TOP2_RATES,
    evaluate_defended_half_policy,
    evaluate_tail_policy,
    search_defended_half_policy,
    search_tail_policy,
)


def _prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
    }


def evaluate_referee_rule(rows, *, max_margin, min_distance,
                          min_goal_distance_m, max_top2_rate):
    predictions = []
    for row in rows:
        predicted = row["base_role"]
        if predicted == "unknown" and row["appearance_group"] == "RESIDUAL":
            goal_distance = row.get("median_goal_distance_m")
            top2_rate = row.get("goal_top2_rate")
            if (row["margin"] <= max_margin
                    and row["nearest_distance"] >= min_distance
                    and isinstance(goal_distance, (int, float))
                    and math.isfinite(goal_distance)
                    and goal_distance >= min_goal_distance_m
                    and isinstance(top2_rate, (int, float))
                    and math.isfinite(top2_rate)
                    and top2_rate <= max_top2_rate):
                predicted = "referee"
        predictions.append((row, predicted))
    tp = sum(row["gt_role"] == "referee" and pred == "referee"
             for row, pred in predictions)
    fp = sum(row["gt_role"] != "referee" and pred == "referee"
             for row, pred in predictions)
    fn = sum(row["gt_role"] == "referee" and pred != "referee"
             for row, pred in predictions)
    result = _prf(tp, fp, fn)
    players = [row for row, _ in predictions if row["gt_role"] == "player"]
    player_fp = sum(row["gt_role"] == "player" and pred == "referee"
                    for row, pred in predictions)
    result.update({
        "player_to_referee": player_fp,
        "player_to_referee_rate": player_fp / len(players) if players else None,
        "assigned_referees": tp + fp,
    })
    return result


def search_referee_diagnostic_ceiling(rows):
    margins = tuple(sorted(set(REFEREE_MARGINS + (0.30, 0.40, 0.50, 0.60))))
    distances = tuple(sorted(set((0.0, 0.20, 0.30) + REFEREE_MIN_DISTANCES)))
    goal_distances = tuple(sorted(set(REFEREE_GOAL_DISTANCES + (35.0, 40.0))))
    top2_rates = tuple(sorted(set((0.05,) + REFEREE_TOP2_RATES + (0.40, 0.50))))
    candidates = []
    for margin, distance, goal, top2 in itertools.product(
            margins, distances, goal_distances, top2_rates):
        metrics = evaluate_referee_rule(
            rows, max_margin=margin, min_distance=distance,
            min_goal_distance_m=goal, max_top2_rate=top2)
        candidates.append({
            "parameters": {
                "residual_referee_max_margin": margin,
                "residual_referee_min_distance": distance,
                "referee_min_goal_distance_m": goal,
                "referee_max_top2_rate": top2,
            },
            "metrics": metrics,
        })
    candidates.sort(key=lambda candidate: (
        candidate["metrics"]["f1"] or 0,
        candidate["metrics"]["precision"] or 0,
        candidate["metrics"]["recall"] or 0,
        -(candidate["metrics"]["player_to_referee_rate"] or 0),
    ), reverse=True)
    return {
        "selected": candidates[0],
        "evaluated_count": len(candidates),
        "label_use": "VALID_GT_DIAGNOSTIC_CEILING_NOT_FREEZABLE",
    }


def _stored_rows(dataset, prediction_dir, sequence_ids):
    role_rows = []
    goalkeeper_rows = []
    for sequence_id in sequence_ids:
        sequence = dataset.load(sequence_id)
        prediction = json.loads((
            prediction_dir / "sequences" / sequence_id / "prediction.json"
        ).read_text(encoding="utf-8"))
        metrics = json.loads((
            prediction_dir / "sequences" / sequence_id / "metrics.json"
        ).read_text(encoding="utf-8"))
        records = {str(row["track_id"]): row for row in prediction["tracks"]}
        gt_roles = sequence.role_by_track()
        gt_teams = sequence.team_gt_by_track()
        mapping = {str(key): int(value) for key, value in (
            metrics.get("mapping_pred_to_gt") or {}).items()}
        mapping_available = set(mapping) == {"0", "1"}
        for track_id, gt_role in gt_roles.items():
            record = records[track_id]
            distances = np.asarray(record.get("distances") or [], dtype=float)
            nearest_distance = (float(np.min(distances))
                                if distances.shape == (2,) and np.isfinite(distances).all()
                                else math.inf)
            goal = record.get("goal_context") or {}
            if str(record.get("appearance_group", "")).startswith("TEAM_"):
                base_role = "player"
            elif (record.get("stage5_role") == "goalkeeper"
                  and record.get("stage5_role_status") == "VALID"):
                base_role = "goalkeeper"
            else:
                # Player recovery is deliberately reset for this isolated
                # referee audit. Dominant-team core assignments remain players.
                base_role = "unknown"
            role_rows.append({
                "sequence_id": sequence_id,
                "track_id": track_id,
                "gt_role": gt_role,
                "appearance_group": record.get("appearance_group"),
                "base_role": base_role,
                "margin": float(record.get("margin") or 0.0),
                "nearest_distance": nearest_distance,
                "median_goal_distance_m": goal.get("median_goal_distance_m"),
                "goal_top2_rate": goal.get("top2_rate"),
            })
            if gt_role == "goalkeeper":
                assignment = record.get("goalkeeper_assignment") or {}
                goalkeeper_rows.append({
                    "sequence_id": sequence_id,
                    "track_id": track_id,
                    "gt_role": gt_role,
                    "gt_team": gt_teams.get(track_id),
                    "mapping": mapping,
                    "mapping_available": mapping_available,
                    "base_role": record.get("stage5_role"),
                    "base_role_status": record.get("stage5_role_status"),
                    "tail_deltas_m": assignment.get("tail_deltas_m") or [],
                    "half_deltas_m": assignment.get("half_deltas_m") or [],
                })
    return role_rows, goalkeeper_rows


def _current_goalkeeper_team(summary):
    group = summary["groups"]["goalkeeper"]
    return {
        "tracks": group["tracks"],
        "assigned": group["valid_predictions"],
        "correct": group["correct_valid"],
        "coverage": group["micro_coverage"],
        "selective_accuracy": group["micro_selective_accuracy"],
        "overall_accuracy": group["micro_overall_accuracy"],
    }


def audit_role_components(*, dataset_root, split, prediction_dir, output_dir):
    prediction_dir = Path(prediction_dir).resolve()
    manifest = json.loads((prediction_dir / "run_manifest.json").read_text(
        encoding="utf-8"))
    summary = json.loads((prediction_dir / "benchmark_summary.json").read_text(
        encoding="utf-8"))
    if manifest.get("status") != "COMPLETE":
        raise ValueError("Role audit requires a COMPLETE stored prediction run")
    if summary.get("split") != split:
        raise ValueError("Stored prediction split mismatch")
    dataset = SoccerNetGSRDataset(dataset_root, split)
    sequence_ids = list(manifest.get("sequence_ids") or [])
    if sequence_ids != dataset.discover():
        raise ValueError("Role audit requires the complete local split in canonical order")
    out = Path(output_dir).resolve()
    if out.exists():
        raise FileExistsError(f"Use a NEW output directory: {out}")
    out.mkdir(parents=True)

    role_rows, goalkeeper_rows = _stored_rows(
        dataset, prediction_dir, sequence_ids)
    referee_current = summary["roles"]["referee"]
    goalkeeper_role_current = summary["roles"]["goalkeeper"]
    goalkeeper_team_current = _current_goalkeeper_team(summary)
    referee_ceiling = search_referee_diagnostic_ceiling(role_rows)
    tail_ceiling = search_tail_policy(
        goalkeeper_rows, min_observations=5, minimum_overall_accuracy=0.80)
    half_ceiling = search_defended_half_policy(
        goalkeeper_rows, min_observations=5, minimum_overall_accuracy=0.80)
    best_goalkeeper_ceiling = max(
        (("DEFENSIVE_TAIL", tail_ceiling["selected"]),
         ("DEFENDED_HALF", half_ceiling["selected"])),
        key=lambda item: item[1]["metrics"]["overall_accuracy"] or 0)

    checks = {
        "goalkeeper_role_f1_ge_90pct": (
            (goalkeeper_role_current.get("f1") or 0) >= 0.90),
        "referee_role_f1_ge_80pct": (
            (referee_current.get("f1") or 0) >= 0.80),
        "goalkeeper_team_overall_accuracy_ge_80pct": (
            (goalkeeper_team_current.get("overall_accuracy") or 0) >= 0.80),
    }
    ceiling_checks = {
        "referee_role_f1_ge_80pct": (
            (referee_ceiling["selected"]["metrics"].get("f1") or 0) >= 0.80),
        "goalkeeper_team_overall_accuracy_ge_80pct": (
            (best_goalkeeper_ceiling[1]["metrics"].get("overall_accuracy") or 0) >= 0.80),
    }
    report = {
        "schema_version": "stage5-role-component-audit-0.3.2",
        "package_version": __version__,
        "status": "COMPLETE",
        "split": split,
        "sequences": len(sequence_ids),
        "source_prediction_dir": str(prediction_dir),
        "outfield_metrics_intentionally_excluded": True,
        "current": {
            "goalkeeper_role": goalkeeper_role_current,
            "referee_role": referee_current,
            "goalkeeper_team": goalkeeper_team_current,
            "role_component_gate": {
                "status": "PASS" if all(checks.values()) else "FAIL",
                "checks": checks,
            },
        },
        "diagnostic_ceiling": {
            "warning": (
                "Uses VALID labels to measure signal ceiling; never freeze or "
                "select production thresholds from this section."),
            "referee_role": referee_ceiling,
            "goalkeeper_team": {
                "best_method": best_goalkeeper_ceiling[0],
                "best": best_goalkeeper_ceiling[1],
                "defensive_tail": tail_ceiling,
                "defended_half": half_ceiling,
            },
            "checks": ceiling_checks,
        },
    }
    _write_json(out / "role_component_audit.json", report)
    return report
