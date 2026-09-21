"""Re-evaluate a residual policy from stored predictions without reading images."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

from . import __version__
from .gsr_benchmark import SoccerNetGSRDataset, _write_json
from .heldout_v027 import heldout_component_gate
from .residual_benchmark import _baseline_gate
from .residual_evaluation import aggregate_residual, evaluate_residual_sequence
from .residual_pipeline import recover_residual_appearance


def _same_path(left, right) -> bool:
    return os.path.normcase(str(Path(left).resolve())) == os.path.normcase(
        str(Path(right).resolve()))


def reset_and_recover_prediction(prediction: dict, config) -> dict:
    """Remove only the old appearance recovery, then apply the new policy."""
    updated = copy.deepcopy(prediction)
    records = {str(record["track_id"]): record for record in updated["tracks"]}
    for record in records.values():
        if record.get("stage5_role_method") != "RESIDUAL_HIGH_MARGIN_PLAYER_RECOVERY":
            continue
        record.update(
            stage5_role="unknown_residual",
            stage5_role_status="UNKNOWN",
            stage5_role_method="APPEARANCE_RESIDUAL",
            role_decision_reason="CACHED_POLICY_RESET",
            team_id=None,
            team_status="UNKNOWN",
            assignment_method="UNAVAILABLE",
        )
    recover_residual_appearance(records, config)
    updated["tracks"] = list(records.values())
    updated["cache_reevaluation"] = {
        "schema_version": "stage5-residual-cache-reevaluation-0.2.8",
        "source_detections_and_features_reused": True,
        "images_read": False,
        "policy_only": True,
    }
    return updated


def reevaluate_residual_cache(
        *, dataset_root, split, source_dir, output_dir, residual_config,
        baseline_summary, calibration_provenance, progress_every=10):
    if str(split).lower() != "valid":
        raise ValueError("Cached development reevaluation is VALID-only")
    residual_config.validate()
    source = Path(source_dir).resolve()
    manifest = json.loads((source / "run_manifest.json").read_text(encoding="utf-8"))
    source_summary = json.loads(
        (source / "benchmark_summary.json").read_text(encoding="utf-8"))
    source_cfg = source_summary.get("configuration") or {}
    if bool(source_cfg.get("multiregion_appearance_enabled", False)) != bool(
            residual_config.multiregion_appearance_enabled):
        raise ValueError(
            "Cache appearance descriptor mismatch; multiregion changes require "
            "a fresh benchmark run that reads the images")
    if residual_config.multiregion_appearance_enabled:
        for key in ("multiregion_torso_weight", "multiregion_lower_weight",
                    "multiregion_require_lower"):
            if source_cfg.get(key) != getattr(residual_config, key):
                raise ValueError(
                    f"Cache multiregion setting mismatch for {key}; run fresh feature extraction")
    if manifest.get("status") != "COMPLETE" or manifest.get("variant") != "V3":
        raise ValueError("Cache source must be a COMPLETE residual-v3 run")
    if source_summary.get("split") != split:
        raise ValueError("Cache source split mismatch")
    if not _same_path(source_summary.get("dataset_root", ""), dataset_root):
        raise ValueError("Cache source dataset root mismatch")

    dataset = SoccerNetGSRDataset(dataset_root, split)
    discovered = dataset.discover()
    ids = list(manifest.get("sequence_ids") or [])
    if ids != discovered or len(manifest.get("completed_sequences") or []) != len(ids):
        raise ValueError("Cache source must cover the complete local VALID split")

    out = Path(output_dir).resolve()
    if out.exists():
        raise FileExistsError(f"Use a NEW output directory: {out}")
    out.mkdir(parents=True)
    output_manifest = {
        "status": "RUNNING",
        "split": split,
        "variant": "V3",
        "package_version": __version__,
        "evaluation_mode": "CACHED_VALID_DEVELOPMENT_REEVALUATION",
        "source_dir": str(source),
        "sequence_ids": ids,
        "completed_sequences": [],
        "calibration_provenance": calibration_provenance,
        "images_read": False,
        "independent_heldout_claim": False,
    }
    _write_json(out / "run_manifest.json", output_manifest)
    rows = []
    try:
        for index, sid in enumerate(ids, 1):
            sequence = dataset.load(sid)
            prediction = json.loads(
                (source / "sequences" / sid / "prediction.json").read_text(
                    encoding="utf-8"))
            prediction = reset_and_recover_prediction(prediction, residual_config)
            metrics = evaluate_residual_sequence(sequence, prediction)
            metrics["method"] = "residual-v3-cache-v028"
            rows.append(metrics)
            _write_json(out / "sequences" / sid / "prediction.json", prediction)
            _write_json(out / "sequences" / sid / "metrics.json", metrics)
            output_manifest["completed_sequences"].append(sid)
            _write_json(out / "run_manifest.json", output_manifest)
            if progress_every and (index % progress_every == 0 or index == len(ids)):
                print(
                    f"[{index}/{len(ids)}] {sid}: "
                    f"overall={metrics['all_team_tracks']['overall_accuracy']:.4f}",
                    flush=True,
                )
        summary = aggregate_residual(rows, "residual-v3-cache-v028", split)
        baseline_gates = _baseline_gate(
            baseline_summary, rows, summary, dataset_root, split)
        component_gate = heldout_component_gate(
            summary, baseline_gates, complete_valid_split=True)
        summary.update({
            "package_version": __version__,
            "dataset_root": str(Path(dataset_root).resolve()),
            "configuration": residual_config.to_dict(),
            "evaluation_mode": "CACHED_VALID_DEVELOPMENT_REEVALUATION",
            "source_dir": str(source),
            "calibration_provenance": calibration_provenance,
            "gates": {
                **baseline_gates,
                "development_component_gate": component_gate,
                "independent_heldout_claim": False,
            },
            "protocol": {
                "schema_version": "stage5-residual-cache-reevaluation-0.2.8",
                "images_read": False,
                "detections_features_geometry_reused": True,
                "policy_reapplied": True,
                "valid_previously_observed": True,
                "eligible_for_final_unbiased_claim": False,
            },
        })
        _write_json(out / "benchmark_summary.json", summary)
        (out / "sequence_metrics.jsonl").write_text(
            "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        output_manifest.update(status="COMPLETE", real_accuracy_claim=False)
        _write_json(out / "run_manifest.json", output_manifest)
        return summary
    except Exception as exc:
        output_manifest.update(status="FAILED", error=str(exc))
        _write_json(out / "run_manifest.json", output_manifest)
        raise
