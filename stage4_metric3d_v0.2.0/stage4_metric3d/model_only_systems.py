from __future__ import annotations

"""Fair comparison of independent Stage-4 model-only systems."""

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .evaluation import mpjpe, procrustes_mpjpe, root_aligned_mpjpe
from .model_only import load_model_only_manifest, sha256_file
from .model_only_evaluation import _load_and_validate_output, _stats


MODEL_ONLY_SYSTEMS_SCHEMA = "stage4-model-only-systems-1.0"


def _load_system_manifest(path: str | Path) -> tuple[Path, dict[str, Any], str]:
    manifest_path = Path(path).expanduser().resolve()
    raw = manifest_path.read_bytes()
    try:
        data = json.loads(raw)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot parse model-only systems manifest: {manifest_path}") from exc
    if not isinstance(data, dict) or data.get("schema_version") != MODEL_ONLY_SYSTEMS_SCHEMA:
        raise ValueError(f"Unsupported model-only systems schema: {manifest_path}")
    return manifest_path, data, hashlib.sha256(raw).hexdigest()


def _resolve(base: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty path")
    candidate = Path(value).expanduser()
    resolved = (candidate if candidate.is_absolute() else base / candidate).resolve()
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise FileNotFoundError(f"Missing {label}: {resolved}")
    return resolved


def _system_metric(pred: np.ndarray, gt: np.ndarray, visible: np.ndarray) -> dict[str, Any]:
    mask = np.asarray(visible, dtype=bool) & np.isfinite(pred).all(axis=1) & np.isfinite(gt).all(axis=1)
    count = int(mask.sum())
    if count < 3:
        return {"status": "NOT_EVALUATED_INSUFFICIENT_VALID_JOINTS", "valid_joint_count": count}
    p = pred.astype(np.float64).copy()
    g = gt.astype(np.float64).copy()
    p[~mask] = np.nan
    g[~mask] = np.nan
    x = np.abs(p[mask, 0] - g[mask, 0])
    return {
        "status": "EVALUATED",
        "valid_joint_count": count,
        "global_mpjpe_m": mpjpe(p, g),
        "root_aligned_mpjpe_m": root_aligned_mpjpe(p, g),
        "pa_mpjpe_m": procrustes_mpjpe(p, g),
        "longitudinal_abs_error_m": _stats(x),
    }


def compare_model_only_systems(*, systems_manifest_path: str | Path, report_path: str | Path | None = None) -> dict[str, Any]:
    systems_path, raw, systems_sha256 = _load_system_manifest(systems_manifest_path)
    base = systems_path.parent
    benchmark_path = _resolve(base, raw.get("benchmark_manifest"), "benchmark_manifest")
    benchmark = load_model_only_manifest(benchmark_path)
    if raw.get("benchmark_manifest_sha256") not in (None, benchmark.sha256):
        raise ValueError("benchmark_manifest_sha256 does not match benchmark_manifest")
    entries = raw.get("systems")
    if not isinstance(entries, list) or not entries:
        raise ValueError("systems must be a non-empty list")

    loaded: list[dict[str, Any]] = []
    names: set[str] = set()
    expected_ids = tuple(record.record_id for record in benchmark.records)
    for index, entry in enumerate(entries):
        label = f"systems[{index}]"
        if not isinstance(entry, dict):
            raise ValueError(f"{label} must be an object")
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError(f"{label}.name must be non-empty and unique")
        names.add(name)
        variant = entry.get("variant", "direct")
        if variant not in {"direct", "ground_first"}:
            raise ValueError(f"{label}.variant must be direct or ground_first")
        backend = entry.get("backend")
        if not isinstance(backend, str) or not backend.strip():
            raise ValueError(f"{label}.backend is required")
        scope = entry.get("coordinate_scope", "PITCH_WORLD_METRIC")
        if scope != "PITCH_WORLD_METRIC":
            raise ValueError("model-only systems comparison currently requires PITCH_WORLD_METRIC")
        checkpoint_sha256 = entry.get("checkpoint_sha256")
        if not isinstance(checkpoint_sha256, str) or not checkpoint_sha256.strip():
            raise ValueError(f"{label}.checkpoint_sha256 is required")
        output_path = _resolve(base, entry.get("output"), f"{label}.output")
        output_manifest = load_model_only_manifest(benchmark_path)
        arrays, metadata, output_sha256 = _load_and_validate_output(output_path, output_manifest)
        output_ids = tuple(str(value) for value in arrays["record_ids"].tolist())
        if output_ids != expected_ids:
            raise ValueError(f"{label}.output record cohort does not match benchmark manifest")
        if metadata.get("checkpoint_sha256") != checkpoint_sha256:
            raise ValueError(f"{label}.checkpoint_sha256 does not match output metadata")
        if entry.get("output_sha256") not in (None, output_sha256):
            raise ValueError(f"{label}.output_sha256 does not match output file")
        poses = arrays["pose23_world_m"] if variant == "direct" else arrays["pose23_ground_first_world_m"]
        loaded.append({
            "name": name,
            "backend": backend,
            "variant": variant,
            "coordinate_scope": scope,
            "output": str(output_path),
            "output_sha256": output_sha256,
            "checkpoint_sha256": checkpoint_sha256,
            "repository_revision": entry.get("repository_revision"),
            "config_sha256": entry.get("config_sha256"),
            "poses": poses.astype(np.float64),
            "valid_mask": np.asarray(arrays["valid_mask"], dtype=bool),
        })

    per_system: list[dict[str, Any]] = []
    metric_records: dict[str, list[dict[str, Any]]] = {}
    for system in loaded:
        records: list[dict[str, Any]] = []
        for index, record in enumerate(benchmark.records):
            metrics = _system_metric(system["poses"][index], record.gt_xyz_world_m, record.gt_visible)
            metrics.update({"record_id": record.record_id, "sequence_id": record.sequence_id, "valid_model_output": bool(system["valid_mask"][index])})
            records.append(metrics)
        metric_records[system["name"]] = records
        evaluated = [r for r in records if r.get("status") == "EVALUATED"]
        per_system.append({
            "name": system["name"],
            "backend": system["backend"],
            "variant": system["variant"],
            "coordinate_scope": system["coordinate_scope"],
            "output": system["output"],
            "output_sha256": system["output_sha256"],
            "checkpoint_sha256": system["checkpoint_sha256"],
            "repository_revision": system["repository_revision"],
            "config_sha256": system["config_sha256"],
            "coverage": {
                "record_count": len(records),
                "valid_model_output_count": int(system["valid_mask"].sum()),
                "evaluated_record_count": len(evaluated),
            },
            "metrics": {
                "global_mpjpe_m": _stats([r.get("global_mpjpe_m") for r in evaluated]),
                "root_aligned_mpjpe_m": _stats([r.get("root_aligned_mpjpe_m") for r in evaluated]),
                "pa_mpjpe_m": _stats([r.get("pa_mpjpe_m") for r in evaluated]),
                "longitudinal_abs_error_m": _stats([
                    (r.get("longitudinal_abs_error_m") or {}).get("mean") for r in evaluated
                ]),
            },
        })

    pairwise: list[dict[str, Any]] = []
    for left_index in range(len(loaded)):
        for right_index in range(left_index + 1, len(loaded)):
            left = loaded[left_index]
            right = loaded[right_index]
            deltas: list[float] = []
            paired_count = 0
            for index, record in enumerate(benchmark.records):
                left_metric = metric_records[left["name"]][index]
                right_metric = metric_records[right["name"]][index]
                if left_metric.get("status") != "EVALUATED" or right_metric.get("status") != "EVALUATED":
                    continue
                paired_count += 1
                deltas.append(float(left_metric["global_mpjpe_m"] - right_metric["global_mpjpe_m"]))
            pairwise.append({
                "left": left["name"],
                "right": right["name"],
                "paired_record_count": paired_count,
                "left_minus_right_global_mpjpe_m": _stats(deltas),
                "interpretation": "negative means left has lower error; paired descriptive comparison only",
            })

    report = {
        "schema_version": "stage4-model-only-systems-report-1.0",
        "systems_manifest": str(systems_path),
        "systems_manifest_sha256": systems_sha256,
        "benchmark_manifest": str(benchmark.path),
        "benchmark_manifest_sha256": benchmark.sha256,
        "dataset": benchmark.dataset,
        "split": benchmark.split,
        "systems": per_system,
        "pairwise": pairwise,
        "raw_per_record": metric_records,
        "metric_computation_completed": any(item["coverage"]["evaluated_record_count"] > 0 for item in per_system),
        "accuracy_claim_allowed": False,
        "research_accuracy_frozen": False,
        "upstream_dependencies": None,
        "limitations": [
            "Systems are compared only on the common benchmark manifest and declared PITCH_WORLD_METRIC scope.",
            "Pairwise deltas are descriptive; no backend is promoted without locked holdout evidence and confidence intervals.",
        ],
    }
    if report_path is not None:
        target = Path(report_path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    return report
