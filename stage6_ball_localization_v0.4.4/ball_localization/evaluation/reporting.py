from __future__ import annotations

from pathlib import Path
from ..version import PACKAGE_VERSION, runtime_provenance
from typing import Any, Mapping
import json

from .common import atomic_json, flatten_mapping, load_json, write_csv


def write_summary_bundle(
    out_dir: str | Path,
    *,
    summary_name: str,
    summary: Mapping[str, Any],
    frame_rows: list[Mapping[str, Any]] | None = None,
    failures: list[Mapping[str, Any]] | None = None,
    stratification: Mapping[str, Any] | None = None,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / summary_name, summary)
    write_csv(out / summary_name.replace(".json", ".csv"), [flatten_mapping(summary)])
    if frame_rows is not None:
        write_csv(out / "frame_metrics.csv", frame_rows)
    if failures is not None:
        atomic_json(out / "failure_cases.json", failures)
    if stratification is not None:
        atomic_json(out / "stratification.json", stratification)
        rows = []
        for dimension, groups in stratification.items():
            for group, metrics in groups.items():
                row = {"dimension": dimension, "group": group}
                row.update(metrics)
                rows.append(row)
        write_csv(out / "stratification.csv", rows)


def _load_optional(path: str | Path | None, filename: str) -> dict[str, Any] | None:
    if path is None:
        return None
    p = Path(path) / filename
    return load_json(p) if p.is_file() else None




def _require_canonical_3d_report(payload: dict[str, Any] | None, label: str) -> None:
    if payload is None:
        return
    transform = payload.get("coordinate_transform") or (payload.get("dataset") or {}).get("coordinate_transform")
    if not transform:
        raise RuntimeError(
            f"{label} does not declare the Stage-6 canonical SoccerNet coordinate transform. "
            "Do not combine pre-v0.3.2 3D summaries; rerun the 3D benchmark with v0.3.2 or later."
        )
    target = str(transform.get("target_frame", ""))
    if target != "STAGE6_CANONICAL_PITCH_XYZ":
        raise RuntimeError(f"{label} uses an unexpected 3D coordinate frame: {target!r}")


def compose_combined_report(
    output_dir: str | Path,
    *,
    detector_2d_dir: str | Path | None = None,
    oracle_original_dir: str | Path | None = None,
    oracle_optimized_dir: str | Path | None = None,
    e2e_dir: str | Path | None = None,
) -> dict[str, Any]:
    two_d = _load_optional(detector_2d_dir, "benchmark_2d_summary.json")
    if two_d is None and detector_2d_dir is not None:
        comparison = _load_optional(detector_2d_dir, "benchmark_2d_gt_comparison.json")
        if comparison is not None:
            two_d = {
                "schema_version": comparison.get("schema_version"),
                "stage6_version": comparison.get("stage6_version"),
                "gt_box_mode": "optimized",
                "metrics": (comparison.get("metrics") or {}).get("optimized", {}),
                "gt_comparison": comparison,
            }
    oracle_original = _load_optional(oracle_original_dir, "benchmark_3d_oracle_summary.json")
    oracle_optimized = _load_optional(oracle_optimized_dir, "benchmark_3d_oracle_summary.json")
    e2e = _load_optional(e2e_dir, "benchmark_3d_e2e_summary.json")
    _require_canonical_3d_report(oracle_original, "oracle_original")
    _require_canonical_3d_report(oracle_optimized, "oracle_optimized")
    _require_canonical_3d_report(e2e, "end_to_end_3d")
    report: dict[str, Any] = {
        "schema_version": "stage6-ball-combined-report-1.1",
        "stage6_version": PACKAGE_VERSION,
        "runtime_provenance": runtime_provenance(),
        "detector_2d": two_d,
        "oracle_original": oracle_original,
        "oracle_optimized": oracle_optimized,
        "end_to_end_3d": e2e,
        "error_budget": {},
        "interpretation": {
            "note": "Oracle-vs-end-to-end gaps isolate detector/box-selection effects from monocular geometry under the same camera/size-prior implementation.",
            "gate_policy": "v0.3 baselines are retained as the single-frame reference; v0.4 adds temporal refinement without changing their historical interpretation.",
        },
    }
    if oracle_optimized and e2e:
        om = oracle_optimized.get("metrics") or {}
        em = e2e.get("metrics") or {}
        if om.get("MAE_3D_m") is not None and em.get("MAE_3D_m") is not None:
            report["error_budget"]["e2e_minus_oracle_MAE_3D_m"] = float(em["MAE_3D_m"]) - float(om["MAE_3D_m"])
        if om.get("BLE_X_MAE_m") is not None and em.get("BLE_X_MAE_m") is not None:
            report["error_budget"]["e2e_minus_oracle_BLE_X_MAE_m"] = float(em["BLE_X_MAE_m"]) - float(om["BLE_X_MAE_m"])
        if om.get("coverage") is not None and em.get("coverage") is not None:
            report["error_budget"]["coverage_drop_vs_oracle"] = float(om["coverage"]) - float(em["coverage"])
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / "benchmark_report.json", report)

    lines = [
        "# Stage 6 v0.4.0 benchmark report",
        "",
        "This report combines available Stage-6 detector and geometry baselines. Missing experiments are left blank rather than inferred.",
        "",
        "| Experiment | AP50 | CandRecall@K | 3D MAE (m) | BLE-X MAE (m) | Coverage |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    if two_d:
        m = two_d.get("metrics") or {}
        cand = next((v for k, v in m.items() if str(k).startswith("CandidateRecall@")), None)
        lines.append(f"| 2D detector | {m.get('AP50','')} | {cand if cand is not None else ''} |  |  |  |")
    for label, payload in (("Oracle original bbox", oracle_original), ("Oracle optimized diameter", oracle_optimized), ("End-to-end top1", e2e)):
        if payload:
            m = payload.get("metrics") or {}
            lines.append(f"| {label} |  |  | {m.get('MAE_3D_m','')} | {m.get('BLE_X_MAE_m','')} | {m.get('coverage','')} |")
    if report["error_budget"]:
        lines += ["", "## Error-budget deltas", "", "```json", json.dumps(report["error_budget"], indent=2), "```"]
    (out / "benchmark_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
