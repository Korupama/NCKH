from __future__ import annotations

"""Stage-6 reduced-pipeline benchmark reporting.

This module deliberately separates publishable dataset benchmarks from project QA.
It does not invent PASS/FAIL accuracy thresholds.  The primary scientific benchmark
is SoccerNet-v3D on its fixed test split; ISSIA-3D is a secondary temporal/cross-domain
benchmark and contact association is evaluated from a frozen annotation manifest.
"""

from pathlib import Path
from typing import Any, Mapping
import json

from ..version import runtime_provenance


def _metric(metrics: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in metrics and metrics[key] is not None:
            return metrics[key]
    return None


def _candidate_recall(metrics: Mapping[str, Any], preferred_k: int = 5) -> Any:
    direct = metrics.get(f"CandidateRecall@{preferred_k}")
    if direct is not None:
        return direct
    pairs = []
    for key, value in metrics.items():
        if str(key).startswith("CandidateRecall@") and value is not None:
            try:
                k = int(str(key).split("@", 1)[1])
            except Exception:
                continue
            pairs.append((abs(k - preferred_k), k, value))
    return min(pairs)[2] if pairs else None


def _ball_x_block(payload: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not payload:
        return None
    m = payload.get("metrics") or {}
    return {
        "mae_m": _metric(m, "Ball_X_MAE_m", "BLE_X_MAE_m"),
        "rmse_m": _metric(m, "Ball_X_RMSE_m", "BLE_X_RMSE_m"),
        "median_m": _metric(m, "Ball_X_median_m", "BLE_X_median_m"),
        "p90_m": _metric(m, "Ball_X_P90_m", "BLE_X_P90_m"),
        "p95_m": _metric(m, "Ball_X_P95_m", "BLE_X_P95_m"),
        "coverage": m.get("coverage"),
        "mae_3d_m_secondary": m.get("MAE_3D_m"),
        "p95_3d_m_secondary": m.get("P95_3D_m"),
    }


def compose_primary_protocol_report(
    *,
    detector_report: Mapping[str, Any],
    oracle_report: Mapping[str, Any],
    e2e_report: Mapping[str, Any],
    split: str,
    smoke: bool,
) -> dict[str, Any]:
    """Compose the publishable Stage-6A/6B benchmark report.

    `smoke=True` is used whenever the caller truncates a dataset.  Such a report is
    useful for QA but is explicitly not eligible for an accuracy claim.
    """
    dm = detector_report.get("metrics") or {}
    oracle_x = _ball_x_block(oracle_report)
    e2e_x = _ball_x_block(e2e_report)
    missing_images = int(detector_report.get("missing_images") or 0)
    components_complete = all(
        str(x.get("status")) == "COMPLETE"
        for x in (detector_report, oracle_report, e2e_report)
    )
    full_test = str(split).lower() == "test" and not smoke
    scientific_claim_ready = bool(components_complete and full_test and missing_images == 0)

    error_budget = {}
    if oracle_x and e2e_x:
        if oracle_x.get("mae_m") is not None and e2e_x.get("mae_m") is not None:
            error_budget["e2e_minus_oracle_ball_x_mae_m"] = (
                float(e2e_x["mae_m"]) - float(oracle_x["mae_m"])
            )
        if oracle_x.get("coverage") is not None and e2e_x.get("coverage") is not None:
            error_budget["coverage_drop_vs_oracle"] = (
                float(oracle_x["coverage"]) - float(e2e_x["coverage"])
            )

    return {
        "schema_version": "stage6-reduced-benchmark-protocol-1.0",
        "runtime_provenance": runtime_provenance(),
        "status": "SMOKE" if smoke else ("COMPLETE" if components_complete else "PARTIAL"),
        "scientific_claim_ready": scientific_claim_ready,
        "protocol": {
            "name": "STAGE6_REDUCED_OFFSIDE_BENCHMARK_V1",
            "primary_dataset": "SoccerNet-v3D",
            "split": str(split),
            "temporal_rows_chained": False,
            "gt_box_convention": "optimized",
            "e2e_candidate_selection": "top1",
            "primary_target": "ball longitudinal coordinate X at the selected frame",
            "qa_policy": "truncated runs and project replay frames are QA/smoke only",
        },
        "stage6a_ball_detection": {
            "AP50": dm.get("AP50"),
            "mAP50_95": dm.get("mAP50_95"),
            "precision_iou50": dm.get("precision"),
            "recall_iou50": dm.get("recall"),
            "candidate_recall_at_5": _candidate_recall(dm, 5),
            "center_error_px_mean": dm.get("CenterErrorPx_mean"),
            "center_error_px_median": dm.get("CenterErrorPx_median"),
            "center_error_px_p90": dm.get("CenterErrorPx_P90"),
            "evaluated_images": detector_report.get("evaluated_images"),
            "missing_images": missing_images,
        },
        "stage6b_oracle_geometry": oracle_x,
        "stage6b_e2e_top1": e2e_x,
        "error_budget": error_budget,
        "interpretation": {
            "oracle": "GT/optimized ball box + dataset calibration; detector error excluded.",
            "e2e": "Predicted top-1 ball box + dataset calibration; valid single-frame detector-to-X benchmark.",
            "3d_metrics": "Reported only as secondary diagnostics; Ball-X is the reduced-pipeline primary metric.",
            "temporal": "SoccerNet-v3D action/replay CSV rows are synchronized multi-view observations and must not be chained as a video trajectory.",
            "contact": "Contact-player/region accuracy requires a separate frozen contact annotation manifest.",
        },
    }


def write_primary_protocol_report(
    output_dir: str | Path,
    *,
    detector_report: Mapping[str, Any],
    oracle_report: Mapping[str, Any],
    e2e_report: Mapping[str, Any],
    split: str,
    smoke: bool,
) -> dict[str, Any]:
    report = compose_primary_protocol_report(
        detector_report=detector_report,
        oracle_report=oracle_report,
        e2e_report=e2e_report,
        split=split,
        smoke=smoke,
    )
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "stage6_primary_benchmark.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    det = report["stage6a_ball_detection"]
    ox = report["stage6b_oracle_geometry"] or {}
    ex = report["stage6b_e2e_top1"] or {}
    lines = [
        "# Stage 6 reduced-pipeline primary benchmark",
        "",
        f"Status: **{report['status']}**",
        "",
        f"Scientific-claim ready: **{report['scientific_claim_ready']}**",
        "",
        "Primary dataset: **SoccerNet-v3D fixed split**.  Random project frames are not benchmark samples.",
        "",
        "## Stage 6A — ball detection",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| AP50 | {det.get('AP50')} |",
        f"| mAP50:95 | {det.get('mAP50_95')} |",
        f"| Recall@IoU0.5 | {det.get('recall_iou50')} |",
        f"| CandidateRecall@5 | {det.get('candidate_recall_at_5')} |",
        f"| Center error median (px) | {det.get('center_error_px_median')} |",
        f"| Center error P90 (px) | {det.get('center_error_px_p90')} |",
        "",
        "## Stage 6B — longitudinal ball localization",
        "",
        "| Experiment | Ball-X MAE (m) | Median (m) | P90 (m) | P95 (m) | Coverage | 3D MAE secondary (m) |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| Oracle optimized box | {ox.get('mae_m')} | {ox.get('median_m')} | {ox.get('p90_m')} | {ox.get('p95_m')} | {ox.get('coverage')} | {ox.get('mae_3d_m_secondary')} |",
        f"| End-to-end top1 | {ex.get('mae_m')} | {ex.get('median_m')} | {ex.get('p90_m')} | {ex.get('p95_m')} | {ex.get('coverage')} | {ex.get('mae_3d_m_secondary')} |",
        "",
        "## Error budget",
        "",
        "```json",
        json.dumps(report["error_budget"], indent=2),
        "```",
        "",
        "SoccerNet-v3D rows are **not** used as a temporal sequence.  Temporal/cross-domain geometry is evaluated separately on ISSIA-3D.",
    ]
    (out / "stage6_primary_benchmark.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def _decomp_variant(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    return ((payload.get("variants_all_evaluable") or {}).get(name) or {})


def compose_geometry_diagnostic_report(
    *,
    decomposition_report: Mapping[str, Any],
    best_iou_report: Mapping[str, Any],
    split: str,
    smoke: bool,
    primary_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compose the non-production geometry attribution report for Stage 6.

    This report is scientifically useful for error attribution, but the best-IoU branch
    uses GT to select a detector proposal and therefore must never be reported as a
    production end-to-end result.
    """
    oracle = _decomp_variant(decomposition_report, "GT_CENTER_GT_DIAMETER")
    center_only = _decomp_variant(decomposition_report, "PRED_CENTER_GT_DIAMETER")
    diameter_only = _decomp_variant(decomposition_report, "GT_CENTER_PRED_DIAMETER")
    top1 = _decomp_variant(decomposition_report, "PRED_CENTER_PRED_DIAMETER")
    best_iou = _ball_x_block(best_iou_report) or {}
    complete = (
        str(decomposition_report.get("status")) in {"COMPLETE", "SMOKE"}
        and str(best_iou_report.get("status")) == "COMPLETE"
    )
    diagnostic_ready = bool(complete and str(split).lower() == "test" and not smoke)

    best_iou_gain = None
    if top1.get("mae_m") is not None and best_iou.get("mae_m") is not None:
        best_iou_gain = float(top1["mae_m"]) - float(best_iou["mae_m"])

    consistency = None
    if primary_report:
        p_oracle = primary_report.get("stage6b_oracle_geometry") or {}
        p_e2e = primary_report.get("stage6b_e2e_top1") or {}
        checks = {}
        for label, a, b in (
            ("oracle_mae", oracle.get("mae_m"), p_oracle.get("mae_m")),
            ("oracle_coverage", oracle.get("coverage"), p_oracle.get("coverage")),
            ("top1_e2e_mae", top1.get("mae_m"), p_e2e.get("mae_m")),
            ("top1_e2e_coverage", top1.get("coverage"), p_e2e.get("coverage")),
        ):
            if a is None or b is None:
                checks[label] = {"status": "NOT_AVAILABLE", "delta_m": None}
            else:
                delta = float(a) - float(b)
                checks[label] = {"status": "MATCH" if abs(delta) <= 1e-9 else "MISMATCH", "delta_m": delta}
        consistency = checks

    return {
        "schema_version": "stage6-geometry-diagnostics-1.2",
        "runtime_provenance": runtime_provenance(),
        "status": "SMOKE" if smoke else ("COMPLETE" if complete else "PARTIAL"),
        "scientific_diagnostic_ready": diagnostic_ready,
        "production_metric": False,
        "protocol": {
            "name": "STAGE6_GEOMETRY_ERROR_ATTRIBUTION_V1_2",
            "dataset": "SoccerNet-v3D",
            "split": str(split),
            "top1_decomposition": True,
            "best_iou_is_gt_assisted": True,
            "use_for": "error attribution and method development",
            "do_not_use_for": "production end-to-end performance claim",
        },
        "center_diameter_decomposition": {
            "oracle_gt_center_gt_diameter": oracle,
            "pred_center_gt_diameter": center_only,
            "gt_center_pred_diameter": diameter_only,
            "pred_center_pred_diameter": top1,
            "common_frame_attribution": decomposition_report.get("error_attribution_common_frames"),
            "top1_iou50_matched": decomposition_report.get("variants_top1_iou50_matched"),
            "counts": decomposition_report.get("counts"),
            "top1_observation_error": decomposition_report.get("top1_observation_error"),
            "e2e_by_absolute_diameter_relative_error": decomposition_report.get("e2e_by_absolute_diameter_relative_error"),
        },
        "best_iou_diagnostic": {
            **best_iou,
            "ball_x_mae_gain_vs_top1_m": best_iou_gain,
            "interpretation": "GT-assisted best proposal among cached detections; diagnostic upper bound for candidate ranking, not a deployable result.",
        },
        "primary_consistency_check": consistency,
    }


def write_geometry_diagnostic_report(
    output_dir: str | Path,
    *,
    decomposition_report: Mapping[str, Any],
    best_iou_report: Mapping[str, Any],
    split: str,
    smoke: bool,
    primary_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    report = compose_geometry_diagnostic_report(
        decomposition_report=decomposition_report,
        best_iou_report=best_iou_report,
        split=split,
        smoke=smoke,
        primary_report=primary_report,
    )
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "stage6_geometry_diagnostics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    d = report["center_diameter_decomposition"]
    b = report["best_iou_diagnostic"]
    rows = (
        ("GT center + GT diameter", d["oracle_gt_center_gt_diameter"]),
        ("Pred center + GT diameter", d["pred_center_gt_diameter"]),
        ("GT center + Pred diameter", d["gt_center_pred_diameter"]),
        ("Pred center + Pred diameter", d["pred_center_pred_diameter"]),
        ("Best-IoU proposal (GT-assisted)", b),
    )
    lines = [
        "# Stage 6 geometry diagnostics v1.2",
        "",
        f"Status: **{report['status']}**",
        "",
        f"Scientific diagnostic ready: **{report['scientific_diagnostic_ready']}**",
        "",
        "> This report is for error attribution. Best-IoU uses ground truth to select a proposal and is not a production metric.",
        "",
        "| Experiment | Ball-X MAE (m) | Median (m) | P90 (m) | P95 (m) | Coverage |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, metric in rows:
        lines.append(
            f"| {label} | {metric.get('mae_m')} | {metric.get('median_m')} | {metric.get('p90_m')} | {metric.get('p95_m')} | {metric.get('coverage')} |"
        )
    lines += [
        "",
        "## Common-frame center/diameter attribution",
        "",
        "```json",
        json.dumps(d.get("common_frame_attribution"), indent=2),
        "```",
        "",
        f"Best-IoU Ball-X MAE gain vs top-1: **{b.get('ball_x_mae_gain_vs_top1_m')} m**",
    ]
    if report.get("primary_consistency_check") is not None:
        lines += [
            "",
            "## Consistency with the existing primary report",
            "",
            "```json",
            json.dumps(report["primary_consistency_check"], indent=2),
            "```",
        ]
    (out / "stage6_geometry_diagnostics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
