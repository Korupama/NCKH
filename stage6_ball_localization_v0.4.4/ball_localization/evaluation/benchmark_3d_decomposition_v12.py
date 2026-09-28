from __future__ import annotations

"""Center/box-size error decomposition for Stage-6 single-frame Ball-X geometry.

The benchmark reuses a completed SoccerNet-v3D detector cache and the dataset camera
calibration.  It creates four size-prior observations per evaluable row:

1. GT center + GT optimized diameter      (oracle geometry)
2. Predicted center + GT optimized diameter
3. GT center + predicted diameter
4. Predicted center + predicted diameter  (top-1 end-to-end)

This isolates how much Ball-X error is associated with image-center selection versus
predicted box size/shape.  The v1.2 protocol preserves the exact detector box for the
end-to-end branch so it reproduces the canonical run_3d_e2e benchmark.  Because MAE is nonlinear, reported deltas are diagnostic
rather than an additive causal decomposition.  Common-frame summaries are included so
all four variants are compared on exactly the same samples.
"""

from pathlib import Path
from typing import Any, Mapping
import json
import math

import numpy as np

from ..camera import camera_from_soccernet_calibration
from ..contracts import BallCandidate2D
from ..coordinates import coordinate_transform_metadata
from ..datasets import SoccerNetV3DCSV, gt_bbox_for_record
from ..datasets.soccernet_v3d import parse_literal
from ..geometry import estimate_frame
from ..version import PACKAGE_VERSION, runtime_provenance
from .common import load_json, record_key, sha256_file, write_csv
from .metrics2d import center, diameter, iou
from .metrics3d import evaluate_record_3d, summarize_3d


VARIANT_ORACLE = "GT_CENTER_GT_DIAMETER"
VARIANT_CENTER = "PRED_CENTER_GT_DIAMETER"
VARIANT_DIAMETER = "GT_CENTER_PRED_DIAMETER"
VARIANT_E2E = "PRED_CENTER_PRED_DIAMETER"
VARIANTS = (VARIANT_ORACLE, VARIANT_CENTER, VARIANT_DIAMETER, VARIANT_E2E)


def _load_2d_checkpoint(benchmark_2d_dir: str | Path) -> dict[str, Any]:
    path = Path(benchmark_2d_dir).expanduser().resolve() / "checkpoint.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return load_json(path)


def _load_prediction(benchmark_2d_dir: str | Path, record_id: str) -> dict[str, Any] | None:
    path = Path(benchmark_2d_dir).expanduser().resolve() / "predictions" / f"{record_key(record_id)}.json"
    if not path.is_file():
        return None
    payload = load_json(path)
    if str(payload.get("record_id")) != str(record_id):
        raise RuntimeError(f"Stale/colliding 2D prediction checkpoint: {path}")
    return payload


def _translate_box(box: list[float], new_center: np.ndarray) -> list[float]:
    """Translate a box without changing width/height.

    v1.1 rebuilt every mixed observation as a square.  That changed the detector
    geometry seen by ``estimate_frame`` and prevented the Pred+Pred branch from
    reproducing the canonical top-1 E2E benchmark.  v1.2 preserves the box shape.
    """
    x1, y1, x2, y2 = map(float, box)
    w, h = x2 - x1, y2 - y1
    cx, cy = map(float, new_center)
    return [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]


def _candidate(
    frame_index: int,
    box: list[float],
    *,
    diameter_px: float,
    source: str,
    score: float = 1.0,
) -> BallCandidate2D:
    c = center(box)
    return BallCandidate2D(
        int(frame_index),
        f"decomp-{source}-{int(frame_index)}",
        [float(x) for x in box],
        np.asarray(c, dtype=float).tolist(),
        float(score),
        source,
        float(diameter_px),
    )


def _prediction_geometry(pred: Mapping[str, Any]) -> tuple[list[float], np.ndarray, float]:
    box = [float(x) for x in pred["bbox_xyxy"]]
    c = center(box)
    d = float(pred.get("diameter_px", diameter(box)))
    if not np.isfinite(c).all() or not np.isfinite(box).all() or not math.isfinite(d) or d <= 0.0:
        raise ValueError("Invalid predicted box/center/diameter")
    return box, c, d


def _evaluate_variant(
    *,
    record,
    camera,
    box: list[float],
    diameter_px: float,
    variant: str,
    ball_radius_m: float,
    pitch_margin_m: float,
    max_height_m: float,
) -> dict[str, Any]:
    cand = _candidate(record.row_index, box, diameter_px=diameter_px, source=variant)
    state = estimate_frame(
        camera,
        cand,
        fps=25.0,
        ball_radius_m=ball_radius_m,
        mode="size-prior",
        pitch_margin_m=pitch_margin_m,
        max_size_prior_height_m=max_height_m,
    )
    return {
        "record_id": record.record_id,
        "gt_xyz": record.ball_3d,
        "pred_xyz": state.selected_center_xyz_world_m,
        "status": state.localization_status,
        "geometry_diagnostics": state.diagnostics,
    }


def _summarize_variant(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return summarize_3d(rows)


def _ball_x_view(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "mae_m": metrics.get("BLE_X_MAE_m"),
        "rmse_m": metrics.get("BLE_X_RMSE_m"),
        "median_m": metrics.get("BLE_X_median_m"),
        "p90_m": metrics.get("BLE_X_P90_m"),
        "p95_m": metrics.get("BLE_X_P95_m"),
        "coverage": metrics.get("coverage"),
        "mae_3d_m_secondary": metrics.get("MAE_3D_m"),
        "p95_3d_m_secondary": metrics.get("P95_3D_m"),
    }


def _delta(a: Any, b: Any) -> float | None:
    if a is None or b is None:
        return None
    return float(a) - float(b)


def _common_metrics(rows_by_variant: Mapping[str, list[dict[str, Any]]]) -> tuple[int, dict[str, Any]]:
    valid_sets = []
    for variant in VARIANTS:
        valid_sets.append({
            str(row["record_id"])
            for row in rows_by_variant[variant]
            if row.get("gt_xyz") is not None and row.get("pred_xyz") is not None
        })
    common_ids = set.intersection(*valid_sets) if valid_sets else set()
    metrics = {}
    for variant in VARIANTS:
        subset = [row for row in rows_by_variant[variant] if str(row["record_id"]) in common_ids]
        metrics[variant] = _ball_x_view(_summarize_variant(subset))
    return len(common_ids), metrics


def _attribution(common: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    oracle = common.get(VARIANT_ORACLE) or {}
    center_only = common.get(VARIANT_CENTER) or {}
    diameter_only = common.get(VARIANT_DIAMETER) or {}
    e2e = common.get(VARIANT_E2E) or {}
    o = oracle.get("mae_m")
    c = center_only.get("mae_m")
    d = diameter_only.get("mae_m")
    e = e2e.get("mae_m")
    center_delta = _delta(c, o)
    diameter_delta = _delta(d, o)
    combined_delta = _delta(e, o)
    interaction = None
    if combined_delta is not None and center_delta is not None and diameter_delta is not None:
        interaction = combined_delta - center_delta - diameter_delta
    return {
        "comparison_basis": "intersection of frames with valid predictions for all four variants",
        "oracle_ball_x_mae_m": o,
        "center_only_ball_x_mae_m": c,
        "diameter_only_ball_x_mae_m": d,
        "e2e_ball_x_mae_m": e,
        "center_only_delta_vs_oracle_m": center_delta,
        "diameter_only_delta_vs_oracle_m": diameter_delta,
        "combined_delta_vs_oracle_m": combined_delta,
        "nonadditive_interaction_residual_m": interaction,
        "note": "MAE deltas are diagnostic. They are not assumed to be additive or causal.",
    }


def run_3d_center_diameter_decomposition(
    *,
    csv_path: str | Path,
    benchmark_2d_dir: str | Path,
    split: str,
    output_dir: str | Path,
    ball_radius_m: float = 0.11,
    pitch_margin_m: float = 6.0,
    max_height_m: float = 30.0,
    max_rows: int | None = None,
    allow_partial_2d: bool = False,
    gt_box_mode: str = "optimized",
    progress_every: int = 100,
) -> dict[str, Any]:
    if str(gt_box_mode).lower() != "optimized":
        raise ValueError("v1.1 decomposition is frozen to optimized SoccerNet-v3D GT boxes")

    dataset = SoccerNetV3DCSV(csv_path)
    records = [
        r for r in dataset.split(split)
        if gt_bbox_for_record(r, "optimized") is not None and r.ball_3d is not None
    ]
    if not records:
        raise RuntimeError(f"No decomposition-evaluable rows in split {split!r}")
    if max_rows is not None:
        records = records[: int(max_rows)]

    checkpoint = _load_2d_checkpoint(benchmark_2d_dir)
    if not allow_partial_2d and str(checkpoint.get("status")) != "COMPLETE":
        raise RuntimeError("2D benchmark cache is not COMPLETE")
    identity = dict(checkpoint.get("run_identity") or {})
    csv_hash = sha256_file(csv_path)
    if identity.get("csv_sha256") and str(identity["csv_sha256"]).lower() != csv_hash.lower():
        raise RuntimeError("2D prediction cache was produced from a different SNv3D.csv")
    if identity.get("split") and str(identity["split"]) != str(split):
        raise RuntimeError("2D prediction cache split differs from requested split")

    rows_by_variant: dict[str, list[dict[str, Any]]] = {name: [] for name in VARIANTS}
    frame_rows: list[dict[str, Any]] = []
    missing_checkpoints = 0
    missing_top1 = 0
    matched_top1 = 0

    for idx, record in enumerate(records, start=1):
        gt_box = gt_bbox_for_record(record, "optimized")
        gt_c = center(gt_box)
        gt_d = float(diameter(gt_box))
        calib = parse_literal(record.raw.get("calibration"))
        if not isinstance(calib, dict):
            continue
        camera = camera_from_soccernet_calibration(
            calib,
            frame_index=record.row_index,
            image_width=record.img_w,
            image_height=record.img_h,
        )
        pred_payload = _load_prediction(benchmark_2d_dir, record.record_id)
        pred = None
        if pred_payload is None or bool(pred_payload.get("missing_image")):
            missing_checkpoints += 1
        else:
            preds = list(pred_payload.get("predictions") or [])
            pred = preds[0] if preds else None
        pred_box = None
        pred_c = None
        pred_d = None
        top1_iou = None
        center_error_px = None
        diameter_relative_error = None
        if pred is None:
            missing_top1 += 1
        else:
            pred_box, pred_c, pred_d = _prediction_geometry(pred)
            top1_iou = float(iou(pred["bbox_xyxy"], gt_box))
            matched_top1 += int(top1_iou >= 0.5)
            center_error_px = float(np.linalg.norm(pred_c - gt_c))
            diameter_relative_error = abs(pred_d - gt_d) / max(gt_d, 1e-12)

        observations = {
            # Exact optimized GT box used by the oracle benchmark.
            VARIANT_ORACLE: (list(map(float, gt_box)), gt_d),
            # Move the GT optimized box to the predicted center; size/shape stays GT.
            VARIANT_CENTER: None if pred_c is None else (_translate_box(list(map(float, gt_box)), pred_c), gt_d),
            # Recenter the exact predicted box on the GT center; detector size/shape stays predicted.
            VARIANT_DIAMETER: None if pred_box is None or pred_d is None else (_translate_box(pred_box, gt_c), pred_d),
            # Exact top-1 detector observation. This must reproduce run_3d_e2e(top1).
            VARIANT_E2E: None if pred_box is None or pred_d is None else (pred_box, pred_d),
        }
        detail_row: dict[str, Any] = {
            "record_id": record.record_id,
            "top1_iou": top1_iou,
            "top1_match_iou50": bool(top1_iou is not None and top1_iou >= 0.5),
            "center_error_px": center_error_px,
            "gt_diameter_px": gt_d,
            "pred_diameter_px": pred_d,
            "diameter_relative_error": diameter_relative_error,
            "diameter_ratio": None if pred_d is None else pred_d / max(gt_d, 1e-12),
        }
        for variant, obs in observations.items():
            if obs is None:
                row = {
                    "record_id": record.record_id,
                    "gt_xyz": record.ball_3d,
                    "pred_xyz": None,
                    "status": "NO_TOP1_PREDICTION",
                }
            else:
                row = _evaluate_variant(
                    record=record,
                    camera=camera,
                    box=obs[0],
                    diameter_px=obs[1],
                    variant=variant,
                    ball_radius_m=ball_radius_m,
                    pitch_margin_m=pitch_margin_m,
                    max_height_m=max_height_m,
                )
            rows_by_variant[variant].append(row)
            m = evaluate_record_3d(row)
            detail_row[f"{variant}.status"] = row.get("status")
            detail_row[f"{variant}.abs_dx_m"] = m.get("abs_dx_m")
            detail_row[f"{variant}.error_3d_m"] = m.get("error_3d_m")
        frame_rows.append(detail_row)

        if progress_every > 0 and (idx == 1 or idx % progress_every == 0 or idx == len(records)):
            print(f"[BALL 3D DECOMP {idx}/{len(records)}]", flush=True)

    all_metrics = {variant: _ball_x_view(_summarize_variant(rows_by_variant[variant])) for variant in VARIANTS}
    matched_metrics = {}
    matched_ids = {str(r["record_id"]) for r in frame_rows if r.get("top1_match_iou50")}
    for variant in VARIANTS:
        subset = [row for row in rows_by_variant[variant] if str(row["record_id"]) in matched_ids]
        matched_metrics[variant] = _ball_x_view(_summarize_variant(subset))
    common_count, common_metrics = _common_metrics(rows_by_variant)

    center_errors = [float(r["center_error_px"]) for r in frame_rows if r.get("center_error_px") is not None]
    diameter_errors = [float(r["diameter_relative_error"]) for r in frame_rows if r.get("diameter_relative_error") is not None]
    top1_ious = [float(r["top1_iou"]) for r in frame_rows if r.get("top1_iou") is not None]
    observation_error = {
        "center_error_px_median": float(np.median(center_errors)) if center_errors else None,
        "center_error_px_p90": float(np.percentile(center_errors, 90)) if center_errors else None,
        "diameter_relative_error_median": float(np.median(diameter_errors)) if diameter_errors else None,
        "diameter_relative_error_p90": float(np.percentile(diameter_errors, 90)) if diameter_errors else None,
        "top1_iou_median": float(np.median(top1_ious)) if top1_ious else None,
        "top1_iou_p10": float(np.percentile(top1_ious, 10)) if top1_ious else None,
    }

    ratios = [float(r["diameter_ratio"]) for r in frame_rows if r.get("diameter_ratio") is not None and math.isfinite(float(r["diameter_ratio"]))]
    if ratios:
        observation_error.update({
            "diameter_ratio_median": float(np.median(ratios)),
            "diameter_ratio_p10": float(np.percentile(ratios, 10)),
            "diameter_ratio_p90": float(np.percentile(ratios, 90)),
            "diameter_underestimate_rate": float(np.mean(np.asarray(ratios) < 1.0)),
            "diameter_overestimate_rate": float(np.mean(np.asarray(ratios) > 1.0)),
        })

    size_bins = [
        ("abs_rel_err_le_0.10", 0.0, 0.10),
        ("abs_rel_err_0.10_0.25", 0.10, 0.25),
        ("abs_rel_err_0.25_0.50", 0.25, 0.50),
        ("abs_rel_err_gt_0.50", 0.50, float("inf")),
    ]
    size_error_stratification = {}
    e2e_rows_by_id = {str(r["record_id"]): r for r in rows_by_variant[VARIANT_E2E]}
    for name, lo, hi in size_bins:
        ids = []
        for detail in frame_rows:
            value = detail.get("diameter_relative_error")
            if value is None:
                continue
            value = float(value)
            inside = (value <= hi if lo == 0.0 else (value > lo and value <= hi)) if math.isfinite(hi) else value > lo
            if inside:
                ids.append(str(detail["record_id"]))
        subset = [e2e_rows_by_id[i] for i in ids if i in e2e_rows_by_id]
        block = _ball_x_view(_summarize_variant(subset))
        block["records"] = len(ids)
        size_error_stratification[name] = block

    smoke = max_rows is not None
    report = {
        "schema_version": "stage6-ball-x-center-size-decomposition-1.2",
        "stage6_version": PACKAGE_VERSION,
        "runtime_provenance": runtime_provenance(),
        "status": "SMOKE" if smoke else "COMPLETE",
        "scientific_diagnostic_ready": bool(
            str(split).lower() == "test"
            and not smoke
            and str(checkpoint.get("status")) == "COMPLETE"
            and missing_checkpoints == 0
        ),
        "production_metric": False,
        "dataset": dataset.summary(),
        "protocol": {
            "split": str(split),
            "gt_box_mode": "optimized",
            "candidate_selection": "top1",
            "localization_mode": "size-prior",
            "records": len(records),
            "ball_radius_m": float(ball_radius_m),
            "pitch_margin_m": float(pitch_margin_m),
            "max_height_m": float(max_height_m),
            "coordinate_transform": coordinate_transform_metadata(),
            "2d_cache": str(Path(benchmark_2d_dir).expanduser().resolve()),
            "csv_sha256": csv_hash,
        },
        "counts": {
            "records": len(records),
            "missing_prediction_checkpoints": int(missing_checkpoints),
            "missing_top1_predictions": int(missing_top1),
            "top1_iou50_matches": int(matched_top1),
            "common_valid_prediction_frames": int(common_count),
        },
        "top1_observation_error": observation_error,
        "e2e_by_absolute_diameter_relative_error": size_error_stratification,
        "variants_all_evaluable": all_metrics,
        "variants_top1_iou50_matched": matched_metrics,
        "variants_common_prediction_frames": common_metrics,
        "error_attribution_common_frames": _attribution(common_metrics),
        "interpretation": {
            VARIANT_ORACLE: "GT center + optimized GT diameter; detector excluded.",
            VARIANT_CENTER: "Predicted top-1 center + GT optimized box size/shape; isolates center/selection contribution while holding box geometry to GT.",
            VARIANT_DIAMETER: "GT center + predicted top-1 box size/shape; isolates detector box-size/shape contribution while holding center to GT.",
            VARIANT_E2E: "Exact predicted top-1 detector box and diameter; must reproduce the normal top-1 single-frame E2E geometry under identical settings.",
            "common_frames": "Preferred attribution view because all four variants are scored on the same valid prediction subset.",
        },
    }

    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark_3d_decomposition.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(out / "frame_metrics.csv", frame_rows)
    lines = [
        "# Stage 6 Ball-X center/box-size decomposition v1.2",
        "",
        f"Status: **{report['status']}**",
        "",
        f"Scientific diagnostic ready: **{report['scientific_diagnostic_ready']}**",
        "",
        "| Variant | Ball-X MAE (m) | Median (m) | P90 (m) | P95 (m) | Coverage |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        m = all_metrics[variant]
        lines.append(f"| {variant} | {m.get('mae_m')} | {m.get('median_m')} | {m.get('p90_m')} | {m.get('p95_m')} | {m.get('coverage')} |")
    lines += [
        "",
        "## Common-frame attribution",
        "",
        "```json",
        json.dumps(report["error_attribution_common_frames"], indent=2),
        "```",
        "",
        "The deltas above are diagnostic MAE differences, not an additive causal decomposition. v1.2 preserves the exact detector box in the E2E branch.",
    ]
    (out / "benchmark_3d_decomposition.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
