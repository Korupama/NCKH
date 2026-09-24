from __future__ import annotations

"""Ground-plane Ball-X diagnostic for the Stage-6 contact-aware FOOT branch.

This benchmark reuses SoccerNet-v3D 3D GT, dataset camera calibration and a completed
Stage-6 2D detector cache.  It does *not* claim contact-association accuracy.  Instead,
it asks a narrower geometry question:

    If an oracle tells us that the ball centre is near the physical contact plane
    Z = ball_radius, how accurately can the Stage-6 FOOT geometry recover Ball-X?

The benchmark therefore evaluates four variants on GT-height-defined subsets:

1. ground plane + GT center
2. ground plane + predicted top-1 center
3. size prior + GT optimized bbox
4. size prior + exact predicted top-1 bbox

GT height is used only to define diagnostic subsets, never as an inference input.
Project production still needs a separate contact-association benchmark (e.g. FOOTPASS).
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


GROUND_GT = "GROUND_PLANE_GT_CENTER"
GROUND_PRED = "GROUND_PLANE_PRED_CENTER"
SIZE_ORACLE = "SIZE_PRIOR_GT_BBOX"
SIZE_E2E = "SIZE_PRIOR_TOP1"
VARIANTS = (GROUND_GT, GROUND_PRED, SIZE_ORACLE, SIZE_E2E)

DEFAULT_TOLERANCES_M = (0.05, 0.10, 0.20)
PRIMARY_TOLERANCE_M = 0.10


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


def _candidate(frame_index: int, box: list[float], *, diameter_px: float, source: str, score: float = 1.0) -> BallCandidate2D:
    c = center(box)
    return BallCandidate2D(
        int(frame_index),
        f"grounddiag-{source}-{int(frame_index)}",
        [float(x) for x in box],
        np.asarray(c, dtype=float).tolist(),
        float(score),
        source,
        float(diameter_px),
    )


def _camera_status_usable(camera) -> bool:
    status = str(getattr(camera, "status", "VALID")).upper()
    return status in {"VALID", "DEGRADED"}


def _pitch_bounds(camera, pitch_margin_m: float) -> tuple[float, float]:
    pitch = getattr(camera, "pitch", None) or {}
    half_length = float(pitch.get("length_m", 105.0)) / 2.0 + float(pitch_margin_m)
    half_width = float(pitch.get("width_m", 68.0)) / 2.0 + float(pitch_margin_m)
    return half_length, half_width


def _intersect_ball_contact_plane(camera, uv: np.ndarray, *, ball_radius_m: float, pitch_margin_m: float) -> tuple[list[float] | None, str]:
    """Mirror the v0.5.1 FOOT geometry gate: ray intersect Z=ball radius.

    The function deliberately does not use GT XYZ.  GT is reserved for scoring and
    for selecting near-ground diagnostic subsets outside this function.
    """
    if not _camera_status_usable(camera):
        return None, "CAMERA_UNUSABLE"
    try:
        raw = camera.intersect_z_plane(np.asarray(uv, dtype=float), float(ball_radius_m))
        arr = np.asarray(raw, dtype=float)
        xyz = arr[0] if arr.ndim >= 2 else arr
    except Exception:
        return None, "INTERSECTION_FAILED"
    if xyz.size < 3 or not np.isfinite(xyz[:3]).all():
        return None, "INVALID_INTERSECTION"
    xyz = np.asarray(xyz[:3], dtype=float)
    half_length, half_width = _pitch_bounds(camera, pitch_margin_m)
    if abs(float(xyz[0])) > half_length or abs(float(xyz[1])) > half_width:
        return None, "OUTSIDE_PITCH_MARGIN"
    if abs(float(xyz[2]) - float(ball_radius_m)) > 1e-6:
        return None, "UNEXPECTED_Z"
    return xyz.astype(float).tolist(), "VALID"


def _evaluate_ground(record, camera, uv: np.ndarray | None, *, ball_radius_m: float, pitch_margin_m: float, source: str) -> dict[str, Any]:
    if uv is None:
        return {
            "record_id": record.record_id,
            "gt_xyz": record.ball_3d,
            "pred_xyz": None,
            "status": "NO_2D_OBSERVATION",
        }
    pred_xyz, status = _intersect_ball_contact_plane(
        camera,
        uv,
        ball_radius_m=ball_radius_m,
        pitch_margin_m=pitch_margin_m,
    )
    return {
        "record_id": record.record_id,
        "gt_xyz": record.ball_3d,
        "pred_xyz": pred_xyz,
        "status": status,
        "observation_source": source,
    }


def _evaluate_size_prior(record, camera, box: list[float] | None, diameter_px: float | None, *, ball_radius_m: float, pitch_margin_m: float, max_height_m: float, source: str) -> dict[str, Any]:
    if box is None or diameter_px is None:
        return {
            "record_id": record.record_id,
            "gt_xyz": record.ball_3d,
            "pred_xyz": None,
            "status": "NO_2D_OBSERVATION",
        }
    cand = _candidate(record.row_index, box, diameter_px=diameter_px, source=source)
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
        "observation_source": source,
    }


def _subset_metrics(rows_by_variant: Mapping[str, list[dict[str, Any]]], ids: set[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for variant in VARIANTS:
        subset = [r for r in rows_by_variant[variant] if str(r["record_id"]) in ids]
        out[variant] = _ball_x_view(summarize_3d(subset))
    valid_sets = [
        {str(r["record_id"]) for r in rows_by_variant[v] if str(r["record_id"]) in ids and r.get("pred_xyz") is not None}
        for v in VARIANTS
    ]
    common_ids = set.intersection(*valid_sets) if valid_sets else set()
    common = {}
    for variant in VARIANTS:
        subset = [r for r in rows_by_variant[variant] if str(r["record_id"]) in common_ids]
        common[variant] = _ball_x_view(summarize_3d(subset))

    gp = common.get(GROUND_PRED) or {}
    sg = common.get(SIZE_E2E) or {}
    gg = common.get(GROUND_GT) or {}
    so = common.get(SIZE_ORACLE) or {}
    gain_pred = None
    gain_oracle = None
    center_penalty = None
    if gp.get("mae_m") is not None and sg.get("mae_m") is not None:
        gain_pred = float(sg["mae_m"]) - float(gp["mae_m"])
    if gg.get("mae_m") is not None and so.get("mae_m") is not None:
        gain_oracle = float(so["mae_m"]) - float(gg["mae_m"])
    if gp.get("mae_m") is not None and gg.get("mae_m") is not None:
        center_penalty = float(gp["mae_m"]) - float(gg["mae_m"])
    return {
        "records": len(ids),
        "all_subset_frames": out,
        "common_valid_frames": {
            "records": len(common_ids),
            "metrics": common,
            "ground_pred_mae_gain_vs_size_prior_e2e_m": gain_pred,
            "ground_oracle_mae_gain_vs_size_prior_oracle_m": gain_oracle,
            "pred_center_penalty_on_ground_plane_m": center_penalty,
        },
    }


def _band_name(lo: float, hi: float | None) -> str:
    if lo == 0.0 and hi is not None:
        return f"abs_z_minus_radius_le_{hi:.2f}m"
    if hi is None:
        return f"abs_z_minus_radius_gt_{lo:.2f}m"
    return f"abs_z_minus_radius_{lo:.2f}_{hi:.2f}m"


def _in_band(value: float, lo: float, hi: float | None) -> bool:
    if lo == 0.0 and hi is not None:
        return value <= hi
    if hi is None:
        return value > lo
    return value > lo and value <= hi


def run_ground_plane_diagnostic(
    *,
    csv_path: str | Path,
    benchmark_2d_dir: str | Path,
    split: str,
    output_dir: str | Path,
    ball_radius_m: float = 0.11,
    pitch_margin_m: float = 6.0,
    max_height_m: float = 30.0,
    tolerances_m: tuple[float, ...] = DEFAULT_TOLERANCES_M,
    primary_tolerance_m: float = PRIMARY_TOLERANCE_M,
    max_rows: int | None = None,
    allow_partial_2d: bool = False,
    progress_every: int = 100,
) -> dict[str, Any]:
    tolerances = tuple(sorted({float(v) for v in tolerances_m if float(v) > 0.0}))
    if not tolerances:
        raise ValueError("tolerances_m must contain positive values")
    if float(primary_tolerance_m) not in tolerances:
        tolerances = tuple(sorted(set(tolerances + (float(primary_tolerance_m),))))

    dataset = SoccerNetV3DCSV(csv_path)
    records = [
        r for r in dataset.split(split)
        if gt_bbox_for_record(r, "optimized") is not None and r.ball_3d is not None
    ]
    if not records:
        raise RuntimeError(f"No ground-plane-evaluable rows in split {split!r}")
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

    rows_by_variant: dict[str, list[dict[str, Any]]] = {v: [] for v in VARIANTS}
    details: list[dict[str, Any]] = []
    missing_checkpoints = 0
    missing_top1 = 0

    for idx, record in enumerate(records, start=1):
        gt_box = [float(x) for x in gt_bbox_for_record(record, "optimized")]
        gt_center = center(gt_box)
        gt_d = float(diameter(gt_box))
        gt_xyz = np.asarray(record.ball_3d, dtype=float)
        dz = abs(float(gt_xyz[2]) - float(ball_radius_m))

        calib = parse_literal(record.raw.get("calibration"))
        if not isinstance(calib, dict):
            continue
        camera = camera_from_soccernet_calibration(
            calib,
            frame_index=record.row_index,
            image_width=record.img_w,
            image_height=record.img_h,
        )

        payload = _load_prediction(benchmark_2d_dir, record.record_id)
        pred = None
        if payload is None or bool(payload.get("missing_image")):
            missing_checkpoints += 1
        else:
            preds = list(payload.get("predictions") or [])
            pred = preds[0] if preds else None
        if pred is None:
            missing_top1 += 1

        pred_box = None
        pred_center = None
        pred_d = None
        top1_iou = None
        center_error_px = None
        if pred is not None:
            pred_box = [float(x) for x in pred["bbox_xyxy"]]
            pred_center = center(pred_box)
            pred_d = float(pred.get("diameter_px", diameter(pred_box)))
            top1_iou = float(iou(pred_box, gt_box))
            center_error_px = float(np.linalg.norm(pred_center - gt_center))

        variants = {
            GROUND_GT: _evaluate_ground(
                record, camera, gt_center,
                ball_radius_m=ball_radius_m, pitch_margin_m=pitch_margin_m,
                source="optimized-gt-center",
            ),
            GROUND_PRED: _evaluate_ground(
                record, camera, pred_center,
                ball_radius_m=ball_radius_m, pitch_margin_m=pitch_margin_m,
                source="top1-pred-center",
            ),
            SIZE_ORACLE: _evaluate_size_prior(
                record, camera, gt_box, gt_d,
                ball_radius_m=ball_radius_m, pitch_margin_m=pitch_margin_m,
                max_height_m=max_height_m, source="optimized-gt-bbox",
            ),
            SIZE_E2E: _evaluate_size_prior(
                record, camera, pred_box, pred_d,
                ball_radius_m=ball_radius_m, pitch_margin_m=pitch_margin_m,
                max_height_m=max_height_m, source="top1-pred-bbox",
            ),
        }

        detail = {
            "record_id": record.record_id,
            "gt_z_m": float(gt_xyz[2]),
            "ball_radius_m": float(ball_radius_m),
            "abs_gt_z_minus_radius_m": float(dz),
            "top1_iou": top1_iou,
            "center_error_px": center_error_px,
        }
        for variant, row in variants.items():
            rows_by_variant[variant].append(row)
            metric = evaluate_record_3d(row)
            detail[f"{variant}.status"] = row.get("status")
            detail[f"{variant}.abs_dx_m"] = metric.get("abs_dx_m")
            detail[f"{variant}.error_3d_m"] = metric.get("error_3d_m")
        details.append(detail)

        if progress_every > 0 and (idx == 1 or idx % progress_every == 0 or idx == len(records)):
            print(f"[BALL GROUND-PLANE DIAG {idx}/{len(records)}]", flush=True)

    detail_by_id = {str(r["record_id"]): r for r in details}
    cumulative: dict[str, Any] = {}
    for tol in tolerances:
        ids = {rid for rid, row in detail_by_id.items() if float(row["abs_gt_z_minus_radius_m"]) <= float(tol)}
        cumulative[f"abs_z_minus_radius_le_{tol:.2f}m"] = _subset_metrics(rows_by_variant, ids)

    edges = [0.0, *tolerances]
    bands: dict[str, Any] = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        ids = {
            rid for rid, row in detail_by_id.items()
            if _in_band(float(row["abs_gt_z_minus_radius_m"]), float(lo), float(hi))
        }
        bands[_band_name(float(lo), float(hi))] = _subset_metrics(rows_by_variant, ids)
    last = float(tolerances[-1])
    ids = {rid for rid, row in detail_by_id.items() if float(row["abs_gt_z_minus_radius_m"]) > last}
    bands[_band_name(last, None)] = _subset_metrics(rows_by_variant, ids)

    primary_key = f"abs_z_minus_radius_le_{float(primary_tolerance_m):.2f}m"
    primary = cumulative[primary_key]
    smoke = max_rows is not None
    report = {
        "schema_version": "stage6-ground-plane-diagnostic-1.0",
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
            "name": "STAGE6_CONTACT_GROUND_PLANE_PROXY_V1",
            "dataset": "SoccerNet-v3D",
            "split": str(split),
            "2d_cache": str(Path(benchmark_2d_dir).expanduser().resolve()),
            "csv_sha256": csv_hash,
            "ball_radius_m": float(ball_radius_m),
            "pitch_margin_m": float(pitch_margin_m),
            "primary_gt_height_tolerance_m": float(primary_tolerance_m),
            "gt_height_tolerances_m": list(tolerances),
            "coordinate_transform": coordinate_transform_metadata(),
            "oracle_usage": "GT Z is used only to select near-ground diagnostic subsets; inference variants never consume GT XYZ.",
            "scope": "geometry proxy for v0.5.1 FOOT contact branch; contact association is not evaluated here",
        },
        "counts": {
            "records": len(records),
            "missing_prediction_checkpoints": int(missing_checkpoints),
            "missing_top1_predictions": int(missing_top1),
            "primary_near_ground_records": int(primary["records"]),
            "primary_common_valid_frames": int(primary["common_valid_frames"]["records"]),
        },
        "primary_near_ground_proxy": {
            "key": primary_key,
            **primary,
        },
        "cumulative_gt_height_tolerance": cumulative,
        "gt_height_error_bands": bands,
        "interpretation": {
            GROUND_GT: "Ray intersect Z=ball_radius using optimized GT 2D center. Geometry ceiling for the FOOT plane assumption.",
            GROUND_PRED: "Ray intersect Z=ball_radius using predicted top-1 center. Approximates FOOT geometry after successful contact classification, but bypasses contact association.",
            SIZE_ORACLE: "Current size-prior baseline with optimized GT bbox on the exact same GT-height subset.",
            SIZE_E2E: "Current size-prior baseline with exact predicted top-1 bbox on the exact same GT-height subset.",
            "positive_gain": "Positive *_gain_vs_size_prior_* means ground-plane geometry has lower Ball-X MAE on common valid frames.",
            "not_production": "The subset is selected with GT ball height, so this is a geometry diagnostic, not an end-to-end contact-aware score.",
        },
    }

    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "stage6_ground_plane_diagnostic.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_csv(out / "frame_metrics.csv", details)

    p = report["primary_near_ground_proxy"]
    allm = p["all_subset_frames"]
    common = p["common_valid_frames"]
    lines = [
        "# Stage 6 contact-ground-plane geometry diagnostic",
        "",
        f"Status: **{report['status']}**",
        "",
        f"Scientific diagnostic ready: **{report['scientific_diagnostic_ready']}**",
        "",
        f"Primary proxy subset: `|GT_Z - ball_radius| <= {primary_tolerance_m:.2f} m` ({p['records']} records)",
        "",
        "> GT height is used only to define the diagnostic subset. This does not evaluate contact association and is not a production metric.",
        "",
        "| Variant | Ball-X MAE (m) | Median (m) | P90 (m) | P95 (m) | Coverage |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        m = allm[variant]
        lines.append(
            f"| {variant} | {m.get('mae_m')} | {m.get('median_m')} | {m.get('p90_m')} | {m.get('p95_m')} | {m.get('coverage')} |"
        )
    lines += [
        "",
        "## Common-valid-frame comparison",
        "",
        "```json",
        json.dumps(common, indent=2),
        "```",
        "",
        "A positive ground-plane gain versus size prior means the ground-contact plane is the better longitudinal geometry on the same frames.",
    ]
    (out / "stage6_ground_plane_diagnostic.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
