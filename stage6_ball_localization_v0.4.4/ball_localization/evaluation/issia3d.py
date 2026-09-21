"""Oracle-2D temporal geometry benchmark for the public ISSIA-3D release.

The predictor never receives ``ball_3D``.  It is only converted into the
canonical Stage-6 coordinate system after prediction, for evaluation.
"""
from __future__ import annotations

import ast
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable
from dataclasses import asdict, dataclass

import numpy as np

from ..camera import CameraStateLite, camera_from_soccernet_calibration
from ..contracts import BallCandidate2D
from ..coordinates import coordinate_transform_metadata, soccernet_xyz_to_stage6
from ..geometry import refine_temporal_trajectory
from ..geometry.localization import estimate_frame
from ..version import runtime_provenance
from .metrics3d import summarize_3d


METHODS = ("V03_SIZE_PRIOR", "GROUND_PLANE", "V04_TEMPORAL", "BALLISTIC_G9_81", "BALLISTIC_FIT_G")


@dataclass(frozen=True)
class HybridConfig:
    """Frozen, GT-free selector parameters for Stage 6 v0.4.4."""
    ground_proxy_height_m: float = 0.55
    ballistic_consensus_distance_m: float = 3.0
    median_window_frames: int = 3

    @classmethod
    def from_mapping(cls, value: dict[str, Any] | None) -> "HybridConfig":
        value = value or {}
        return cls(
            ground_proxy_height_m=float(value.get("ground_proxy_height_m", cls.ground_proxy_height_m)),
            ballistic_consensus_distance_m=float(value.get("ballistic_consensus_distance_m", cls.ballistic_consensus_distance_m)),
            median_window_frames=int(value.get("median_window_frames", cls.median_window_frames)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _number(value: Any) -> float | None:
    try:
        text = str(value).strip()
        if not text:
            return None
        x = float(text)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _xyz(value: Any) -> np.ndarray | None:
    try:
        point = np.asarray(ast.literal_eval(str(value)), dtype=float).reshape(3)
    except (SyntaxError, ValueError, TypeError):
        return None
    return point if np.all(np.isfinite(point)) else None


def _candidate(frame_index: int, x: float, y: float, diameter: float) -> BallCandidate2D:
    half = diameter / 2.0
    return BallCandidate2D(
        frame_index=int(frame_index), candidate_id=f"issia-{frame_index}",
        bbox_xyxy=[x-half, y-half, x+half, y+half], center_uv=[x, y],
        detector_score=1.0, ranking_score=1.0, pitch_prior=1.0,
        diameter_px=diameter, source="ISSIA_ORACLE_2D",
    )


def _read_rows(csv_path: str | Path) -> list[dict[str, Any]]:
    with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def inspect_issia3d(csv_path: str | Path, calibration_path: str | Path) -> dict[str, Any]:
    rows = _read_rows(csv_path)
    calib = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
    counts: dict[str, int] = {}
    for camera in range(1, 7):
        counts[f"cam{camera}"] = sum(
            _number(row.get(f"x_cam{camera}")) is not None
            and _number(row.get(f"y_cam{camera}")) is not None
            and (_number(row.get(f"opt_d_cam{camera}")) or 0.0) > 0.0
            for row in rows
        )
    gt = sum(_xyz(row.get("ball_3D")) is not None for row in rows)
    missing = [f"cam{i}" for i in range(1, 7) if f"cam{i}" not in calib]
    return {
        # The public table intentionally includes rows where triangulation was
        # unavailable.  Those are not benchmark samples, rather than a broken
        # download; readiness requires at least one complete GT sample.
        "status": "READY" if rows and gt > 0 and not missing else "INCOMPLETE",
        "csv": str(Path(csv_path).expanduser().resolve()),
        "calibration": str(Path(calibration_path).expanduser().resolve()),
        "rows": len(rows), "rows_with_gt_3d": gt,
        "oracle_2d_diameter_observations": counts,
        "calibration_cameras": sorted(calib.keys()), "missing_calibration_cameras": missing,
        "coordinate_transform": coordinate_transform_metadata(),
        "note": "ISSIA ball_3D is withheld from all predictors and used only for final scoring.",
    }


def _segments(items: list[dict[str, Any]], max_frames: int | None = None) -> Iterable[list[dict[str, Any]]]:
    """Split independent clips; a discontinuous frame number has no temporal link."""
    current: list[dict[str, Any]] = []
    previous: int | None = None
    for item in items:
        frame = int(item["frame"])
        if current and previous is not None and (frame != previous + 1 or (max_frames is not None and len(current) >= max_frames)):
            yield current
            current = []
        current.append(item)
        previous = frame
    if current:
        yield current


def _ballistic(segment: list[dict[str, Any]], *, fitted_g: bool, fps: float) -> dict[int, list[float] | None]:
    """Fit a parabola to image rays, without any 3-D labels.

    The residual is the component of the trajectory point perpendicular to its
    observed ray. A weak apparent-size depth anchor removes the unavoidable
    monocular scale null-space; it is an observation, not a 3-D label.
    """
    answer: dict[int, list[float] | None] = {item["key"]: None for item in segment}
    if len(segment) < (4 if fitted_g else 3):
        return answer
    t = np.asarray([(int(item["frame"]) - int(segment[0]["frame"])) / float(fps) for item in segment], float)
    columns = 7 if fitted_g else 6
    a_rows: list[np.ndarray] = []
    b_rows: list[np.ndarray] = []
    for item, ti in zip(segment, t):
        camera: CameraStateLite = item["camera"]
        _origin, direction = camera.world_ray(np.asarray(item["candidate"].center_uv, float))
        projector = np.eye(3) - np.outer(direction[0], direction[0])
        row = np.zeros((3, columns), float)
        row[:, :3] = projector
        row[:, 3:6] = float(ti) * projector
        if fitted_g:
            row[:, 6] = projector @ np.asarray([0.0, 0.0, -0.5 * ti * ti])
            target = projector @ camera.camera_center_world_m
        else:
            target = projector @ (camera.camera_center_world_m + np.asarray([0.0, 0.0, 0.5 * 9.81 * ti * ti]))
        a_rows.append(row); b_rows.append(target)
        # A single calibrated camera cannot determine the absolute scale of a
        # constant-velocity/parabolic path from rays alone.  Keep the fit close
        # to the independently observed ball-size depth, but substantially less
        # tightly than exact reprojection.  This makes the comparator meaningful
        # on real ISSIA clips without allowing an arbitrary 100m solution.
        size_state = estimate_frame(camera, item["candidate"], fps=fps, mode="size-prior")
        if size_state.size_prior_xyz_world_m is not None:
            prior = np.zeros((3, columns), float)
            prior[:, :3] = np.eye(3)
            prior[:, 3:6] = float(ti) * np.eye(3)
            if fitted_g:
                prior[:, 6] = np.asarray([0.0, 0.0, -0.5 * ti * ti])
                prior_target = np.asarray(size_state.size_prior_xyz_world_m, float)
            else:
                prior_target = np.asarray(size_state.size_prior_xyz_world_m, float) + np.asarray([0.0, 0.0, 0.5 * 9.81 * ti * ti])
            a_rows.append(0.25 * prior); b_rows.append(0.25 * prior_target)
    try:
        solution = np.linalg.lstsq(np.vstack(a_rows), np.concatenate(b_rows), rcond=None)[0]
    except np.linalg.LinAlgError:
        return answer
    g = float(solution[6]) if fitted_g else 9.81
    if not math.isfinite(g) or not 0.0 < g < 30.0:
        return answer
    p0, velocity = solution[:3], solution[3:6]
    for item, ti in zip(segment, t):
        point = p0 + velocity * ti + np.asarray([0.0, 0.0, -0.5 * g * ti * ti])
        if np.all(np.isfinite(point)) and -60 <= point[0] <= 60 and -45 <= point[1] <= 45 and 0 <= point[2] <= 30:
            answer[item["key"]] = point.astype(float).tolist()
    return answer


def _extended_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = summarize_3d(rows)
    errors = [float(np.linalg.norm(np.asarray(row["pred_xyz"], float) - np.asarray(row["gt_xyz"], float)))
              for row in rows if row.get("pred_xyz") is not None and row.get("gt_xyz") is not None]
    for threshold in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0):
        summary[f"Precision_at_{threshold:g}m"] = None if not errors else float(np.mean(np.asarray(errors) <= threshold))
    return summary


def _hybrid_rows(collected: dict[str, list[dict[str, Any]]], config: HybridConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select ground/airborne geometry without inspecting the ISSIA GT.

    ``V03_SIZE_PRIOR`` supplies only an observation-derived height proxy.  The
    ground branch uses the ground-plane solution; airborne prefers a ballistic
    result only if it agrees with the independent apparent-size depth estimate.
    """
    by_method = {method: {row["record_id"]: row for row in rows} for method, rows in collected.items()}
    source_rows = collected["V03_SIZE_PRIOR"]
    groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        groups[(int(row["camera"]), int(row["segment"]))].append(row)
    output: list[dict[str, Any]] = []
    regime_counts: Counter[str] = Counter()
    method_counts: Counter[str] = Counter()
    half = max(0, int(config.median_window_frames) // 2)
    for sequence in groups.values():
        sequence.sort(key=lambda row: int(row["frame"]))
        proxy = [None if row.get("pred_xyz") is None else float(row["pred_xyz"][2]) for row in sequence]
        for index, size_row in enumerate(sequence):
            window = [z for z in proxy[max(0, index-half):index+half+1] if z is not None and math.isfinite(z)]
            stabilized_height = None if not window else float(np.median(window))
            regime = "GROUND" if stabilized_height is not None and stabilized_height <= config.ground_proxy_height_m else "AIRBORNE"
            record_id = size_row["record_id"]
            ground = by_method["GROUND_PLANE"][record_id].get("pred_xyz")
            ballistic_fit = by_method["BALLISTIC_FIT_G"][record_id].get("pred_xyz")
            ballistic_fixed = by_method["BALLISTIC_G9_81"][record_id].get("pred_xyz")
            temporal = by_method["V04_TEMPORAL"][record_id].get("pred_xyz")
            size = size_row.get("pred_xyz")
            prediction = None
            selected = "HYBRID_NO_PREDICTION"
            consensus = None
            if regime == "GROUND" and ground is not None:
                prediction, selected = ground, "HYBRID_GROUND_PLANE"
            elif size is not None:
                for candidate, label in ((ballistic_fit, "HYBRID_BALLISTIC_FIT_G"), (ballistic_fixed, "HYBRID_BALLISTIC_G9_81")):
                    if candidate is None:
                        continue
                    distance = float(np.linalg.norm(np.asarray(candidate, float) - np.asarray(size, float)))
                    if distance <= config.ballistic_consensus_distance_m:
                        prediction, selected, consensus = candidate, label, distance
                        break
                if prediction is None:
                    prediction, selected = size, "HYBRID_SIZE_PRIOR_AIRBORNE"
            elif temporal is not None:
                prediction, selected = temporal, "HYBRID_TEMPORAL_FALLBACK"
            regime_counts[regime] += 1
            method_counts[selected] += 1
            row = dict(size_row)
            row["pred_xyz"] = prediction
            row["status"] = "VALID" if prediction is not None else "NO_PREDICTION"
            row["hybrid"] = {"regime": regime, "size_height_proxy_m": stabilized_height, "selected_method": selected, "ballistic_consensus_distance_m": consensus}
            output.append(row)
    return output, {"config": config.to_dict(), "regime_counts": dict(regime_counts), "selected_method_counts": dict(method_counts)}


def _stratum(z: float) -> str:
    if z <= 0.25:
        return "ground"
    if z <= 1.0:
        return "low_airborne"
    return "high_airborne"


def _dynamics(rows: list[dict[str, Any]], fps: float) -> dict[str, Any]:
    velocity_errors: list[float] = []
    acceleration_errors: list[float] = []
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("pred_xyz") is not None:
            grouped[(int(row["camera"]), int(row["segment"]))].append(row)
    for sequence in grouped.values():
        sequence.sort(key=lambda row: int(row["frame"]))
        pred = np.asarray([r["pred_xyz"] for r in sequence], float)
        gt = np.asarray([r["gt_xyz"] for r in sequence], float)
        if len(sequence) >= 2:
            velocity_errors.extend(np.linalg.norm(np.diff(pred, axis=0) * fps - np.diff(gt, axis=0) * fps, axis=1).tolist())
        if len(sequence) >= 3:
            acceleration_errors.extend(np.linalg.norm(np.diff(pred, n=2, axis=0) * fps * fps - np.diff(gt, n=2, axis=0) * fps * fps, axis=1).tolist())
    return {
        "velocity_MAE_mps": None if not velocity_errors else float(np.mean(velocity_errors)),
        "acceleration_MAE_mps2": None if not acceleration_errors else float(np.mean(acceleration_errors)),
    }


def benchmark_issia3d_temporal(
    *, csv_path: str | Path, calibration_path: str | Path, output_dir: str | Path,
    cameras: Iterable[int] = (1, 2), fps: float = 25.0, ball_radius_m: float = 0.11,
    protocol: str = "v043-compatible", hybrid_config: dict[str, Any] | HybridConfig | None = None,
    write_artifacts: bool = True, include_internal_rows: bool = False, max_segment_frames: int | None = None,
) -> dict[str, Any]:
    if protocol != "v043-compatible":
        raise ValueError("only protocol v043-compatible is supported")
    preflight = inspect_issia3d(csv_path, calibration_path)
    if preflight["status"] != "READY":
        raise ValueError("ISSIA-3D preflight is incomplete")
    raw_rows = _read_rows(csv_path)
    calibration = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
    prepared: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(raw_rows):
        gt_sn = _xyz(row.get("ball_3D"))
        frame = _number(row.get("frame"))
        if gt_sn is None or frame is None:
            continue
        gt = soccernet_xyz_to_stage6(gt_sn).astype(float).tolist()
        for cam in cameras:
            x, y, diameter = (_number(row.get(f"x_cam{cam}")), _number(row.get(f"y_cam{cam}")), _number(row.get(f"opt_d_cam{cam}")))
            if x is None or y is None or diameter is None or diameter <= 0.0:
                continue
            camera = camera_from_soccernet_calibration(calibration[f"cam{cam}"], frame_index=int(frame), image_width=1920, image_height=1080)
            camera.timestamp_sec = float(frame) / float(fps)
            prepared[int(cam)].append({
                "key": index, "frame": int(frame), "camera": camera,
                "candidate": _candidate(int(frame), x, y, diameter), "gt_xyz": gt,
            })

    collected: dict[str, list[dict[str, Any]]] = {method: [] for method in METHODS}
    segment_count = Counter()
    for cam, observations in prepared.items():
        observations.sort(key=lambda item: (item["frame"], item["key"]))
        for segment_id, segment in enumerate(_segments(observations, max_segment_frames)):
            segment_count[str(cam)] += 1
            frames = list(range(len(segment)))
            by_frame: dict[int, CameraStateLite] = {}
            selected: dict[int, BallCandidate2D] = {}
            for local, item in enumerate(segment):
                item["local"] = local
                camera = item["camera"]
                camera.frame_index = local
                camera.timestamp_sec = local / float(fps)
                candidate = item["candidate"]
                candidate.frame_index = local
                by_frame[local] = camera; selected[local] = candidate
            temporal = refine_temporal_trajectory(cameras_by_frame=by_frame, selected=selected, frame_indices=frames, fps=fps, ball_radius_m=ball_radius_m)
            ballistic_g = _ballistic(segment, fitted_g=False, fps=fps)
            ballistic_fit = _ballistic(segment, fitted_g=True, fps=fps)
            for item in segment:
                local = item["local"]
                size = estimate_frame(by_frame[local], selected[local], fps=fps, ball_radius_m=ball_radius_m, mode="size-prior")
                ground = estimate_frame(by_frame[local], selected[local], fps=fps, ball_radius_m=ball_radius_m, mode="ground-only")
                temporal_result = temporal.frames[local]
                predictions = {
                    "V03_SIZE_PRIOR": size.selected_center_xyz_world_m,
                    "GROUND_PLANE": ground.selected_center_xyz_world_m,
                    "V04_TEMPORAL": temporal_result.xyz_world_m,
                    "BALLISTIC_G9_81": ballistic_g[item["key"]],
                    "BALLISTIC_FIT_G": ballistic_fit[item["key"]],
                }
                for method, prediction in predictions.items():
                    collected[method].append({
                        "record_id": f"cam{cam}/frame{item['frame']}/row{item['key']}", "camera": cam,
                        "frame": item["frame"], "segment": segment_id, "gt_xyz": item["gt_xyz"],
                        "pred_xyz": prediction, "status": "VALID" if prediction is not None else "NO_PREDICTION",
                    })

    config = hybrid_config if isinstance(hybrid_config, HybridConfig) else HybridConfig.from_mapping(hybrid_config)
    hybrid_rows, hybrid_diagnostics = _hybrid_rows(collected, config)
    collected["V044_HYBRID"] = hybrid_rows
    methods: dict[str, Any] = {}
    for method, rows in collected.items():
        by_height = {name: _extended_summary([r for r in rows if _stratum(float(r["gt_xyz"][2])) == name])
                     for name in ("ground", "low_airborne", "high_airborne")}
        methods[method] = {"all": _extended_summary(rows), "by_gt_height": by_height, "dynamics": _dynamics(rows, fps)}
    common_ids = set.intersection(*[{row["record_id"] for row in rows if row.get("pred_xyz") is not None} for rows in collected.values()])
    common = {method: _extended_summary([row for row in rows if row["record_id"] in common_ids]) for method, rows in collected.items()}
    report = {
        "schema_version": "stage6-issia3d-temporal-benchmark-1.0", "status": "COMPLETE",
        "runtime_provenance": runtime_provenance(), "preflight": preflight,
        "protocol": {"name": protocol, "cameras": list(map(int, cameras)), "fps": float(fps), "ball_radius_m": float(ball_radius_m),
                     "gt_usage": "evaluation_only", "segments_per_camera": dict(segment_count),
                     "methods": [*METHODS, "V044_HYBRID"]},
        "methods": methods, "hybrid_diagnostics": hybrid_diagnostics,
        "all_method_common_frames": {"frames": len(common_ids), "metrics": common},
    }
    if include_internal_rows:
        # In-memory-only cache used by calibration. It is deliberately omitted
        # from JSON reports because it contains raw prediction rows.
        report["_internal_collected"] = collected
    if not write_artifacts:
        return report
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark_issia3d_temporal.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (out / "frame_errors.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "record_id", "camera", "frame", "segment", "gt_x", "gt_y", "gt_z", "pred_x", "pred_y", "pred_z", "error_3d_m"])
        writer.writeheader()
        for method, rows in collected.items():
            for row in rows:
                gt = row["gt_xyz"]; pred = row.get("pred_xyz")
                writer.writerow({"method": method, "record_id": row["record_id"], "camera": row["camera"], "frame": row["frame"], "segment": row["segment"],
                                 "gt_x": gt[0], "gt_y": gt[1], "gt_z": gt[2], "pred_x": None if pred is None else pred[0], "pred_y": None if pred is None else pred[1], "pred_z": None if pred is None else pred[2],
                                 "error_3d_m": None if pred is None else float(np.linalg.norm(np.asarray(pred)-np.asarray(gt)))})
    lines = ["# ISSIA-3D temporal geometry benchmark", "", f"Common predicted frames: {len(common_ids)}", "", "| Method | Coverage | MAE 3D (m) | MAE Z (m) |", "|---|---:|---:|---:|"]
    for method, result in methods.items():
        metric = result["all"]
        lines.append(f"| {method} | {metric['coverage']:.3f} | {metric['MAE_3D_m'] if metric['MAE_3D_m'] is not None else '—'} | {metric['MAE_Z_m'] if metric['MAE_Z_m'] is not None else '—'} |")
    (out / "benchmark_issia3d_temporal.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def _calibration_score(report: dict[str, Any]) -> dict[str, float]:
    metrics = report["methods"]["V044_HYBRID"]
    all_metrics = metrics["all"]
    strata = metrics["by_gt_height"]
    heights = [strata[name]["MAE_3D_m"] for name in ("ground", "low_airborne", "high_airborne")]
    if all_metrics["MAE_3D_m"] is None or all(value is None for value in heights):
        return {"objective": float("inf"), "height_balanced_mae_m": float("inf")}
    usable_heights = [float(value) for value in heights if value is not None]
    balanced = float(np.mean(usable_heights))
    objective = (
        0.35 * float(all_metrics["MAE_3D_m"])
        + 0.50 * balanced
        + 0.15 * float(all_metrics["P90_3D_m"])
        + 3.0 * max(0.0, 0.98 - float(all_metrics["coverage"]))
    )
    return {"objective": float(objective), "height_balanced_mae_m": balanced}


def calibrate_issia3d_hybrid(
    *, csv_path: str | Path, calibration_path: str | Path, output_dir: str | Path,
    cameras: Iterable[int] = (3, 4, 5, 6), fps: float = 25.0, protocol: str = "v043-compatible",
    ground_proxy_height_grid_m: Iterable[float] = (0.35, 0.50, 0.65, 0.80),
    ballistic_consensus_grid_m: Iterable[float] = (1.5, 2.5, 3.5, 5.0),
) -> dict[str, Any]:
    selected = tuple(sorted({int(camera) for camera in cameras}))
    if set(selected) & {1, 2}:
        raise ValueError("camera 1–2 are held-out test cameras and cannot be used for hybrid calibration")
    if not selected or any(camera not in {3, 4, 5, 6} for camera in selected):
        raise ValueError("calibration requires a non-empty subset of cameras 3–6")
    # Compute all five frozen v0.4.3 baselines once. Grid search below touches
    # only the lightweight GT-free selector, so it cannot subtly alter them.
    base = benchmark_issia3d_temporal(
        csv_path=csv_path, calibration_path=calibration_path, output_dir=output_dir,
        cameras=selected, fps=fps, protocol=protocol, write_artifacts=False, include_internal_rows=True,
        max_segment_frames=80,
    )
    collected = base.pop("_internal_collected")
    height_grid = tuple(float(value) for value in ground_proxy_height_grid_m)
    consensus_grid = tuple(float(value) for value in ballistic_consensus_grid_m)
    if not height_grid or not consensus_grid or min(height_grid) <= 0.0 or min(consensus_grid) <= 0.0:
        raise ValueError("hybrid grids must contain positive values")
    trials: list[dict[str, Any]] = []
    for height in height_grid:
        for consensus in consensus_grid:
            config = HybridConfig(height, consensus)
            rows, diagnostics = _hybrid_rows(collected, config)
            metrics = {
                "all": _extended_summary(rows),
                "by_gt_height": {name: _extended_summary([row for row in rows if _stratum(float(row["gt_xyz"][2])) == name]) for name in ("ground", "low_airborne", "high_airborne")},
                "dynamics": _dynamics(rows, fps),
            }
            report = {"methods": {"V044_HYBRID": metrics}, "hybrid_diagnostics": diagnostics}
            score = _calibration_score(report)
            trials.append({"config": config.to_dict(), **score, "metrics": metrics, "hybrid_diagnostics": diagnostics})
    best = min(trials, key=lambda trial: trial["objective"])
    result = {
        "schema_version": "stage6-hybrid-calibration-1.0", "status": "COMPLETE",
        "runtime_provenance": runtime_provenance(),
        "protocol": {"name": protocol, "cameras": list(selected), "fps": float(fps), "held_out_cameras": [1, 2], "calibration_max_segment_frames": 80},
        "grid": {"ground_proxy_height_m": list(height_grid), "ballistic_consensus_distance_m": list(consensus_grid)},
        "objective": "0.35*MAE_all + 0.50*height_balanced_MAE + 0.15*P90 + 3*max(0, 0.98-coverage)",
        "best": best, "trials": trials,
        "runtime_note": "The saved config has no ball_3D or GT-height inputs; they are used only to choose this train/validation setting.",
    }
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "hybrid_calibration.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
