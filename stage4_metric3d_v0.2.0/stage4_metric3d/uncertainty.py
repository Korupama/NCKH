from __future__ import annotations

from collections import Counter
from dataclasses import replace
import hashlib
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

from .optimizer import FrameInput, OptimizationResult, solve_metric_pose
from .schemas import Stage4Config
from .wholebody import LEGAL_GEOMETRY_CANDIDATE_23, POSE23_NAMES


STATE_SIGMA_MULTIPLIER = {
    "VALID": 1.0,
    "LOW_MODEL_EVIDENCE": 1.35,
    "GEOMETRIC_OUTLIER": 2.0,
    "TEMPORAL_OUTLIER": 1.65,
    "LEFT_RIGHT_SUSPECT": 1.8,
    "TEMPORAL_IMPUTED": 2.0,
    "MISSING": 3.0,
}


def _stable_seed(base_seed: int, frame_index: int, track_id: str) -> int:
    digest = hashlib.sha256(str(track_id).encode("utf-8")).digest()
    track_component = int.from_bytes(digest[:8], "little", signed=False)
    return int((int(base_seed) + 1_000_003 * int(frame_index) + track_component) % (2**63 - 1))


def _pixel_sigma(frame: FrameInput, states23: Sequence[str], config: Stage4Config) -> np.ndarray:
    bbox = np.asarray(frame.bbox_xyxy, dtype=np.float64)
    h = max(float(bbox[3] - bbox[1]), 1.0) if np.isfinite(bbox).all() else 80.0
    resolution_mult = float(np.clip(np.sqrt(80.0 / h), 0.8, 2.0))
    return np.asarray([
        np.clip(
            config.base_keypoint_sigma_px
            * resolution_mult
            * STATE_SIGMA_MULTIPLIER.get(str(states23[j]), 1.5),
            0.5,
            config.max_keypoint_sigma_px,
        )
        for j in range(23)
    ], dtype=np.float64)


def _clone_with_pixel_noise(
    frame: FrameInput,
    states23: Sequence[str],
    config: Stage4Config,
    rng: np.random.Generator,
) -> Tuple[FrameInput, np.ndarray]:
    sigma = _pixel_sigma(frame, states23, config)
    uv = np.asarray(frame.uv23, dtype=np.float64).copy()
    good = np.isfinite(uv).all(axis=1) & (frame.state_weights23 > 0)
    uv[good] += rng.normal(0.0, sigma[good, None], size=(int(good.sum()), 2))
    return FrameInput(
        frame_index=frame.frame_index,
        camera=frame.camera,
        uv23=uv,
        state_weights23=frame.state_weights23.copy(),
        bbox_xyxy=frame.bbox_xyxy.copy(),
        contact=frame.contact,
        initializer=frame.initializer,
        uv_source23=frame.uv_source23,
    ), sigma


def _sample_acceptance(
    result: OptimizationResult,
    *,
    mode: str,
    config: Stage4Config,
) -> Tuple[bool, str]:
    geometry = result.diagnostics.get("geometry_quality_gate") or {}
    if not bool(geometry.get("plausible")):
        # A track-level negative-Z fraction is nearly continuous over hundreds
        # of joints. At one frame it changes in 1/23 increments, so the default
        # 8% threshold rejects exactly two mildly negative joints (8.70%). Keep
        # that boundary sample as explicitly degraded evidence when every other
        # geometry component passes; full-window mode retains the strict gate.
        other_geometry_ok = all(bool(geometry.get(key)) for key in (
            "finite_xyz", "height_in_bounds", "bone_plausible", "ground_plausible",
        ))
        negative_fraction = float(geometry.get("negative_z_joint_fraction", float("inf")))
        single_frame_z_limit = max(float(config.geometry_max_negative_z_fraction), 2.0 / 23.0 + 1e-12)
        if mode == "selected_frame" and other_geometry_ok and negative_fraction <= single_frame_z_limit:
            return True, "DEGRADED_SINGLE_FRAME_Z"
        return False, "IMPLAUSIBLE_GEOMETRY"
    if result.status == "CONVERGED":
        return True, "VALID"
    if result.status == "MAX_NFEV":
        return True, "DEGRADED_MAX_NFEV"
    return False, str(result.status or "FAILED")


def _distribution(values: np.ndarray) -> Dict[str, float | int | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return {"count": 0, "mean_m": None, "std_m": None, "q025_m": None, "q50_m": None, "q975_m": None}
    return {
        "count": int(finite.size),
        "mean_m": float(np.mean(finite)),
        "std_m": float(np.std(finite, ddof=1 if finite.size > 1 else 0)),
        "q025_m": float(np.percentile(finite, 2.5)),
        "q50_m": float(np.percentile(finite, 50.0)),
        "q975_m": float(np.percentile(finite, 97.5)),
    }


def _longitudinal_summary(arr: np.ndarray) -> Dict[str, object]:
    legal = np.asarray(LEGAL_GEOMETRY_CANDIDATE_23, dtype=bool)
    per_anchor: Dict[str, object] = {}
    for j, name in enumerate(POSE23_NAMES):
        if legal[j]:
            per_anchor[name] = _distribution(arr[:, j, 0])

    legal_x = arr[:, legal, 0]
    min_x = np.full(arr.shape[0], np.nan, dtype=np.float64)
    max_x = np.full(arr.shape[0], np.nan, dtype=np.float64)
    for i, row in enumerate(legal_x):
        finite = row[np.isfinite(row)]
        if finite.size:
            min_x[i] = float(np.min(finite))
            max_x[i] = float(np.max(finite))
    return {
        "attack_direction_applied": False,
        "per_legal_anchor_x": per_anchor,
        "legal_extrema_x": {
            "min_legal_x": _distribution(min_x),
            "max_legal_x": _distribution(max_x),
        },
    }


def _sampling_stability(arr: np.ndarray) -> Dict[str, object]:
    """Report prefix stability for deterministic adaptive sample schedules."""
    checkpoints = [n for n in (16, 32, 64, 128, 256) if n <= arr.shape[0]]
    if arr.shape[0] not in checkpoints:
        checkpoints.append(int(arr.shape[0]))
    checkpoints = sorted(set(checkpoints))
    legal = np.asarray(LEGAL_GEOMETRY_CANDIDATE_23, dtype=bool)
    rows = []
    previous = None
    for count in checkpoints:
        subset = arr[:count]
        std_x = np.nanstd(subset[:, legal, 0], axis=0, ddof=1 if count > 1 else 0)
        median_std_x = float(np.nanmedian(std_x))
        legal_x = subset[:, legal, 0]
        min_x = np.nanmin(legal_x, axis=1)
        max_x = np.nanmax(legal_x, axis=1)
        summary = {
            "samples": int(count),
            "median_legal_anchor_std_x_m": median_std_x,
            "min_legal_x_std_m": float(np.nanstd(min_x, ddof=1 if count > 1 else 0)),
            "max_legal_x_std_m": float(np.nanstd(max_x, ddof=1 if count > 1 else 0)),
            "relative_change_from_previous": None,
        }
        if previous is not None:
            denom = max(abs(previous), 1e-12)
            summary["relative_change_from_previous"] = float(abs(median_std_x - previous) / denom)
        rows.append(summary)
        previous = median_std_x
    return {
        "schedule": rows,
        "provisional_stable_at_10pct": bool(
            len(rows) >= 2
            and rows[-1]["relative_change_from_previous"] is not None
            and rows[-1]["relative_change_from_previous"] <= 0.10
        ),
    }


def _not_run(samples_requested: int, mode: str) -> Dict[str, object]:
    scope = "stage3_pixel_only_camera_fixed_selected_frame" if mode == "selected_frame" else "stage3_pixel_only_camera_fixed_full_window"
    return {
        "schema_version": "stage4-longitudinal-sensitivity-1.0",
        "status": "NOT_RUN",
        "scope": scope,
        "calibrated": False,
        "interpretation": "engineering sensitivity, not calibrated posterior probability",
        "samples_requested": int(samples_requested),
        "samples_usable": 0,
        "samples_successful": 0,
        "usable_fraction": 0.0,
        "std_xyz_m": None,
        "q025_xyz_m": None,
        "q975_xyz_m": None,
        "longitudinal": None,
    }


def pixel_sensitivity_monte_carlo(
    frames: Sequence[FrameInput],
    solved_track: OptimizationResult,
    config: Stage4Config,
    *,
    selected_frame_index: int,
    states23_by_frame: Mapping[int, Sequence[str]],
    track_id: str,
    mode: str | None = None,
) -> Dict[str, object]:
    """Estimate Stage-3 pixel sensitivity while keeping the camera fixed.

    ``selected_frame`` is the cheap conditional smoke mode. ``full_window``
    perturbs every usable temporal observation and reruns the production
    temporal objective. Both are sensitivity analyses, not calibrated
    posterior intervals.
    """
    selected_mode = str(mode or config.uncertainty_mode)
    if selected_mode not in {"selected_frame", "full_window"}:
        raise ValueError(f"Unsupported uncertainty mode: {selected_mode}")
    n = int(config.uncertainty_samples)
    if n <= 0:
        return _not_run(0, selected_mode)

    source_frames = list(frames)
    selected_pos = next((i for i, frame in enumerate(source_frames) if int(frame.frame_index) == int(selected_frame_index)), None)
    if selected_pos is None:
        result = _not_run(n, selected_mode)
        result.update({"status": "NOT_APPLICABLE", "reason": "selected_frame_input_missing"})
        return result

    if selected_mode == "selected_frame":
        sampled_source = [source_frames[selected_pos]]
        selected_sample_pos = 0
        temporal_weight = 0.0
    else:
        sampled_source = source_frames
        selected_sample_pos = selected_pos
        temporal_weight = config.temporal_weight

    effective_seed = _stable_seed(config.uncertainty_seed, selected_frame_index, track_id)
    rng = np.random.default_rng(effective_seed)
    samples = []
    termination_counts: Counter[str] = Counter()
    acceptance_counts: Counter[str] = Counter()
    failure_counts: Counter[str] = Counter()
    rejection_examples = []
    selected_sigma = None
    local_cfg = replace(
        config,
        temporal_weight=temporal_weight,
        uncertainty_samples=0,
        optimizer_max_nfev=min(config.optimizer_max_nfev, config.uncertainty_optimizer_max_nfev),
        optimizer_verbose=0,
    )
    frozen_xyz_by_frame = {}
    if hasattr(solved_track, "frame_indices") and hasattr(solved_track, "xyz_world_m"):
        frame_indices = np.asarray(solved_track.frame_indices, dtype=np.int64).reshape(-1)
        frozen_xyz = np.asarray(solved_track.xyz_world_m, dtype=np.float64)
        if frozen_xyz.shape == (len(frame_indices), 23, 3):
            frozen_xyz_by_frame = {
                int(frame_index): frozen_xyz[i]
                for i, frame_index in enumerate(frame_indices)
            }
    initial_xyz = None
    if all(int(frame.frame_index) in frozen_xyz_by_frame for frame in sampled_source):
        initial_xyz = np.stack([frozen_xyz_by_frame[int(frame.frame_index)] for frame in sampled_source])

    for _ in range(n):
        perturbed_frames = []
        for frame in sampled_source:
            states = states23_by_frame.get(int(frame.frame_index), ("MISSING",) * 23)
            perturbed, sigma = _clone_with_pixel_noise(frame, states, config, rng)
            perturbed_frames.append(perturbed)
            if int(frame.frame_index) == int(selected_frame_index):
                selected_sigma = sigma
        try:
            solved = solve_metric_pose(
                perturbed_frames,
                local_cfg,
                fixed_height_m=solved_track.estimated_height_m,
                fixed_depth_beta_m=(solved_track.depth_beta_m if solved_track.initializer_used else None),
                initial_xyz_world_m=initial_xyz,
            )
        except Exception as exc:
            message = " ".join(str(exc).split())[:160]
            failure_counts[f"EXCEPTION_{type(exc).__name__}: {message}"] += 1
            continue

        termination_counts[str(solved.status)] += 1
        usable, sample_quality = _sample_acceptance(solved, mode=selected_mode, config=config)
        if not usable:
            failure_counts[sample_quality] += 1
            if len(rejection_examples) < 3:
                rejection_examples.append({
                    "reason": sample_quality,
                    "termination": str(solved.status),
                    "geometry_quality_gate": solved.diagnostics.get("geometry_quality_gate"),
                })
            continue
        xyz = np.asarray(solved.xyz_world_m[selected_sample_pos], dtype=np.float64)
        if xyz.shape != (23, 3) or not np.isfinite(xyz).any():
            failure_counts["NONFINITE_SELECTED_POSE"] += 1
            continue
        acceptance_counts[sample_quality] += 1
        samples.append(xyz)

    scope = (
        "stage3_pixel_only_camera_fixed_selected_frame"
        if selected_mode == "selected_frame"
        else "stage3_pixel_only_camera_fixed_full_window"
    )
    usable_fraction = len(samples) / max(n, 1)
    common = {
        "schema_version": "stage4-longitudinal-sensitivity-1.0",
        "scope": scope,
        "mode": selected_mode,
        "calibrated": False,
        "interpretation": "engineering sensitivity, not calibrated posterior probability",
        "camera_sampled": False,
        "attack_direction_applied": False,
        "track_id": str(track_id),
        "selected_frame": int(selected_frame_index),
        "base_seed": int(config.uncertainty_seed),
        "effective_seed": effective_seed,
        "samples_requested": n,
        "samples_usable": len(samples),
        "samples_successful": len(samples),
        "usable_fraction": float(usable_fraction),
        "sample_acceptance_counts": dict(acceptance_counts),
        "degraded_sample_fraction": float(
            sum(value for key, value in acceptance_counts.items() if key.startswith("DEGRADED"))
            / max(len(samples), 1)
        ),
        "sample_termination_counts": dict(termination_counts),
        "sample_failure_counts": dict(failure_counts),
        "sample_rejection_examples": rejection_examples,
        "selected_frame_pixel_sigma_px": None if selected_sigma is None else selected_sigma.tolist(),
    }
    if not samples:
        return {
            **common,
            "status": "FAILED",
            "std_xyz_m": None,
            "q025_xyz_m": None,
            "q975_xyz_m": None,
            "longitudinal": None,
        }

    arr = np.stack(samples, axis=0)
    std = np.nanstd(arr, axis=0, ddof=1 if len(arr) > 1 else 0)
    q025 = np.nanpercentile(arr, 2.5, axis=0)
    q975 = np.nanpercentile(arr, 97.5, axis=0)
    degraded_fraction = float(common["degraded_sample_fraction"])
    status = (
        "OK"
        if usable_fraction >= float(config.uncertainty_min_usable_fraction) and degraded_fraction <= 0.20
        else "DEGRADED"
    )
    return {
        **common,
        "status": status,
        "std_xyz_m": std.tolist(),
        "q025_xyz_m": q025.tolist(),
        "q975_xyz_m": q975.tolist(),
        "longitudinal": _longitudinal_summary(arr),
        "sampling_stability": _sampling_stability(arr),
    }


def selected_frame_pixel_monte_carlo(
    frame: FrameInput,
    solved_track: OptimizationResult,
    config: Stage4Config,
    *,
    states23: Tuple[str, ...],
    track_id: str = "unknown",
) -> Dict[str, object]:
    """Backward-compatible selected-frame wrapper used by the v0.1 runner."""
    return pixel_sensitivity_monte_carlo(
        [frame],
        solved_track,
        config,
        selected_frame_index=frame.frame_index,
        states23_by_frame={frame.frame_index: states23},
        track_id=track_id,
        mode="selected_frame",
    )
