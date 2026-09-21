from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence
import math

import numpy as np

from ..camera import CameraStateLite
from ..contracts import BallCandidate2D
from .localization import angular_radius_from_bbox


@dataclass
class TemporalRefinementConfig:
    """Configuration for Stage-6 v0.4 temporal ball geometry.

    The optimizer intentionally has no dependency on scipy.  Both stages are
    weighted least-squares systems solved with NumPy so the production package
    remains light and deterministic.
    """

    # Pass A: log-diameter smoothing / short-gap interpolation.
    diameter_second_diff_weight: float = 8.0
    diameter_first_diff_weight: float = 0.10
    diameter_huber_delta_log: float = 0.12
    diameter_irls_iterations: int = 4

    # Short gaps can be bridged. Longer gaps start a new temporal segment.
    max_gap_frames: int = 3
    min_segment_observations: int = 3

    # Pass B: ray-range trajectory fit.
    range_sigma_fraction: float = 0.08
    range_huber_delta_sigma: float = 2.5
    range_irls_iterations: int = 4
    position_second_diff_sigma_m: float = 0.75
    position_third_diff_sigma_m: float = 0.50
    ground_consistency_log_tolerance: float = 0.12
    ground_anchor_sigma_m: float = 0.75
    max_range_deviation_fraction: float = 0.35

    # Physical feasibility box in canonical Stage-6 coordinates.
    pitch_margin_m: float = 6.0
    min_height_m: float = 0.0
    max_height_m: float = 30.0
    max_range_m: float = 220.0

    # Hard-bound enforcement inside the least-squares system.
    bound_weight: float = 1.0e5
    ridge: float = 1.0e-8
    min_observation_confidence: float = 0.05

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TemporalFrameResult:
    frame_index: int
    status: str
    center_uv: Optional[list[float]] = None
    center_imputed: bool = False
    raw_diameter_px: Optional[float] = None
    refined_diameter_px: Optional[float] = None
    optimized_diameter_px: Optional[float] = None
    diameter_imputed: bool = False
    observation_confidence: float = 0.0
    raw_size_range_m: Optional[float] = None
    optimized_range_m: Optional[float] = None
    range_bounds_m: Optional[list[float]] = None
    ground_compatible: bool = False
    ground_range_m: Optional[float] = None
    xyz_world_m: Optional[list[float]] = None
    size_residual_sigma: Optional[float] = None
    segment_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TemporalRefinementResult:
    frames: Dict[int, TemporalFrameResult]
    diagnostics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frames": {str(k): v.to_dict() for k, v in self.frames.items()},
            "diagnostics": self.diagnostics,
        }


def _candidate_confidence(candidate: BallCandidate2D | None, floor: float) -> float:
    if candidate is None:
        return 0.0
    detector = float(np.clip(candidate.detector_score, 0.0, 1.0))
    ranking = float(np.clip(candidate.ranking_score, 0.0, 1.0))
    pitch = float(np.clip(candidate.pitch_prior, 0.0, 1.0))
    # Ranking already includes the pitch prior in the current contract.  Blend
    # it with raw detector confidence rather than multiplying twice.
    confidence = math.sqrt(max(detector, 1e-9) * max(ranking, 1e-9))
    confidence *= max(0.25, pitch)
    return float(np.clip(confidence, floor, 1.0))


def _segments_from_observations(mask: Sequence[bool], max_gap_frames: int, min_observations: int) -> list[tuple[int, int]]:
    observed = [i for i, flag in enumerate(mask) if flag]
    if not observed:
        return []
    segments: list[tuple[int, int]] = []
    start = observed[0]
    prev = observed[0]
    obs_count = 1
    for idx in observed[1:]:
        missing_between = idx - prev - 1
        if missing_between > max_gap_frames:
            if obs_count >= min_observations:
                segments.append((start, prev))
            start = idx
            obs_count = 1
        else:
            obs_count += 1
        prev = idx
    if obs_count >= min_observations:
        segments.append((start, prev))
    return segments


def _difference_matrix(n: int, order: int) -> np.ndarray:
    if order == 1:
        if n < 2:
            return np.zeros((0, n), dtype=float)
        out = np.zeros((n - 1, n), dtype=float)
        for i in range(n - 1):
            out[i, i] = -1.0
            out[i, i + 1] = 1.0
        return out
    if order == 2:
        if n < 3:
            return np.zeros((0, n), dtype=float)
        out = np.zeros((n - 2, n), dtype=float)
        for i in range(n - 2):
            out[i, i] = 1.0
            out[i, i + 1] = -2.0
            out[i, i + 2] = 1.0
        return out
    raise ValueError("order must be 1 or 2")


def _solve_log_diameter_segment(
    raw_diameters: np.ndarray,
    confidences: np.ndarray,
    config: TemporalRefinementConfig,
) -> np.ndarray:
    n = len(raw_diameters)
    observed = np.isfinite(raw_diameters) & (raw_diameters > 0.0)
    y = np.zeros(n, dtype=float)
    y[observed] = np.log(raw_diameters[observed])
    base_weights = np.where(observed, np.clip(confidences, config.min_observation_confidence, 1.0), 0.0)
    d1 = _difference_matrix(n, 1)
    d2 = _difference_matrix(n, 2)
    robust = np.ones(n, dtype=float)
    x = np.where(observed, y, 0.0)

    for _ in range(max(1, int(config.diameter_irls_iterations))):
        w = base_weights * robust
        normal = np.diag(w + config.ridge)
        rhs = w * y
        if len(d1):
            normal += float(config.diameter_first_diff_weight) * (d1.T @ d1)
        if len(d2):
            normal += float(config.diameter_second_diff_weight) * (d2.T @ d2)
        try:
            x = np.linalg.solve(normal, rhs)
        except np.linalg.LinAlgError:
            x = np.linalg.lstsq(normal, rhs, rcond=None)[0]
        residual = np.zeros(n, dtype=float)
        residual[observed] = x[observed] - y[observed]
        abs_res = np.abs(residual)
        robust = np.ones(n, dtype=float)
        big = observed & (abs_res > float(config.diameter_huber_delta_log))
        robust[big] = float(config.diameter_huber_delta_log) / np.maximum(abs_res[big], 1e-12)
    return np.exp(x)


def refine_diameter_series(
    frame_indices: Sequence[int],
    selected: Mapping[int, BallCandidate2D | None],
    *,
    config: TemporalRefinementConfig | None = None,
) -> Dict[int, Dict[str, Any]]:
    """Robustly smooth observed ball diameter and bridge only short gaps."""
    cfg = config or TemporalRefinementConfig()
    fis = [int(x) for x in frame_indices]
    raw = np.asarray([
        float(selected.get(fi).diameter_px) if selected.get(fi) is not None and float(selected.get(fi).diameter_px) > 0.0 else np.nan
        for fi in fis
    ], dtype=float)
    conf = np.asarray([_candidate_confidence(selected.get(fi), cfg.min_observation_confidence) for fi in fis], dtype=float)
    observed = np.isfinite(raw)
    segments = _segments_from_observations(observed.tolist(), cfg.max_gap_frames, cfg.min_segment_observations)
    out: Dict[int, Dict[str, Any]] = {
        fi: {
            "raw_diameter_px": None if not np.isfinite(raw[i]) else float(raw[i]),
            "refined_diameter_px": None,
            "diameter_imputed": False,
            "observation_confidence": float(conf[i]),
            "segment_id": None,
        }
        for i, fi in enumerate(fis)
    }
    for segment_id, (start, end) in enumerate(segments):
        seg_raw = raw[start : end + 1]
        seg_conf = conf[start : end + 1]
        refined = _solve_log_diameter_segment(seg_raw, seg_conf, cfg)
        for local, idx in enumerate(range(start, end + 1)):
            fi = fis[idx]
            out[fi]["refined_diameter_px"] = float(refined[local])
            out[fi]["diameter_imputed"] = not bool(np.isfinite(raw[idx]))
            out[fi]["segment_id"] = int(segment_id)
    # Observations in tiny isolated segments are kept raw; they are not used to
    # bridge missing frames because temporal evidence is insufficient.
    for i, fi in enumerate(fis):
        if out[fi]["refined_diameter_px"] is None and np.isfinite(raw[i]):
            out[fi]["refined_diameter_px"] = float(raw[i])
            out[fi]["segment_id"] = -1
    return out


def interpolate_center_series(
    frame_indices: Sequence[int],
    selected: Mapping[int, BallCandidate2D | None],
    *,
    max_gap_frames: int = 3,
) -> Dict[int, Dict[str, Any]]:
    fis = [int(x) for x in frame_indices]
    centers: list[Optional[np.ndarray]] = []
    for fi in fis:
        cand = selected.get(fi)
        centers.append(None if cand is None else np.asarray(cand.center_uv, dtype=float))
    out: Dict[int, Dict[str, Any]] = {
        fi: {"center_uv": None if centers[i] is None else centers[i].astype(float).tolist(), "center_imputed": False}
        for i, fi in enumerate(fis)
    }
    observed = [i for i, c in enumerate(centers) if c is not None]
    for left, right in zip(observed[:-1], observed[1:]):
        gap = right - left - 1
        if gap <= 0 or gap > int(max_gap_frames):
            continue
        a = centers[left]
        b = centers[right]
        assert a is not None and b is not None
        for idx in range(left + 1, right):
            t = (idx - left) / float(right - left)
            uv = (1.0 - t) * a + t * b
            fi = fis[idx]
            out[fi] = {"center_uv": uv.astype(float).tolist(), "center_imputed": True}
    return out


def _centered_square_bbox(center_uv: Sequence[float], diameter_px: float) -> list[float]:
    cx, cy = map(float, center_uv)
    half = 0.5 * float(diameter_px)
    return [cx - half, cy - half, cx + half, cy + half]


def _range_interval_for_box(
    camera: CameraStateLite,
    ray_direction: np.ndarray,
    *,
    ball_radius_m: float,
    pitch_margin_m: float,
    min_height_m: float,
    max_height_m: float,
    max_range_m: float,
) -> Optional[tuple[float, float]]:
    half_l = float(camera.pitch.get("length_m", 105.0)) / 2.0 + float(pitch_margin_m)
    half_w = float(camera.pitch.get("width_m", 68.0)) / 2.0 + float(pitch_margin_m)
    bounds = [(-half_l, half_l), (-half_w, half_w), (float(min_height_m), float(max_height_m))]
    c = np.asarray(camera.camera_center_world_m, dtype=float)
    d = np.asarray(ray_direction, dtype=float)
    lo = max(float(ball_radius_m) * 1.001, 1e-3)
    hi = float(max_range_m)
    for axis, (lower, upper) in enumerate(bounds):
        if abs(float(d[axis])) < 1e-12:
            if float(c[axis]) < lower or float(c[axis]) > upper:
                return None
            continue
        r0 = (lower - float(c[axis])) / float(d[axis])
        r1 = (upper - float(c[axis])) / float(d[axis])
        axis_lo, axis_hi = min(r0, r1), max(r0, r1)
        lo = max(lo, axis_lo)
        hi = min(hi, axis_hi)
    lo = max(lo, 1e-6)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    return float(lo), float(hi)


def _trajectory_segments(usable: Sequence[bool]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    start: Optional[int] = None
    for i, flag in enumerate(usable):
        if flag and start is None:
            start = i
        if (not flag) and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(usable) - 1))
    return out


def _solve_range_segment(
    cameras: Sequence[CameraStateLite],
    ray_dirs: np.ndarray,
    rho_obs: np.ndarray,
    ground_rho: np.ndarray,
    confidences: np.ndarray,
    bounds: Sequence[tuple[float, float]],
    timestamps: np.ndarray,
    config: TemporalRefinementConfig,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(cameras)
    if n == 1:
        rho = np.asarray([float(np.clip(rho_obs[0], bounds[0][0], bounds[0][1]))], dtype=float)
        return rho, np.asarray([0.0], dtype=float)

    centers = np.asarray([c.camera_center_world_m for c in cameras], dtype=float)
    sigma = np.maximum(float(config.range_sigma_fraction) * rho_obs, 0.05)
    base_size_scale = np.sqrt(np.clip(confidences, config.min_observation_confidence, 1.0)) / sigma
    robust = np.ones(n, dtype=float)
    rho = np.asarray([np.clip(rho_obs[i], bounds[i][0], bounds[i][1]) for i in range(n)], dtype=float)

    def build_system(active_bounds: Optional[Mapping[int, float]] = None) -> tuple[np.ndarray, np.ndarray]:
        rows: list[np.ndarray] = []
        rhs: list[float] = []
        for i in range(n):
            row = np.zeros(n, dtype=float)
            scale = float(base_size_scale[i] * math.sqrt(max(robust[i], 1e-9)))
            row[i] = scale
            rows.append(row)
            rhs.append(scale * float(rho_obs[i]))

        # Camera calibration is not perfectly frame-stable.  Penalize per-frame
        # world-position differences directly instead of dividing by dt^2/dt^3;
        # the latter amplifies sub-pixel camera jitter by ~fps^2 and caused the
        # v0.4 prototype to overfit physical bounds on real broadcast replay.
        if n >= 3 and config.position_second_diff_sigma_m > 0:
            scale = 1.0 / float(config.position_second_diff_sigma_m)
            for i in range(1, n - 1):
                const = centers[i - 1] - 2.0 * centers[i] + centers[i + 1]
                coeff = np.stack([ray_dirs[i - 1], -2.0 * ray_dirs[i], ray_dirs[i + 1]], axis=0)
                for axis in range(3):
                    row = np.zeros(n, dtype=float)
                    row[i - 1] = scale * coeff[0, axis]
                    row[i] = scale * coeff[1, axis]
                    row[i + 1] = scale * coeff[2, axis]
                    rows.append(row)
                    rhs.append(scale * (-const[axis]))

        if n >= 4 and config.position_third_diff_sigma_m > 0:
            scale = 1.0 / float(config.position_third_diff_sigma_m)
            for i in range(n - 3):
                const = -centers[i] + 3.0 * centers[i + 1] - 3.0 * centers[i + 2] + centers[i + 3]
                coeff = np.stack([-ray_dirs[i], 3.0 * ray_dirs[i + 1], -3.0 * ray_dirs[i + 2], ray_dirs[i + 3]], axis=0)
                for axis in range(3):
                    row = np.zeros(n, dtype=float)
                    row[i] = scale * coeff[0, axis]
                    row[i + 1] = scale * coeff[1, axis]
                    row[i + 2] = scale * coeff[2, axis]
                    row[i + 3] = scale * coeff[3, axis]
                    rows.append(row)
                    rhs.append(scale * (-const[axis]))

        if config.ground_anchor_sigma_m > 0:
            for i in range(n):
                if not np.isfinite(ground_rho[i]):
                    continue
                row = np.zeros(n, dtype=float)
                scale = math.sqrt(max(float(confidences[i]), config.min_observation_confidence)) / float(config.ground_anchor_sigma_m)
                row[i] = scale
                rows.append(row)
                rhs.append(scale * float(ground_rho[i]))

        if active_bounds:
            scale = math.sqrt(float(config.bound_weight))
            for i, value in active_bounds.items():
                row = np.zeros(n, dtype=float)
                row[int(i)] = scale
                rows.append(row)
                rhs.append(scale * float(value))

        # Tiny ridge improves conditioning for short or nearly parallel-ray spans.
        ridge_scale = math.sqrt(max(float(config.ridge), 0.0))
        if ridge_scale > 0:
            for i in range(n):
                row = np.zeros(n, dtype=float)
                row[i] = ridge_scale
                rows.append(row)
                rhs.append(ridge_scale * float(rho_obs[i]))
        return np.vstack(rows), np.asarray(rhs, dtype=float)

    for _ in range(max(1, int(config.range_irls_iterations))):
        a, b = build_system()
        candidate = np.linalg.lstsq(a, b, rcond=None)[0]
        active: dict[int, float] = {}
        for i, (lo, hi) in enumerate(bounds):
            if candidate[i] < lo:
                active[i] = lo
            elif candidate[i] > hi:
                active[i] = hi
        if active:
            a, b = build_system(active)
            candidate = np.linalg.lstsq(a, b, rcond=None)[0]
        rho = np.asarray([np.clip(candidate[i], bounds[i][0], bounds[i][1]) for i in range(n)], dtype=float)
        standardized = np.abs((rho - rho_obs) / sigma)
        robust = np.ones(n, dtype=float)
        big = standardized > float(config.range_huber_delta_sigma)
        robust[big] = float(config.range_huber_delta_sigma) / np.maximum(standardized[big], 1e-12)

    standardized_signed = (rho - rho_obs) / sigma
    return rho, standardized_signed


def refine_temporal_trajectory(
    *,
    cameras_by_frame: Mapping[int, CameraStateLite],
    selected: Mapping[int, BallCandidate2D | None],
    frame_indices: Sequence[int],
    fps: float,
    ball_radius_m: float = 0.11,
    config: TemporalRefinementConfig | None = None,
) -> TemporalRefinementResult:
    """Two-pass temporal ball refinement used by Stage 6 v0.4.

    Pass A smooths log apparent diameter and bridges short missing spans. Pass B
    estimates one positive range value along each observed/interpolated camera ray
    while regularizing world-space acceleration and jerk.  The ball center remains
    on the measured image ray, so the optimizer does not invent lateral image
    displacement for directly observed frames.
    """
    cfg = config or TemporalRefinementConfig()
    fis = [int(x) for x in frame_indices]
    diameter_info = refine_diameter_series(fis, selected, config=cfg)
    center_info = interpolate_center_series(fis, selected, max_gap_frames=cfg.max_gap_frames)

    frame_results: Dict[int, TemporalFrameResult] = {}
    usable: list[bool] = []
    ray_dirs: list[Optional[np.ndarray]] = []
    rho_obs_values: list[Optional[float]] = []
    bounds_values: list[Optional[tuple[float, float]]] = []
    ground_rho_values: list[Optional[float]] = []
    confidences: list[float] = []
    timestamps: list[float] = []

    for fi in fis:
        cand = selected.get(fi)
        dinfo = diameter_info[fi]
        cinfo = center_info[fi]
        center_uv = cinfo["center_uv"]
        refined_d = dinfo["refined_diameter_px"]
        confidence = float(dinfo["observation_confidence"])
        if cinfo["center_imputed"]:
            confidence = max(cfg.min_observation_confidence, confidence * 0.35 if confidence > 0 else 0.10)
        result = TemporalFrameResult(
            frame_index=fi,
            status="NO_TEMPORAL_GEOMETRY",
            center_uv=center_uv,
            center_imputed=bool(cinfo["center_imputed"]),
            raw_diameter_px=dinfo["raw_diameter_px"],
            refined_diameter_px=refined_d,
            diameter_imputed=bool(dinfo["diameter_imputed"]),
            observation_confidence=confidence,
            segment_id=dinfo["segment_id"],
        )
        frame_results[fi] = result
        camera = cameras_by_frame.get(fi)
        has_temporal_support = dinfo.get("segment_id") not in (None, -1)
        if camera is None or camera.status not in {"VALID", "DEGRADED"} or not has_temporal_support or center_uv is None or refined_d is None or refined_d <= 0:
            usable.append(False)
            ray_dirs.append(None)
            rho_obs_values.append(None)
            bounds_values.append(None)
            ground_rho_values.append(None)
            confidences.append(confidence)
            timestamps.append(float(fi) / max(float(fps), 1e-9))
            continue
        bbox = _centered_square_bbox(center_uv, float(refined_d))
        ang = angular_radius_from_bbox(camera, bbox)
        if ang is None:
            usable.append(False)
            ray_dirs.append(None)
            rho_obs_values.append(None)
            bounds_values.append(None)
            ground_rho_values.append(None)
            confidences.append(confidence)
            timestamps.append(camera.timestamp_sec if camera.timestamp_sec is not None else float(fi) / max(float(fps), 1e-9))
            result.status = "INVALID_ANGULAR_SIZE"
            continue
        sin_alpha = math.sin(float(ang["alpha_rad"]))
        if sin_alpha <= 1e-10:
            usable.append(False)
            ray_dirs.append(None)
            rho_obs_values.append(None)
            bounds_values.append(None)
            ground_rho_values.append(None)
            confidences.append(confidence)
            timestamps.append(camera.timestamp_sec if camera.timestamp_sec is not None else float(fi) / max(float(fps), 1e-9))
            result.status = "INVALID_ANGULAR_SIZE"
            continue
        rho_obs = float(ball_radius_m / sin_alpha)
        _origins, dirs = camera.world_ray(np.asarray(center_uv, dtype=float))
        direction = dirs[0]
        interval = _range_interval_for_box(
            camera,
            direction,
            ball_radius_m=ball_radius_m,
            pitch_margin_m=cfg.pitch_margin_m,
            min_height_m=cfg.min_height_m,
            max_height_m=cfg.max_height_m,
            max_range_m=cfg.max_range_m,
        )
        if interval is None:
            usable.append(False)
            ray_dirs.append(direction)
            rho_obs_values.append(rho_obs)
            bounds_values.append(None)
            ground_rho_values.append(None)
            confidences.append(confidence)
            timestamps.append(camera.timestamp_sec if camera.timestamp_sec is not None else float(fi) / max(float(fps), 1e-9))
            result.raw_size_range_m = rho_obs
            result.status = "NO_FEASIBLE_RAY_RANGE"
            continue
        # Optional grounded-ball anchor.  It is activated only when the refined
        # apparent size agrees with the z=ball_radius ray intersection.
        ground_rho = None
        ground = camera.intersect_z_plane(np.asarray(center_uv, dtype=float), ball_radius_m)[0]
        if np.all(np.isfinite(ground)):
            ground_range = float(np.linalg.norm(ground - camera.camera_center_world_m))
            if ground_range > ball_radius_m:
                ground_alpha = math.asin(min(0.999999999, ball_radius_m / ground_range))
                size_log_error = abs(math.log(max(float(ang["alpha_rad"]), 1e-12) / max(ground_alpha, 1e-12)))
                if size_log_error <= float(cfg.ground_consistency_log_tolerance):
                    ground_rho = ground_range
                    result.ground_compatible = True
                    result.ground_range_m = ground_range

        # Guardrail: temporal motion may correct the size estimate, but it must
        # not move arbitrarily far from the observed apparent-size range.
        dev = float(cfg.max_range_deviation_fraction)
        if dev > 0:
            measurement_interval = (max(ball_radius_m * 1.001, rho_obs * (1.0 - dev)), rho_obs * (1.0 + dev))
            interval = (max(interval[0], measurement_interval[0]), min(interval[1], measurement_interval[1]))
            if interval[1] <= interval[0]:
                interval = measurement_interval

        usable.append(True)
        ray_dirs.append(direction)
        rho_obs_values.append(rho_obs)
        bounds_values.append(interval)
        ground_rho_values.append(ground_rho)
        confidences.append(confidence)
        timestamps.append(camera.timestamp_sec if camera.timestamp_sec is not None else float(fi) / max(float(fps), 1e-9))
        result.raw_size_range_m = rho_obs
        result.range_bounds_m = [float(interval[0]), float(interval[1])]

    segments = _trajectory_segments(usable)
    optimized_count = 0
    imputed_count = 0
    bound_active_count = 0
    residuals: list[float] = []
    segment_reports: list[dict[str, Any]] = []

    for segment_id, (start, end) in enumerate(segments):
        if end < start:
            continue
        seg_fis = fis[start : end + 1]
        seg_cameras = [cameras_by_frame[fi] for fi in seg_fis]
        seg_dirs = np.asarray([ray_dirs[i] for i in range(start, end + 1)], dtype=float)
        seg_rho_obs = np.asarray([rho_obs_values[i] for i in range(start, end + 1)], dtype=float)
        seg_bounds = [bounds_values[i] for i in range(start, end + 1)]
        seg_ground = np.asarray([np.nan if ground_rho_values[i] is None else ground_rho_values[i] for i in range(start, end + 1)], dtype=float)
        seg_conf = np.asarray(confidences[start : end + 1], dtype=float)
        seg_ts = np.asarray(timestamps[start : end + 1], dtype=float)
        if any(x is None for x in seg_bounds):
            continue
        seg_bounds_typed = [x for x in seg_bounds if x is not None]
        rho_opt, standardized = _solve_range_segment(
            seg_cameras,
            seg_dirs,
            seg_rho_obs,
            seg_ground,
            seg_conf,
            seg_bounds_typed,
            seg_ts,
            cfg,
        )
        for local, fi in enumerate(seg_fis):
            camera = seg_cameras[local]
            xyz = camera.camera_center_world_m + float(rho_opt[local]) * seg_dirs[local]
            result = frame_results[fi]
            result.optimized_range_m = float(rho_opt[local])
            result.xyz_world_m = xyz.astype(float).tolist()
            result.size_residual_sigma = float(standardized[local])
            result.segment_id = int(segment_id)
            result.status = "VALID_TEMPORAL_3D_INTERPOLATED" if (result.center_imputed or result.diameter_imputed) else "VALID_TEMPORAL_3D"
            # Convert optimized range back to an equivalent local apparent diameter.
            raw_alpha = math.asin(min(0.999999999, ball_radius_m / max(float(seg_rho_obs[local]), ball_radius_m * 1.000001)))
            opt_alpha = math.asin(min(0.999999999, ball_radius_m / max(float(rho_opt[local]), ball_radius_m * 1.000001)))
            if result.refined_diameter_px is not None and raw_alpha > 1e-12:
                result.optimized_diameter_px = float(result.refined_diameter_px * opt_alpha / raw_alpha)
            optimized_count += 1
            imputed_count += int(result.center_imputed or result.diameter_imputed)
            lo, hi = seg_bounds_typed[local]
            if abs(float(rho_opt[local]) - lo) < 1e-5 or abs(float(rho_opt[local]) - hi) < 1e-5:
                bound_active_count += 1
            residuals.append(float(standardized[local]))
        segment_reports.append({
            "segment_id": int(segment_id),
            "frame_start": int(seg_fis[0]),
            "frame_end": int(seg_fis[-1]),
            "frames": len(seg_fis),
            "observed_centers": int(sum(selected.get(fi) is not None for fi in seg_fis)),
        })

    raw_diameters = [r.raw_diameter_px for r in frame_results.values() if r.raw_diameter_px is not None]
    refined_diameters = [r.refined_diameter_px for r in frame_results.values() if r.refined_diameter_px is not None]
    diagnostics = {
        "schema_version": "stage6-temporal-refinement-1.0",
        "config": cfg.to_dict(),
        "frames": len(fis),
        "frames_with_direct_candidate": int(sum(selected.get(fi) is not None for fi in fis)),
        "frames_with_refined_diameter": len(refined_diameters),
        "frames_with_temporal_3d": int(optimized_count),
        "frames_with_imputed_temporal_3d": int(imputed_count),
        "bound_active_frames": int(bound_active_count),
        "ground_compatible_frames": int(sum(r.ground_compatible for r in frame_results.values())),
        "segments": segment_reports,
        "size_residual_sigma_median": None if not residuals else float(np.median(np.abs(residuals))),
        "size_residual_sigma_p90": None if not residuals else float(np.percentile(np.abs(residuals), 90)),
        "raw_diameter_px_median": None if not raw_diameters else float(np.median(raw_diameters)),
        "refined_diameter_px_median": None if not refined_diameters else float(np.median(refined_diameters)),
        "note": "Temporal optimizer preserves each frame's measured/interpolated image ray; it refines apparent size/range and world-space trajectory, not the 2D center of directly observed frames.",
    }
    return TemporalRefinementResult(frame_results, diagnostics)
