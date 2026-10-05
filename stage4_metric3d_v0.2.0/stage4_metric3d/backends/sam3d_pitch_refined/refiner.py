from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np
from scipy.optimize import least_squares

from ...camera import CameraStateLite
from ...stage3_adapter import PoseObservation2D
from .config import Sam3DPitchRefinedConfig
from .geometry import project_camera_points, camera_to_world
from .ground_anchor import GroundAnchorCandidate
from .joint_mapping import DEFAULT_MAPPING


@dataclass
class FrameEvidence:
    frame_index: int
    camera: CameraStateLite
    observation: PoseObservation2D
    relative_joints_m: np.ndarray
    sam_prior_cam_m: np.ndarray
    sam_2d_px: np.ndarray
    ground_anchor: GroundAnchorCandidate | None


@dataclass(frozen=True)
class RefinementResult:
    refined_cam_m: np.ndarray
    initial_cost: float
    final_cost: float
    success: bool
    status: int
    message: str
    nfev: int
    temporal_triplet_count: int
    skipped_temporal_triplet_count: int
    reprojection_joint_count: int
    invalid_reprojection_joint_count: int
    bounds_active_count: int
    fallback_to_initial: bool


def temporal_triplet_indices(frames: Sequence[FrameEvidence]) -> tuple[tuple[int, int, int], ...]:
    """Return only exact consecutive-frame triplets with usable cameras."""
    active: list[tuple[int, int, int]] = []
    for i in range(1, len(frames) - 1):
        f0, f1, f2 = (int(frames[i - 1].frame_index), int(frames[i].frame_index), int(frames[i + 1].frame_index))
        camera_ok = all(str(frames[j].camera.status).upper() == "VALID" for j in (i - 1, i, i + 1))
        if f1 - f0 == 1 and f2 - f1 == 1 and camera_ok:
            active.append((i - 1, i, i + 1))
    return tuple(active)


def _reprojection_counts(
    frames: Sequence[FrameEvidence], roots: np.ndarray, cfg: Sam3DPitchRefinedConfig,
) -> tuple[int, int]:
    valid_count = 0
    invalid_count = 0
    for ev, root in zip(frames, roots):
        for entry in DEFAULT_MAPPING:
            uv = np.asarray(ev.observation.uv23[entry.rtmw_index], dtype=np.float64)
            weight = float(ev.observation.state_weights23[entry.rtmw_index])
            rel = np.asarray(ev.relative_joints_m[entry.sam3d_index], dtype=np.float64)
            if weight < cfg.min_rtmw_joint_weight or not np.isfinite(uv).all() or not np.isfinite(rel).all():
                continue
            valid_count += 1
            projected = project_camera_points(ev.camera, rel + root, distort=True)
            if not np.isfinite(projected).all():
                invalid_count += 1
    return valid_count - invalid_count, invalid_count


def _frame_residual(root: np.ndarray, ev: FrameEvidence, cfg: Sam3DPitchRefinedConfig) -> list[np.ndarray]:
    residuals: list[np.ndarray] = []
    mapped_indices = []
    observed_uv = []
    weights = []
    for entry in DEFAULT_MAPPING:
        uv = np.asarray(ev.observation.uv23[entry.rtmw_index], dtype=np.float64)
        w = float(ev.observation.state_weights23[entry.rtmw_index])
        rel = np.asarray(ev.relative_joints_m[entry.sam3d_index], dtype=np.float64)
        if w < cfg.min_rtmw_joint_weight or not np.isfinite(uv).all() or not np.isfinite(rel).all():
            continue
        mapped_indices.append(entry.sam3d_index)
        observed_uv.append(uv)
        weights.append(w)
    if mapped_indices:
        pts_cam = ev.relative_joints_m[np.asarray(mapped_indices, dtype=int)] + root[None, :]
        pred = project_camera_points(ev.camera, pts_cam, distort=True)
        obs = np.asarray(observed_uv, dtype=np.float64)
        w = np.sqrt(np.asarray(weights, dtype=np.float64))[:, None]
        diff = (pred - obs) * w / float(cfg.reprojection_sigma_px)
        diff[~np.isfinite(diff)] = 1e3
        residuals.append(diff.reshape(-1))

    sam_sigma = np.asarray(cfg.sam_prior_sigma_xyz_m, dtype=np.float64)
    if ev.ground_anchor is not None and ev.ground_anchor.usable:
        sam_sigma = np.full(3, cfg.grounded_sam_prior_sigma_m)
    residuals.append((root - ev.sam_prior_cam_m) / sam_sigma)

    if ev.ground_anchor is not None and ev.ground_anchor.usable:
        ground_sigma = np.asarray(cfg.ground_sigma_xyz_m, dtype=np.float64)
        residuals.append((root - ev.ground_anchor.root_cam_m) / ground_sigma)
    return residuals


def _residual_vector(x: np.ndarray, frames: Sequence[FrameEvidence], cfg: Sam3DPitchRefinedConfig) -> np.ndarray:
    roots = np.asarray(x, dtype=np.float64).reshape(len(frames), 3)
    blocks: list[np.ndarray] = []
    for root, ev in zip(roots, frames):
        blocks.extend(_frame_residual(root, ev, cfg))

    if cfg.use_temporal and len(frames) >= 3:
        sigma = float(cfg.temporal_second_difference_sigma_m)
        for i0, i1, i2 in temporal_triplet_indices(frames):
            p0 = camera_to_world(frames[i0].camera, roots[i0])
            p1 = camera_to_world(frames[i1].camera, roots[i1])
            p2 = camera_to_world(frames[i2].camera, roots[i2])
            blocks.append((p2 - 2.0 * p1 + p0) / sigma)
    if not blocks:
        return np.zeros((0,), dtype=np.float64)
    return np.concatenate([np.asarray(b, dtype=np.float64).reshape(-1) for b in blocks])


def refine_translation_sequence(frames: Sequence[FrameEvidence], cfg: Sam3DPitchRefinedConfig) -> RefinementResult:
    cfg.validate()
    if not frames:
        raise ValueError("No frame evidence supplied")
    grounded = [ev.ground_anchor is not None and ev.ground_anchor.usable for ev in frames]
    x0 = np.concatenate([np.asarray(ev.ground_anchor.root_cam_m if g else ev.sam_prior_cam_m, dtype=np.float64) for ev, g in zip(frames, grounded)])
    temporal_triplets = temporal_triplet_indices(frames)
    skipped_temporal = max(0, len(frames) - 2 - len(temporal_triplets)) if cfg.use_temporal else 0
    initial_res = _residual_vector(x0, frames, cfg)
    initial_cost = float(0.5 * np.sum(initial_res * initial_res))
    # Component bounds inscribe a cube in the requested Euclidean-radius ball.
    bound = np.repeat([cfg.max_ground_refinement_m / np.sqrt(3) if g else cfg.max_translation_correction_m for g in grounded], 3)
    lower = x0 - bound
    upper = x0 + bound
    try:
        result = least_squares(
            lambda x: _residual_vector(x, frames, cfg),
            x0,
            bounds=(lower, upper),
            loss=cfg.optimizer_loss,
            f_scale=1.0,
            max_nfev=int(cfg.optimizer_max_nfev),
        )
        candidate = np.asarray(result.x, dtype=np.float64)
        success = bool(result.success)
        status = int(result.status)
        message = str(result.message)
        nfev = int(result.nfev)
        fallback_to_initial = not success or candidate.shape != x0.shape or not np.isfinite(candidate).all()
    except Exception as exc:
        candidate = x0.copy()
        success = False
        status = -1
        message = f"optimizer_exception:{type(exc).__name__}:{exc}"
        nfev = 0
        fallback_to_initial = True
    solution = x0.copy() if fallback_to_initial else candidate
    final_res = _residual_vector(solution, frames, cfg)
    final_cost = float(0.5 * np.sum(final_res * final_res))
    reprojection_joint_count, invalid_reprojection_joint_count = _reprojection_counts(
        frames, solution.reshape(len(frames), 3), cfg,
    )
    delta = np.abs(solution - x0)
    bounds = np.repeat([cfg.max_ground_refinement_m / np.sqrt(3) if g else cfg.max_translation_correction_m for g in grounded], 3)
    bounds_active_count = int(np.count_nonzero(delta >= (bounds - 1e-7)))
    return RefinementResult(
        refined_cam_m=solution.reshape(len(frames), 3),
        initial_cost=initial_cost,
        final_cost=final_cost,
        success=success,
        status=status,
        message=message,
        nfev=nfev,
        temporal_triplet_count=len(temporal_triplets) if cfg.use_temporal else 0,
        skipped_temporal_triplet_count=skipped_temporal,
        reprojection_joint_count=reprojection_joint_count,
        invalid_reprojection_joint_count=invalid_reprojection_joint_count,
        bounds_active_count=bounds_active_count,
        fallback_to_initial=fallback_to_initial,
    )
