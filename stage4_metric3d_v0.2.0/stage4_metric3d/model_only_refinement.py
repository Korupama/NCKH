from __future__ import annotations

"""Stage-4-only root translation refinement for the benchmark lane.

The refinement uses only benchmark camera geometry plus native SAM3D MHR70
outputs. It deliberately does not import Stage 1/3 adapters or benchmark GT.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from .backends.sam3d_pitch_refined.joint_mapping import DEFAULT_MAPPING
from .camera import CameraStateLite


GROUND_PRIMARY = ("left_big_toe", "left_small_toe", "left_heel", "right_big_toe", "right_small_toe", "right_heel")
GROUND_FALLBACK = ("left_ankle", "right_ankle")
GROUND_CONSENSUS_RADIUS_M = 0.75
GROUND_MIN_CONSENSUS_COUNT = 2
GROUND_MIN_CONSENSUS_FRACTION = 0.5
MODEL_ONLY_REFINEMENT_POLICY = "ground-consensus-gated-v0.5.2"


@dataclass(frozen=True)
class GroundCandidate:
    canonical_name: str
    root_cam_m: np.ndarray
    ground_world_m: np.ndarray
    fallback: bool


@dataclass(frozen=True)
class ModelOnlyRefinementResult:
    refined_root_cam_m: np.ndarray
    anchor_root_cam_m: np.ndarray | None
    ground_world_m: np.ndarray | None
    candidate_count: int
    consensus_count: int
    candidate_spread_m: float | None
    status: str
    reason: str | None
    source: str
    candidate_names: tuple[str, ...]
    consensus_names: tuple[str, ...]
    consensus_fraction: float | None
    rejection_reasons: tuple[str, ...]
    correction_bounded: bool


def _intersect_pitch_undistorted(camera: CameraStateLite, uv: np.ndarray) -> np.ndarray:
    """Intersect a SAM3D pixel prediction with z=0 using benchmark K only.

    Official SAM3D's supplied-camera output is compared in the existing worker
    without lens distortion. Using zero distortion here avoids applying the
    benchmark distortion model a second time to SAM3D's camera-space pixels.
    """
    point = np.asarray(uv, dtype=np.float64).reshape(1, 1, 2)
    und = cv2.undistortPoints(point, camera.K, np.zeros_like(camera.distortion), P=camera.K).reshape(2)
    direction_cam = np.linalg.inv(camera.K) @ np.array([und[0], und[1], 1.0], dtype=np.float64)
    direction_world = camera.R_world_to_camera.T @ direction_cam
    direction_world /= max(float(np.linalg.norm(direction_world)), 1e-12)
    origin = np.asarray(camera.camera_center_world_m, dtype=np.float64)
    denom = float(direction_world[2])
    if abs(denom) <= 1e-10:
        return np.full(3, np.nan, dtype=np.float64)
    lam = -float(origin[2]) / denom
    if not np.isfinite(lam) or lam <= 0:
        return np.full(3, np.nan, dtype=np.float64)
    return origin + lam * direction_world


def _candidate_for_joint(
    *, camera: CameraStateLite, canonical_name: str, native_2d: np.ndarray,
    native_3d_relative_m: np.ndarray, fallback: bool,
) -> tuple[GroundCandidate | None, str | None]:
    mapping = next(entry for entry in DEFAULT_MAPPING if entry.canonical_name == canonical_name)
    uv = np.asarray(native_2d[mapping.sam3d_index], dtype=np.float64)
    relative = np.asarray(native_3d_relative_m[mapping.sam3d_index], dtype=np.float64)
    if not np.isfinite(uv).all() or not np.isfinite(relative).all():
        return None, "invalid_joint_evidence"
    ground_world = _intersect_pitch_undistorted(camera, uv)
    if not np.isfinite(ground_world).all():
        return None, "invalid_ground_intersection"
    pitch_length_raw = camera.pitch.get("length_m")
    pitch_width_raw = camera.pitch.get("width_m")
    if camera.pitch.get("origin") != "center" or pitch_length_raw is None or pitch_width_raw is None:
        return None, "invalid_pitch_geometry"
    pitch_length = float(pitch_length_raw)
    pitch_width = float(pitch_width_raw)
    if not np.isfinite([pitch_length, pitch_width]).all() or pitch_length <= 0 or pitch_width <= 0:
        return None, "invalid_pitch_geometry"
    if abs(float(ground_world[0])) > pitch_length / 2.0 + 2.0 or abs(float(ground_world[1])) > pitch_width / 2.0 + 2.0:
        return None, "outside_pitch_bounds"
    ground_cam = np.asarray(camera.world_to_camera(ground_world), dtype=np.float64)
    root = ground_cam - relative
    if not np.isfinite(root).all():
        return None, "invalid_root_candidate"
    return GroundCandidate(canonical_name, root, ground_world, fallback), None


def refine_model_only_translation(
    *, camera: CameraStateLite, native_2d: np.ndarray, native_3d_relative_m: np.ndarray,
    sam_prior_cam_m: np.ndarray, max_correction_m: float = 0.75,
) -> ModelOnlyRefinementResult:
    native_2d = np.asarray(native_2d, dtype=np.float64)
    native_3d_relative_m = np.asarray(native_3d_relative_m, dtype=np.float64)
    prior = np.asarray(sam_prior_cam_m, dtype=np.float64).reshape(3)
    if not np.isfinite(max_correction_m) or max_correction_m < 0:
        return ModelOnlyRefinementResult(
            prior, None, None, 0, 0, None, "MISSING", "invalid_correction_bound",
            "NONE", (), (), None, ("invalid_correction_bound",), False,
        )
    if native_2d.shape != (70, 2) or native_3d_relative_m.shape != (70, 3) or not np.isfinite(prior).all():
        return ModelOnlyRefinementResult(
            prior, None, None, 0, 0, None, "MISSING", "invalid_native_sam3d_input",
            "NONE", (), (), None, ("invalid_native_sam3d_input",), False,
        )
    if str(camera.status).upper() != "VALID":
        return ModelOnlyRefinementResult(
            prior, None, None, 0, 0, None, "GROUND_UNAVAILABLE", "invalid_camera_state",
            "NONE", (), (), None, (f"camera_status:{camera.status}",), False,
        )

    rejection_reasons: list[str] = []
    primary_candidates: list[GroundCandidate] = []
    for name in GROUND_PRIMARY:
        candidate, rejection = _candidate_for_joint(
            camera=camera, canonical_name=name, native_2d=native_2d,
            native_3d_relative_m=native_3d_relative_m, fallback=False,
        )
        if candidate is not None:
            primary_candidates.append(candidate)
        elif rejection is not None:
            rejection_reasons.append(f"{name}:{rejection}")
    source = "DISTAL_FOOT"
    candidates = primary_candidates
    if not candidates:
        fallback_candidates: list[GroundCandidate] = []
        for name in GROUND_FALLBACK:
            candidate, rejection = _candidate_for_joint(
                camera=camera, canonical_name=name, native_2d=native_2d,
                native_3d_relative_m=native_3d_relative_m, fallback=True,
            )
            if candidate is not None:
                fallback_candidates.append(candidate)
            elif rejection is not None:
                rejection_reasons.append(f"{name}:{rejection}")
        candidates = fallback_candidates
        source = "ANKLE_FALLBACK"
    if not candidates:
        return ModelOnlyRefinementResult(
            prior, None, None, 0, 0, None, "GROUND_UNAVAILABLE", "no_usable_ground_candidate",
            "NONE", (), (), None, tuple(rejection_reasons), False,
        )

    roots = np.stack([candidate.root_cam_m for candidate in candidates])
    pairwise_distances = np.linalg.norm(roots[:, None, :] - roots[None, :, :], axis=2)
    neighborhood_counts = np.sum(pairwise_distances <= GROUND_CONSENSUS_RADIUS_M, axis=1)
    seed = int(np.argmax(neighborhood_counts))
    inliers = pairwise_distances[seed] <= GROUND_CONSENSUS_RADIUS_M
    consensus = np.median(roots[inliers], axis=0)
    spread = float(np.max(np.linalg.norm(roots[inliers] - consensus[None, :], axis=1)))
    candidate_names = tuple(candidate.canonical_name for candidate in candidates)
    consensus_names = tuple(candidate.canonical_name for candidate, keep in zip(candidates, inliers) if bool(keep))
    consensus_count = int(inliers.sum())
    consensus_fraction = float(consensus_count / len(candidates))
    if consensus_count < GROUND_MIN_CONSENSUS_COUNT:
        return ModelOnlyRefinementResult(
            prior, None, None, len(candidates), consensus_count, spread,
            "CONTACT_AMBIGUOUS", "insufficient_ground_consensus",
            source, candidate_names, consensus_names, consensus_fraction,
            tuple(rejection_reasons), False,
        )
    if consensus_fraction <= GROUND_MIN_CONSENSUS_FRACTION:
        return ModelOnlyRefinementResult(
            prior, None, None, len(candidates), consensus_count, spread,
            "CONTACT_AMBIGUOUS", "minority_ground_consensus",
            source, candidate_names, consensus_names, consensus_fraction,
            tuple(rejection_reasons), False,
        )
    correction = consensus - prior
    correction_norm = float(np.linalg.norm(correction))
    correction_bounded = correction_norm > max_correction_m
    if correction_norm > max_correction_m:
        correction = correction * (float(max_correction_m) / correction_norm)
    refined = prior + correction
    representative = candidates[int(np.flatnonzero(inliers)[0])]
    return ModelOnlyRefinementResult(
        refined_root_cam_m=refined,
        anchor_root_cam_m=consensus,
        ground_world_m=representative.ground_world_m,
        candidate_count=len(candidates),
        consensus_count=consensus_count,
        candidate_spread_m=spread,
        status="CONTACT_SUPPORTED",
        reason=None,
        source=source,
        candidate_names=candidate_names,
        consensus_names=consensus_names,
        consensus_fraction=consensus_fraction,
        rejection_reasons=tuple(rejection_reasons),
        correction_bounded=correction_bounded,
    )
