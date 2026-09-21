from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np

from ...camera import CameraStateLite
from ...stage3_adapter import PoseObservation2D
from .joint_mapping import MAPPING_BY_CANONICAL, GROUND_PRIMARY, GROUND_FALLBACK


@dataclass(frozen=True)
class GroundAnchorCandidate:
    canonical_name: str
    root_cam_m: np.ndarray
    ground_world_m: np.ndarray
    rtmw_weight: float
    distance_to_sam_prior_m: float
    usable: bool
    fallback: bool
    candidate_spread_m: float = 0.0
    consensus_count: int = 1
    rejection_reason: str | None = None


def _candidates_for_names(
    *,
    names: tuple[str, ...],
    camera: CameraStateLite,
    observation: PoseObservation2D,
    relative_joints_m: np.ndarray,
    sam_prior_cam_m: np.ndarray,
    min_joint_weight: float,
    max_prior_distance_m: float,
    fallback: bool,
) -> list[GroundAnchorCandidate]:
    out: list[GroundAnchorCandidate] = []
    for name in names:
        mapping = MAPPING_BY_CANONICAL[name]
        uv = np.asarray(observation.uv23[mapping.rtmw_index], dtype=np.float64)
        weight = float(observation.state_weights23[mapping.rtmw_index])
        joint_rel = np.asarray(relative_joints_m[mapping.sam3d_index], dtype=np.float64)
        if weight < min_joint_weight or not np.isfinite(uv).all() or not np.isfinite(joint_rel).all():
            continue
        ground_world = np.asarray(camera.intersect_pitch(uv), dtype=np.float64)
        if not np.isfinite(ground_world).all():
            continue
        ground_cam = np.asarray(camera.world_to_camera(ground_world), dtype=np.float64)
        root = ground_cam - joint_rel
        d = float(np.linalg.norm(root - np.asarray(sam_prior_cam_m, dtype=np.float64)))
        # Ground geometry is assessed independently of the monocular depth prior.
        foot_ids = [MAPPING_BY_CANONICAL[n].sam3d_index for n in GROUND_PRIMARY]
        feet_world = (relative_joints_m[foot_ids] + root) @ camera.R_world_to_camera + camera.camera_center_world_m
        reason = None
        if camera.pitch.get("origin", "center") != "center":
            reason = "unsupported_pitch_origin"
        elif abs(ground_world[0]) > float(camera.pitch.get("length_m", 105.0))/2 + 2.0 or abs(ground_world[1]) > float(camera.pitch.get("width_m", 68.0))/2 + 2.0:
            reason = "outside_pitch_bounds"
        elif np.nanmin(feet_world[:, 2]) < -0.10:
            reason = "other_foot_below_pitch"
        out.append(GroundAnchorCandidate(
            canonical_name=name,
            root_cam_m=root,
            ground_world_m=ground_world,
            rtmw_weight=weight,
            distance_to_sam_prior_m=d,
            usable=reason is None,
            fallback=fallback,
            rejection_reason=reason,
        ))
    return out


def choose_ground_anchor(
    *,
    camera: CameraStateLite,
    observation: PoseObservation2D,
    relative_joints_m: np.ndarray,
    sam_prior_cam_m: np.ndarray,
    min_joint_weight: float,
    max_prior_distance_m: float,
) -> tuple[GroundAnchorCandidate | None, list[GroundAnchorCandidate]]:
    primary = _candidates_for_names(
        names=GROUND_PRIMARY, camera=camera, observation=observation,
        relative_joints_m=relative_joints_m, sam_prior_cam_m=sam_prior_cam_m,
        min_joint_weight=min_joint_weight, max_prior_distance_m=max_prior_distance_m,
        fallback=False,
    )
    candidates = primary
    if not any(c.usable for c in candidates):
        candidates += _candidates_for_names(
            names=GROUND_FALLBACK, camera=camera, observation=observation,
            relative_joints_m=relative_joints_m, sam_prior_cam_m=sam_prior_cam_m,
            min_joint_weight=min_joint_weight, max_prior_distance_m=max_prior_distance_m,
            fallback=True,
        )
    if not candidates:
        return None, []
    usable = [c for c in candidates if c.usable]
    if not usable:
        return candidates[0], candidates
    roots = np.stack([c.root_cam_m for c in usable])
    weights = np.asarray([c.rtmw_weight for c in usable])
    distances = np.linalg.norm(roots[:, None] - roots[None, :], axis=-1)
    neighborhoods = distances <= 0.75
    seed = int(np.argmax(neighborhoods @ weights))
    inliers = neighborhoods[seed]
    consensus = np.median(roots[inliers], axis=0)
    representative = usable[int(np.argmin(np.linalg.norm(roots-consensus, axis=1)))]
    spread = float(np.max(np.linalg.norm(roots[inliers]-consensus, axis=1)))
    # Require a dominant cluster when multiple mutually inconsistent roots exist.
    consistent = float(weights[inliers].sum()) > float(weights.sum()) * 0.5 or len(usable) == 1
    foot_ids = [MAPPING_BY_CANONICAL[n].sam3d_index for n in GROUND_PRIMARY]
    feet_world = (relative_joints_m[foot_ids] + consensus) @ camera.R_world_to_camera + camera.camera_center_world_m
    consistent = consistent and bool(np.nanmin(feet_world[:, 2]) >= -0.10)
    return replace(representative, root_cam_m=consensus,
                   distance_to_sam_prior_m=float(np.linalg.norm(consensus-sam_prior_cam_m)),
                   candidate_spread_m=spread, consensus_count=int(inliers.sum()),
                   usable=consistent, rejection_reason=None if consistent else "inconsistent_ground_candidates"), candidates
