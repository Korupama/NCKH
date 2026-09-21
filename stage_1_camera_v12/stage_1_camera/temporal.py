from __future__ import annotations

from typing import List, Optional
import copy
import numpy as np
import cv2
from scipy.spatial.transform import Rotation, Slerp

from .contracts import CameraState, CameraStatus, CameraTimeline
from .capabilities import annotate_capabilities
from .evidence import reprojection_error_from_correspondences


def _rot_to_vec(R):
    r, _ = cv2.Rodrigues(R)
    return r.reshape(3)


def _vec_to_rot(v):
    R, _ = cv2.Rodrigues(np.asarray(v, dtype=np.float64).reshape(3,1))
    return R


def smooth_same_shot(timeline: CameraTimeline, radius: int = 2) -> CameraTimeline:
    """Robust local smoothing for non-invalid states inside one already-isolated shot.

    Raw cameras are preserved in diagnostics.  This function does *not* promote
    a DEGRADED camera to VALID; use ``rescue_degraded_bracketed`` for the stricter
    reprojection-validated rescue path.
    """
    states = timeline.states
    out = []
    for i, s in enumerate(states):
        if s.status == CameraStatus.INVALID:
            out.append(s); continue
        neigh = [x for x in states[max(0,i-radius):min(len(states),i+radius+1)] if x.status != CameraStatus.INVALID]
        if len(neigh) < 2:
            out.append(s); continue
        K = s.K.copy()
        K[0,0] = np.median([x.K[0,0] for x in neigh]); K[1,1] = np.median([x.K[1,1] for x in neigh])
        K[0,2] = np.median([x.K[0,2] for x in neigh]); K[1,2] = np.median([x.K[1,2] for x in neigh])
        C = np.median(np.stack([x.camera_center_world_m for x in neigh]), axis=0)
        rv = np.median(np.stack([_rot_to_vec(x.R_world_to_camera) for x in neigh]), axis=0)
        ns = CameraState(
            frame_index=s.frame_index, image_width=s.image_width, image_height=s.image_height,
            K=K, R_world_to_camera=_vec_to_rot(rv), camera_center_world_m=C,
            distortion=s.distortion, pitch=s.pitch, timestamp_sec=s.timestamp_sec, status=s.status,
            source=copy.deepcopy(s.source), evidence=copy.deepcopy(s.evidence), audit=copy.deepcopy(s.audit),
            diagnostics=copy.deepcopy(s.diagnostics),
            temporal={**copy.deepcopy(s.temporal), "source":"smoothed", "window":len(neigh)},
            uncertainty=copy.deepcopy(s.uncertainty), symmetry=copy.deepcopy(s.symmetry),
            schema_version=s.schema_version,
        )
        ns.diagnostics["raw_camera"] = {"K":s.K.tolist(), "R":s.R_world_to_camera.tolist(), "C":s.camera_center_world_m.tolist()}
        annotate_capabilities(ns)
        out.append(ns)
    return CameraTimeline(out, shot_id=timeline.shot_id, metadata={**timeline.metadata, "smoothed":True})


def interpolate_short_gaps(timeline: CameraTimeline, max_gap: int = 2) -> CameraTimeline:
    """Fill only short INVALID gaps; interpolated states stay DEGRADED."""
    states = [s for s in timeline.states]
    for i, s in enumerate(list(states)):
        if s.status != CameraStatus.INVALID:
            continue
        left = next((states[j] for j in range(i-1,-1,-1) if states[j].status != CameraStatus.INVALID), None)
        right = next((states[j] for j in range(i+1,len(states)) if states[j].status != CameraStatus.INVALID), None)
        if left is None or right is None:
            continue
        gap = right.frame_index - left.frame_index - 1
        if gap > max_gap:
            continue
        a = (s.frame_index-left.frame_index)/(right.frame_index-left.frame_index)
        K = (1-a)*left.K + a*right.K
        C = (1-a)*left.camera_center_world_m + a*right.camera_center_world_m
        R = _slerp_pair(left.R_world_to_camera, right.R_world_to_camera, a)
        states[i] = CameraState(
            frame_index=s.frame_index, image_width=s.image_width, image_height=s.image_height,
            K=K, R_world_to_camera=R, camera_center_world_m=C,
            distortion=left.distortion, pitch=left.pitch, timestamp_sec=s.timestamp_sec,
            status=CameraStatus.DEGRADED,
            source={"backend":"temporal_interpolation"},
            temporal={"source":"interpolated", "left_frame":left.frame_index, "right_frame":right.frame_index},
            schema_version=s.schema_version,
        )
        annotate_capabilities(states[i])
    return CameraTimeline(states, shot_id=timeline.shot_id, metadata={**timeline.metadata, "interpolated_short_gaps":True})


def rescue_degraded_bracketed(
    timeline: CameraTimeline,
    *,
    target_frame: Optional[int] = None,
    max_neighbor_distance: int = 5,
    min_correspondences: int = 6,
    min_non_ground_correspondences: int = 2,
    max_median_reproj_px: float = 5.0,
    max_p95_reproj_px: float = 10.0,
) -> CameraTimeline:
    """Conservatively rescue DEGRADED vertical-3D cameras with valid neighbors.

    A rescue is promoted only when all of the following hold:
    1. the target is bracketed by vertical-3D VALID direct cameras in the same
       ``CameraTimeline`` (which must already represent one shot),
    2. both neighbors are within ``max_neighbor_distance`` frames,
    3. interpolated K/C and SO(3)-slerped R reproject the target frame's own
       stored calibration correspondences well enough, and
    4. at least ``min_non_ground_correspondences`` are truly elevated points.

    Thresholds are provisional engineering gates pending the SoccerNet temporal
    benchmark.  Failed rescues never overwrite the direct camera.
    """
    states = list(timeline.states)
    result = []

    for i, s in enumerate(states):
        if target_frame is not None and s.frame_index != target_frame:
            result.append(s); continue
        caps = (s.diagnostics or {}).get("capabilities", {}) or {}
        vstat = ((caps.get("vertical_3d") or {}).get("status"))
        if s.status == CameraStatus.INVALID or vstat == CameraStatus.VALID.value:
            result.append(s); continue

        left = next((states[j] for j in range(i-1, -1, -1) if _vertical_valid(states[j])), None)
        right = next((states[j] for j in range(i+1, len(states)) if _vertical_valid(states[j])), None)
        if left is None or right is None:
            result.append(_record_failed_rescue(s, "missing_bracketing_vertical_valid_neighbors")); continue
        if s.frame_index-left.frame_index > max_neighbor_distance or right.frame_index-s.frame_index > max_neighbor_distance:
            result.append(_record_failed_rescue(s, "bracketing_neighbors_too_far")); continue

        denom = right.frame_index - left.frame_index
        if denom <= 0:
            result.append(_record_failed_rescue(s, "invalid_neighbor_order")); continue
        a = (s.frame_index - left.frame_index) / denom
        K = (1-a)*left.K + a*right.K
        K[2] = np.array([0.0, 0.0, 1.0])
        C = (1-a)*left.camera_center_world_m + a*right.camera_center_world_m
        R = _slerp_pair(left.R_world_to_camera, right.R_world_to_camera, a)

        ns = CameraState(
            frame_index=s.frame_index, image_width=s.image_width, image_height=s.image_height,
            K=K, R_world_to_camera=R, camera_center_world_m=C,
            distortion=s.distortion, pitch=s.pitch, timestamp_sec=s.timestamp_sec,
            status=s.status, source=copy.deepcopy(s.source), evidence=copy.deepcopy(s.evidence),
            audit=copy.deepcopy(s.audit), diagnostics=copy.deepcopy(s.diagnostics),
            temporal=copy.deepcopy(s.temporal), uncertainty=copy.deepcopy(s.uncertainty),
            symmetry=copy.deepcopy(s.symmetry), schema_version=s.schema_version,
        )
        correspondences = ns.evidence.get("keypoint_correspondences", []) or []
        reproj = reprojection_error_from_correspondences(ns, correspondences)
        validated = (
            reproj.get("count", 0) >= min_correspondences
            and reproj.get("non_ground_count", 0) >= min_non_ground_correspondences
            and reproj.get("median_px") is not None and reproj["median_px"] <= max_median_reproj_px
            and reproj.get("p95_px") is not None and reproj["p95_px"] <= max_p95_reproj_px
        )
        ns.temporal["direct_status"] = s.status.value
        ns.temporal["rescue"] = {
            "attempted": True,
            "validated": bool(validated),
            "method": "bracketed_linear_KC_so3_slerp",
            "left_frame": left.frame_index,
            "right_frame": right.frame_index,
            "alpha": float(a),
            "target_reprojection": reproj,
            "thresholds": {
                "max_neighbor_distance": int(max_neighbor_distance),
                "min_correspondences": int(min_correspondences),
                "min_non_ground_correspondences": int(min_non_ground_correspondences),
                "max_median_reproj_px": float(max_median_reproj_px),
                "max_p95_reproj_px": float(max_p95_reproj_px),
                "provisional": True,
            },
        }
        if validated:
            ns.status = CameraStatus.VALID
            ns.source["camera_solution_source"] = "temporal_rescue"
            ns.diagnostics["direct_camera_before_rescue"] = {
                "K": s.K.tolist(), "R": s.R_world_to_camera.tolist(), "C": s.camera_center_world_m.tolist(),
            }
        annotate_capabilities(ns)
        result.append(ns)

    return CameraTimeline(result, shot_id=timeline.shot_id, metadata={**timeline.metadata, "temporal_rescue_applied": True})


def _record_failed_rescue(s: CameraState, reason: str) -> CameraState:
    ns = copy.deepcopy(s)
    ns.temporal.setdefault("rescue", {})
    ns.temporal["rescue"].update({"attempted": True, "validated": False, "reason": reason})
    annotate_capabilities(ns)
    return ns


def _vertical_valid(s: CameraState) -> bool:
    caps = (s.diagnostics or {}).get("capabilities", {}) or {}
    return ((caps.get("vertical_3d") or {}).get("status")) == CameraStatus.VALID.value


def _slerp_pair(R0: np.ndarray, R1: np.ndarray, alpha: float) -> np.ndarray:
    rotations = Rotation.from_matrix(np.stack([R0, R1], axis=0))
    slerp = Slerp([0.0, 1.0], rotations)
    return slerp([float(alpha)]).as_matrix()[0]
