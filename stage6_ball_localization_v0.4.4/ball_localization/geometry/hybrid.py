from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from ..camera import CameraStateLite
from ..contracts import BallCandidate2D, BallFrameState
from .localization import estimate_frame
from .temporal import TemporalRefinementConfig, refine_temporal_trajectory


@dataclass(frozen=True)
class HybridGeometryConfig:
    ground_proxy_height_m: float = 0.55
    ballistic_consensus_distance_m: float = 3.0
    median_window_frames: int = 3

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "HybridGeometryConfig":
        value = value or {}
        return cls(float(value.get("ground_proxy_height_m", 0.55)), float(value.get("ballistic_consensus_distance_m", 3.0)), int(value.get("median_window_frames", 3)))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _ballistic_segment(items: list[tuple[int, CameraStateLite, BallCandidate2D, BallFrameState]], *, fitted_g: bool, fps: float) -> dict[int, list[float] | None]:
    out = {fi: None for fi, *_ in items}
    if len(items) < (4 if fitted_g else 3):
        return out
    origin_frame = items[0][0]
    columns = 7 if fitted_g else 6
    rows: list[np.ndarray] = []; rhs: list[np.ndarray] = []
    for fi, camera, candidate, size_state in items:
        t = (fi - origin_frame) / float(fps)
        _, direction = camera.world_ray(np.asarray(candidate.center_uv, float))
        projector = np.eye(3) - np.outer(direction[0], direction[0])
        ray_row = np.zeros((3, columns), float); ray_row[:, :3] = projector; ray_row[:, 3:6] = t * projector
        if fitted_g:
            ray_row[:, 6] = projector @ np.asarray([0.0, 0.0, -0.5*t*t]); ray_target = projector @ camera.camera_center_world_m
        else:
            ray_target = projector @ (camera.camera_center_world_m + np.asarray([0.0, 0.0, 0.5*9.81*t*t]))
        rows.append(ray_row); rhs.append(ray_target)
        if size_state.size_prior_xyz_world_m is not None:
            anchor = np.zeros((3, columns), float); anchor[:, :3] = np.eye(3); anchor[:, 3:6] = t*np.eye(3)
            if fitted_g:
                anchor[:, 6] = np.asarray([0.0, 0.0, -0.5*t*t]); target = np.asarray(size_state.size_prior_xyz_world_m, float)
            else:
                target = np.asarray(size_state.size_prior_xyz_world_m, float) + np.asarray([0.0, 0.0, 0.5*9.81*t*t])
            rows.append(0.25*anchor); rhs.append(0.25*target)
    try:
        solution = np.linalg.lstsq(np.vstack(rows), np.concatenate(rhs), rcond=None)[0]
    except np.linalg.LinAlgError:
        return out
    g = float(solution[6]) if fitted_g else 9.81
    if not math.isfinite(g) or not 0.0 < g < 30.0:
        return out
    for fi, *_ in items:
        t = (fi-origin_frame)/float(fps); point = solution[:3] + solution[3:6]*t + np.asarray([0.0,0.0,-0.5*g*t*t])
        if np.all(np.isfinite(point)) and -60 <= point[0] <= 60 and -45 <= point[1] <= 45 and 0 <= point[2] <= 30:
            out[fi] = point.astype(float).tolist()
    return out


def hybrid_trajectory_states(*, cameras_by_frame: Mapping[int, CameraStateLite], selected: Mapping[int, BallCandidate2D | None], frame_indices: Sequence[int], fps: float, ball_radius_m: float, config: HybridGeometryConfig, temporal_config: TemporalRefinementConfig | None = None) -> tuple[list[BallFrameState], dict[str, Any]]:
    fis = [int(fi) for fi in frame_indices]
    raw = {fi: estimate_frame(cameras_by_frame[fi], selected.get(fi), fps=fps, ball_radius_m=ball_radius_m, mode="size-prior") for fi in fis}
    ground = {fi: estimate_frame(cameras_by_frame[fi], selected.get(fi), fps=fps, ball_radius_m=ball_radius_m, mode="ground-only") for fi in fis}
    temporal = refine_temporal_trajectory(cameras_by_frame=cameras_by_frame, selected=selected, frame_indices=fis, fps=fps, ball_radius_m=ball_radius_m, config=temporal_config)
    fixed: dict[int, list[float] | None] = {}; fitted: dict[int, list[float] | None] = {}
    segment: list[tuple[int, CameraStateLite, BallCandidate2D, BallFrameState]] = []
    for fi in fis + [None]:
        candidate = None if fi is None else selected.get(fi)
        if fi is not None and candidate is not None:
            segment.append((fi, cameras_by_frame[fi], candidate, raw[fi])); continue
        if segment:
            fixed.update(_ballistic_segment(segment, fitted_g=False, fps=fps)); fitted.update(_ballistic_segment(segment, fitted_g=True, fps=fps)); segment=[]
    regime_counts: dict[str, int] = {}; method_counts: dict[str, int] = {}; frames: list[BallFrameState] = []
    half = max(0, config.median_window_frames//2)
    for index, fi in enumerate(fis):
        heights = [raw[fis[j]].size_prior_xyz_world_m[2] for j in range(max(0,index-half), min(len(fis),index+half+1)) if raw[fis[j]].size_prior_xyz_world_m is not None]
        proxy = None if not heights else float(np.median(heights)); regime = "GROUND" if proxy is not None and proxy <= config.ground_proxy_height_m else "AIRBORNE"
        size, grd, tmp = raw[fi].selected_center_xyz_world_m, ground[fi].selected_center_xyz_world_m, temporal.frames[fi].xyz_world_m
        prediction=None; method="HYBRID_NO_PREDICTION"; consensus=None
        if regime == "GROUND" and grd is not None:
            prediction, method = grd, "HYBRID_GROUND_PLANE"
        elif size is not None:
            for candidate, label in ((fitted.get(fi), "HYBRID_BALLISTIC_FIT_G"), (fixed.get(fi), "HYBRID_BALLISTIC_G9_81")):
                if candidate is not None:
                    distance=float(np.linalg.norm(np.asarray(candidate)-np.asarray(size)))
                    if distance <= config.ballistic_consensus_distance_m:
                        prediction, method, consensus = candidate, label, distance; break
            if prediction is None: prediction, method = size, "HYBRID_SIZE_PRIOR_AIRBORNE"
        elif tmp is not None:
            prediction, method = tmp, "HYBRID_TEMPORAL_FALLBACK"
        diag=dict(raw[fi].diagnostics); diag["hybrid"]={"config":config.to_dict(),"regime":regime,"size_height_proxy_m":proxy,"selected_method":method,"ballistic_consensus_distance_m":consensus,"temporal":temporal.frames[fi].to_dict()}
        frames.append(BallFrameState(fi, raw[fi].timestamp_sec, raw[fi].candidate, raw[fi].observation_status, raw[fi].camera_status, "VALID_"+method if prediction is not None else "NO_HYBRID_PREDICTION", raw[fi].ground_contact_xyz_world_m, raw[fi].ground_center_xyz_world_m, raw[fi].size_prior_xyz_world_m, prediction, method if prediction is not None else None, diag))
        regime_counts[regime]=regime_counts.get(regime,0)+1; method_counts[method]=method_counts.get(method,0)+1
    return frames, {"config":config.to_dict(),"regime_counts":regime_counts,"selected_method_counts":method_counts,"temporal_refinement":temporal.diagnostics}
