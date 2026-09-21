from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import hashlib
import json
import math
import numpy as np

from .camera import CameraTimelineLite
from .foot_contact import contact_to_dict, infer_contact_sequence
from .initializers.cache import InitializerCache
from .optimizer import FrameInput, OptimizationResult, solve_metric_pose
from .schemas import Stage4Config
from .stage3_adapter import PoseObservation2D, Stage3State, Stage3Track, load_stage3_state
from .stage3_bridge_qa import audit_stage3_bridge
from .uncertainty import pixel_sensitivity_monte_carlo
from .wholebody import LEGAL_GEOMETRY_CANDIDATE_23, POSE23_NAMES, derive_pose_features
from .output_semantics import build_stage5_handoff, selected_frame_availability

STAGE4_VERSION = "stage4-metric3d-0.2.0"
OUTPUT_SCHEMA = "metric-pose-3d-state-1.0"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _as_json_float(value: float):
    value = float(value)
    return value if math.isfinite(value) else None


def _vec_or_none(vec: np.ndarray):
    arr = np.asarray(vec, dtype=np.float64)
    if not np.isfinite(arr).all():
        return None
    return [float(v) for v in arr]


def _json_safe(value):
    """Recursively replace non-finite numpy/Python numbers with None."""
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def preflight_stage4(
    *,
    stage3_state: str | Path,
    camera_dir: str | Path,
    initializer_cache: str | Path | None = None,
    config: Optional[Stage4Config] = None,
) -> Dict[str, Any]:
    cfg = config or Stage4Config()
    s3 = load_stage3_state(stage3_state)
    cams = CameraTimelineLite.load_dir(camera_dir)
    init = InitializerCache.load_jsonl(initializer_cache) if initializer_cache else None
    bridge_qa = audit_stage3_bridge(s3)

    width = int(s3.replay_context.get("image_width", -1))
    height = int(s3.replay_context.get("image_height", -1))
    selected_cam = cams.by_frame(s3.selected_frame)
    resolution_match = bool(
        selected_cam is not None
        and selected_cam.image_width == width
        and selected_cam.image_height == height
    )
    selected_status_ok = bool(
        selected_cam is not None
        and (
            selected_cam.status == "VALID"
            or (cfg.allow_degraded_camera and selected_cam.status == "DEGRADED")
        )
    )

    window_lo = s3.selected_frame - cfg.window_radius_frames
    window_hi = s3.selected_frame + cfg.window_radius_frames
    requested_frames = sorted({
        o.frame_index
        for t in s3.tracks
        for o in t.observations
        if window_lo <= o.frame_index <= window_hi
    })
    missing_camera_frames = [f for f in requested_frames if cams.by_frame(f) is None]
    invalid_camera_frames = [
        f for f in requested_frames
        if cams.by_frame(f) is not None
        and cams.by_frame(f).status == "INVALID"
    ]
    mismatched_resolution_frames = [
        f for f in requested_frames
        if cams.by_frame(f) is not None
        and (cams.by_frame(f).image_width != width or cams.by_frame(f).image_height != height)
    ]

    candidate_at_t0 = [
        t.track_id for t in s3.tracks
        if any(o.frame_index == s3.selected_frame for o in t.observations)
    ]
    missing_init_at_t0 = []
    if init is not None:
        missing_init_at_t0 = [tid for tid in candidate_at_t0 if init.get(tid, s3.selected_frame) is None]
    elif cfg.require_initializer:
        missing_init_at_t0 = list(candidate_at_t0)

    errors: List[str] = []
    if selected_cam is None:
        errors.append("selected_frame_camera_missing")
    if not resolution_match:
        errors.append("stage3_camera_image_resolution_mismatch")
    if not selected_status_ok:
        errors.append("selected_frame_camera_status_not_usable")
    if not candidate_at_t0:
        errors.append("no_stage3_track_at_selected_frame")
    if cfg.require_initializer and missing_init_at_t0:
        errors.append("initializer_missing_at_selected_frame")

    warnings: List[str] = []
    if missing_camera_frames:
        warnings.append("some_temporal_frames_missing_camera")
    if invalid_camera_frames:
        warnings.append("some_temporal_frames_have_invalid_camera")
    if mismatched_resolution_frames:
        warnings.append("some_temporal_camera_frames_resolution_mismatch")
    if init is None:
        warnings.append("initializer_cache_absent_geometry_only_fallback")
    elif missing_init_at_t0:
        warnings.append("some_selected_tracks_missing_initializer")
    if bridge_qa["summary"]["raw_pose_missing_tracks"]:
        warnings.append("stage3_selected_frame_raw_pose_missing")
    if bridge_qa["summary"]["edge_risk_tracks"]:
        warnings.append("stage3_selected_frame_edge_risk")
    if bridge_qa["summary"]["crowded_risk_tracks"]:
        warnings.append("stage3_selected_frame_crowded_or_occluded_risk")
    if bridge_qa["summary"]["incomplete_core_anchor_tracks"]:
        warnings.append("stage3_selected_frame_core_anchor_incomplete")

    return {
        "schema_version": "stage4-preflight-1.0",
        "ready": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "selected_frame": s3.selected_frame,
        "stage3_coordinate_space": s3.raw.get("coordinate_space"),
        "stage3_image_size": [width, height],
        "selected_camera_image_size": None if selected_cam is None else [selected_cam.image_width, selected_cam.image_height],
        "selected_camera_status": None if selected_cam is None else selected_cam.status,
        "resolution_match": resolution_match,
        "candidate_tracks_at_t0": candidate_at_t0,
        "requested_temporal_frames": requested_frames,
        "missing_camera_frames": missing_camera_frames,
        "invalid_camera_frames": invalid_camera_frames,
        "mismatched_resolution_frames": mismatched_resolution_frames,
        "initializer_cache": None if init is None else init.path,
        "missing_initializer_at_t0": missing_init_at_t0,
        "no_extra_undistortion": True,
        "camera_geometry_source": "Stage1 CameraState v1.2 JSON",
        "stage3_bridge_qa": bridge_qa,
    }


def _clone_obs_with_temporal_fill(
    track: Stage3Track,
    obs: PoseObservation2D,
    *,
    max_gap: int,
) -> Tuple[np.ndarray, np.ndarray, Tuple[str, ...]]:
    uv = obs.uv23.copy()
    weights = obs.state_weights23.copy()
    source = ["STAGE3_RAW" if np.isfinite(uv[j]).all() and weights[j] > 0 else "MISSING" for j in range(23)]
    by_frame = track.by_frame()
    frames = sorted(by_frame)
    for j in range(23):
        if np.isfinite(uv[j]).all() and weights[j] > 0:
            continue
        left = [f for f in frames if f < obs.frame_index and obs.frame_index - f <= max_gap and np.isfinite(by_frame[f].uv23[j]).all() and by_frame[f].state_weights23[j] > 0]
        right = [f for f in frames if f > obs.frame_index and f - obs.frame_index <= max_gap and np.isfinite(by_frame[f].uv23[j]).all() and by_frame[f].state_weights23[j] > 0]
        if left and right:
            fl, fr = max(left), min(right)
            alpha = (obs.frame_index - fl) / max(fr - fl, 1)
            uv[j] = (1.0 - alpha) * by_frame[fl].uv23[j] + alpha * by_frame[fr].uv23[j]
            weights[j] = 0.30 * min(by_frame[fl].state_weights23[j], by_frame[fr].state_weights23[j])
            source[j] = "STAGE4_TEMPORAL_2D_INTERPOLATION"
    return uv, weights, tuple(source)


def _frames_for_track(
    s3: Stage3State,
    track: Stage3Track,
    cams: CameraTimelineLite,
    init: Optional[InitializerCache],
    cfg: Stage4Config,
):
    lo = s3.selected_frame - cfg.window_radius_frames
    hi = s3.selected_frame + cfg.window_radius_frames
    raw_obs = [o for o in track.observations if lo <= o.frame_index <= hi]
    contacts = infer_contact_sequence(
        raw_obs,
        likelihood_threshold=cfg.contact_likelihood_threshold,
        both_y_tolerance_ratio=cfg.both_contact_y_tolerance_ratio,
    )
    frames: List[FrameInput] = []
    source_obs: Dict[int, PoseObservation2D] = {}
    for obs in raw_obs:
        cam = cams.by_frame(obs.frame_index)
        if cam is None or cam.status == "INVALID":
            continue
        if cam.image_width != int(s3.replay_context.get("image_width", cam.image_width)) or cam.image_height != int(s3.replay_context.get("image_height", cam.image_height)):
            continue
        uv, weights, uv_source = _clone_obs_with_temporal_fill(track, obs, max_gap=cfg.max_temporal_fill_gap_frames)
        frames.append(FrameInput(
            frame_index=obs.frame_index,
            camera=cam,
            uv23=uv,
            state_weights23=weights,
            bbox_xyxy=obs.bbox_xyxy.copy(),
            contact=contacts[obs.frame_index],
            initializer=None if init is None else init.get(track.track_id, obs.frame_index),
            uv_source23=uv_source,
        ))
        source_obs[obs.frame_index] = obs
    return frames, contacts, source_obs


def _track_quality(result: OptimizationResult, frame_count: int, initializer_available: bool, cfg: Stage4Config) -> str:
    geometry = result.diagnostics.get("geometry_quality_gate") or {}
    if not bool(geometry.get("plausible")):
        return "REJECTED"
    if result.status == "MAX_NFEV":
        return "DEGRADED"
    if result.status != "CONVERGED":
        return "REJECTED"
    if not initializer_available:
        return "DEGRADED"
    if frame_count < cfg.min_temporal_frames:
        return "DEGRADED"
    return "VALID"


def _joint_records(
    frame: FrameInput,
    xyz23: np.ndarray,
    uncertainty: Mapping[str, Any] | None,
    source_obs: PoseObservation2D,
) -> List[Dict[str, Any]]:
    std = None
    q025 = None
    q975 = None
    if uncertainty and uncertainty.get("status") in {"OK", "DEGRADED"}:
        std = np.asarray(uncertainty.get("std_xyz_m"), dtype=np.float64)
        q025 = np.asarray(uncertainty.get("q025_xyz_m"), dtype=np.float64)
        q975 = np.asarray(uncertainty.get("q975_xyz_m"), dtype=np.float64)
    out = []
    for j, name in enumerate(POSE23_NAMES):
        xyz = xyz23[j]
        item = {
            "index": j,
            "name": name,
            "xyz_world_m": _vec_or_none(xyz),
            "stage3_uv_raw_distorted_px": _vec_or_none(source_obs.uv23[j]),
            "uv_source_for_stage4": None if frame.uv_source23 is None else frame.uv_source23[j],
            "stage3_keypoint_state": source_obs.states23[j],
            "candidate_for_legal_body_geometry": bool(LEGAL_GEOMETRY_CANDIDATE_23[j]),
            "anatomical_note": (
                "arm_branch_excluded_by_project_scope" if not LEGAL_GEOMETRY_CANDIDATE_23[j] else "candidate_anchor_only_stage5_builds_surface"
            ),
        }
        if std is not None and np.isfinite(std[j]).all():
            item["std_xyz_m"] = std[j].tolist()
            item["q025_xyz_m"] = q025[j].tolist()
            item["q975_xyz_m"] = q975[j].tolist()
        else:
            item["std_xyz_m"] = None
        if frame.initializer is not None:
            rel = frame.initializer.relative_depth_23[j]
            item["initializer_relative_depth_raw"] = _as_json_float(rel)
            item["initializer_backend"] = frame.initializer.backend
        else:
            item["initializer_relative_depth_raw"] = None
            item["initializer_backend"] = None
        out.append(item)
    return out


def run_stage4(
    *,
    stage3_state: str | Path,
    camera_dir: str | Path,
    output_dir: str | Path,
    initializer_cache: str | Path | None = None,
    config: Optional[Stage4Config] = None,
    strict_preflight: bool = True,
) -> Dict[str, Any]:
    cfg = config or Stage4Config()
    preflight = preflight_stage4(
        stage3_state=stage3_state,
        camera_dir=camera_dir,
        initializer_cache=initializer_cache,
        config=cfg,
    )
    if strict_preflight and not preflight["ready"]:
        raise RuntimeError("Stage4 preflight failed: " + ", ".join(preflight["errors"]))

    s3 = load_stage3_state(stage3_state)
    bridge_qa = audit_stage3_bridge(s3)
    cams = CameraTimelineLite.load_dir(camera_dir)
    init = InitializerCache.load_jsonl(initializer_cache) if initializer_cache else None
    outdir = Path(output_dir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    track_outputs: List[Dict[str, Any]] = []
    selected_outputs: List[Dict[str, Any]] = []
    quality_counts = {"VALID": 0, "DEGRADED": 0, "REJECTED": 0, "MISSING": 0}
    reproj_values: List[float] = []
    ground_values: List[float] = []

    total_tracks = len(s3.tracks)

    for track_idx, track in enumerate(s3.tracks, start=1):

        print(
            f"\n[Stage4] [{track_idx}/{total_tracks}] "
            f"{track.track_id} - preparing...",
            flush=True,
        )

        frames, contacts, source_obs = _frames_for_track(
            s3, track, cams, init, cfg
        )

        print(
            f"[Stage4] [{track_idx}/{total_tracks}] "
            f"{track.track_id} - optimizing {len(frames)} frames...",
            flush=True,
        )

        selected_frame_input = next((f for f in frames if f.frame_index == s3.selected_frame), None)
        if selected_frame_input is None:
            availability = selected_frame_availability(None, "MISSING")
            quality_counts["MISSING"] += 1
            track_outputs.append({
                "track_id": track.track_id,
                "upstream_role": track.role,
                "status": "MISSING",
                "reason": "no usable Stage3+Camera observation at selected frame",
                "observations": [],
                "selected_frame_pose_status": "MISSING",
                "selected_frame_stage3_pose_status": "MISSING",
                "selected_frame_availability": availability,
                "track_optimizer_status": None,
            })
            continue

        try:
            result = solve_metric_pose(frames, cfg)
        except Exception as exc:
            availability = selected_frame_availability(None, "REJECTED")
            quality_counts[availability["status"]] += 1
            track_outputs.append({
                "track_id": track.track_id,
                "upstream_role": track.role,
                "status": "REJECTED",
                "reason": f"optimizer_error: {type(exc).__name__}: {exc}",
                "observations": [],
                "selected_frame_pose_status": availability["status"],
                "selected_frame_stage3_pose_status": source_obs.get(s3.selected_frame).pose_status if s3.selected_frame in source_obs else "MISSING",
                "selected_frame_availability": availability,
                "track_optimizer_status": None,
            })
            continue

        initializer_available = any(f.initializer is not None for f in frames)
        quality = _track_quality(result, len(frames), initializer_available, cfg)
        print(
            f"[Stage4] [{track_idx}/{total_tracks}] {track.track_id} - {quality} | "
            f"termination={result.status} | nfev={result.nfev} | njev={result.njev} | "
            f"elapsed={result.elapsed_s:.2f}s | cost={result.cost:.3f}",
            flush=True,
        )
        if np.isfinite(result.median_reprojection_error_px):
            reproj_values.append(result.median_reprojection_error_px)
        if result.ground_residual_median_m is not None:
            ground_values.append(result.ground_residual_median_m)

        uncertainty = pixel_sensitivity_monte_carlo(
            frames,
            result,
            cfg,
            selected_frame_index=s3.selected_frame,
            states23_by_frame={frame_index: obs.states23 for frame_index, obs in source_obs.items()},
            track_id=track.track_id,
            mode=cfg.uncertainty_mode,
        )

        observations_out = []
        for ti, frame in enumerate(frames):
            src = source_obs[frame.frame_index]
            unc = uncertainty if frame.frame_index == s3.selected_frame else None
            xyz = result.xyz_world_m[ti]
            rec = {
                "frame_index": frame.frame_index,
                "metric_pose23": _joint_records(frame, xyz, unc, src),
                "derived": derive_pose_features(xyz),
                "contact": contact_to_dict(frame.contact),
                "camera_status": frame.camera.status,
                "quality": quality if frame.frame_index == s3.selected_frame else ("VALID" if result.success else "REJECTED"),
            }
            observations_out.append(rec)
            if frame.frame_index == s3.selected_frame:
                availability = selected_frame_availability(rec, quality)
                rec["quality"] = availability["status"]
                selected_outputs.append({
                    "track_id": track.track_id,
                    "upstream_role": track.role,
                    "observation": rec,
                    "uncertainty": uncertainty,
                    "selected_frame_pose_status": availability["status"],
                    "selected_frame_availability": availability,
                })

        track_out = {
            "track_id": track.track_id,
            "upstream_role": track.role,
            "upstream_identity_confidence": track.identity_confidence,
            "track_optimizer_status": result.status,
            "selected_frame_pose_status": availability["status"],
            "selected_frame_stage3_pose_status": source_obs[s3.selected_frame].pose_status,
            "selected_frame_availability": availability,
            "track_scale": {
                "estimated_height_m": result.estimated_height_m,
                "height_prior_nominal_m": cfg.nominal_height_m,
                "height_prior_sigma_m": cfg.height_sigma_m,
                "depth_beta_m": result.depth_beta_m if result.initializer_used else None,
            },
            "initializer": {
                "used": result.initializer_used,
                "required": cfg.require_initializer,
                "primary_expected_backend": "RTMW3D-L-384x288",
                "global_position_trusted": False,
                "stage3_xy_source_of_truth": True,
            },
            "optimizer": {
                "status": result.status,
                "success": result.success,
                "cost": result.cost,
                "nfev": result.nfev,
                "njev": result.njev,
                "elapsed_s": result.elapsed_s,
                "termination_reason": result.status,
                "message": result.message,
                "median_reprojection_error_px": _as_json_float(result.median_reprojection_error_px),
                "p90_reprojection_error_px": _as_json_float(result.p90_reprojection_error_px),
                "median_bone_abs_error_m": _as_json_float(result.median_bone_abs_error_m),
                "ground_residual_median_m": result.ground_residual_median_m,
                "negative_z_joint_fraction": result.negative_z_joint_fraction,
                "diagnostics": result.diagnostics,
            },
            "temporal_window": {
                "frames_used": result.frame_indices.tolist(),
                "frame_count": len(frames),
                "minimum_for_full_temporal_quality": cfg.min_temporal_frames,
            },
            "selected_frame_uncertainty": uncertainty,
            "observations": observations_out,
        }
        track_outputs.append(track_out)
        quality_counts[availability["status"]] += 1

    candidate_n = len(s3.tracks)
    selected_available = sum(v for k, v in quality_counts.items() if k in ("VALID", "DEGRADED"))
    state_path = outdir / "metric_pose_3d_state.json"
    handoff_path = outdir / "stage5_handoff.json"
    state: Dict[str, Any] = {
        "schema_version": OUTPUT_SCHEMA,
        "stage4_version": STAGE4_VERSION,
        "source_stage3": {
            "path": str(s3.path),
            "sha256": _sha256(s3.path),
            "schema_version": s3.raw.get("schema_version"),
            "stage3_version": s3.raw.get("stage3_version"),
        },
        "source_stage1": {
            "camera_state_directory": str(Path(camera_dir).expanduser().resolve()),
            "camera_state_schema": "1.2",
            "geometry_policy": "use CameraState raw-pixel distortion semantics; no downstream independent undistortion",
        },
        "source_initializer": {
            "cache": None if init is None else init.path,
            "role": "relative_3d_depth_prior_only",
            "global_metric_position_trusted": False,
        },
        "replay_context": s3.replay_context,
        "world_frame": {
            "units": "metres",
            "origin": "pitch_center",
            "x_axis": "goal_to_goal",
            "y_axis": "touchline_to_touchline",
            "z_axis": "up",
        },
        "pose_schema": {
            "name": "METRIC_POSE_23",
            "names": list(POSE23_NAMES),
            "source_2d": "COCO_WHOLEBODY_133 first 23 points",
            "arms_below_shoulders_excluded_from_downstream_legal_geometry": True,
            "legal_geometry_candidate_mask": list(LEGAL_GEOMETRY_CANDIDATE_23),
        },
        "configuration": cfg.to_dict(),
        "preflight": preflight,
        "stage3_bridge_qa": bridge_qa,
        "tracks": track_outputs,
        "selected_frame_poses": selected_outputs,
        "metrics": {
            "candidate_tracks_from_stage3": candidate_n,
            "MetricPoseCoverageAtT0_given_stage3_candidate": float(selected_available / max(candidate_n, 1)),
            "selected_frame_status_counts": quality_counts,
            "median_track_reprojection_error_px": float(np.median(reproj_values)) if reproj_values else None,
            "median_track_ground_residual_m": float(np.median(ground_values)) if ground_values else None,
            "research_accuracy_requires_ground_truth": True,
        },
        "acceptance_gate": {
            "structural_passed": bool(preflight["ready"]),
            "research_accuracy_frozen": False,
            "uncertainty_calibrated": False,
            "note": "v0.2 adds camera-fixed pixel sensitivity; numeric 3D accuracy and calibrated uncertainty require real ground truth.",
        },
        "diagnostics": {
            "hard_camera_ray_parameterization": True,
            "stage3_xy_moved_by_optimizer": False,
            "independent_undistortion_applied": False,
            "attack_direction_applied": False,
            "team_assignment_applied": False,
            "legal_body_surface_constructed": False,
        },
        "artifacts": {},
    }
    state_path.write_text(json.dumps(_json_safe(state), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")

    handoff = build_stage5_handoff(state, str(state_path))
    handoff_path.write_text(json.dumps(_json_safe(handoff), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    state["artifacts"] = {
        "metric_pose_3d_state": str(state_path),
        "stage5_handoff": str(handoff_path),
    }

    try:
        from .visualization import save_topdown_selected_frame
        viz = save_topdown_selected_frame(state, outdir / "selected_frame_metric3d_topdown.png")
        state["artifacts"]["selected_frame_metric3d_topdown"] = str(viz)
    except Exception as exc:
        state["diagnostics"]["visualization_error"] = f"{type(exc).__name__}: {exc}"

    state_path.write_text(json.dumps(_json_safe(state), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return state
