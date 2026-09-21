from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from .camera import CameraStateLite, CameraTimelineLite
from .height_profiles import HeightProfile, load_height_profile
from .proxy_evaluation import anatomical_group, evaluate_proxy_state, structural_gate
from .proxy_quality import compactness_diagnostics, robust_ground_anchor
from .proxy_schemas import Stage4ProjectionConfig
from .stage3_adapter import PoseObservation2D, Stage3State, load_stage3_state
from .stage3_bridge_qa import audit_stage3_bridge
from .wholebody import POSE23_NAMES


STAGE4_VERSION = "stage4-metric-body-proxy-0.3.1"
OUTPUT_SCHEMA = "metric-body-proxy-state-1.1"
HANDOFF_SCHEMA = "stage4-body-proxy-handoff-1.1"
PROJECTION_METHOD = "CAMERA_RAY_X_CANONICAL_Z_PLANE"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(value), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def _camera_usable(camera: Optional[CameraStateLite], config: Stage4ProjectionConfig) -> bool:
    if camera is None or camera.status == "INVALID":
        return False
    return camera.status == "VALID" or (config.allow_degraded_camera and camera.status == "DEGRADED")


def _pitch_dimensions(camera: CameraStateLite) -> tuple[float, float]:
    return float(camera.pitch.get("length_m", 105.0)), float(camera.pitch.get("width_m", 68.0))


def _pitch_plausible(xyz: np.ndarray, camera: CameraStateLite, margin_m: float) -> bool:
    length, width = _pitch_dimensions(camera)
    return bool(
        abs(float(xyz[0])) <= length / 2.0 + margin_m
        and abs(float(xyz[1])) <= width / 2.0 + margin_m
    )


def _profile_metadata(profile: HeightProfile, reference_height_m: float) -> Dict[str, Any]:
    return {
        "profile_id": profile.profile_id,
        "profile_source": str(profile.source_path),
        "profile_sha256": _sha256(profile.source_path),
        "reference_player_height_m": float(reference_height_m),
        "provenance": profile.raw.get("provenance"),
    }


def _neutral_stage3_bridge_qa(stage3: Stage3State) -> Dict[str, Any]:
    """Reuse image-evidence QA while removing legacy legal-body semantics."""

    report = audit_stage3_bridge(stage3)
    report["schema_version"] = "stage3-to-stage4-proxy-bridge-qa-1.0"
    for track in report.get("tracks", []):
        track.pop("raw_valid_legal_joint_count", None)
    report["policy"] = (
        "read_only_audit; flags do not alter Stage-3 pixels or Stage-4 projection; "
        "legal-body membership is not evaluated"
    )
    return report


def preflight_stage4(
    *,
    stage3_state: str | Path,
    camera_dir: str | Path,
    config: Optional[Stage4ProjectionConfig] = None,
) -> Dict[str, Any]:
    cfg = config or Stage4ProjectionConfig()
    cfg.validate()
    stage3 = load_stage3_state(stage3_state)
    cameras = CameraTimelineLite.load_dir(camera_dir)
    profile = load_height_profile(cfg.height_profile)
    bridge_qa = _neutral_stage3_bridge_qa(stage3)

    image_width = int(stage3.replay_context.get("image_width", -1))
    image_height = int(stage3.replay_context.get("image_height", -1))
    selected_camera = cameras.by_frame(stage3.selected_frame)
    resolution_match = bool(
        selected_camera is not None
        and selected_camera.image_width == image_width
        and selected_camera.image_height == image_height
    )
    selected_status_ok = _camera_usable(selected_camera, cfg)

    lower = stage3.selected_frame - cfg.window_radius_frames
    upper = stage3.selected_frame + cfg.window_radius_frames
    requested_frames = sorted({
        obs.frame_index
        for track in stage3.tracks
        for obs in track.observations
        if lower <= obs.frame_index <= upper
    })
    missing_camera_frames = [frame for frame in requested_frames if cameras.by_frame(frame) is None]
    unusable_camera_frames = [
        frame for frame in requested_frames
        if cameras.by_frame(frame) is not None and not _camera_usable(cameras.by_frame(frame), cfg)
    ]
    mismatched_resolution_frames = [
        frame for frame in requested_frames
        if cameras.by_frame(frame) is not None
        and (
            cameras.by_frame(frame).image_width != image_width
            or cameras.by_frame(frame).image_height != image_height
        )
    ]
    candidates_at_t0 = [
        track.track_id for track in stage3.tracks
        if stage3.selected_frame in track.by_frame()
    ]

    errors: List[str] = []
    if selected_camera is None:
        errors.append("selected_frame_camera_missing")
    if not resolution_match:
        errors.append("stage3_camera_image_resolution_mismatch")
    if not selected_status_ok:
        errors.append("selected_frame_camera_status_not_usable")
    if not candidates_at_t0:
        errors.append("no_stage3_track_at_selected_frame")

    warnings: List[str] = []
    if missing_camera_frames:
        warnings.append("some_temporal_frames_missing_camera")
    if unusable_camera_frames:
        warnings.append("some_temporal_frames_have_unusable_camera")
    if mismatched_resolution_frames:
        warnings.append("some_temporal_camera_frames_resolution_mismatch")
    if bridge_qa["summary"]["raw_pose_missing_tracks"]:
        warnings.append("stage3_selected_frame_raw_pose_missing")
    if bridge_qa["summary"]["edge_risk_tracks"]:
        warnings.append("stage3_selected_frame_edge_risk")
    if bridge_qa["summary"]["crowded_risk_tracks"]:
        warnings.append("stage3_selected_frame_crowded_or_occluded_risk")

    return {
        "schema_version": "stage4-proxy-preflight-1.1",
        "stage4_version": STAGE4_VERSION,
        "ready": not errors,
        "errors": errors,
        "warnings": warnings,
        "selected_frame": stage3.selected_frame,
        "stage3_coordinate_space": stage3.raw.get("coordinate_space"),
        "stage3_image_size": [image_width, image_height],
        "selected_camera_image_size": None if selected_camera is None else [selected_camera.image_width, selected_camera.image_height],
        "selected_camera_status": None if selected_camera is None else selected_camera.status,
        "resolution_match": resolution_match,
        "candidate_tracks_at_t0": candidates_at_t0,
        "requested_temporal_frames": requested_frames,
        "missing_camera_frames": missing_camera_frames,
        "unusable_camera_frames": unusable_camera_frames,
        "mismatched_resolution_frames": mismatched_resolution_frames,
        "height_profile": _profile_metadata(profile, cfg.reference_player_height_m),
        "no_extra_undistortion": True,
        "camera_geometry_source": "Stage1 CameraState v1.2 JSON",
        "stage3_bridge_qa": bridge_qa,
    }


def _project_point(
    *,
    observation: PoseObservation2D,
    index: int,
    camera: Optional[CameraStateLite],
    profile: HeightProfile,
    config: Stage4ProjectionConfig,
) -> Dict[str, Any]:
    name = POSE23_NAMES[index]
    uv = observation.uv23[index]
    stage3_state = observation.states23[index]
    score = observation.raw_scores23[index]
    eligible = bool(np.isfinite(uv).all() and observation.state_weights23[index] > 0.0)
    z_m = profile.z_m(name, config.reference_player_height_m)
    base = {
        "index": index,
        "name": name,
        "anatomical_group": anatomical_group(name),
        "stage3_uv_raw_distorted_px": uv.tolist() if np.isfinite(uv).all() else None,
        "stage3_keypoint_state": stage3_state,
        "stage3_raw_model_score": float(score) if np.isfinite(score) else None,
        "height_prior": {
            "fraction": float(profile.fractions[name]),
            "reference_player_height_m": float(config.reference_player_height_m),
            "z_m": float(z_m),
            "profile_id": profile.profile_id,
            "kind": "engineering_prior",
        },
        "projection_method": PROJECTION_METHOD,
        "semantic_role": "RAW_HEIGHT_PLANE_PROXY",
        "downstream_legal_body_eligible": False,
        "xyz_proxy_world_m": None,
        "xy_pitch_m": None,
        "qa": None,
    }
    if not eligible:
        base["projection_status"] = "MISSING_STAGE3"
        return base
    if not _camera_usable(camera, config):
        base["projection_status"] = "CAMERA_UNAVAILABLE_OR_UNUSABLE"
        return base
    assert camera is not None
    xyz = np.asarray(camera.intersect_z_plane(uv, z_m), dtype=np.float64)
    if xyz.shape != (3,) or not np.isfinite(xyz).all():
        base["projection_status"] = "NO_FORWARD_RAY_PLANE_INTERSECTION"
        return base

    reprojected = np.asarray(camera.project_world(xyz, distort=True), dtype=np.float64)
    reprojection_error = float(np.linalg.norm(reprojected - uv))
    height_low = config.reference_player_height_m - config.height_sensitivity_delta_m
    height_high = config.reference_player_height_m + config.height_sensitivity_delta_m
    xyz_low = np.asarray(camera.intersect_z_plane(uv, profile.z_m(name, height_low)), dtype=np.float64)
    xyz_high = np.asarray(camera.intersect_z_plane(uv, profile.z_m(name, height_high)), dtype=np.float64)
    dx_candidates = [
        abs(float(candidate[0]) - float(xyz[0]))
        for candidate in (xyz_low, xyz_high)
        if candidate.shape == (3,) and np.isfinite(candidate).all()
    ]
    base.update({
        "projection_status": "PROJECTED",
        "xyz_proxy_world_m": xyz.tolist(),
        "xy_pitch_m": xyz[:2].tolist(),
        "qa": {
            "reprojection_error_px": reprojection_error,
            "pitch_plausible": _pitch_plausible(xyz, camera, config.pitch_margin_m),
            "longitudinal_height_sensitivity_m": max(dx_candidates) if dx_candidates else None,
            "height_sensitivity_delta_m": float(config.height_sensitivity_delta_m),
        },
    })
    return base


def _project_observation(
    observation: PoseObservation2D,
    camera: Optional[CameraStateLite],
    profile: HeightProfile,
    config: Stage4ProjectionConfig,
    expected_width: int,
    expected_height: int,
) -> Dict[str, Any]:
    resolution_match = bool(
        camera is not None
        and camera.image_width == expected_width
        and camera.image_height == expected_height
    )
    projection_camera = camera if resolution_match else None
    points = [
        _project_point(
            observation=observation,
            index=index,
            camera=projection_camera,
            profile=profile,
            config=config,
        )
        for index in range(23)
    ]
    eligible = sum(point["stage3_uv_raw_distorted_px"] is not None and point["stage3_keypoint_state"] != "MISSING" for point in points)
    projected = sum(point["xyz_proxy_world_m"] is not None for point in points)
    if eligible == 0 or projected == 0:
        projection_status = "MISSING"
    elif projected < eligible or camera is None or camera.status == "DEGRADED":
        projection_status = "DEGRADED"
    else:
        projection_status = "VALID"
    compactness = compactness_diagnostics(points, config)
    ground_anchor = robust_ground_anchor(points, config)
    body_quality_status = compactness["status"]
    if projection_status == "MISSING":
        body_quality_status = "MISSING"
    elif projection_status == "DEGRADED" and body_quality_status == "VALID":
        body_quality_status = "DEGRADED"
    return {
        "frame_index": observation.frame_index,
        "status": body_quality_status,
        "projection_status": projection_status,
        "body_proxy_quality_status": body_quality_status,
        "camera_status": None if camera is None else camera.status,
        "camera_resolution_match": resolution_match,
        "stage3_pose_status": observation.pose_status,
        "eligible_keypoints": int(eligible),
        "projected_keypoints": int(projected),
        "compactness": compactness,
        "ground_anchor": ground_anchor,
        "raw_height_plane_proxies_23": points,
    }


def _build_handoff(state: Mapping[str, Any], state_path: Path) -> Dict[str, Any]:
    selected = int((state.get("replay_context") or {}).get("selected_frame"))
    tracks = []
    for track in state.get("tracks", []):
        observation = next(
            (item for item in track.get("observations", []) if int(item.get("frame_index", -1)) == selected),
            None,
        )
        ground_anchor = None if observation is None else observation.get("ground_anchor")
        raw_projection_available = bool(
            observation is not None and int(observation.get("projected_keypoints", 0)) > 0
        )
        tracks.append({
            "track_id": track.get("track_id"),
            "upstream_role": track.get("upstream_role"),
            "upstream_identity_confidence": track.get("upstream_identity_confidence"),
            "selected_frame_body_proxy_status": track.get("selected_frame_proxy_status"),
            "selected_frame_ground_anchor": ground_anchor,
            "raw_projection_available": raw_projection_available,
            "raw_height_plane_proxies_exported": False,
        })
    ground_available = sum(
        (track.get("selected_frame_ground_anchor") or {}).get("status") in {"VALID", "DEGRADED"}
        for track in tracks
    )
    raw_available = sum(bool(track.get("raw_projection_available")) for track in tracks)
    return {
        "schema_version": HANDOFF_SCHEMA,
        "stage4_version": STAGE4_VERSION,
        "source_metric_body_proxy_state": str(state_path),
        "selected_frame": selected,
        "coordinate_frame": state.get("coordinate_frame"),
        "height_profile": state.get("height_profile"),
        "semantics": {
            "values_are": "quality-gated player ground-state proxies",
            "values_are_not": "true reconstructed 3D joints or legal-body boundaries",
            "legal_body_membership_assigned": False,
            "team_or_attack_direction_assigned": False,
            "raw_independent_height_plane_proxies_withheld": True,
        },
        "readiness": {
            "stage5_ground_state_available_tracks": int(ground_available),
            "stage5_ground_state_required_tracks": int(raw_available),
            "stage5_ground_state_candidate_tracks": len(tracks),
            "stage5_ground_state_ready": raw_available > 0 and ground_available == raw_available,
            "stage8_legal_body_handoff_ready": False,
            "stage8_blocker": "independent upper-body height-plane proxies are not physically coherent or accuracy-frozen",
        },
        "downstream_ownership": {
            "stage5": "team affiliation and player state",
            "stage6": "ball detection/tracking/metric localization",
            "stage8": "legal-body filtering and offside reference geometry",
            "stage9": "offside-position decision",
        },
        "tracks": tracks,
    }


def run_stage4(
    *,
    stage3_state: str | Path,
    camera_dir: str | Path,
    output_dir: str | Path,
    config: Optional[Stage4ProjectionConfig] = None,
    strict_preflight: bool = True,
) -> Dict[str, Any]:
    cfg = config or Stage4ProjectionConfig()
    cfg.validate()
    preflight = preflight_stage4(stage3_state=stage3_state, camera_dir=camera_dir, config=cfg)
    if strict_preflight and not preflight["ready"]:
        raise RuntimeError("Stage4 v0.3.1 preflight failed: " + ", ".join(preflight["errors"]))

    stage3 = load_stage3_state(stage3_state)
    cameras = CameraTimelineLite.load_dir(camera_dir)
    profile = load_height_profile(cfg.height_profile)
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "metric_body_proxy_state.json"
    handoff_path = output / "stage4_downstream_handoff.json"
    visualization_path = output / "selected_frame_metric_body_proxy_topdown.png"

    width = int(stage3.replay_context.get("image_width", -1))
    height = int(stage3.replay_context.get("image_height", -1))
    lower = stage3.selected_frame - cfg.window_radius_frames
    upper = stage3.selected_frame + cfg.window_radius_frames
    track_outputs = []
    selected_outputs = []
    status_counts = {"VALID": 0, "DEGRADED": 0, "REJECTED": 0, "MISSING": 0}

    for track in stage3.tracks:
        observations = []
        for observation in track.observations:
            if not lower <= observation.frame_index <= upper:
                continue
            projected = _project_observation(
                observation,
                cameras.by_frame(observation.frame_index),
                profile,
                cfg,
                width,
                height,
            )
            observations.append(projected)
        selected_observation = next(
            (item for item in observations if item["frame_index"] == stage3.selected_frame),
            None,
        )
        selected_status = "MISSING" if selected_observation is None else selected_observation["status"]
        status_counts[selected_status] += 1
        record = {
            "track_id": track.track_id,
            "upstream_role": track.role,
            "upstream_identity_confidence": track.identity_confidence,
            "status": selected_status,
            "selected_frame_proxy_status": selected_status,
            "observations": observations,
        }
        track_outputs.append(record)
        if selected_observation is not None:
            selected_outputs.append({
                "track_id": track.track_id,
                "upstream_role": track.role,
                "status": selected_status,
                "observation": selected_observation,
            })

    selected_camera = cameras.by_frame(stage3.selected_frame)
    pitch = {} if selected_camera is None else dict(selected_camera.pitch)
    state: Dict[str, Any] = {
        "schema_version": OUTPUT_SCHEMA,
        "stage4_version": STAGE4_VERSION,
        "mission": "POSE_AWARE_METRIC_BODY_PROJECTION",
        "method": {
            "name": PROJECTION_METHOD,
            "description": "Intersect each calibrated Stage-1 camera ray with the keypoint's canonical height plane.",
            "pretrained_model": None,
            "temporal_3d_optimization": False,
            "full_3d_pose_reconstruction": False,
            "raw_proxy_downstream_policy": "audit_only_not_exported_to_stage8",
        },
        "configuration": cfg.to_dict(),
        "height_profile": _profile_metadata(profile, cfg.reference_player_height_m),
        "source": {
            "stage3_state": str(stage3.path),
            "stage3_state_sha256": _sha256(stage3.path),
            "camera_state_directory": cameras.source_dir,
        },
        "replay_context": dict(stage3.replay_context),
        "coordinate_frame": {
            "units": "metres",
            "origin": pitch.get("origin", "center"),
            "x_axis": pitch.get("x_axis", "goal_to_goal"),
            "y_axis": pitch.get("y_axis", "touchline_to_touchline"),
            "z_axis": pitch.get("z_axis", "up"),
            "pitch": pitch,
        },
        "preflight": preflight,
        "stage3_bridge_qa": _neutral_stage3_bridge_qa(stage3),
        "selected_frame_status_counts": status_counts,
        "tracks": track_outputs,
        "selected_frame_proxies": selected_outputs,
        "limitations": [
            "Canonical height planes are engineering priors, not measured player anatomy.",
            "Outputs are body proxies, not true 3D joint reconstructions.",
            "Camera uncertainty and player-specific height uncertainty are not calibrated here.",
            "Legal-body membership, team, attack direction and offside decisions belong downstream.",
            "Only robust ground anchors are exported downstream; raw upper-body height-plane proxies remain audit evidence.",
        ],
        "artifacts": {
            "metric_body_proxy_state": str(state_path),
            "stage4_downstream_handoff": str(handoff_path),
            "selected_frame_topdown": None,
        },
    }
    metrics = evaluate_proxy_state(state)
    state["metrics"] = metrics
    state["acceptance_gate"] = structural_gate(state, metrics)

    handoff = _build_handoff(state, state_path)
    _write_json(handoff_path, handoff)
    if cfg.export_visualization:
        try:
            from .proxy_visualization import save_proxy_topdown_selected_frame

            save_proxy_topdown_selected_frame(state, visualization_path)
            state["artifacts"]["selected_frame_topdown"] = str(visualization_path)
        except Exception as exc:
            state["visualization_warning"] = (
                f"top-down PNG was skipped: {type(exc).__name__}: {exc}"
            )
    _write_json(state_path, state)
    return _json_safe(state)
