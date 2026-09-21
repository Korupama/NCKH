"""Selected-frame availability and the stable Stage-4 to Stage-5 handoff."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence
import math


CORE_ANCHOR_NAMES = frozenset({
    "left_shoulder", "right_shoulder", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
})

LEGAL_ANCHOR_GROUPS = {
    "head": ["nose", "left_eye", "right_eye", "left_ear", "right_ear"],
    "torso": ["left_shoulder", "right_shoulder", "left_hip", "right_hip"],
    "left_leg": ["left_hip", "left_knee", "left_ankle"],
    "right_leg": ["right_hip", "right_knee", "right_ankle"],
    "left_foot": ["left_ankle", "left_big_toe", "left_small_toe", "left_heel"],
    "right_foot": ["right_ankle", "right_big_toe", "right_small_toe", "right_heel"],
}

FOOT_PRIMARY_ANCHOR = {
    "left": "left_distal_foot_xyz_m",
    "right": "right_distal_foot_xyz_m",
    "fallback": "left_foot_proxy_xyz_m or right_foot_proxy_xyz_m",
}


def _finite_xyz(value: object) -> bool:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        return False
    try:
        return all(math.isfinite(float(v)) for v in value)
    except (TypeError, ValueError):
        return False


def selected_frame_availability(observation: Mapping[str, Any] | None, track_quality: str) -> dict[str, Any]:
    """Classify the actual availability at t0, independently of track convergence."""
    joints = [] if observation is None else list(observation.get("metric_pose23") or [])
    finite_names = [str(joint.get("name")) for joint in joints if _finite_xyz(joint.get("xyz_world_m"))]
    legal_names = [
        str(joint.get("name")) for joint in joints
        if bool(joint.get("candidate_for_legal_body_geometry")) and _finite_xyz(joint.get("xyz_world_m"))
    ]
    core_names = sorted(set(finite_names).intersection(CORE_ANCHOR_NAMES))

    if not finite_names:
        status, reason = "MISSING", "no_finite_metric_xyz_at_selected_frame"
    elif not legal_names:
        status, reason = "DEGRADED", "no_finite_legal_anchor_at_selected_frame"
    elif len(core_names) < 4:
        status, reason = "DEGRADED", "incomplete_core_metric_anchors_at_selected_frame"
    elif track_quality in {"VALID", "DEGRADED"}:
        status, reason = track_quality, "selected_frame_metric_pose_available"
    else:
        status, reason = "REJECTED", "track_geometry_not_accepted"

    return {
        "status": status,
        "reason": reason,
        "finite_metric_joint_count": len(finite_names),
        "finite_legal_anchor_count": len(legal_names),
        "finite_core_anchor_count": len(core_names),
        "finite_metric_joint_names": finite_names,
        "finite_core_anchor_names": core_names,
    }


def selected_observation(track: Mapping[str, Any], selected_frame: int) -> Mapping[str, Any] | None:
    for observation in track.get("observations") or []:
        if int(observation.get("frame_index", -1)) == int(selected_frame):
            return observation
    return None


def _track_handoff_entry(track: Mapping[str, Any], selected_frame: int) -> dict[str, Any]:
    observation = selected_observation(track, selected_frame)
    track_quality = str(track.get("selected_frame_pose_status", "MISSING"))
    availability = dict(track.get("selected_frame_availability") or selected_frame_availability(observation, track_quality))
    status = str(availability["status"])
    optimizer = dict(track.get("optimizer") or {})
    stage5_eligible = status in {"VALID", "DEGRADED"} and availability["finite_legal_anchor_count"] > 0

    blocking_qa: list[str] = []
    non_blocking_qa: list[str] = []
    if availability["finite_metric_joint_count"] == 0:
        blocking_qa.append("NO_METRIC_POSE_AT_T0")
    elif availability["finite_core_anchor_count"] < 4:
        blocking_qa.append("INSUFFICIENT_CORE_METRIC_ANCHORS_AT_T0")
    if status == "REJECTED":
        blocking_qa.append("SELECTED_FRAME_GEOMETRY_REJECTED")
    if status == "DEGRADED":
        non_blocking_qa.append("SELECTED_FRAME_DEGRADED")
    if str(track.get("selected_frame_stage3_pose_status", "VALID")) == "MISSING":
        non_blocking_qa.append("STAGE3_RAW_POSE_MISSING_AT_T0")

    uncertainty = dict(track.get("selected_frame_uncertainty") or {})
    uncertainty_status = str(uncertainty.get("status", "NOT_RUN"))
    longitudinal = uncertainty.get("longitudinal") or {}
    compact_uncertainty = {
        "status": uncertainty_status,
        "scope": uncertainty.get("scope"),
        "calibrated": bool(uncertainty.get("calibrated", False)),
        "samples_requested": uncertainty.get("samples_requested", 0),
        "samples_usable": uncertainty.get("samples_usable", uncertainty.get("samples_successful", 0)),
        "usable_fraction": uncertainty.get("usable_fraction"),
        "legal_extrema_x": longitudinal.get("legal_extrema_x"),
    }
    if stage5_eligible:
        if uncertainty_status not in {"OK", "DEGRADED"}:
            non_blocking_qa.append("LONGITUDINAL_UNCERTAINTY_UNAVAILABLE")
        elif uncertainty_status == "DEGRADED":
            non_blocking_qa.append("LONGITUDINAL_UNCERTAINTY_DEGRADED")
        if uncertainty_status in {"OK", "DEGRADED"} and not compact_uncertainty["calibrated"]:
            non_blocking_qa.append("UNCERTAINTY_NOT_CALIBRATED")

    return {
        "track_id": track.get("track_id"),
        "upstream_role": track.get("upstream_role"),
        "selected_frame_pose_status": status,
        "stage5_eligible": stage5_eligible,
        "selected_frame_availability": availability,
        "legal_anchor_groups": LEGAL_ANCHOR_GROUPS,
        "foot_primary_anchor": FOOT_PRIMARY_ANCHOR,
        "metric_pose_path": f"tracks[{track.get('track_id')}].observations[frame_index={selected_frame}].metric_pose23",
        "refinement_evidence": {
            "track_optimizer_status": track.get("track_optimizer_status", optimizer.get("status")),
            "optimizer_success": optimizer.get("success"),
            "termination_reason": optimizer.get("termination_reason"),
            "geometry_quality_gate": (optimizer.get("diagnostics") or {}).get("geometry_quality_gate"),
            "temporal_frame_count": (track.get("temporal_window") or {}).get("frame_count"),
            "initializer_used": (track.get("initializer") or {}).get("used"),
        },
        "longitudinal_uncertainty": compact_uncertainty,
        "qa": {"blocking": blocking_qa, "non_blocking": non_blocking_qa},
    }


def build_stage5_handoff(state: Mapping[str, Any], state_path: str) -> dict[str, Any]:
    selected_frame = int((state.get("replay_context") or {}).get("selected_frame"))
    tracks = [_track_handoff_entry(track, selected_frame) for track in state.get("tracks") or []]
    status = {entry["track_id"]: entry["selected_frame_pose_status"] for entry in tracks}
    return {
        "schema_version": "stage4-to-stage5-handoff-1.2",
        "metric_pose_3d_state": state_path,
        "selected_frame": selected_frame,
        "world_frame": state.get("world_frame"),
        "pose_schema": (state.get("pose_schema") or {}).get("name", "METRIC_POSE_23"),
        "legal_anchor_groups": LEGAL_ANCHOR_GROUPS,
        "foot_primary_anchor": FOOT_PRIMARY_ANCHOR,
        "tracks": tracks,
        "selected_track_ids": [entry["track_id"] for entry in tracks],
        "stage5_eligible_track_ids": [entry["track_id"] for entry in tracks if entry["stage5_eligible"]],
        "valid_track_ids": [track_id for track_id, value in status.items() if value == "VALID"],
        "degraded_track_ids": [track_id for track_id, value in status.items() if value == "DEGRADED"],
        "rejected_or_missing_track_ids": [track_id for track_id, value in status.items() if value in {"REJECTED", "MISSING"}],
        "uncertainty_contract": {
            "quantity": "camera-fixed Stage-3 pixel sensitivity of pitch-longitudinal legal anchors",
            "attack_direction_applied": False,
            "calibrated": False,
            "consumer_policy": "Stage 5 selects min_legal_x or max_legal_x only after attack direction is known",
        },
        "stage5_responsibility": "construct legal-body volumes/surfaces and goalward extent; do not use arm branch below shoulder",
    }


def recompute_selected_frame_metrics(state: dict[str, Any]) -> None:
    selected_frame = int((state.get("replay_context") or {}).get("selected_frame"))
    counts: Counter[str] = Counter()
    selected_outputs = {entry.get("track_id"): entry for entry in state.get("selected_frame_poses") or []}
    for track in state.get("tracks") or []:
        observation = selected_observation(track, selected_frame)
        track_quality = str(track.get("selected_frame_pose_status", "MISSING"))
        availability = selected_frame_availability(observation, track_quality)
        track["track_optimizer_status"] = (track.get("optimizer") or {}).get("status")
        track["selected_frame_availability"] = availability
        track["selected_frame_pose_status"] = availability["status"]
        selected_output = selected_outputs.get(track.get("track_id"))
        if selected_output is not None:
            selected_output["selected_frame_pose_status"] = availability["status"]
            selected_output["selected_frame_availability"] = availability
            observation_out = selected_output.get("observation")
            if isinstance(observation_out, dict):
                observation_out["quality"] = availability["status"]
        counts[availability["status"]] += 1
    metrics = state.setdefault("metrics", {})
    total = len(state.get("tracks") or [])
    metrics["MetricPoseCoverageAtT0_given_stage3_candidate"] = (counts["VALID"] + counts["DEGRADED"]) / max(total, 1)
    metrics["selected_frame_status_counts"] = {key: counts[key] for key in ("VALID", "DEGRADED", "REJECTED", "MISSING")}
