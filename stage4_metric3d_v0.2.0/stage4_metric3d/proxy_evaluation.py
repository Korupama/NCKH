from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping
import math
import numpy as np

from .proxy_quality import proxy_points


GROUP_BY_NAME = {
    "nose": "head",
    "left_eye": "head", "right_eye": "head",
    "left_ear": "head", "right_ear": "head",
    "left_shoulder": "shoulder", "right_shoulder": "shoulder",
    "left_elbow": "arm", "right_elbow": "arm",
    "left_wrist": "arm", "right_wrist": "arm",
    "left_hip": "hip", "right_hip": "hip",
    "left_knee": "knee", "right_knee": "knee",
    "left_ankle": "ankle", "right_ankle": "ankle",
    "left_big_toe": "foot", "left_small_toe": "foot", "left_heel": "foot",
    "right_big_toe": "foot", "right_small_toe": "foot", "right_heel": "foot",
}


def anatomical_group(name: str) -> str:
    return GROUP_BY_NAME[name]


def _coverage(projected: int, eligible: int) -> Dict[str, Any]:
    return {
        "projected": int(projected),
        "eligible": int(eligible),
        "rate": None if eligible == 0 else float(projected / eligible),
    }


def _distribution(values: Iterable[float]) -> Dict[str, Any]:
    arr = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return {"count": 0, "median": None, "p95": None, "max": None}
    return {
        "count": int(arr.size),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def iter_observations(state: Mapping[str, Any]):
    for track in state.get("tracks", []):
        for observation in track.get("observations", []):
            yield track, observation


def evaluate_proxy_state(state: Mapping[str, Any]) -> Dict[str, Any]:
    selected_frame = int((state.get("replay_context") or {}).get("selected_frame", -1))
    total_eligible = total_projected = 0
    t0_eligible = t0_projected = 0
    t0_group = {group: [0, 0] for group in ("head", "shoulder", "arm", "hip", "knee", "ankle", "foot")}
    reprojection = []
    pitch_plausible = 0
    pitch_projected = 0
    sensitivity = {group: [] for group in ("overall", "head", "shoulder", "arm", "hip", "knee", "ankle", "foot")}
    compactness_values = {
        "longitudinal_span_m": [],
        "transverse_span_m": [],
        "planar_diameter_m": [],
        "foot_to_upper_center_m": [],
    }
    compactness_counts = {"VALID": 0, "DEGRADED": 0, "REJECTED": 0, "MISSING": 0}
    compactness_per_track = []
    ground_counts = {"VALID": 0, "DEGRADED": 0, "REJECTED": 0, "MISSING": 0}
    raw_tracks_at_t0 = set()
    ground_tracks_at_t0 = set()

    for track, observation in iter_observations(state):
        at_t0 = int(observation.get("frame_index", -1)) == selected_frame
        points = proxy_points(observation)
        if at_t0 and any(point.get("xyz_proxy_world_m") is not None for point in points):
            raw_tracks_at_t0.add(str(track.get("track_id")))
        for point in points:
            eligible = point.get("stage3_uv_raw_distorted_px") is not None and point.get("stage3_keypoint_state") != "MISSING"
            projected = point.get("xyz_proxy_world_m") is not None
            group = str(point.get("anatomical_group", anatomical_group(str(point["name"]))))
            total_eligible += int(eligible)
            total_projected += int(projected)
            if at_t0:
                t0_eligible += int(eligible)
                t0_projected += int(projected)
                t0_group[group][1] += int(eligible)
                t0_group[group][0] += int(projected)
            if projected:
                qa = point.get("qa") or {}
                if qa.get("reprojection_error_px") is not None:
                    reprojection.append(float(qa["reprojection_error_px"]))
                pitch_projected += 1
                pitch_plausible += int(bool(qa.get("pitch_plausible")))
                if at_t0 and qa.get("longitudinal_height_sensitivity_m") is not None:
                    value = float(qa["longitudinal_height_sensitivity_m"])
                    sensitivity["overall"].append(value)
                    sensitivity[group].append(value)

        if at_t0:
            compactness = observation.get("compactness") or {}
            compactness_status = str(compactness.get("status", "MISSING"))
            if compactness_status not in compactness_counts:
                compactness_status = "MISSING"
            compactness_counts[compactness_status] += 1
            for key in compactness_values:
                if compactness.get(key) is not None:
                    compactness_values[key].append(float(compactness[key]))
            compactness_per_track.append({
                "track_id": str(track.get("track_id")),
                "status": compactness_status,
                "reasons": compactness.get("reasons") or [],
                **{key: compactness.get(key) for key in compactness_values},
            })
            ground = observation.get("ground_anchor") or {}
            ground_status = str(ground.get("status", "MISSING"))
            if ground_status not in ground_counts:
                ground_status = "MISSING"
            ground_counts[ground_status] += 1
            if ground_status in {"VALID", "DEGRADED"} and ground.get("xyz_ground_m") is not None:
                ground_tracks_at_t0.add(str(track.get("track_id")))

    tracks = list(state.get("tracks", []))
    available_players = sum(
        str(track.get("selected_frame_proxy_status")) in {"VALID", "DEGRADED"}
        for track in tracks
    )
    return {
        "ProjectionCoverage": _coverage(total_projected, total_eligible),
        "ProjectionCoverageAtT0": _coverage(t0_projected, t0_eligible),
        "ProjectionCoverageAtT0ByGroup": {
            group: _coverage(counts[0], counts[1]) for group, counts in t0_group.items()
        },
        "PlayerMetricProxyCoverageAtT0": {
            "available_tracks": int(available_players),
            "candidate_tracks": int(len(tracks)),
            "rate": None if not tracks else float(available_players / len(tracks)),
            "note": "Available means the selected-frame body proxy is VALID/DEGRADED after compactness checks. Denominator is every Stage-3 track; downstream role filtering belongs to Stage 5/7.",
        },
        "RawHeightPlaneProjectionTrackCoverageAtT0": {
            "available_tracks": len(raw_tracks_at_t0),
            "candidate_tracks": int(len(tracks)),
            "rate": None if not tracks else float(len(raw_tracks_at_t0) / len(tracks)),
            "downstream_eligible": False,
        },
        "GroundAnchorCoverageAtT0": {
            "available_tracks": len(ground_tracks_at_t0),
            "candidate_tracks": int(len(tracks)),
            "rate": None if not tracks else float(len(ground_tracks_at_t0) / len(tracks)),
            "status_counts": ground_counts,
        },
        "BodyProxyCompactnessAtT0": {
            "status_counts": compactness_counts,
            "longitudinal_span_m": _distribution(compactness_values["longitudinal_span_m"]),
            "transverse_span_m": _distribution(compactness_values["transverse_span_m"]),
            "planar_diameter_m": _distribution(compactness_values["planar_diameter_m"]),
            "foot_to_upper_center_m": _distribution(compactness_values["foot_to_upper_center_m"]),
            "per_track": compactness_per_track,
            "thresholds_are_provisional": True,
        },
        "ReprojectionConsistencyPx": _distribution(reprojection),
        "PitchPlausibilityRate": {
            "plausible": int(pitch_plausible),
            "projected": int(pitch_projected),
            "rate": None if pitch_projected == 0 else float(pitch_plausible / pitch_projected),
        },
        "LongitudinalHeightSensitivity": {
            "definition": "max absolute change in proxy X under reference player height H +/- delta",
            **{group: _distribution(values) for group, values in sensitivity.items()},
        },
    }


def structural_gate(state: Mapping[str, Any], metrics: Mapping[str, Any]) -> Dict[str, Any]:
    preflight_ready = bool((state.get("preflight") or {}).get("ready"))
    t0 = metrics["ProjectionCoverageAtT0"]
    reproj = metrics["ReprojectionConsistencyPx"]
    implementation_checks = [
        {"name": "preflight_ready", "passed": preflight_ready, "observed": preflight_ready},
        {"name": "selected_frame_has_projected_points", "passed": int(t0["projected"]) > 0, "observed": int(t0["projected"])},
        {"name": "selected_frame_projection_complete_for_eligible_points", "passed": t0["rate"] == 1.0, "observed": t0["rate"], "limit": 1.0},
        {"name": "reprojection_p95_px", "passed": reproj["p95"] is not None and float(reproj["p95"]) <= 1e-5, "observed": reproj["p95"], "limit": 1e-5},
    ]
    raw_tracks = int(metrics["RawHeightPlaneProjectionTrackCoverageAtT0"]["available_tracks"])
    usable_body_tracks = int(metrics["PlayerMetricProxyCoverageAtT0"]["available_tracks"])
    ground_tracks = int(metrics["GroundAnchorCoverageAtT0"]["available_tracks"])
    quality_checks = [
        {
            "name": "all_raw_projected_tracks_pass_body_compactness",
            "passed": raw_tracks > 0 and usable_body_tracks == raw_tracks,
            "observed_usable_body_tracks": usable_body_tracks,
            "observed_raw_projected_tracks": raw_tracks,
        },
        {
            "name": "safe_ground_anchor_for_each_raw_projected_track",
            "passed": raw_tracks > 0 and ground_tracks == raw_tracks,
            "observed_ground_anchor_tracks": ground_tracks,
            "observed_raw_projected_tracks": raw_tracks,
        },
    ]
    implementation_pass = all(check["passed"] for check in implementation_checks)
    quality_pass = all(check["passed"] for check in quality_checks)
    return {
        "status": "PASS" if implementation_pass and quality_pass else "FAIL",
        "implementation_status": "PASS" if implementation_pass else "FAIL",
        "body_proxy_quality_status": "PASS" if quality_pass else "FAIL",
        "checks": implementation_checks + quality_checks,
        "downstream_readiness": {
            "stage5_ground_state_ready": raw_tracks > 0 and ground_tracks == raw_tracks,
            "stage8_legal_body_ready": False,
            "stage8_blocker": "upper-body proxy accuracy and physical coherence are not frozen",
        },
        "research_accuracy_frozen": False,
        "note": "Implementation consistency and physical compactness are separate; neither establishes real-world offside accuracy.",
    }
