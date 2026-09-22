from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import json
import hashlib
import numpy as np

from .schemas import Stage3Config
from .stage2_adapter import Stage2Bundle, load_stage2_bundle
from .wholebody133 import WHOLEBODY_KEYPOINT_NAMES, keypoint_records_to_arrays
from .quality import evaluate_pose, pose_selection_score, status_rank, summarize_t0_anatomy_evidence
from .temporal import annotate_temporal

STAGE3_VERSION = "stage3-pose2d-0.1.0"
OUTPUT_SCHEMA = "tracked-pose-2d-state-1.0"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _entity_bbox_index(track: Mapping[str, Any]) -> Dict[int, Dict[str, Any]]:
    return {int(o["frame_index"]): o for o in track.get("observations", [])}


def _convert_cache_observation(track_id: str, cache_obs: Mapping[str, Any], entity_obs: Mapping[str, Any], config: Stage3Config) -> Dict[str, Any]:
    pose = cache_obs.get("pose") or {}
    records = pose.get("keypoints") or []
    xy, scores = keypoint_records_to_arrays(records)
    bbox = entity_obs.get("bbox_xyxy") or cache_obs.get("bbox_xyxy") or pose.get("bbox_xyxy")
    if not bbox:
        raise ValueError(f"No bbox for {track_id} frame {cache_obs.get('frame_index')}")
    qa, kp_records = evaluate_pose(xy, scores, bbox, config)
    return {
        "frame_index": int(cache_obs["frame_index"]),
        "pose_cache_key": cache_obs.get("pose_cache_key"),
        "physical_human_id": cache_obs.get("physical_human_id"),
        "source_bbox_xyxy": [float(x) for x in bbox],
        "source_expanded_pose_bbox_xyxy": pose.get("expanded_pose_bbox_xyxy"),
        "source_backend": pose.get("backend", "rtmw_unknown"),
        "source_score_semantics": pose.get("score_semantics", "RTMW_RAW_SIMCC_MAX_NOT_CALIBRATED_PROBABILITY"),
        "keypoints_133": kp_records,
        "qa": qa,
        "pose_status": qa["pose_status"],
        "provenance": {"source": "STAGE2_RTMW_CACHE", "re_inferred": False},
    }


def _missing_pose_observation(frame_index: int, entity_obs: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "frame_index": int(frame_index),
        "pose_cache_key": entity_obs.get("pose_cache_key"),
        "physical_human_id": entity_obs.get("physical_human_id"),
        "source_bbox_xyxy": [float(x) for x in entity_obs.get("bbox_xyxy", [])],
        "source_expanded_pose_bbox_xyxy": None,
        "source_backend": None,
        "source_score_semantics": None,
        "keypoints_133": [
            {
                "index": i,
                "name": name,
                "x": None,
                "y": None,
                "raw_model_score": None,
                "state": "MISSING",
                "source": "NO_RTMW_CACHE",
                "coordinate_evidence_kind": "RAW_MISSING",
                "temporal_estimate_xy": None,
            }
            for i, name in enumerate(WHOLEBODY_KEYPOINT_NAMES)
        ],
        "qa": {"pose_status": "MISSING", "reason": "no_rtmw_cache_observation"},
        "pose_status": "MISSING",
        "provenance": {"source": "NO_RTMW_CACHE", "re_inferred": False},
    }


def _fallback_reinfer(observation: Dict[str, Any], frame, model, config: Stage3Config) -> Dict[str, Any]:
    """Try controlled re-crops without losing the Stage-2 raw evidence.

    If a re-inferred pose wins, ``keypoints_133`` becomes the accepted Stage-3
    pose while ``upstream_raw_pose`` preserves the exact pre-fallback pose/QA.
    The replacement is therefore explicit and auditable, never silent.
    """
    original_score = pose_selection_score(observation.get("qa") or {})
    original_snapshot = {
        "keypoints_133": observation.get("keypoints_133"),
        "qa": observation.get("qa"),
        "pose_status": observation.get("pose_status"),
        "source_backend": observation.get("source_backend"),
        "source_score_semantics": observation.get("source_score_semantics"),
        "provenance": observation.get("provenance"),
    }
    best = observation
    best_score = original_score
    best_scale = None
    bbox = observation["source_bbox_xyxy"]
    attempts = []
    candidates = []
    for crop_scale in config.fallback_crop_scales:
        result = model.infer_one(frame, bbox, crop_scale=float(crop_scale))
        qa, kps = evaluate_pose(result.keypoints_xy, result.scores, bbox, config)
        score = pose_selection_score(qa)
        attempts.append({"crop_scale": float(crop_scale), "pose_status": qa["pose_status"], "selection_score": score})
        candidate = dict(observation)
        candidate["keypoints_133"] = kps
        for kp in candidate["keypoints_133"]:
            kp["source"] = "RTMW_REINFERENCE"
        candidate["qa"] = qa
        candidate["pose_status"] = qa["pose_status"]
        candidates.append((candidate, score, float(crop_scale)))
        if (status_rank(qa["pose_status"]), score) > (status_rank(best.get("pose_status")), best_score):
            best, best_score, best_scale = candidate, score, float(crop_scale)
    if best is not observation:
        best["upstream_raw_pose"] = original_snapshot
        best["provenance"] = {
            "source": "RTMW_REINFERENCE",
            "re_inferred": True,
            "selected_crop_scale": best_scale,
            "attempts": attempts,
            "raw_cache_preserved": True,
        }
    else:
        best = dict(best)
        best["provenance"] = dict(best.get("provenance") or {})
        best["provenance"]["fallback_attempts"] = attempts
        best["provenance"]["fallback_selected"] = False
    return best


def run_stage3(
    *,
    stage2_dir: str | Path | None = None,
    entity_state: str | Path | None = None,
    rtmw_cache: str | Path | None = None,
    handoff: str | Path | None = None,
    output_dir: str | Path,
    config: Optional[Stage3Config] = None,
) -> Dict[str, Any]:
    config = config or Stage3Config()
    bundle = load_stage2_bundle(stage2_dir=stage2_dir, entity_state=entity_state, rtmw_cache=rtmw_cache, handoff=handoff, strict=True)
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    tracks = bundle.track_by_id()
    cache_tracks = bundle.cache_track_by_id()
    selected_frame = bundle.selected_frame
    fallback_model = None
    video_path = str(bundle.replay_context.get("video_path", ""))
    if config.enable_fallback_reinference:
        if not config.rtmw_model:
            raise ValueError("Fallback re-inference requires config.rtmw_model")
        from .rtmw_onnx import RTMWOpenCVDNN
        fallback_model = RTMWOpenCVDNN(
            config.rtmw_model,
            input_width=config.rtmw_input_width,
            input_height=config.rtmw_input_height,
            device=config.rtmw_device,
        )
        if not Path(video_path).is_file():
            raise FileNotFoundError(f"Fallback requires replay_context.video_path; not found: {video_path}")

    output_tracks: List[Dict[str, Any]] = []
    selected_poses: List[Dict[str, Any]] = []
    counts = {"VALID": 0, "DEGRADED": 0, "REJECTED": 0, "MISSING": 0}
    fallback_frames_cache: Dict[int, Any] = {}

    for tid in bundle.candidate_track_ids:
        entity_track = tracks[tid]
        entity_idx = _entity_bbox_index(entity_track)
        raw_cache_obs = {int(x["frame_index"]): x for x in cache_tracks.get(tid, {}).get("observations", [])}
        pose_obs: List[Dict[str, Any]] = []
        for frame_index, eobs in sorted(entity_idx.items()):
            if int(eobs.get("frame_index", frame_index)) != frame_index:
                continue
            cobs = raw_cache_obs.get(frame_index)
            if cobs is None:
                obs = _missing_pose_observation(frame_index, eobs)
            else:
                obs = _convert_cache_observation(tid, cobs, eobs, config)
            if fallback_model is not None and obs["pose_status"] in ("MISSING", "DEGRADED", "REJECTED"):
                if frame_index not in fallback_frames_cache:
                    from .video import read_frame
                    fallback_frames_cache[frame_index] = read_frame(video_path, frame_index)
                obs = _fallback_reinfer(obs, fallback_frames_cache[frame_index], fallback_model, config)
            pose_obs.append(obs)

        pose_obs, temporal_summary = annotate_temporal(pose_obs, config)
        selected = next((o for o in pose_obs if int(o["frame_index"]) == selected_frame), None)
        selected_status = "MISSING" if selected is None else str(selected["pose_status"])
        t0_anatomy_evidence = None if selected is None else summarize_t0_anatomy_evidence(
            selected.get("keypoints_133") or [],
            pose_status=selected_status,
        )
        counts[selected_status] = counts.get(selected_status, 0) + 1
        tr_out = {
            "track_id": tid,
            "upstream_role": entity_track.get("role"),
            "upstream_role_score": entity_track.get("role_score"),
            "upstream_identity_confidence": entity_track.get("identity_confidence"),
            "candidate_for_stage3": True,
            "selected_frame_pose_status": selected_status,
            "t0_anatomy_evidence": t0_anatomy_evidence,
            "temporal_qa": temporal_summary,
            "observations": pose_obs,
        }
        output_tracks.append(tr_out)
        if selected is not None:
            selected_poses.append({"track_id": tid, "upstream_role": entity_track.get("role"), "observation": selected})

    candidate_n = len(bundle.candidate_track_ids)
    selected_available = sum(x["selected_frame_pose_status"] != "MISSING" for x in output_tracks)
    selected_accepted = sum(x["selected_frame_pose_status"] in ("VALID", "DEGRADED") for x in output_tracks)
    selected_valid = sum(x["selected_frame_pose_status"] == "VALID" for x in output_tracks)
    feet_good = 0
    t0_evidence = [x.get("t0_anatomy_evidence") for x in output_tracks if x.get("t0_anatomy_evidence")]
    for x in output_tracks:
        obs = next((o for o in x["observations"] if int(o["frame_index"]) == selected_frame), None)
        if obs and float((obs.get("qa") or {}).get("feet_completeness", 0.0)) >= config.min_feet_completeness_valid:
            feet_good += 1

    anatomy_coverage = {
        group: float(sum(
            bool((entry.get("groups") or {}).get(group, {}).get("all_raw_observed"))
            for entry in t0_evidence
        ) / max(1, candidate_n))
        for group in ("head", "torso", "left_leg", "right_leg", "left_foot", "right_foot")
    }
    complete_stage4_core = sum(
        bool((entry.get("stage4_core_metric_anchors") or {}).get("all_raw_observed"))
        for entry in t0_evidence
    )
    temporal_estimate_tracks = sum(
        bool(entry.get("temporal_estimate_only_keypoint_names"))
        for entry in t0_evidence
    )

    state: Dict[str, Any] = {
        "schema_version": OUTPUT_SCHEMA,
        "stage3_version": STAGE3_VERSION,
        "source_stage2": {
            "schema_version": bundle.entity_state.get("schema_version"),
            "stage2_version": bundle.entity_state.get("stage2_version"),
            "entity_state_path": str(bundle.entity_state_path),
            "rtmw_cache_path": str(bundle.rtmw_cache_path),
            "stage3_handoff_path": str(bundle.handoff_path),
            "entity_state_sha256": _sha256(bundle.entity_state_path),
            "rtmw_cache_sha256": _sha256(bundle.rtmw_cache_path),
        },
        "replay_context": bundle.replay_context,
        "coordinate_space": "RAW_DISTORTED_PIXEL",
        "keypoint_schema": {"name": "COCO_WHOLEBODY_133", "count": 133, "names": list(WHOLEBODY_KEYPOINT_NAMES)},
        "score_semantics": {
            "raw_model_score": "RTMW_SIMCC_MAX",
            "is_probability": False,
            "quality_policy": "relative score evidence + geometry + temporal diagnostics; no probability claim",
        },
        "configuration": config.to_dict(),
        "preflight": bundle.validation,
        "tracks": output_tracks,
        "selected_frame_poses": selected_poses,
        "metrics": {
            "candidate_tracks": candidate_n,
            "PoseCoverageAtT0_given_stage2_candidate": float(selected_available / max(1, candidate_n)),
            "AcceptedPoseCoverageAtT0_given_stage2_candidate": float(selected_accepted / max(1, candidate_n)),
            "ValidPoseCoverageAtT0_given_stage2_candidate": float(selected_valid / max(1, candidate_n)),
            "FootPoseCoverageAtT0_given_stage2_candidate": float(feet_good / max(1, candidate_n)),
            "T0AnatomyEvidenceCoverage_given_stage2_candidate": {
                "meaning": "raw-observed 2D evidence coverage for downstream World-State consumers; not ground-truth accuracy or Law-11 legal-body coverage",
                "complete_group_coverage": anatomy_coverage,
                "complete_stage4_core_metric_anchor_coverage": float(complete_stage4_core / max(1, candidate_n)),
                "tracks_with_temporal_estimate_only_at_t0": int(temporal_estimate_tracks),
            },
            "selected_frame_status_counts": counts,
            "upstream_stage2_candidate_recall_not_recomputed": True,
        },
        "acceptance_gate": {
            "structural": {
                "stage2_preflight_ready": bool(bundle.validation.get("ready")),
                "coordinate_space_raw_distorted_pixel": bundle.validation.get("coordinate_space") == "RAW_DISTORTED_PIXEL",
                "wholebody_schema_count_133": True,
                "ambiguous_track_frame_pose_mapping": bool(bundle.validation.get("ambiguous_track_frame_pose_mapping")),
                "silent_coordinate_transform": False,
                "silent_temporal_imputation": False,
            },
            "engineering": {
                "PoseCoverageAtT0_given_stage2_candidate>=0.98": bool(float(selected_available / max(1, candidate_n)) >= 0.98),
            },
            "structural_passed": bool(bundle.validation.get("ready")),
            "research_accuracy_frozen": False,
            "note": "Structural readiness is frozen in v0.1; football-specific accuracy thresholds remain provisional until real 3DSP/COCO/Pose23 baselines are collected.",
        },
        "diagnostics": {
            "silent_coordinate_transform": False,
            "silent_temporal_imputation": False,
            "legal_body_semantics_applied": False,
            "team_semantics_applied": False,
            "world_geometry_applied": False,
        },
        "artifacts": {},
    }

    state_path = out / "tracked_pose_2d_state.json"
    state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    downstream = {
        "schema_version": "stage3-downstream-handoff-1.0",
        "tracked_pose_2d_state": str(state_path),
        "selected_frame": selected_frame,
        "coordinate_space": "RAW_DISTORTED_PIXEL",
        "keypoint_schema": "COCO_WHOLEBODY_133",
        "t0_anatomy_evidence_path": "tracks[].t0_anatomy_evidence",
        "observed_coordinate_policy": "Only keypoints with coordinate_evidence_kind=RAW_OBSERVED are raw observed pixels; TEMPORAL_ESTIMATE_ONLY never replaces x/y.",
        "valid_track_ids": [x["track_id"] for x in output_tracks if x["selected_frame_pose_status"] == "VALID"],
        "degraded_track_ids": [x["track_id"] for x in output_tracks if x["selected_frame_pose_status"] == "DEGRADED"],
        "rejected_track_ids": [x["track_id"] for x in output_tracks if x["selected_frame_pose_status"] in ("REJECTED", "MISSING")],
        "note": "Downstream stages must preserve raw image coordinates and must not interpret Stage-3 anatomy as IFAB legal-body semantics.",
    }
    handoff_path = out / "stage3_downstream_handoff.json"
    handoff_path.write_text(json.dumps(downstream, indent=2, ensure_ascii=False), encoding="utf-8")
    state["artifacts"]["tracked_pose_2d_state"] = str(state_path)
    state["artifacts"]["downstream_handoff"] = str(handoff_path)

    # Optional visualization if replay video is available.
    if video_path and Path(video_path).is_file() and selected_poses:
        try:
            from .visualization import save_selected_frame_overlay
            overlay = save_selected_frame_overlay(video_path, selected_frame, selected_poses, out / "selected_frame_pose_qa.png")
            state["artifacts"]["selected_frame_pose_qa"] = str(overlay)
        except Exception as exc:
            state["diagnostics"]["visualization_error"] = repr(exc)
    state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    return state
